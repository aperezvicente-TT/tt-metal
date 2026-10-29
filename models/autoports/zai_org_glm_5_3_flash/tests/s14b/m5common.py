# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14b job helpers: the mHC arguments of an EXPLICIT layer (a long-lived process must never read a module-level
layer constant), device copies of weights, the L1 allocator state."""

from __future__ import annotations

import copy
import os

import torch

import ttnn


def mhc_args(source, cfg, layer: int, sub: str) -> dict:
    """MhcPre arguments of `layer`'s attention ("attn") or FFN ("ffn") hyper-connection (dev_mk_a / dev_mk_ffn)."""
    n = cfg.hc_mult
    norm = "input_layernorm.weight" if sub == "attn" else "post_attention_layernorm.weight"
    scale = source.host(layer, f"hc_{sub}_scale").float().reshape(-1)
    return dict(
        fn_t=source.host(layer, f"hc_{sub}_fn").reshape(n * cfg.hidden_size, (2 + n) * n),
        scale=torch.cat([scale[0].repeat(n), scale[1].repeat(n), scale[2].repeat(n * n)]),
        bias=source.host(layer, f"hc_{sub}_base").float().reshape(-1),
        norm_w=source.host(layer, norm).float().reshape(-1),
        hc_eps=cfg.hc_eps, x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps, iters=cfg.hc_sinkhorn_iters,
        hidden=cfg.hidden_size, n=n,
    )  # fmt: skip


def dup(mesh, t):
    """a device copy of t at a new address (host round trip of the packed storage: identical bytes)"""
    return ttnn.to_device(ttnn.from_device(t), mesh, memory_config=t.memory_config())


KDA_TENSORS = ("tap_cur", "taps_hist", "w_fb", "w_gb", "dt_bias", "decay_rate", "o_norm")


def clone_kda(mesh, kda):
    c = copy.copy(kda)
    for name in KDA_TENSORS:
        if isinstance(getattr(kda, name, None), ttnn.Tensor):
            setattr(c, name, dup(mesh, getattr(kda, name)))
    return c


def clone_lin(mesh, lin):
    c = copy.copy(lin)
    c.weight = dup(mesh, lin.weight)
    return c


def l1_state(mesh) -> dict:
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    blocks = [b for b in mv.block_table if str(b.get("allocated", "")).lower() in ("yes", "true", "1")]
    return dict(allocated_per_bank=int(mv.total_bytes_allocated_per_bank),
                lowest=min((int(b["address"]) for b in blocks), default=None), n_blocks=len(blocks))  # fmt: skip


KNOBS = ("GLM53_MK_KB_QKV", "GLM53_MK_KB_O", "GLM53_MK_RING_KB_QKV", "GLM53_MK_RING_KB_O", "GLM53_MK_RING_KB",
         "GLM53_MK_OPT", "GLM53_MK_READERS", "GLM53_MK_KB_ROUTER", "GLM53_MK_KB_SGU", "GLM53_MKX_RING",
         "GLM53_MKX_MBUF", "GLM53_MKX_INFLIGHT", "GLM53_MK_PARK", "GLM53_MK_PARK_SLOTS", "GLM53_MK_PARK_PHASES",
         "GLM53_MK_DEFER_INV", "GLM53_REC_EARLY_SIGNAL", "GLM53_REC_DEPTH", "GLM53_MK_FOLD_Y2")  # fmt: skip
DENSE_KNOBS = {"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3", "GLM53_MK_RING_KB_QKV": "700"}
SPARSE_KNOBS = {"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3", "GLM53_MKX_INFLIGHT": "3", "GLM53_MKX_RING": "4"}


def set_env(env: dict):
    """the megakernel knobs of one layer kind (read at construction / build time): reset, then `env`"""
    import os

    for k in KNOBS:
        os.environ.pop(k, None)
    assert set(env) <= set(KNOBS), env
    os.environ.update(env)


def build_layers(mesh, cfg, ctx, dense: list, sparse: list, env_d: dict, env_s: dict, dsa: list = ()) -> list:
    """the real single-chip megakernel layers (7a's MkLayer for dense_kda, MkSparseLayer for sparse_kda; sparse_dsa =
    Stage 7b's MkMla (the attention half) + 7a's MkMoE (the FFN half), two programs) of the given layer ids in id
    order: [dict(L, mk, sparse, kda)] (dsa: dict(L, mla, moe, attn, sparse="dsa")); one set of persistent hand-off
    buffers per KDA kind."""
    from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST
    from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import cached_decoder
    from models.autoports.zai_org_glm_5_3_flash.tt import mk_layer as MKL
    from models.autoports.zai_org_glm_5_3_flash.tt import mk_sparse as MKSP
    from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    seq = []
    for L in sorted(list(dense) + list(sparse) + list(dsa)):
        dec = cached_decoder(cfg, L, mesh, ctx)
        if L in dsa:
            from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MKA
            from models.autoports.zai_org_glm_5_3_flash.tt import mk_moe as MKM

            assert not dec.is_kda, f"layer {L} is not an MLA layer"
            set_env(env_s)
            a_args, f_args = mhc_args(source, cfg, L, "attn"), mhc_args(source, cfg, L, "ffn")
            seq.append(dict(L=L, mla=MKA.MkMla(mesh, a_args, dec.layer.attn), moe=MKM.MkMoE(mesh, f_args, dec.layer.mlp),
                            attn=dec.layer.attn, sparse="dsa", kda=None))  # fmt: skip
            continue
        kda, mlp = dec.layer.attn, dec.layer.mlp
        lins = {d.role: d for d in dec.layer.ds_linears()}
        a_args, f_args = mhc_args(source, cfg, L, "attn"), mhc_args(source, cfg, L, "ffn")
        if L in dense:
            set_env(env_d)
            mk = MKL.MkLayer(mesh, a_args, f_args, kda, {r: lins[r] for r in ("kda_qkv", "kda_lowrank", "kda_o",
                                                                               "mlp_gate_up", "mlp_down")},
                             intermediate=mlp.intermediate, limit=mlp.limit)  # fmt: skip
        else:
            set_env(env_s)
            mk = MKSP.MkSparseLayer(mesh, a_args, f_args, kda, {r: lins[r] for r in ("kda_qkv", "kda_lowrank",
                                                                                     "kda_o")}, mlp)  # fmt: skip
        seq.append(dict(L=L, mk=mk, sparse=L in sparse, kda=kda))
    for kind in (False, True):
        same = [e["mk"] for e in seq if e["sparse"] is kind]
        for mk_ in same[1:]:
            mk_.persist = same[0].persist
    return seq


def kind_runs(seq: list) -> list:
    """consecutive layers of one kind -> [[index, ...], ...]"""
    runs, cur = [], []
    for i, e in enumerate(seq):
        if cur and (seq[cur[-1]]["sparse"] != e["sparse"] or e["sparse"] == "dsa"):  # dsa: runs of 1
            runs.append(cur)
            cur = []
        cur.append(i)
    return runs + [cur]


def keep_decoders(ctx, keep) -> list:
    """drop the long-lived process's cached Stage 3 decoders (ctx ("dec", L)) of layers not in `keep` (DRAM: a sparse
    decoder holds ~2.7 GB of experts) -> the dropped layer ids"""
    import gc

    drop = [k for k in ctx if isinstance(k, tuple) and len(k) == 2 and k[0] == "dec" and k[1] not in set(keep)]
    for k in drop:
        ctx.pop(k)
    gc.collect()
    return [k[1] for k in drop]


def prune_ctx(ctx, keep=()) -> list:
    """drop every Stage 14b object cached in the long-lived process's ctx (keys ("m5...", ...): layer sets, stacks with
    their L1 intermediates, heads) except the `keep` keys and the host-side lm_head / embedding copies -> dropped keys
    """
    import gc

    keep = set(keep) | {("m5_head_host",)}
    drop = [k for k in ctx if k not in keep and ((isinstance(k, tuple) and k and str(k[0]).startswith("m5"))
                                               or (isinstance(k, str) and k.startswith("m5")))]  # fmt: skip
    for k in drop:  # resident traces (m5_sweep) are released, not just dropped
        v = ctx.get(k)
        for tr in v.get("traces", {}).values() if isinstance(v, dict) and isinstance(v.get("traces"), dict) else ():
            tr.release()
        be = v.get("backend") if isinstance(v, dict) else None  # a StackBackend's traces (a failed job leaks them)
        if be is not None and hasattr(be, "close"):
            try:
                be.close()
            except Exception as e:  # noqa: BLE001 -- best effort; the trace region is what matters
                print(f"prune_ctx: {k}: close {e!r}", flush=True)
    for k in drop:
        ctx.pop(k)
    gc.collect()
    return drop


def l1_tensors(limit: int = 12) -> dict:
    """the live Python-held L1 tensors (gc walk): count, bytes, the largest (shape, dtype) — what pins the L1 floor"""
    import gc

    out = []
    for o in gc.get_objects():
        try:
            if isinstance(o, ttnn.Tensor) and o.is_allocated() and o.storage_type() == ttnn.StorageType.DEVICE:
                if o.memory_config().buffer_type == ttnn.BufferType.L1:
                    out.append((int(o.buffer().size()) if hasattr(o, "buffer") else 0, tuple(o.shape), str(o.dtype)))
        except Exception:  # noqa: BLE001 (diagnostic only)
            continue
    out.sort(key=lambda v: -v[0])
    return dict(n=len(out), bytes=sum(v[0] for v in out), largest=out[:limit])


def dram_state(mesh) -> dict:
    """the DRAM allocated per chip (every mesh tensor is allocated lock-step: chip 0's view = every chip's)"""
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    per_bank, banks = int(mv.total_bytes_allocated_per_bank), int(mv.num_banks)
    return dict(allocated_gib_per_chip=round(per_bank * banks / 2**30, 3), banks=banks,
                total_gib_per_chip=round(int(mv.total_bytes_per_bank) * banks / 2**30, 3))  # fmt: skip


def serving(ctx, mesh, SM, G, *, B: int, ids, cfg, native: bool, max_seq_len: int = 2048, shared_weights: bool = True,
            rev=None, log=print):
    """the gate batch's ONE StackBackend + generator per (B, layers, native, rev), shared by the step-time / TF /
    lagged-TF / generator-determinism jobs (job args "serving": true; returns the ctx key): no rebuild or capture between gate jobs
    (program 2026-10-20 19:31Z: the gate batch within 10 min of the open). Every other cached backend is closed."""
    import time

    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    key = ("m5_serving", int(B), tuple(int(v) for v in ids), int(mesh.get_num_devices()), bool(native), rev)
    prune_ctx(ctx, [key])
    if key not in ctx:
        t0 = time.time()
        keep_decoders(ctx, [])
        wc = ctx.setdefault("s14b_weights", {}) if shared_weights else None
        build = lambda: SM.build_model_backend(None, mesh, max_batch=int(B), layers=list(ids), hf_config=cfg,  # noqa: E731
                                               native_t=bool(native), weight_cache=wc)  # fmt: skip
        if os.environ.get("GLM53_WEIGHT_SOURCE") == "image" and not wc:
            # Stage 12 fast start (tt/tp_image.py): the process's FIRST load replays the allocation map of a per-chip
            # DRAM image (resident restart: only stale blocks move); no image / any mismatch -> the tensor loader
            # (GLM53_IMAGE_BUILD=1: and write the image for the next open)
            from models.autoports.zai_org_glm_5_3_flash.tt import tp_image as TI

            info = {}
            be, info = TI.build_or_restore(build, mesh, signature=dict(B=int(B), layers=list(ids), native=bool(native)),
                                           teardown=lambda b: b.close(), timeline=info,
                                           on_fail=lambda: wc.clear() if wc is not None else None)  # fmt: skip
            log(f"fast start: weights from {info.get('source')} {info.get('image') or info.get('image_built') or ''} "
                f"stale {info.get('stale_blocks')} clobbered {info.get('clobbered_blocks')} "
                f"restore {info.get('restore_s')} tried {info.get('tried')}")  # fmt: skip
        else:
            be = build()
        gen = G.build_generator(None, mesh, backend=be, tokenizer=load_tokenizer(), max_batch=int(B),
                                max_seq_len=max_seq_len, stop_token_ids=())  # fmt: skip
        t1 = time.time()
        gen.warmup()
        ctx[key] = dict(backend=be, gen=gen)
        log(f"serving backend B={B} native={bool(native)} layers={len(ids)}: {be.trace_report()} built in "
            f"{t1 - t0:.1f}s, warmup {time.time() - t1:.1f}s")  # fmt: skip
    return key  # ctx[key] = {"backend", "gen"}: the jobs read it under this key (an alias key would be pruned = closed)
