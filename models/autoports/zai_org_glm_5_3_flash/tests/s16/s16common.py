# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 16 job helpers on top of Stage 14's s14common (evidence rows, mesh open, traced timing, PCC): the backend +
prefill build kept resident in the server's CTX, the CPU prefill goldens (Stage 00c), host <-> mesh helpers."""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "s14b"))
import s14common as C  # noqa: E402,F401

import ttnn  # noqa: E402

_ctx_s14 = C._ctx
# every evidence row names the host it ran on (host-8chip or host-p300: the d0 cards run the same jobs)
C._ctx = lambda: {**_ctx_s14(), "host": os.uname().nodename}

GOLDENS = os.environ.get("GLM53_GOLDENS", "/opt/glm53/kit/goldens")
MILESTONE = "P2"
EVIDENCE = (
    "/opt/glm53/kit/doc/ttnn/16-prefill-production"  # rows of jobs run elsewhere
)


def log(*a):
    print(f"S16 {C.now()}", *a, flush=True)


def golden_layer(L: int, n: int):
    d = os.path.join(GOLDENS, "layer_io", f"L{L:02d}", str(n))
    return (torch.load(os.path.join(d, "streams_in.pt"), map_location="cpu", weights_only=True),
            torch.load(os.path.join(d, "streams_out.pt"), map_location="cpu", weights_only=True))  # fmt: skip


def golden_kda_state(L: int, n: int):
    return torch.load(os.path.join(GOLDENS, "kda_state", str(n), f"L{L:02d}.pt"), map_location="cpu",
                      weights_only=True)  # fmt: skip


def golden_prompt(n: int) -> torch.Tensor:
    return torch.load(os.path.join(GOLDENS, "prompts", f"{n}.pt"), map_location="cpu", weights_only=True).reshape(-1)


def golden_logits(n: int) -> dict:
    return torch.load(os.path.join(GOLDENS, "logits", f"{n}.prefill.pt"), map_location="cpu", weights_only=True)


def to_mesh_rows(mesh, x: torch.Tensor, sp: bool, dtype=ttnn.float32):
    """[rows, W] host -> [1, 1, rows(/N), W] mesh tensor (sp: sharded over the mesh along the rows)"""
    n = int(mesh.get_num_devices())
    x = x.reshape(1, 1, x.shape[-2], x.shape[-1])
    mp = (ttnn.ShardTensorToMesh(mesh, dim=2) if sp else ttnn.ReplicateTensorToMesh(mesh)) if n > 1 else None
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=mp)  # fmt: skip


def from_mesh_rows(t, sp: bool) -> torch.Tensor:
    parts = ttnn.get_device_tensors(t) if t.device().get_num_devices() > 1 else [t]
    hs = [ttnn.to_torch(p).float() for p in parts]
    return torch.cat(hs, dim=2) if sp else hs[0]


def adopt_full(ctx, layers, tp) -> bool:
    """register p2_full's backend (the "s16-full" CTX entry: the generator's backend, its kv cache / decode io and
    TPPrefill) as build()'s entry for `layers`, so a later job in the SAME process (op_profile, p2_layer_check) reuses
    it instead of building a second backend (a full-model build per process: ~5 min, and a second build's L1 floor).
    build() still rebuilds the TPPrefill on that backend if the job's prefill knobs differ. -> True if adopted."""
    full = [k for k in ctx if isinstance(k, tuple) and k and k[0] == "s16-full" and list(k[2]) == list(layers)]
    key = backend_key(layers, 1, tp)
    if not full or key in ctx:
        return key in ctx
    ent = ctx[full[-1]]
    be = ent["backend"]
    kv = getattr(be, "_kv", None)
    if kv is None or getattr(be, "tpp", None) is None:
        return False
    ctx[key] = dict(backend=be, kv=kv, io=kv.extra.get("io"), pf=be.tpp, build_s=0.0, backend_s=0.0,
                    pf_rev=os.environ.get("GLM53_PF_REV", "0"), pf_attn=_attn_knobs())  # fmt: skip
    log(f"adopted p2_full's backend ({len(layers)} layers) for build()")
    return True


def backend_key(layers, b, tp, native=False):
    return ("s16-backend", tuple(layers), int(b), int(tp), bool(native))


def _attn_knobs():
    """the P3 attention / projection knobs a TPPrefill reads at construction (a change rebuilds it)"""
    return tuple(os.environ.get(k, "") for k in ("GLM53_PF_MLA_FORM", "GLM53_PF_MHA_SDPA", "GLM53_PF_MHA_MIN",
                                                 "GLM53_PF_SDPA", "GLM53_PF_KDA_GROUPS", "GLM53_PF_KDA_GROUPED_MEM",
                                                 "GLM53_PF_WARM_CAPTURE", "GLM53_PF_HEAD_L1ACC",
                                                 "GLM53_PF_MM_FID", "GLM53_PF_MOE_FID",
                                                 "GLM53_PF_SPARSE_ATTN", "GLM53_PF_SPARSE_RESHARD",
                                                 "GLM53_PF_NOHEAD", "GLM53_PF_PACKED_DEVMASK"))  # fmt: skip


def _shard_on() -> bool:
    """GLM53_DSA_SHARD: every unit's TPPrefill then REQUIRES a published owner-aware page table (TPPrefill.page_row
    raises instead of falling back to the owners' null blocks; pmk45_alias.publish_page_table publishes one)"""
    return os.environ.get("GLM53_DSA_SHARD", "0") == "1"


def build(mesh, ctx, layers, *, b=1, max_seq=4096, buckets=(128, 512, 2048), native=False):
    """StackBackend (the stack's TP layers on the mesh) + kv cache + decode io + TPPrefill, kept in ctx (one build per
    process: the weights stay resident; layer subsets reuse the weight cache)."""
    import importlib

    import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF
    import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM
    from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config

    key = backend_key(layers, b, int(mesh.get_num_devices()), native)
    if key in ctx and int(ctx[key]["kv"].max_seq_len) < int(max_seq):  # a longer context: new kv cache + prefill
        old = ctx.pop(key)
        old["pf"].release()
        old["backend"].close()
    if key in ctx:
        ent = ctx[key]
        # the env the backend build left (the stack knob profile of the last layer kind: GLM53_MHC_* / GLM53_MK_* /
        # GLM53_REC_* ..., read lazily by the stack's programs): re-applied for every job on this resident build, since
        # serve.py restores each job's env at its end (5f3462c0935) -- a cache hit sees the same env as the build did
        os.environ.update(ent.get("build_env") or {})
        if (
            ent.get("pf_rev") != os.environ.get("GLM53_PF_REV", "0")
            or ent["pf"].mhc_mode != os.environ.get("GLM53_PF_MHC", "comp")
            or ent["pf"].dsa != (os.environ.get("GLM53_PF_DSA", "0") == "1")
            or ent["pf"].mhc_post != os.environ.get("GLM53_PF_MHC_POST", "ops")
            or ent.get("pf_attn") != _attn_knobs()
        ):
            ent["pf"].release()
            importlib.reload(PF)
            ent["pf"] = PF.TPPrefill(ent["backend"], ent["kv"], buckets=buckets)
            ent["pf_rev"] = os.environ.get("GLM53_PF_REV", "0")
            ent["pf_attn"] = _attn_knobs()
            ent.pop("captured", None)  # the new prefill has no traces yet
        ent["pf"].buckets = tuple(sorted(set(ent["pf"].buckets) | {int(b_) for b_ in buckets}))
        ent["pf"].require_pt = _shard_on()
        return ent
    for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] in ("s16-backend", "s16-full", "s16-tf")]:
        old = ctx.pop(k)
        try:
            old["backend"].close()
        except Exception as e:  # noqa: BLE001
            log(f"close {k}: {e!r}")
    importlib.reload(PF)
    t0 = time.time()
    env0 = dict(os.environ)
    wc = ctx.setdefault("weight_cache", {})
    backend = SM.build_model_backend(None, mesh, max_batch=b, layers=list(layers), hf_config=hf_config(),
                                     weight_cache=wc, native_t=native)  # fmt: skip
    kv = backend.allocate_kv_cache(max_batch=b, max_seq_len=max_seq)
    io = backend.allocate_decode_io(kv)
    t_backend = time.time() - t0
    pf = PF.TPPrefill(backend, kv, buckets=buckets)
    ent = dict(backend=backend, kv=kv, io=io, pf=pf, build_s=round(time.time() - t0, 1), backend_s=round(t_backend, 1),
               pf_rev=os.environ.get("GLM53_PF_REV", "0"), pf_attn=_attn_knobs())  # fmt: skip
    ent["build_env"] = {k_: v_ for k_, v_ in os.environ.items() if env0.get(k_) != v_}  # (see the cache-hit path)
    pf.require_pt = _shard_on()
    ctx[key] = ent
    log(f"built backend layers={list(layers)} b={b} in {ent['build_s']} s (backend {ent['backend_s']} s)")
    return ent


def set_attn(args):
    """the P3 attention knobs from a job's args (before build): mla_form ("absorbed" | "mha"), mha_sdpa / sdpa
    ("q,k,fidelity"), mm_nq (pf_mm per-core N quantum), mm_kb (pf_mm CB budget, KB), mm_fid (pf_mm math fidelity of the
    block-float weights, "" = the decode's), moe_fid (EpMoe's routed-expert fidelity, "" = HiFi2), sparse_attn (sparse
    chunks: "mask" = dense SDPA + [C, T] mask, "sdpa" = sparse_sdpa over the selected ids); env ({}: any further
    environment for the build, e.g. 14b's served construction knobs GLM53_TP_XSL / GLM53_TP_XSL_SLICED /
    GLM53_STACK_KNOBS)"""
    for k, v in (args.get("env") or {}).items():
        os.environ[str(k)] = str(v)
    os.environ["GLM53_PF_MLA_FORM"] = str(args.get("mla_form", "absorbed"))
    os.environ["GLM53_PF_MHA_SDPA"] = str(args.get("mha_sdpa", ""))
    os.environ["GLM53_PF_SDPA"] = str(args.get("sdpa", ""))
    os.environ["GLM53_PF_MM_NQ"] = str(args.get("mm_nq", 4))
    os.environ["GLM53_PF_MM_KB"] = str(args.get("mm_kb", 640))
    os.environ["GLM53_PF_REP_BUCKETS"] = ",".join(str(int(v)) for v in args.get("rep_buckets", []))
    os.environ["GLM53_PF_WARM_CAPTURE"] = str(args.get("warm_capture", 1))  # 0: no eager forward before a capture
    os.environ["GLM53_PF_HEAD_L1ACC"] = str(args.get("head_l1acc", 1))
    os.environ["GLM53_PF_MM_FID"] = str(args.get("mm_fid", ""))
    os.environ["GLM53_PF_MOE_FID"] = str(args.get("moe_fid", ""))
    os.environ["GLM53_PF_SPARSE_ATTN"] = str(args.get("sparse_attn", "gather"))
    os.environ["GLM53_PF_SPARSE_RESHARD"] = str(args.get("sparse_reshard", "pad"))


def set_moe_v2(pf, on):
    """EP MoE v2 for the next forwards: on = False / True (both halves) / "dispatch" / "combine" (one half, the hang
    diagnosis); the env read by EpMoe's constructor, and every layer's cached EpMoe dropped (rebuilt lazily)"""
    mode = {False: "0", True: "1", "0": "0", "1": "1"}.get(on, on)
    os.environ["GLM53_PF_MOE_V2"] = "1" if mode == "1" else "0"
    os.environ["GLM53_PF_MOE_V2_DISPATCH"] = "1" if mode == "dispatch" else "0"
    os.environ["GLM53_PF_MOE_V2_COMBINE"] = "1" if mode == "combine" else "0"
    for pl in pf.layers:
        if getattr(pl, "epmoe", None) is not None:
            pl.epmoe = None


# program 2026-10-21 03:20Z (DEVICE DUTY): every timing row replays its trace >= 2,000 times (>= 1 s of device
# time, medians over the steady state) and cites the replay count; a watched unit ends with a 30-s unwatched soak
SOAK_REPLAYS = int(os.environ.get("GLM53_S16_SOAK_REPLAYS", "2000"))


def replay_stats(mesh, tid, *, replays: int | None = None, min_s: float = 1.0, batch: int = 10,
                 max_s: float | None = None, label: str = "", every: int = 200) -> dict:  # fmt: skip
    """steady-state replay time of a captured trace: >= `replays` (default SOAK_REPLAYS) replays AND >= min_s seconds,
    in batches of `batch` non-blocking replays + one synchronize (sample = batch wall / batch); the first batch (warm)
    is excluded. max_s: a wall cap (the row then reports the replays it got). -> dict(replays, median_us, min_us,
    p90_us, soak_s); {} on the mock. A progress line every `every` replays (label: which trace) -- a hang inside a
    soak is then pinned to a replay window (03:49Z card2 hang: the index was unknown). tid may be a list: one "replay"
    = every trace of it in order (P4: a decode step with an interleaved prefill chunk)."""
    if C.MOCK or tid is None or (replays is not None and int(replays) == 0):
        return {}  # replays=0: no soak (correctness-only runs; the 03:53Z rule keeps sparse_dsa soaks off card2)
    target = SOAK_REPLAYS if replays is None else int(replays)
    tids = list(tid) if isinstance(tid, (list, tuple)) else [tid]
    samples, done = [], 0
    t_start = time.perf_counter()
    while done < target or time.perf_counter() - t_start < min_s:
        t0 = time.perf_counter()
        for _ in range(batch):
            for t_ in tids:
                ttnn.execute_trace(mesh, t_, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        samples.append((time.perf_counter() - t0) * 1e6 / batch)
        done += batch
        if every and done % every == 0:
            log(f"soak {label or tid}: {done} replays, {time.perf_counter() - t_start:.1f} s")
        if max_s is not None and time.perf_counter() - t_start > max_s:
            break
    steady = sorted(samples[1:] or samples)
    return dict(replays=done, median_us=round(steady[len(steady) // 2], 1), min_us=round(steady[0], 1),
                p90_us=round(steady[min(len(steady) - 1, int(0.9 * len(steady)))], 1),
                soak_s=round(time.perf_counter() - t_start, 1))  # fmt: skip


def soak(mesh, tids, seconds: float) -> dict:
    """the post-watched-unit soak: replay every trace in `tids` round-robin (non-blocking, synchronize per round) for
    `seconds` of wall time -> dict(soak_s, soak_replays)"""
    tids = [t_ for t_ in tids if t_ is not None]
    if C.MOCK or not tids or seconds <= 0:
        return {}
    n, t_start = 0, time.perf_counter()
    while time.perf_counter() - t_start < seconds:
        for t_ in tids:
            ttnn.execute_trace(mesh, t_, cq_id=0, blocking=False)
            n += 1
        ttnn.synchronize_device(mesh)
    return dict(soak_s=round(time.perf_counter() - t_start, 1), soak_replays=n)


def timing(us) -> dict:
    """C.stats with its sample count as `reps_n` (C.stats' `n` would overwrite a row's prompt length)"""
    st = dict(C.stats(us))
    if "n" in st:
        st["reps_n"] = st.pop("n")
    return st
