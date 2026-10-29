# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel, the WHOLE dense_kda layer as two programs (M1): the attention half (tt/mk_dense.py MkStepB step
D: mhc_pre -> qkv + low-rank -> kda_pre -> kda_rec -> o_proj -> mhc_post -> s1) and the FFN half (tt/mk_ffn.py MkFFN:
mhc_pre -> gate_up -> swiglu -> down -> mhc_post -> the layer output) vs the chain they replace (the same as
dev_mk_c step D + dev_mk_ffn), L0, T = 1 / 2 / 4, device vs device, traced (every case warmed before any capture).

Numerics: one traced call with the conv history and the recurrent state restored inside it; hist bitwise, s1 / state /
the layer output PCC. Timing: separate captures WITHOUT the state restores (the state drifts; timing only), median of
REPLAYS: attention program alone, FFN program alone, both back to back (the second program's kernel-config write can
overlap the first's execution when both fit the worker kernel-config ring: GLM53_WORKER_L1), and the chain.
Rows -> pcc.jsonl / tests.jsonl / perf.jsonl."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_kda_pre import masks
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_mk_a import mhc_args
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_mk_ffn import ffn_mhc_args
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced, cached_decoder
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
from models.autoports.zai_org_glm_5_3_flash.tt.kda_pre import KdaPre
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import OUT_FLAT, OUT_STACKED
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import X_FLAT as PX_FLAT
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import X_STACKED, MhcPost
from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import MkStepB
from models.autoports.zai_org_glm_5_3_flash.tt.mk_ffn import MkFFN
from models.autoports.zai_org_glm_5_3_flash.tt.mk_layer import MkLayer
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "0"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
TS = [int(v) for v in os.environ.get("GLM53_MK_TS", "1,2,4").split(",")]
ONE = os.environ.get("GLM53_MK_ONE") == "1"  # the whole layer as ONE program (tt/mk_layer.py MkLayer, M1)
REPLAYS = 30
SPAN_REPLAYS = int(os.environ.get("GLM53_MK_SPAN_REPLAYS", "10"))
ROWS = []
_FLUSHED = 0


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, layer=LAYER, test=name,
             watcher=bool(os.environ.get("TT_METAL_WATCHER")), rec_depth=os.environ.get("GLM53_REC_DEPTH", "1"),
             **kw)  # fmt: skip
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)


def flush():
    global _FLUSHED
    new, _FLUSHED = ROWS[_FLUSHED:], len(ROWS)
    with open(os.path.join(OUT, "pcc.jsonl"), "a") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT, "tests.jsonl"), "a") as f:
        for r in new:
            f.write(json.dumps({k: r[k] for k in ("time_utc", "run", "test", "status", "t") if k in r}) + "\n")
    with open(os.path.join(OUT, "perf.jsonl"), "a") as f:
        for r in new:
            if r.get("timing_us"):
                f.write(json.dumps({k: r[k] for k in ("time_utc", "run", "test", "t", "execution", "timing_us",
                                                       "worker_l1", "mk_opt", "watcher", "readers", "ring_slots", "park", "rec_depth",
                                                       "mk_one_per_layer_steady_us", "chain_per_layer_steady_us", "mk_layer_per_layer_steady_us",
                                                       "one_speedup", "layer_speedup", "mk_one_span_restored_us",
                                                       "mk_one_span_restored_minmax", "code") if k in r}) + "\n")  # fmt: skip


def probe_span_us(probe_host, mhz=1350.0):
    """Last mark - first mark over every core / RISC of one launch (low 32 bits of the AICLK wall clock; a spread over
    half the range = one wrap inside the launch)."""
    p = probe_host.reshape(-1, 16).long() & 0xFFFFFFFF
    v = p[p > 0]
    if not v.numel():
        return None
    if int(v.max() - v.min()) > (1 << 31):
        v = torch.where(v < (1 << 31), v + (1 << 32), v)
    return (int(v.max()) - int(v.min())) / mhz


def host(t):
    return ttnn.to_torch(t).float()


TWO_PROG = ("mk_att", "mk_ffn", "mk_layer", "chain_layer", "mk_layer_x4", "chain_x2")


def chain_env(fn):
    """GLM53_CHAIN_MHC_MODE (debug, set = also when empty): the chain's ops (standalone mhc_pre / mhc_post) are built
    and run with this GLM53_MHC_MODE instead of the megakernel's — which side of a mode disagreement is wrong."""
    mode = os.environ.get("GLM53_CHAIN_MHC_MODE")
    if mode is None:
        return fn

    def wrapped(*a, **k):
        old = os.environ.get("GLM53_MHC_MODE")
        os.environ["GLM53_MHC_MODE"] = mode
        try:
            return fn(*a, **k)
        finally:
            if old is None:
                os.environ.pop("GLM53_MHC_MODE", None)
            else:
                os.environ["GLM53_MHC_MODE"] = old

    return wrapped


def chained(layer, x, n):
    """n layer calls, each fed the previous output (the first gets x)"""
    for _ in range(n):
        x = layer(x)
    return x


def timed(tr):
    if not tr.enabled:
        return None
    d = []
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        tr.replay()
        d.append((time.perf_counter() - t0) * 1e6)
    return round(sorted(d)[len(d) // 2], 1)


def main(mesh=None, ctx=None):
    """mesh / ctx: run inside the long-lived device process (serve.py): no open / close, the decoder cached."""
    own = mesh is None
    cfg = hf_config()
    wl1 = int(os.environ.get("GLM53_WORKER_L1", "0")) if own else (ctx or {}).get("worker_l1", 0)
    extra = dict(worker_l1_size=wl1) if wl1 else {}
    if own:
        mesh = (
            ttnn.open_device(device_id=0, **extra)
            if MOCK
            else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000, **extra)
        )
    tr = Traced(mesh)
    try:
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        dec = cached_decoder(cfg, LAYER, mesh, ctx)
        kda, mlp = dec.layer.attn, dec.layer.mlp
        lins = {d.role: d for d in dec.layer.ds_linears()}

        def build():
            a_args, f_args = mhc_args(source, cfg, LAYER), ffn_mhc_args(source, cfg, LAYER)
            att = MkStepB(mesh, a_args, kda, lins["kda_qkv"], lins["kda_lowrank"], ds_o=lins["kda_o"], post=True)
            MOE = dec.layer.spec.mlp == "moe"  # sparse layers: the MoE FFN half (tt/mk_moe.py MkMoE)
            if MOE:
                from models.autoports.zai_org_glm_5_3_flash.tt.mk_moe import MkMoE

                ffn = MkMoE(mesh, f_args, mlp)
            else:
                ffn = MkFFN(mesh, f_args, lins["mlp_gate_up"], lins["mlp_down"], intermediate=mlp.intermediate,
                            limit=mlp.limit)  # fmt: skip
            one = None
            if ONE and MOE:  # the whole sparse_kda layer as ONE program (tt/mk_sparse.py MkSparseLayer, M4)
                from models.autoports.zai_org_glm_5_3_flash.tt.mk_sparse import MkSparseLayer

                one = MkSparseLayer(mesh, a_args, f_args, kda, lins, mlp, probe=os.environ.get("GLM53_MK_PROBE") == "1")
            elif ONE:
                one = MkLayer(mesh, a_args, f_args, kda, lins, intermediate=mlp.intermediate, limit=mlp.limit,
                              probe=os.environ.get("GLM53_MK_PROBE") == "1")  # fmt: skip
            ref_a, ref_f = M.MhcPre(mesh, **a_args), M.MhcPre(mesh, **f_args)
            ref_kda = KdaPre(kda, rows_max=32)  # ROWS mode (the conv-history shift in the program) up to 32 rows
            mpost = MhcPost(mesh)
            return dict(
                a_args=a_args,
                f_args=f_args,
                att=att,
                ffn=ffn,
                one=one,
                ref_a=ref_a,
                ref_f=ref_f,
                ref_kda=ref_kda,
                mpost=mpost,
                MOE=MOE,
            )

        # RESIDENT STATE (program job-pipeline rule): the programs' host objects (and their device-side constants:
        # mHC fn / scale, kda taps, expert address tables) live in the server's CTX, keyed by the layer, the knobs and the
        # class identities (a module reload makes new classes); ONE resident set at a time (their persistent L1 buffers
        # bound the L1 floor), persistent per-width buffers dropped at every job start
        from models.autoports.zai_org_glm_5_3_flash.tt import mk_layer as _ml
        from models.autoports.zai_org_glm_5_3_flash.tt import mk_sparse as _ms

        skip = ("GLM53_MK_TS", "GLM53_MK_DUMP")
        knobs = tuple(sorted((k, v) for k, v in os.environ.items()
                             if k.startswith(("GLM53_MK", "GLM53_MHC", "GLM53_REC")) and k not in skip))  # fmt: skip
        key = (LAYER, ONE, knobs, id(_ml.MkLayer), id(_ms.MkSparseLayer), id(MkStepB), id(MkFFN), id(M.MhcPre))
        res = ctx.get("dev_mk_layer_objs") if ctx is not None else None
        if res is None or res[0] != key:
            if ctx is not None:
                ctx.pop("dev_mk_layer_objs", None)
            res = (key, build())
            if ctx is not None:
                ctx["dev_mk_layer_objs"] = res
            print("DEV_MK_LAYER objects built", flush=True)
        else:
            print("DEV_MK_LAYER objects resident", flush=True)
        o_ = res[1]
        a_args, f_args, att, ffn, one = o_["a_args"], o_["f_args"], o_["att"], o_["ffn"], o_["one"]
        ref_a, ref_f, ref_kda, mpost, MOE = o_["ref_a"], o_["ref_f"], o_["ref_kda"], o_["mpost"], o_["MOE"]
        for obj in (att, ffn, one):
            if obj is not None and hasattr(obj, "persist"):
                obj.persist.clear()  # per-width persistent buffers: re-made lazily for this job's widths
        i_ = mlp.intermediate
        g = golden(LAYER, 2048)["streams_in"].float()
        cases, dbg_fns = {}, {}
        for t in TS:
            # t <= 4: the stacked residual (M1's layout); t > 4: flat / stream-major (the layer programs' residual_modes)
            big = t > 4
            pcm = M.PC_STREAM_MAJOR if big else M.PC_STACKED
            akw = dict(t=t, x_mode=M.X_FLAT, xout_mode=M.XOUT_NONE if big else M.XOUT_STACKED, pc_mode=pcm)
            pxm, s1m = (PX_FLAT, OUT_FLAT) if big else (X_STACKED, OUT_STACKED)
            x = ttnn.from_torch(g[0, 300 : 300 + 37 * t : 37].reshape(1, 1, t, 16384), dtype=ttnn.float32,
                                layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            active = torch.tensor([(i % 3) != 2 for i in range(t)]) if t > 1 else torch.tensor([True])
            act = masks(mesh, active, kda.lower_bound)
            torch.manual_seed(1000 + t)  # the same random history / state in every process and job (comparable rows)
            hist0 = ttnn.from_torch((torch.randn(1, 3, t, 24576) * 0.5).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            st_a, st_b = kda.allocate_state(t), kda.allocate_state(t)
            s0 = ttnn.from_torch(torch.randn(t, 64, 128, 128) * 0.1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                 device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            s1_fix = ttnn.from_torch(g[0, 301 : 301 + 37 * t : 37].reshape((1, 1, t, 16384) if big else (t, 1, 4, 4096)),
                                     dtype=ttnn.float32,
                                     layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip

            def mk_att(x=x, act=act, st=st_a, akw=akw):
                return att(x, st.C, act, state=st.S, **akw)[-1]

            def mk_ffn(s1, t=t):
                return ffn(s1, t=t)["out"] if MOE else ffn(s1, t=t)[-1]

            def chain_att(x=x, act=act, st=st_b, akw=akw, t=t, pxm=pxm, s1m=s1m, big=big):
                r = ref_a(x, hn_dtype=ttnn.bfloat16, **akw)
                q_ = lins["kda_qkv"](r.hn)
                l_ = lins["kda_lowrank"](r.hn)
                ko_ = ref_kda(q_, l_, st.C, act, rows=t)
                a_ = kda._recurrence_and_output(*ko_[:6], st, t)
                return mpost(a_, x if big else r.x_out, r.coef, t=t, x_mode=pxm, out_mode=s1m)

            def chain_ffn(s1, t=t, pxm=pxm, pcm=pcm, big=big):
                r = ref_f(s1, hn_dtype=ttnn.bfloat16, t=t, x_mode=M.X_FLAT if big else M.X_STACKED,
                          xout_mode=M.XOUT_NONE, pc_mode=pcm)  # fmt: skip
                if MOE:  # router + router tail, fused_experts, the shared SwiGLU, mhc_post (y + y2)
                    ids_, sc_ = mlp.router(r.hn, decode=True)
                    y_ = ttnn.reshape(mlp.routed(r.hn, ids_, sc_, mlp.cfg.moe_decode_micro_batch,
                                                 mlp.cfg.experts_block_size), (1, 1, t, mlp.hidden))  # fmt: skip
                    return mpost(y_, s1, r.coef, t=t, x_mode=pxm, out_mode=OUT_FLAT,
                                 y2=mlp.shared(r.hn, decode=True), out_shape=(1, 1, t, 16384),
                                 out_memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
                gu_ = lins["mlp_gate_up"](r.hn, ttnn.L1_MEMORY_CONFIG)
                shape = list(gu_.shape)
                m_ = ttnn.clamped_silu_glu(ttnn.slice(gu_, [0, 0, 0, 0], shape[:-1] + [i_]),
                                           ttnn.slice(gu_, [0, 0, 0, i_], shape[:-1] + [2 * i_]), mlp.limit)  # fmt: skip
                y_ = lins["mlp_down"](m_, ttnn.L1_MEMORY_CONFIG)
                return mpost(y_, s1, r.coef, t=t, x_mode=pxm, out_mode=OUT_FLAT, out_shape=(1, 1, t, 16384),
                             out_memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip

            chain_att, chain_ffn = chain_env(chain_att), chain_env(chain_ffn)

            def restore(st, hist0=hist0, s0=s0):
                ttnn.copy(hist0, st.C)
                ttnn.copy(s0, st.S)

            # every helper bound as a default argument: the closures are called after the loop (late binding)
            def num_mk(st=st_a, restore=restore, mk_att=mk_att, mk_ffn=mk_ffn):
                restore(st)
                s1 = mk_att()
                return s1, mk_ffn(s1)

            def num_chain(st=st_b, restore=restore, chain_att=chain_att, chain_ffn=chain_ffn):
                restore(st)
                s1 = chain_att()
                return s1, chain_ffn(s1)

            def num_one(st=st_a, restore=restore, x=x, act=act, t=t):
                restore(st)
                r_ = one(x, st.C, act, t=t, state=st.S)
                return r_["s1"], r_["out"]

            if ONE:
                num_mk = num_one

            def dbg_mk(st=st_a, restore=restore, x=x, act=act, t=t):
                restore(st)
                r_ = one(x, st.C, act, t=t, state=st.S)
                return {
                    k: r_[k] for k in ("hn", "qkv", "lr", "a", "on", "x_out", "coef_a", "s1", "hn2", "gu", "m", "y")
                }

            def dbg_chain(st=st_b, restore=restore, x=x, act=act, akw=akw, t=t, pxm=pxm, s1m=s1m, big=big):
                restore(st)
                r = ref_a(x, hn_dtype=ttnn.bfloat16, **akw)
                q_ = lins["kda_qkv"](r.hn)
                l_ = lins["kda_lowrank"](r.hn)
                ko_ = ref_kda(q_, l_, st.C, act, rows=t)
                a_ = kda._recurrence_and_output(*ko_[:6], st, t)
                s1_ = mpost(a_, x if big else r.x_out, r.coef, t=t, x_mode=pxm, out_mode=s1m)
                return dict(hn=r.hn, qkv=q_, lr=l_, a=a_, x_out=r.x_out, coef_a=r.coef, s1=s1_)

            dbg_fns[t] = (dbg_mk, dbg_chain)
            timing = dict(
                mk_one=lambda st=st_a, x=x, act=act, t=t: one(x, st.C, act, t=t, state=st.S)["out"],
                # steady state: N layer launches back to back in one trace, each fed the previous layer's output (the
                # decode step's shape: the next program's kernel-config write overlaps the running one)
                mk_one_x2=lambda st=st_a, x=x, act=act, t=t: chained(
                    lambda x_: one(x_, st.C, act, t=t, state=st.S)["out"], x, 2
                ),
                mk_one_x4=lambda st=st_a, x=x, act=act, t=t: chained(
                    lambda x_: one(x_, st.C, act, t=t, state=st.S)["out"], x, 4
                ),
                chain_x2=lambda x=x, chain_att=chain_att, chain_ffn=chain_ffn: chained(
                    lambda x_: chain_ffn(chain_att(x=x_)), x, 2
                ),
                mk_att=lambda mk_att=mk_att: mk_att(),
                mk_ffn=lambda s1_fix=s1_fix, mk_ffn=mk_ffn: mk_ffn(s1_fix),
                mk_layer=lambda mk_att=mk_att, mk_ffn=mk_ffn: mk_ffn(mk_att()),
                # steady state of the two-program layer: 4 layers chained in one trace
                mk_layer_x4=lambda x=x, mk_att=mk_att, mk_ffn=mk_ffn: chained(lambda x_: mk_ffn(mk_att(x=x_)), x, 4),
                chain_layer=lambda chain_att=chain_att, chain_ffn=chain_ffn: chain_ffn(chain_att()),
            )
            if os.environ.get("GLM53_MK_TIMING") == "lean":  # steady state only (the decode step's shape)
                keep = ("mk_one_x4", "chain_x2")
                timing = {k_: v_ for k_, v_ in timing.items() if k_ in keep}  # (probes are read after mk_one)
            if big:  # the two-program halves serve t <= 4 only
                for k_ in ("mk_att", "mk_ffn", "mk_layer", "mk_layer_x4"):
                    timing.pop(k_, None)
            cases[t] = (num_mk, num_chain, timing, st_a, st_b)
        tr.warm(
            *[f for c in cases.values() for f in (c[0], c[1], *[v for k, v in c[2].items() if ONE or k in TWO_PROG])]
        )  # compile all before any capture
        if os.environ.get("GLM53_MK_DUMP") and ONE:
            tr.warm(*[f for fs in dbg_fns.values() for f in fs])
        for t, (num_mk, num_chain, timing, st_a, st_b) in cases.items():
            try:
                s1, out = tr.run(num_mk)
                got = dict(s1=host(s1), out=host(out), hist=host(st_a.C), state=host(st_a.S))
                tr.release()
                s1b, outb = tr.run(num_chain)
                want = dict(s1=host(s1b), out=host(outb), hist=host(st_b.C), state=host(st_b.S))
                if os.environ.get("GLM53_MK_DUMP"):  # debugging: the compared tensors + the intermediates, host side
                    dbg = {}
                    if ONE:
                        for side, fn in (("mk", dbg_fns[t][0]), ("chain", dbg_fns[t][1])):
                            res_ = tr.run(fn)
                            dbg[side] = {k: host(v) for k, v in res_.items()}
                            tr.release()
                    torch.save(dict(got=got, want=want, dbg=dbg), f"{os.environ['GLM53_MK_DUMP']}_t{t}.pt")
                tr.release()
                us = {}
                for name, fn in timing.items():
                    if name in ("mk_one", "mk_one_x2", "mk_one_x4") and not ONE:
                        continue
                    tr.run(fn)
                    us[name] = timed(tr)
                    tr.release()
                # controlled span (14c #22: replays without the state restore drift the KDA state and with it the
                # routing, +160-174 us per sparse layer at B=32): ONE trace = the state restore + one launch, replayed
                # SPAN_REPLAYS times; after each replay the span from the program's own probe marks (first -> last
                # mark over all cores), i.e. the restore is outside the span; median. The probe rows come from the
                # last restored launch.
                spans = []
                if ONE and one is not None and getattr(one, "probe", None) is not None and tr.enabled:
                    from models.autoports.zai_org_glm_5_3_flash.tt.mk_layer import decode_probe

                    tr.run(num_mk)
                    for _ in range(SPAN_REPLAYS):
                        tr.replay()
                        ph = ttnn.to_torch(one.probe)
                        spans.append(probe_span_us(ph))
                    tr.release()
                    row("mk_one_probe", t=t, status="info", controlled=True, worker_l1=wl1 or None,
                        readers=os.environ.get("GLM53_MK_READERS"), ring_kb=os.environ.get("GLM53_MK_RING_KB"),
                        **decode_probe(ph, one.readers.assign))  # fmt: skip
                    if hasattr(one, "tail_rows"):  # MkSparseLayer: the router-tail cores' marks
                        from models.autoports.zai_org_glm_5_3_flash.tt.mk_sparse import decode_tail_probe

                        row("mk_one_tail_probe", t=t, status="info", controlled=True,
                            **decode_tail_probe(ph, len(one.tail_rows(t))))  # fmt: skip
                r = dict(t=t, execution=tr.mode, timing_us=us, mk_opt=os.environ.get("GLM53_MK_OPT", "size"),
                         readers=os.environ.get("GLM53_MK_READERS"),
                         park=one.park_plan if one is not None else None,
                         ring_slots={str(k): v for k, v in one.ring_slots.items()} if one is not None else None,
                         worker_l1=wl1 or None, alloc_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1",
                         timing_note="mk_one* / chain_x2 / mk_layer*: replays without the state restore (the KDA "
                                     "state and the routing drift: uncontrolled for sparse kinds); "
                                     "mk_one_span_restored_us: the program's probe span of a restored launch (controlled)",
                         code=os.environ.get("GLM53_CODE_SHA"))  # fmt: skip
                if us.get("mk_layer") and us.get("chain_layer"):
                    r["layer_speedup"] = round(us["chain_layer"] / us["mk_layer"], 3)
                if us.get("mk_one") and us.get("chain_layer"):
                    r["one_speedup"] = round(us["chain_layer"] / us["mk_one"], 3)
                if us.get("mk_layer_x4"):
                    r["mk_layer_per_layer_steady_us"] = round(us["mk_layer_x4"] / 4, 1)
                if us.get("mk_one_x4"):
                    r["mk_one_per_layer_steady_us"] = round(us["mk_one_x4"] / 4, 1)
                if us.get("chain_x2"):
                    r["chain_per_layer_steady_us"] = round(us["chain_x2"] / 2, 1)
                if spans:
                    r["mk_one_span_restored_us"] = round(sorted(spans)[len(spans) // 2], 1)
                    r["mk_one_span_restored_minmax"] = [round(min(spans), 1), round(max(spans), 1)]
                if all(us.get(k_) for k_ in ("mk_att", "mk_ffn", "mk_layer")):
                    r["launch_overlap_us"] = round(us["mk_att"] + us["mk_ffn"] - us["mk_layer"], 1)
                for k_ in got:
                    a, b = got[k_].reshape(-1), want[k_].reshape(-1)[: got[k_].numel()]
                    r[f"{k_}_bitwise"] = bool(torch.equal(a, b))
                    r[f"{k_}_pcc"] = pcc(a, b)
                    r[f"{k_}_maxabs"] = float((a - b).abs().max())
                ok = r["hist_bitwise"] and min(r[f"{k_}_pcc"] for k_ in got) >= 0.9999
                r["status"] = "record" if MOCK else ("pass" if ok else "fail")
                row("mk_one_vs_chain" if ONE else "mk_layer_vs_chain", **r)
            except Exception:
                traceback.print_exc()
                err = traceback.format_exc()
                row("mk_layer_vs_chain", t=t, status="fail", error=err[-400:])
                if "TIMEOUT" in err or "hang" in err:
                    flush()
                    print("DEV_MK_LAYER_ABORT device timeout", flush=True)
                    os._exit(3)
                tr.release()
    finally:
        flush()
        if own:
            ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MK_LAYER_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
