# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel step C / D (GLM53_MK_STEP=D: + P6 the attention mhc_post on the owners -> s1) check: P1 mhc_pre (attention) -> P2 qkv + low-rank (16 readers) -> P3 kda_pre (64 owners)
-> P4 kda_rec (recurrence + gated norm, head h on owner h) -> P5 o_proj (readers) in ONE program (tt/mk_dense.py
MkStepB with ds_o) vs the chain it replaces (MhcPre, DSLinear qkv / low-rank, KdaPre, fused_recurrent_kda + gated norm
+ DSLinear o_proj), dense_kda L0, T = 1 / 2 / 4; traced (eager only as the compile step: every case warmed before any
capture; the conv history and the recurrent state are restored inside the traced call). hn / qkv / low-rank / history
/ state bitwise, kda outputs, on (o_proj input) and the attention output PCC; traced time of each.
Rows -> pcc.jsonl / tests.jsonl."""

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
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced, cached_decoder
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
from models.autoports.zai_org_glm_5_3_flash.tt.kda_pre import KdaPre
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import OUT_STACKED, X_STACKED, MhcPost
from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import MkStepB
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "0"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
REPLAYS = 20
ROWS = []
_FLUSHED = 0


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, layer=LAYER, test=name,
             watcher=bool(os.environ.get("TT_METAL_WATCHER")), **kw)  # fmt: skip
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


def host(t):
    return ttnn.to_torch(t).float()


def timed(tr):
    if not tr.enabled:
        return None
    d = []
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        tr.replay()
        d.append((time.perf_counter() - t0) * 1e6)
    return sorted(d)[len(d) // 2]


def main(mesh=None, ctx=None):
    """mesh / ctx: run inside the long-lived device process (serve.py): no open / close, the decoder cached."""
    own = mesh is None
    cfg = hf_config()
    # GLM53_WORKER_L1 (bytes): open with a smaller allocatable L1 -> a larger kernel-config ring (the megakernel's
    # owner binaries at -O3 with GLM53_MK_OPT=O3); 0 = the default
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
        kda = dec.layer.attn
        lins = {d.role: d for d in dec.layer.ds_linears()}
        args = mhc_args(source, cfg, LAYER)
        STEP_D = os.environ.get("GLM53_MK_STEP", "C") == "D"
        PROBE = os.environ.get("GLM53_MK_PROBE") == "1"
        mk = MkStepB(mesh, args, kda, lins["kda_qkv"], lins["kda_lowrank"], ds_o=lins["kda_o"], post=STEP_D,
                     probe=PROBE)  # fmt: skip
        mpost = MhcPost(mesh)
        cap = {}
        orig_lin = kda._lin

        def lin(role, x_, *a, **k):
            if role == "ds_o":
                cap["on"] = x_
            return orig_lin(role, x_, *a, **k)

        kda._lin = lin
        ref_mhc = M.MhcPre(mesh, **args)
        ref_kda = KdaPre(kda, rows_max=4)
        g = golden(LAYER, 2048)["streams_in"].float()
        cases = {}
        for t in (1, 2, 4):
            kw = dict(t=t, x_mode=M.X_FLAT, xout_mode=M.XOUT_STACKED, pc_mode=M.PC_STACKED)
            x = ttnn.from_torch(g[0, 300 : 300 + 37 * t : 37].reshape(1, 1, t, 16384), dtype=ttnn.float32,
                                layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            active = torch.tensor([(i % 3) != 2 for i in range(t)]) if t > 1 else torch.tensor([True])
            act = masks(mesh, active, kda.lower_bound)
            hist0 = ttnn.from_torch((torch.randn(1, 3, t, 24576) * 0.5).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            st_a, st_b = kda.allocate_state(t), kda.allocate_state(t)
            s0 = ttnn.from_torch(torch.randn(t, 64, 128, 128) * 0.1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                 device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip

            def run_mk(x=x, act=act, hist0=hist0, s0=s0, st=st_a, kw=kw):
                ttnn.copy(hist0, st.C)
                ttnn.copy(s0, st.S)
                return mk(x, st.C, act, state=st.S, **kw)

            def run_chain(x=x, act=act, hist0=hist0, s0=s0, st=st_b, kw=kw, t=t):
                ttnn.copy(hist0, st.C)
                ttnn.copy(s0, st.S)
                r = ref_mhc(x, hn_dtype=ttnn.bfloat16, **kw)
                q_ = lins["kda_qkv"](r.hn)
                l_ = lins["kda_lowrank"](r.hn)
                ko_ = ref_kda(q_, l_, st.C, act, rows=t)
                a_ = kda._recurrence_and_output(*ko_[:6], st, t)
                if STEP_D:
                    s1_ = mpost(a_, r.x_out, r.coef, t=t, x_mode=X_STACKED, out_mode=OUT_STACKED)
                    return r, q_, l_, ko_, cap["on"], a_, s1_
                return r, q_, l_, ko_, cap["on"], a_

            cases[t] = (run_mk, run_chain, st_a, st_b)
        tr.warm(*[f for c in cases.values() for f in c[:2]])  # every program compiled before any capture
        for t, (run_mk, run_chain, st_a, st_b) in cases.items():
            try:
                res = tr.run(run_mk)
                o, qkv, lr, ko, on, att = res[:6]
                got = dict(hn=host(o.hn), qkv=host(qkv)[..., :24576], lr=host(lr)[..., :320], hist=host(st_a.C),
                           state=host(st_a.S), on=host(on), attn=host(att)[..., :4096])  # fmt: skip
                if STEP_D:
                    got["s1"] = host(res[6])
                got.update({n: host(v) for n, v in zip(("q", "k", "v", "g", "beta", "gate_out"), ko[:6])})
                us_mk = timed(tr)
                if PROBE:  # the last replay's marks
                    from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import decode_probe_b

                    row("mk_probe", t=t, step="D" if STEP_D else "C", status="info",
                        **decode_probe_b(ttnn.to_torch(mk.probe)))  # fmt: skip
                tr.release()
                res2 = tr.run(run_chain)
                o2, qkv2, lr2, ko2, on2, att2 = res2[:6]
                want = dict(hn=host(o2.hn), qkv=host(qkv2), lr=host(lr2), hist=host(st_b.C), state=host(st_b.S),
                            on=host(on2), attn=host(att2))  # fmt: skip
                if STEP_D:
                    want["s1"] = host(res2[6])
                want.update({n: host(v) for n, v in zip(("q", "k", "v", "g", "beta", "gate_out"), ko2[:6])})
                us_chain = timed(tr)
                tr.release()
                r = dict(t=t, step="D" if STEP_D else "C", execution=tr.mode, mk_us=us_mk, chain_us=us_chain,
                         mk_opt=os.environ.get("GLM53_MK_OPT", "size"), worker_l1=wl1 or None,
                         alloc_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1")  # fmt: skip
                for k_ in got:
                    a, b = got[k_].reshape(-1), want[k_].reshape(-1)[: got[k_].numel()]
                    r[f"{k_}_bitwise"] = bool(torch.equal(a, b))
                    r[f"{k_}_pcc"] = pcc(a, b)
                # bitwise: the data-movement outputs and the qkv phase (same K blocking as DSLinear); the low-rank
                # phase (K block 8 vs DSLinear's) and what derives from it (g, state, on, attn) are arithmetic: PCC
                ok = all(r[f"{k_}_bitwise"] for k_ in ("hn", "qkv", "hist")) and min(
                    r[f"{k_}_pcc"] for k_ in got) >= 0.9999  # fmt: skip
                r["bitwise_required"] = ["hn", "qkv", "hist"]
                r["status"] = "record" if MOCK else ("pass" if ok else "fail")
                row("mk_d_vs_chain" if STEP_D else "mk_c_vs_chain", **r)
            except Exception:
                traceback.print_exc()
                err = traceback.format_exc()
                row("mk_c_vs_chain", t=t, status="fail", error=err[-400:])
                if "TIMEOUT" in err or "hang" in err:
                    flush()
                    print("DEV_MK_C_ABORT device timeout", flush=True)
                    os._exit(3)
                tr.release()
    finally:
        flush()
        if own:
            ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MK_C_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
