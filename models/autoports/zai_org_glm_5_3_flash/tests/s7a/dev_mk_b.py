# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel step B check: P1 mhc_pre (attention) -> P2 qkv + low-rank (16 readers) -> P3 kda_pre (64 owners)
in ONE program (tt/mk_dense.py MkStepB: the owner phases composed into one kernel per RISC) vs the chain it replaces
(MhcPre, DSLinear qkv, DSLinear low-rank, KdaPre), dense_kda L0, T = 1 / 2 / 4 (kda_pre ROWS mode); traced (eager only
as the compile step: every case warmed before any capture; the conv history is restored inside the traced call). hn / qkv bitwise, low-rank (HiFi4 like DSLinear)
and kda outputs PCC, history bitwise; traced time of each. Rows -> pcc.jsonl / tests.jsonl."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import load_optimized_decoder
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_kda_pre import masks
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_mk_a import mhc_args
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
from models.autoports.zai_org_glm_5_3_flash.tt.kda_pre import KdaPre
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


def main():
    cfg = hf_config()
    mesh = (
        ttnn.open_device(device_id=0)
        if MOCK
        else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    )
    tr = Traced(mesh)
    try:
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        dec = load_optimized_decoder(cfg, LAYER, mesh)
        kda = dec.layer.attn
        lins = {d.role: d for d in dec.layer.ds_linears()}
        args = mhc_args(source, cfg, LAYER)
        mk = MkStepB(mesh, args, kda, lins["kda_qkv"], lins["kda_lowrank"])
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

            def run_mk(x=x, act=act, hist0=hist0, st=st_a, kw=kw):
                ttnn.copy(hist0, st.C)
                return mk(x, st.C, act, **kw)

            def run_chain(x=x, act=act, hist0=hist0, st=st_b, kw=kw, t=t):
                ttnn.copy(hist0, st.C)
                r = ref_mhc(x, hn_dtype=ttnn.bfloat16, **kw)
                q_ = lins["kda_qkv"](r.hn)
                l_ = lins["kda_lowrank"](r.hn)
                return r, q_, l_, ref_kda(q_, l_, st.C, act, rows=t)

            cases[t] = (run_mk, run_chain, st_a, st_b)
        tr.warm(*[f for c in cases.values() for f in c[:2]])  # every program compiled before any capture
        for t, (run_mk, run_chain, st_a, st_b) in cases.items():
            try:
                o, qkv, lr, ko = tr.run(run_mk)
                got = dict(hn=host(o.hn), qkv=host(qkv)[..., :24576], lr=host(lr)[..., :320], hist=host(st_a.C))
                got.update({n: host(v) for n, v in zip(("q", "k", "v", "g", "beta", "gate_out"), ko[:6])})
                us_mk = timed(tr)
                tr.release()
                o2, qkv2, lr2, ko2 = tr.run(run_chain)
                want = dict(hn=host(o2.hn), qkv=host(qkv2), lr=host(lr2), hist=host(st_b.C))
                want.update({n: host(v) for n, v in zip(("q", "k", "v", "g", "beta", "gate_out"), ko2[:6])})
                us_chain = timed(tr)
                tr.release()
                r = dict(t=t, execution=tr.mode, mk_us=us_mk, chain_us=us_chain,
                         alloc_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1")  # fmt: skip
                for k_ in got:
                    a, b = got[k_].reshape(-1), want[k_].reshape(-1)[: got[k_].numel()]
                    r[f"{k_}_bitwise"] = bool(torch.equal(a, b))
                    r[f"{k_}_pcc"] = pcc(a, b)
                ok = r["hn_bitwise"] and r["qkv_bitwise"] and r["hist_bitwise"] and min(
                    r[f"{k_}_pcc"] for k_ in got) >= 0.9999  # fmt: skip
                r["status"] = "record" if MOCK else ("pass" if ok else "fail")
                row("mk_b_vs_chain", **r)
            except Exception:
                traceback.print_exc()
                err = traceback.format_exc()
                row("mk_b_vs_chain", t=t, status="fail", error=err[-400:])
                if "TIMEOUT" in err or "hang" in err:
                    flush()
                    print("DEV_MK_B_ABORT device timeout", flush=True)
                    os._exit(3)
                tr.release()
    finally:
        flush()
        ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MK_B_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
