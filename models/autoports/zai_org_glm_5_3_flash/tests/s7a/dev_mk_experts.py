# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a M4: the routed experts as reader phases (tt/mk_experts.py ReaderExperts) vs fused_experts (the op chain's
routed path) on the same x and the same router ids / scores, layer GLM53_S7A_LAYER (default 4), T = GLM53_MK_TS
(default 1,4,8,32); device vs device, traced (every case warmed before any capture): output bitwise / PCC and traced
time of each. Rows -> pcc.jsonl / tests.jsonl / perf.jsonl."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced, cached_decoder
from models.autoports.zai_org_glm_5_3_flash.tt.mk_experts import ReaderExperts

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "4"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
TS = [int(v) for v in os.environ.get("GLM53_MK_TS", "1,4,8,32").split(",")]
REPLAYS = 20
ROWS = []


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, layer=LAYER, test=name,
             watcher=bool(os.environ.get("TT_METAL_WATCHER")), **kw)  # fmt: skip
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)


def flush():
    for name, keys in (("pcc.jsonl", None), ("tests.jsonl", ("time_utc", "run", "test", "status", "t")),
                       ("perf.jsonl", ("time_utc", "run", "test", "t", "execution", "timing_us", "watcher",
                                       "experts_in_union", "readers_per_bank", "ring_slots"))):  # fmt: skip
        with open(os.path.join(OUT, name), "a") as f:
            for r in ROWS:
                if name == "perf.jsonl" and not r.get("timing_us"):
                    continue
                f.write(json.dumps(r if keys is None else {k: r[k] for k in keys if k in r}) + "\n")


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
    """mesh / ctx: run inside the long-lived device process (serve.py)."""
    own = mesh is None
    cfg = hf_config()
    if own:
        mesh = (ttnn.open_device(device_id=0) if MOCK
                else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000))  # fmt: skip
    tr = Traced(mesh)
    try:
        dec = cached_decoder(cfg, LAYER, mesh, ctx)
        mlp = dec.layer.mlp
        fp32_router = dec.layer.fcfg.router_fp32_input
        rx = ReaderExperts(mesh, mlp.gate_up, mlp.down, top_k=mlp.top_k, scaling=mlp.scaling, limit=mlp.limit,
                           intermediate=mlp.intermediate, hidden=mlp.hidden,
                           readers_per_bank=int(os.environ.get("GLM53_MK_READERS", "3")),
                           ring_slots=int(os.environ.get("GLM53_MKX_RING", "12")))  # fmt: skip
        # GLM53_MKX_EP=P: expert parallelism emulated on one chip — P programs, each with experts [c E/P, (c+1) E/P),
        # their partial outputs summed (fp32, host) vs fused_experts; each slice timed alone (the per-chip routed time)
        ep = int(os.environ.get("GLM53_MKX_EP", "0"))
        slices = []
        if ep:
            per = mlp.num_experts // ep
            slices = [ReaderExperts(mesh, mlp.gate_up[c * per:(c + 1) * per], mlp.down[c * per:(c + 1) * per],
                                    top_k=mlp.top_k, scaling=mlp.scaling, limit=mlp.limit,
                                    intermediate=mlp.intermediate, hidden=mlp.hidden, e_lo=c * per,
                                    num_experts=mlp.num_experts,
                                    readers_per_bank=int(os.environ.get("GLM53_MK_READERS", "3")))
                      for c in range(ep)]  # fmt: skip
        torch.manual_seed(7)
        cases = {}
        for t in TS:
            x32 = torch.randn(1, 1, t, mlp.hidden) * 0.5
            xr = ttnn.from_torch(x32, dtype=ttnn.float32 if fp32_router else ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
            xb = ttnn.from_torch(x32.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
            ids, scores = mlp.router(xr, decode=True)  # fixed routing for both sides
            micro, block = mlp.cfg.moe_decode_micro_batch, mlp.cfg.experts_block_size
            cases[t] = (lambda xb=xb, ids=ids, scores=scores, t=t: rx(xb, ids, scores, t=t),
                        lambda xb=xb, ids=ids, scores=scores, t=t: ttnn.reshape(
                            mlp.routed(xb, ids, scores, micro, block), (1, 1, t, mlp.hidden)),
                        ids)  # fmt: skip
        ep_cases = {}
        for t, (run_mk, run_chain, ids) in cases.items():
            if slices:
                xb_, ids_, sc_ = run_mk.__defaults__[:3]
                ep_cases[t] = [lambda sx=sx, xb_=xb_, ids_=ids_, sc_=sc_, t=t: sx(xb_, ids_, sc_, t=t) for sx in slices]
        tr.warm(*[f for c in cases.values() for f in c[:2]], *[f for fs in ep_cases.values() for f in fs])
        for t, fs in ep_cases.items():
            try:
                parts, us = [], []
                for f in fs:
                    yp = tr.run(f)
                    parts.append(ttnn.to_torch(yp).float().reshape(-1, mlp.hidden)[:t])
                    us.append(timed(tr))
                    tr.release()
                yw = tr.run(cases[t][1])
                want = ttnn.to_torch(yw).float().reshape(-1, mlp.hidden)[:t]
                tr.release()
                got = sum(parts)
                r = dict(t=t, ep=ep, execution=tr.mode, timing_us=dict(slices=us, max_slice=max(us) if us[0] else None),
                         pcc=pcc(got, want), maxabs=float((got - want).abs().max()),
                         bitwise=bool(torch.equal(got, want)))  # fmt: skip
                r["status"] = "record" if MOCK else ("pass" if r["pcc"] >= 0.9999 else "fail")
                row("mk_experts_ep_vs_fused_experts", **r)
            except Exception:
                traceback.print_exc()
                row("mk_experts_ep_vs_fused_experts", t=t, status="fail", error=traceback.format_exc()[-400:])
                tr.release()
        for t, (run_mk, run_chain, ids) in cases.items():
            if slices:
                continue
            try:
                y = tr.run(run_mk)
                got = ttnn.to_torch(y).float().reshape(-1, mlp.hidden)[:t]
                us_mk = timed(tr)
                tr.release()
                y2 = tr.run(run_chain)
                want = ttnn.to_torch(y2).float().reshape(-1, mlp.hidden)[:t]
                us_chain = timed(tr)
                tr.release()
                idh = ttnn.to_torch(ids).reshape(-1, ids.shape[-1])[:t].long()
                union = int(torch.unique(idh[idh < mlp.num_experts]).numel())
                r = dict(t=t, execution=tr.mode, timing_us=dict(mk_experts=us_mk, fused_experts=us_chain),
                         experts_in_union=union, readers_per_bank=int(os.environ.get("GLM53_MK_READERS", "3")),
                         ring_slots=rx.ring_slots, bitwise=bool(torch.equal(got, want)), pcc=pcc(got, want),
                         maxabs=float((got - want).abs().max()),
                         n_diff=int((got != want).sum()), numel=got.numel())  # fmt: skip
                r["status"] = "record" if MOCK else ("pass" if r["pcc"] >= 0.9999 else "fail")
                row("mk_experts_vs_fused_experts", **r)
            except Exception:
                traceback.print_exc()
                err = traceback.format_exc()
                row("mk_experts_vs_fused_experts", t=t, status="fail", error=err[-400:])
                tr.release()
                if "TIMEOUT" in err or "hang" in err:
                    flush()
                    print("DEV_MK_EXPERTS_ABORT device timeout", flush=True)
                    os._exit(3)
    finally:
        flush()
        if own:
            ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MK_EXPERTS_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
