# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a check for Stage 14 / 14c (TP8 B = 32): the standalone KdaRec program on a TP8-sized head shard (H =
GLM53_REC_HEADS, default 8 heads, random inputs and state) in core layouts spread x user groups (GLM53_REC_LAYOUTS,
default 1x1,1x2,4x1,4x2,2x4,1x8: spread = cores per head (value columns), user groups = copies of the head cores taking
contiguous row ranges; H * spread * groups <= 64 cores), T = GLM53_MK_TS rows (default 4,8,32; spread > 1 above 4 rows
is what this checks). Per (T, layout), device vs device and vs the fp64 host oracle (kda_rec.reference_step): the
updated state bitwise equal to layout 1x1's (column-independent recurrence), the gated-normed output (the head's sum of
squares is reduced in a fixed tree at spread > 1: PCC / max-abs vs 1x1), both vs the oracle (PCC); time per call =
traced replay of N_CALLS back-to-back calls without the state restore (timing only) / N_CALLS. Rows -> pcc.jsonl /
tests.jsonl / perf.jsonl."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import pcc
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced

MOCK = os.environ.get("GLM53_MOCK") == "1"
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
TS = [int(v) for v in os.environ.get("GLM53_MK_TS", "4,8,32").split(",")]
H = int(os.environ.get("GLM53_REC_HEADS", "8"))
LAYOUTS = [tuple(int(x) for x in v.split("x")) for v in os.environ.get("GLM53_REC_LAYOUTS",
                                                                       "1x1,1x2,4x1,4x2,2x4,1x8").split(",")]  # fmt: skip
DK, N_CALLS, REPLAYS = 128, 4, 20
ROWS = []


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, test=name, heads=H,
             watcher=bool(os.environ.get("TT_METAL_WATCHER")), **kw)  # fmt: skip
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)


def flush():
    for name, keys in (
        ("pcc.jsonl", None),
        ("tests.jsonl", ("time_utc", "run", "test", "status", "t")),
        ("perf.jsonl", ("time_utc", "run", "test", "t", "layout", "heads", "execution", "timing_us", "watcher")),
    ):
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
    return round(sorted(d)[len(d) // 2] / N_CALLS, 1)


def main(mesh=None, ctx=None):
    """mesh / ctx: run inside the long-lived device process (serve.py)."""
    own = mesh is None
    if ctx is not None and ctx.pop("dev_mk_layer_objs", None) is not None:  # its resident objects pin L1: release
        gc.collect()
    if own:
        mesh = (ttnn.open_device(device_id=0) if MOCK
                else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000))  # fmt: skip
    from models.autoports.zai_org_glm_5_3_flash.tt.kda_rec import KdaRec, reference_step

    tr = Traced(mesh)
    dev = lambda x, dt=ttnn.float32: ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,
                                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    host = lambda x: ttnn.to_torch(x).float()
    try:
        for t in TS:
            g = torch.Generator().manual_seed(1000 + t)
            rnd = lambda *s: torch.randn(*s, generator=g)
            w = 1.0 + 0.1 * rnd(DK)
            hs = dict(q=rnd(1, 1, t, H * DK), k=rnd(1, 1, t, H * DK), v=rnd(1, 1, t, H * DK),
                      g=-torch.rand(1, 1, t, H * DK, generator=g) * 0.2, beta=torch.rand(1, 1, t, H, generator=g),
                      go=rnd(1, 1, t, H * DK), S=rnd(t, H, DK, DK) * 0.05)  # fmt: skip
            o_norm = dev(w.reshape(1, 1, 1, DK), ttnn.bfloat16)
            ins = {k: dev(v) for k, v in hs.items() if k != "S"}
            s0 = dev(hs["S"])
            hv = lambda x: x.reshape(t, H, DK)
            ref = reference_step(hv(hs["q"]), hv(hs["k"]), hv(hs["v"]), hv(hs["g"]), hs["beta"].reshape(t, H),
                                 hv(hs["go"]), hs["S"], ttnn.to_torch(o_norm).float().reshape(DK), 1e-6)  # fmt: skip
            base = None
            for spread, ug in LAYOUTS:
                lay = f"{spread}x{ug}"
                try:
                    assert H * spread * ug <= 64, (H, spread, ug)
                    kda = SimpleNamespace(device=mesh, h=H, d=DK, eps=1e-6, o_norm=o_norm, kda_rec_spread=spread,
                                          kda_rec_ugroups=ug)  # fmt: skip
                    rec = KdaRec(kda)
                    st = dev(hs["S"])
                    on = rec.output(t, ttnn.L1_MEMORY_CONFIG)  # outside the capture (user groups: zeroed on host)
                    call = lambda: rec(ins["q"], ins["k"], ins["v"], ins["g"], ins["beta"], ins["go"], st, rows=t,
                                       on=on)  # fmt: skip

                    def numerics():
                        ttnn.copy(s0, st)
                        return call()

                    tr.run(numerics)
                    got_on = host(on)[..., :t, :].reshape(t, H, DK)
                    got_st = host(st).reshape(t, H, DK, DK)
                    tr.release()
                    tr.run(lambda: [call() for _ in range(N_CALLS)])  # timing only: the state drifts
                    us = timed(tr)
                    tr.release()
                    r = dict(t=t, layout=lay, spread=spread, ugroups=ug, cores=H * spread * ug, execution=tr.mode,
                             timing_us=dict(per_call=us), on_pcc_ref=pcc(got_on, ref["on"]),
                             state_pcc_ref=pcc(got_st, ref["state"]),
                             state_maxabs_ref=float((got_st.double() - ref["state"]).abs().max()))  # fmt: skip
                    if base is None:
                        base = (got_on, got_st)
                    else:
                        r.update(state_bitwise_vs_1x1=bool(torch.equal(got_st, base[1])),
                                 on_bitwise_vs_1x1=bool(torch.equal(got_on, base[0])),
                                 on_pcc_vs_1x1=pcc(got_on, base[0]),
                                 on_maxabs_vs_1x1=float((got_on - base[0]).abs().max()))  # fmt: skip
                    ok = (r["on_pcc_ref"] >= 0.9999 and r["state_pcc_ref"] >= 0.99999
                          and r.get("state_bitwise_vs_1x1", True) and r.get("on_pcc_vs_1x1", 1.0) >= 0.99999)  # fmt: skip
                    r["status"] = "record" if MOCK else ("pass" if ok else "fail")
                    row("kda_rec_layouts", **r)
                    del rec, st, on
                except Exception:
                    traceback.print_exc()
                    tr.release()
                    err = traceback.format_exc()
                    row("kda_rec_layouts", t=t, layout=lay, status="fail", error=err[-400:])
                    if "TIMEOUT" in err or "hang" in err:
                        flush()
                        print("DEV_KDA_REC_SPREAD_ABORT device timeout", flush=True)
                        os._exit(3)
            del ins, s0, o_norm
            gc.collect()
    finally:
        flush()
        if own:
            ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_KDA_REC_SPREAD_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
