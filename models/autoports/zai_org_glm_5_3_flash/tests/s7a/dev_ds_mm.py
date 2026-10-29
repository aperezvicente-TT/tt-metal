# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel building block check: the DS matmul program (tt/ds_mm.py) vs the layer's own DSLinear (today's
DRAM-sharded decode matmul) on the SAME weight tensor and input, for every decode projection of a real layer (L0 by
default: qkv, low-rank, o_proj, gate_up, down), M = 1 / 8 / 32 rows; traced (eager only as the compile step):
output PCC / max error vs DSLinear, then the step time of each (median of timed blocking trace replays) and the weight
stream rate. Rows -> $GLM53_STAGE_EVIDENCE/{pcc.jsonl,tests.jsonl,perf.jsonl}. GLM53_MOCK=1: dry run."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import load_optimized_decoder
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tt.ds_mm import DSMatmul

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "0"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
REPLAYS = int(os.environ.get("GLM53_DSMM_REPLAYS", "20"))
ROWS = []
_FLUSHED = 0


def row(name, **kw):
    r = dict(
        time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        run=RUN,
        layer=LAYER,
        test=name,
        watcher=bool(os.environ.get("TT_METAL_WATCHER")),
        **kw,
    )
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)


def timed(tr: Traced, n: int) -> float | None:
    if not tr.enabled:
        return None
    dts = []
    for _ in range(n):
        t0 = time.perf_counter()
        tr.replay()
        dts.append((time.perf_counter() - t0) * 1e6)
    return sorted(dts)[len(dts) // 2]


def main():
    cfg = hf_config()
    mesh = (
        ttnn.open_device(device_id=0)
        if MOCK
        else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=100_000_000)
    )
    tr = Traced(mesh)
    try:
        dec = load_optimized_decoder(cfg, LAYER, mesh)
        roles = [r for r in os.environ.get("GLM53_DSMM_ROLES", "").split(",") if r]
        lins = {ds.role: ds for ds in dec.layer.ds_linears()}
        order = [lins[r] for r in roles if r in lins] if roles else list(lins.values())
        for ds in order:
            fid = getattr(ds.ckc, "math_fidelity", ttnn.MathFidelity.HiFi2)
            try:
                mm = DSMatmul(mesh, ds, fidelity=fid)
            except Exception as e:
                row("ds_mm_vs_dslinear", role=ds.role, status="info", skipped=str(e)[:200])
                continue
            for m in (1, 8, 32):
                try:
                    x = (torch.randn(1, 1, m, ds.k) * 0.5).to(torch.bfloat16)
                    xd = ttnn.from_torch(
                        x,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                    y_ref = tr.run(lambda: ds(xd))
                    ref = ttnn.to_torch(y_ref).float().reshape(m, -1)[:, : ds.n]
                    us_ref = timed(tr, REPLAYS)
                    tr.release()
                    y_mm = tr.run(lambda: mm(xd))
                    got = ttnn.to_torch(y_mm).float().reshape(m, -1)[:, : ds.n]
                    us_mm = timed(tr, REPLAYS)
                    tr.replay()
                    got2 = ttnn.to_torch(y_mm).float().reshape(m, -1)[:, : ds.n]
                    tr.release()
                    wbytes = ds.k * ds.n_pad * {ttnn.bfloat8_b: 1.0625, ttnn.bfloat16: 2.0}.get(ds.weight.dtype, 2.0)
                    r = dict(role=ds.role, m=m, k=ds.k, n=ds.n, w_dtype=str(ds.weight.dtype), fidelity=str(fid),
                             execution=tr.mode, nb=mm.nb, kb=mm.kb, nr=mm.nr, l1_bytes=mm.l1_bytes(),
                             pcc_vs_dslinear=pcc(got, ref), max_abs_vs_dslinear=float((got - ref).abs().max()),
                             ref_max=float(ref.abs().max()), deterministic=bool(torch.equal(got, got2)),
                             dslinear_us=us_ref, ds_mm_us=us_mm,
                             ds_mm_gbs=(wbytes / us_mm / 1e3) if us_mm else None,
                             dslinear_gbs=(wbytes / us_ref / 1e3) if us_ref else None)  # fmt: skip
                    ok = r["pcc_vs_dslinear"] >= 0.9999 and r["deterministic"]
                    r["status"] = "record" if MOCK else ("pass" if ok else "fail")
                    row("ds_mm_vs_dslinear", **r)
                    ttnn.deallocate(xd)
                except Exception:
                    traceback.print_exc()
                    err = traceback.format_exc()
                    row("ds_mm_vs_dslinear", role=ds.role, m=m, status="fail", error=err[-400:])
                    if "TIMEOUT" in err or "hang" in err:
                        # a device hang: every later call would only queue more work behind it and the teardown
                        # waits on the outstanding reads — record and exit at once (a process that throws and exits
                        # needs no reset)
                        flush()
                        print("DEV_DS_MM_ABORT device timeout", flush=True)
                        os._exit(3)
                    tr.release()
    finally:
        flush()
        ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_DS_MM_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


def flush():
    """Append the rows not written yet (idempotent)."""
    global _FLUSHED
    new, _FLUSHED = ROWS[_FLUSHED:], len(ROWS)
    with open(os.path.join(OUT, "pcc.jsonl"), "a") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT, "tests.jsonl"), "a") as f:
        for r in new:
            f.write(json.dumps({k: r[k] for k in ("time_utc", "run", "test", "status", "role", "m") if k in r}) + "\n")


if __name__ == "__main__":
    sys.exit(main())
