# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel M2 building block check: the stream matmul (tt/mm_stream.py, weight in the mk-stream-v1
per-core layout, many worker cores) vs ttnn.matmul on the SAME bf8 weight values in the standard interleaved layout,
for the dense_kda shapes (4096 x 24576 on 96 cores, 8192 x 4096 and 12288 x 4096 on 64 cores), M = 1 / 32 rows;
traced (eager only as the compile step): PCC / max error, determinism, median traced step time and the weight-stream
GB/s. Random weights (the layout and the rate do not depend on the values). Rows -> pcc.jsonl / tests.jsonl /
perf.jsonl, tagged with the watcher state."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import pcc
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tt.mm_stream import StreamMatmul

MOCK = os.environ.get("GLM53_MOCK") == "1"
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
REPLAYS = int(os.environ.get("GLM53_MM_REPLAYS", "20"))
SHAPES = [(4096, 24576, 96, 4), (8192, 4096, 64, 8), (12288, 4096, 64, 12)]  # k, n, cores, kb
ROWS = []
_FLUSHED = 0


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, test=name,
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
            f.write(
                json.dumps({k: r[k] for k in ("time_utc", "run", "test", "status", "k", "n", "m") if k in r}) + "\n"
            )


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
    mesh = (
        ttnn.open_device(device_id=0)
        if MOCK
        else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=100_000_000)
    )
    tr = Traced(mesh)
    ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)  # fmt: skip
    try:
        for k, n, cores, kb in SHAPES:
            g = torch.Generator().manual_seed(k + n)
            w = torch.randn(k, n, generator=g) * 0.02
            try:
                mm = StreamMatmul(mesh, w, cores=cores, kb=kb)
                w_ref = ttnn.from_torch(w.reshape(1, 1, k, n), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                                        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            except Exception:
                traceback.print_exc()
                row("mm_stream_vs_matmul", k=k, n=n, status="fail", error=traceback.format_exc()[-400:])
                continue
            for m in (1, 32):
                try:
                    x = (torch.randn(1, 1, m, k) * 0.5).to(torch.bfloat16)
                    xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                         memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
                    y_ref = tr.run(lambda: ttnn.matmul(xd, w_ref, compute_kernel_config=ckc, dtype=ttnn.bfloat16))
                    ref = ttnn.to_torch(y_ref).float().reshape(m, n)
                    us_ref = timed(tr)
                    tr.release()
                    y = tr.run(lambda: mm(xd))
                    got = ttnn.to_torch(y).float().reshape(m, n)
                    us = timed(tr)
                    tr.replay()
                    got2 = ttnn.to_torch(y).float().reshape(m, n)
                    tr.release()
                    wbytes = k * n * 1.0625
                    r = dict(k=k, n=n, m=m, cores=cores, kb=kb, nt=mm.nt, execution=tr.mode, pcc_vs_matmul=pcc(got, ref),
                             max_abs=float((got - ref).abs().max()), ref_max=float(ref.abs().max()),
                             deterministic=bool(torch.equal(got, got2)), mm_stream_us=us, matmul_us=us_ref,
                             mm_stream_gbs=wbytes / us / 1e3 if us else None,
                             matmul_gbs=wbytes / us_ref / 1e3 if us_ref else None)  # fmt: skip
                    r["status"] = (
                        "record"
                        if MOCK
                        else ("pass" if r["pcc_vs_matmul"] >= 0.9999 and r["deterministic"] else "fail")
                    )
                    row("mm_stream_vs_matmul", **r)
                except Exception:
                    traceback.print_exc()
                    err = traceback.format_exc()
                    row("mm_stream_vs_matmul", k=k, n=n, m=m, status="fail", error=err[-400:])
                    if "TIMEOUT" in err or "hang" in err:
                        flush()
                        print("DEV_MM_STREAM_ABORT device timeout", flush=True)
                        os._exit(3)
                    tr.release()
    finally:
        flush()
        ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MM_STREAM_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
