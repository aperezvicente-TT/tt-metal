# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cost of one EAGER call of the Stage 7a generic_op programs (the Stage 3 gates with the flags on ran 5-15x
slower than with them off): per program, the wall time of N eager calls with fresh output tensors (synchronised), and
the split between building the ProgramDescriptor in Python and the ttnn.generic_op call (program-cache hit or miss).
The eager calls here ARE the measurement of the host path (the operator's no-eager rule concerns device validation;
nothing is validated). Rows -> $GLM53_STAGE_EVIDENCE/perf.jsonl."""

from __future__ import annotations

import json
import os
import sys
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, hf_config
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_mk_a import mhc_args
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_post as P
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

OUT = os.environ.get("GLM53_STAGE_EVIDENCE", "/tmp")
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
N = 20


def main():
    cfg = hf_config()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    rows = []
    try:
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        prog = M.MhcPre(mesh, **mhc_args(source, cfg))
        mp = P.MhcPost(mesh)
        t = 1
        x = ttnn.from_torch(torch.randn(1, 1, t, 16384), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        kw = dict(t=t, x_mode=M.X_FLAT, xout_mode=M.XOUT_STACKED, pc_mode=M.PC_STACKED)
        o = prog(x, **kw)
        ttnn.synchronize_device(mesh)
        y = ttnn.from_torch(torch.randn(1, 1, t, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                            memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
        mp(y, o.x_out, o.coef, t=t, x_mode=P.X_STACKED, out_mode=P.OUT_STACKED)
        ttnn.synchronize_device(mesh)
        for name, fn in (("mhc_pre", lambda: prog(x, **kw)),
                         ("mhc_post", lambda: mp(y, o.x_out, o.coef, t=t, x_mode=P.X_STACKED, out_mode=P.OUT_STACKED)),
                         ("ttnn_add_ref", lambda: ttnn.add(y, y))):  # fmt: skip
            d = []
            for _ in range(N):
                t0 = time.perf_counter()
                r = fn()
                ttnn.synchronize_device(mesh)
                d.append((time.perf_counter() - t0) * 1e3)
                del r
            # descriptor build alone (mhc_pre)
            row = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, test="eager_call_cost",
                       program=name, calls=N, median_ms=sorted(d)[N // 2], max_ms=max(d), min_ms=min(d), status="info")  # fmt: skip
            rows.append(row)
            print("ROW", json.dumps(row), flush=True)
        hn = ttnn.empty((1, 1, t, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.L1_MEMORY_CONFIG)
        d = []
        for _ in range(N):
            t0 = time.perf_counter()
            prog.descriptor(x, hn, o.post, o.comb, o.x_out, o.coef, **kw)
            d.append((time.perf_counter() - t0) * 1e3)
        row = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, test="eager_call_cost",
                   program="mhc_pre_descriptor_build_only", calls=N, median_ms=sorted(d)[N // 2], status="info")  # fmt: skip
        rows.append(row)
        print("ROW", json.dumps(row), flush=True)
    finally:
        with open(os.path.join(OUT, "perf.jsonl"), "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        ttnn.close_mesh_device(mesh)
    print("DEV_EAGER_COST_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
