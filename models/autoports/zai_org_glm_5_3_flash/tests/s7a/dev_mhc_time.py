# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a glue trims (Stage 14's top item: mhc_pre <= 30 us at B=1): the mhc_pre program alone, traced, 8 calls back
to back per replay (per-call time = replay / 8), layer GLM53_S7A_LAYER (default 0), attention + FFN instances, flat
DRAM input (the layer input's layout), T = GLM53_MK_TS (default 1,4,32). Variants per case (one job, same inputs):
`tiles` = whole x tiles from DRAM (today), `rows` = GLM53_MHC_ROWREAD=1 (Stage 14's decode-row read, T <= 8). hn of the two
variants must be bitwise equal. Rows -> pcc.jsonl / tests.jsonl / perf.jsonl."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_mk_a import mhc_args
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_mk_ffn import ffn_mhc_args
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

LAYER = int(os.environ.get("GLM53_S7A_LAYER", "0"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
TS = [int(v) for v in os.environ.get("GLM53_MK_TS", "1,4,32").split(",")]
N_CALLS, REPLAYS = 8, 30
ROWS = []


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, layer=LAYER, test=name,
             watcher=bool(os.environ.get("TT_METAL_WATCHER")), **kw)  # fmt: skip
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)


def timed(tr):
    d = []
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        tr.replay()
        d.append((time.perf_counter() - t0) * 1e6)
    return sorted(d)[len(d) // 2]


def main(mesh=None, ctx=None):
    own = mesh is None
    if ctx is not None and ctx.pop("dev_mk_layer_objs", None) is not None:  # its resident objects pin L1: release
        import gc

        gc.collect()
    if own:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    tr = Traced(mesh)
    saved = os.environ.get("GLM53_MHC_ROWREAD")
    try:
        cfg = hf_config()
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        pres = dict(attn=M.MhcPre(mesh, **mhc_args(source, cfg, LAYER)),
                    ffn=M.MhcPre(mesh, **ffn_mhc_args(source, cfg, LAYER)))  # fmt: skip
        g = golden(LAYER, 2048)["streams_in"].float()
        for t in TS:
            x = ttnn.from_torch(g[0, 300 : 300 + 37 * t : 37].reshape(1, 1, t, 16384), dtype=ttnn.float32,
                                layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            pc = M.PC_STACKED if t <= 4 else M.PC_STREAM_MAJOR
            for name, pre in pres.items():
                try:
                    res = {}
                    for variant in ("tiles", "rows"):
                        os.environ["GLM53_MHC_ROWREAD"] = "1" if variant == "rows" else "0"
                        fn = lambda pre=pre, x=x, t=t, pc=pc: [
                            pre(x, t=t, x_mode=M.X_FLAT, xout_mode=M.XOUT_NONE, pc_mode=pc).hn for _ in range(N_CALLS)
                        ]
                        outs = tr.run(fn)
                        hn = ttnn.to_torch(outs[-1])
                        us = timed(tr) / N_CALLS
                        tr.release()
                        res[variant] = (hn, round(us, 1))
                    same = bool(torch.equal(res["tiles"][0], res["rows"][0]))
                    row("mhc_pre_time", t=t, instance=name, execution="traced",
                        timing_us=dict(tiles=res["tiles"][1], rows=res["rows"][1]), hn_bitwise_rows_vs_tiles=same,
                        status="pass" if same else "fail")  # fmt: skip
                except Exception:
                    traceback.print_exc()
                    tr.release()
                    row("mhc_pre_time", t=t, instance=name, status="fail", error=traceback.format_exc()[-400:])
    finally:
        if saved is None:
            os.environ.pop("GLM53_MHC_ROWREAD", None)
        else:
            os.environ["GLM53_MHC_ROWREAD"] = saved
        for name, keys in (("pcc.jsonl", None), ("tests.jsonl", ("time_utc", "run", "test", "status", "t")),
                           ("perf.jsonl", ("time_utc", "run", "test", "t", "instance", "execution", "timing_us",
                                           "watcher"))):  # fmt: skip
            with open(os.path.join(OUT, name), "a") as f:
                for r in ROWS:
                    if name == "perf.jsonl" and not r.get("timing_us"):
                        continue
                    f.write(json.dumps(r if keys is None else {k: r[k] for k in keys if k in r}) + "\n")
        if own:
            ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MHC_TIME_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
