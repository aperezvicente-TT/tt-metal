# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a DRAM-saturation demo (operator request 2026-10-19 17:11Z): megakernel step R — the five dense_kda weight
matmuls of a layer (qkv, low-rank, o_proj, gate_up, down; L0 weights, DSLinear's DRAM-sharded tensors in place) in ONE
program on the 16 bank-adjacent DS readers (tt/mk_readers.py) — captured once (traced, every case warmed first) and
replayed back to back for GLM53_SAT_S seconds with no host readback between replays: GLM53_SAT_BATCH non-blocking
replays per synchronize. Prints / writes START / END (UTC), replays, mean us per replay, weight bytes per replay and the
resulting GB/s, so tt-mgmt's MEMRD for the device can be read over the window. Row -> perf.jsonl."""

from __future__ import annotations

import json
import os
import sys
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config
from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import load_optimized_decoder
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import MMPhase, ReaderPhases

LAYER = int(os.environ.get("GLM53_S7A_LAYER", "0"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", "/tmp")
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("sat_%Y%m%dT%H%M%SZ", time.gmtime()))
SECONDS = float(os.environ.get("GLM53_SAT_S", "60"))
BATCH = int(os.environ.get("GLM53_SAT_BATCH", "50"))
REQ = os.environ.get("GLM53_SAT_REQUEST", "")
KB = dict(kda_qkv=2, kda_lowrank=8, kda_o=8, mlp_gate_up=2, mlp_down=8)
utc = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def note(line):
    print(line, flush=True)
    if REQ:
        with open(REQ, "a") as f:
            f.write(line + "\n")


def main():
    cfg = hf_config()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    tr = Traced(mesh)
    try:
        dec = load_optimized_decoder(cfg, LAYER, mesh)
        lins = {d.role: d for d in dec.layer.ds_linears()}
        phases = [MMPhase(ds=lins[r], kb=KB[r]) for r in KB if r in lins]
        rp = ReaderPhases(mesh, phases)
        from models.autoports.zai_org_glm_5_3_flash.tt.ds_mm import TILE_BYTES

        wbytes = sum((p.ds.k // 32) * (p.ds.n_pad // 32) * TILE_BYTES[p.ds.weight.dtype] for p in phases)
        xs = [ttnn.from_torch((torch.randn(1, 1, 1, p.ds.k) * 0.5).to(torch.bfloat16), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG)
              for p in phases]  # fmt: skip
        fn = lambda: rp(xs)
        tr.warm(fn)
        tr.run(fn, replays=3)
        ttnn.synchronize_device(mesh)
        n, t_end = 0, time.time() + SECONDS
        start = utc()
        note(f"{start} STAGE 7a DRAM-SATURATION DEMO START (chip {os.environ.get('GLM53_CHIP', '?')}): step R = 5 "
             f"dense_kda L{LAYER} weight matmuls in ONE program, {wbytes / 1e6:.1f} MB of weights per replay, replays "
             f"back to back ({BATCH} non-blocking per synchronize) for {SECONDS:.0f} s")  # fmt: skip
        t0 = time.perf_counter()
        while time.time() < t_end:
            for _ in range(BATCH):
                ttnn.execute_trace(mesh, tr.tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            n += BATCH
        dt = time.perf_counter() - t0
        end = utc()
        us = dt / n * 1e6
        gbs = wbytes / (us * 1e-6) / 1e9
        note(f"{end} STAGE 7a DRAM-SATURATION DEMO END: {n} replays in {dt:.2f} s, mean {us:.1f} us per replay, "
             f"{wbytes / 1e6:.1f} MB per replay -> {gbs:.1f} GB/s of weight reads (wall clock incl. per-replay "
             f"launch; peak 512 GB/s); window {start} .. {end}")  # fmt: skip
        tr.release()
        with open(os.path.join(OUT, "perf.jsonl"), "a") as f:
            f.write(json.dumps(dict(time_utc=end, run=RUN, test="dram_saturation_demo", layer=LAYER, start=start,
                                    end=end, replays=n, seconds=round(dt, 2), us_per_replay=round(us, 1),
                                    bytes_per_replay=int(wbytes), gbs=round(gbs, 1), batch=BATCH, watcher=False,
                                    execution="traced", status="info")) + "\n")  # fmt: skip
    finally:
        ttnn.close_mesh_device(mesh)
    print("DEV_MK_SAT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
