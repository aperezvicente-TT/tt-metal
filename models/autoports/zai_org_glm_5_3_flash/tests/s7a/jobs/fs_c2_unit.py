# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a "coll" piece unit, C2 (prefill_megakernel.md §4b H5 / H6, TP2 DRAM form): FabricStreamC1(mode="c2").unit on
the pair. Every chip's partial P_p [1, 1, n B 32, 4096] bf16 random -> stub producers fill every MM core's OUT with its
block -> lanes send the PEER-owned blocks into the owner's landing buffer r (credited launch, fused FS_REDUCED incs per
16-tile slice) -> r on chip p must equal P_q[p's rows] BITWISE; the consumer's sum P_p[p's rows] + r (fp32 -> bf16, on
the host) vs the traced ttnn.reduce_scatter (max |diff|). Traced timing. ARGS: blocks_local ([8, 32]), lanes, nb, replays,
record, tag."""

from __future__ import annotations

import importlib
import json
import os
import time

import torch

import ttnn
import models.autoports.zai_org_glm_5_3_flash.tt.fabric_stream as FS
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced

FS = importlib.reload(FS)
A = globals().get("ARGS", {}) or {}
mesh = globals().get("MESH")
MOCK = os.environ.get("GLM53_MOCK") == "1"
BLOCKS = [int(b) for b in A.get("blocks_local", [8, 32])]
LANES = [tuple(int(v) for v in xy) for xy in A.get("lanes", [[0, 0], [1, 0]])]
NB = int(A.get("nb", 16))
REPLAYS = int(A.get("replays", 50))
OUT = os.environ.get("GLM53_OUT_DIR", "/tmp")
TAG = A.get("tag", "7a-fs-c2-unit")
REC = bool(A.get("record", False))
DRAM = ttnn.DRAM_MEMORY_CONFIG
ROWS_OUT = []


def row(**kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test="fs_c2_unit", tag=TAG, mock=MOCK,
             ndev=int(mesh.get_num_devices()), **kw)  # fmt: skip
    ROWS_OUT.append(r)
    print("ROW", json.dumps(r, default=str), flush=True)
    return r


def timed(tr, fn):
    tr.run(fn)
    if not tr.enabled:
        return None
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh, tr.tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    return round((time.perf_counter() - t0) / REPLAYS * 1e6, 1)


def mm_cores(n_needed: int) -> list:
    """the first worker cores row-major that are not lanes"""
    g = mesh.compute_with_storage_grid_size()
    cores = [(x, y) for y in range(int(g.y)) for x in range(int(g.x)) if (x, y) not in LANES]
    assert len(cores) >= n_needed, (len(cores), n_needed)
    return cores[:n_needed]


def main():
    n = int(mesh.get_num_devices())
    assert n == 2, "the pair"
    fails = 0
    for bl in BLOCKS:
        rows = bl * 32
        gen = torch.Generator().manual_seed(321 + bl)
        ph = torch.randn(n, 1, n * rows, 4096, generator=gen).to(torch.bfloat16)  # chip p's partial = ph[p]
        x = ttnn.from_torch(ph, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))  # fmt: skip
        ref_out = {}

        def ref_fn():
            ref_out["y"] = ttnn.reduce_scatter(x, dim=2, cluster_axis=None, memory_config=DRAM)
            return ref_out["y"]

        tr = Traced(mesh)
        tr.warm(ref_fn)
        us_ref = timed(tr, ref_fn)
        refs = [ttnn.to_torch(t).float().reshape(rows, 4096) for t in ttnn.get_device_tensors(ref_out["y"])] if tr.enabled else None
        tr.release()
        c2 = FS.FabricStreamC1(mesh, bl, mm_cores=mm_cores(n * bl), lanes=LANES, nb=NB, mode="c2")
        fn = lambda: c2.unit(x)  # noqa: E731
        tr = Traced(mesh)
        tr.warm(fn)
        us = timed(tr, fn)
        nbytes = rows * 4096 * 2
        res = dict(blocks_local=bl, rows_local=rows, lanes=len(LANES), nb=NB, us=us, us_ttnn_rs=us_ref,
                   recv_gb_s=None if us is None else round(nbytes / us / 1e3, 2),
                   ttnn_recv_gb_s=None if us_ref is None else round(nbytes / us_ref / 1e3, 2), replays=REPLAYS,
                   note="unit span = stub DRAM->OUT fill of all 64 blocks + C2 (the transfer alone is shorter)")  # fmt: skip
        if tr.enabled:
            rs = [ttnn.to_torch(t).reshape(rows, 4096) for t in ttnn.get_device_tensors(c2.r)]
            res["landed_bitwise"] = [bool(torch.equal(rs[p], ph[1 - p, 0, p * rows:(p + 1) * rows])) for p in range(n)]
            red = [(ph[p, 0, p * rows:(p + 1) * rows].float() + rs[p].float()).to(torch.bfloat16).float() for p in range(n)]
            if refs is not None:
                res["reduced_vs_ttnn_maxabs"] = [float((red[p] - refs[p]).abs().max()) for p in range(n)]
            res["status"] = "pass" if all(res["landed_bitwise"]) else "fail"
            fails += res["status"] == "fail"
        tr.release()
        row(**res)
        c2.release()
        ttnn.deallocate(x)
    if REC and not MOCK:
        with open(os.path.join(OUT, "perf.jsonl"), "a") as fh:
            for r in ROWS_OUT:
                fh.write(json.dumps(r, default=str) + "\n")
    return fails


RESULT = None if main() == 0 else "fail"
