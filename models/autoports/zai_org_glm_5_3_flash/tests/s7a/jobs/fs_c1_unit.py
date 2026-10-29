# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a "coll" piece unit, C1 (prefill_megakernel.md §4b H1 / H2): FabricStreamC1.unit on the pair. x̂ rows [1, 1, n B
32, 4096] bf16 random, sharded by rows (B blocks per chip) -> stub producers fill the local blocks' XS on their MM cores ->
lanes write every local block into the same XS of the peer (credited launch, fused FS_ARRIVE incs per 16-tile slice) ->
stub consumers dump every MM core's XS to y. y on every chip must equal x BITWISE. Traced (one capture, replays timed),
against the traced ttnn.all_gather of the same rows. ARGS: blocks_local ([8, 32]), lanes ([[0, 0], [1, 0]]), nb (16),
replays (50), record, tag."""

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
TAG = A.get("tag", "7a-fs-c1-unit")
REC = bool(A.get("record", False))
DRAM = ttnn.DRAM_MEMORY_CONFIG
ROWS_OUT = []


def row(**kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test="fs_c1_unit", tag=TAG, mock=MOCK,
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
    fails = 0
    for bl in BLOCKS:
        rows = bl * 32
        gen = torch.Generator().manual_seed(123 + bl)
        xh = torch.randn(1, 1, n * rows, 4096, generator=gen).to(torch.bfloat16)
        x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2))  # fmt: skip
        def ref_fn():  # the prefill_tp gather (both axes on a 2-D mesh)
            y = x
            for ax in ([1, 0] if int(mesh.shape[0]) > 1 and int(mesh.shape[1]) > 1 else [None]):
                y = ttnn.all_gather(y, dim=2, cluster_axis=ax, memory_config=DRAM)
            return y

        tr = Traced(mesh)
        tr.warm(ref_fn)
        us_ref = timed(tr, ref_fn)
        tr.release()
        two_axes = int(mesh.shape[0]) == 2 and int(mesh.shape[1]) > 1  # the 2x4 (TP8): FabricStreamC1X
        cls = FS.FabricStreamC1X if two_axes else FS.FabricStreamC1
        c1 = cls(mesh, bl, mm_cores=mm_cores(n * bl), lanes=LANES, nb=NB)
        fn = lambda: c1.unit(x)  # noqa: E731
        tr = Traced(mesh)
        tr.warm(fn)
        us = timed(tr, fn)
        nbytes = rows * 4096 * 2
        res = dict(blocks_local=bl, rows_local=rows, lanes=len(LANES), nb=NB, us=us, us_ttnn_ag=us_ref,
                   recv_gb_s=None if us is None else round(nbytes * (n - 1) / us / 1e3, 2),
                   ttnn_recv_gb_s=None if us_ref is None else round(nbytes * (n - 1) / us_ref / 1e3, 2), replays=REPLAYS,
                   note="unit span = stub DRAM->XS fill + C1 + stub XS->DRAM dump (the C1 transfer alone is shorter)")  # fmt: skip
        if tr.enabled:
            ys = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(c1.y)]
            res["bitwise"] = [bool(torch.equal(y, xh)) for y in ys]
            res["status"] = "pass" if all(res["bitwise"]) else "fail"
            fails += res["status"] == "fail"
        tr.release()
        row(**res)
        c1.release()
        ttnn.deallocate(x)
    if REC and not MOCK:
        with open(os.path.join(OUT, "perf.jsonl"), "a") as fh:
            for r in ROWS_OUT:
                fh.write(json.dumps(r, default=str) + "\n")
    return fails


RESULT = None if main() == 0 else "fail"
