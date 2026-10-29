# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a "coll" piece unit, C2 with the OWNER-LANE REDUCE (prefill_megakernel.md §4b H5 -> H6): FabricStreamRSR on the
pair. Every chip's partial P_p [1, 1, n B 32, 4096] bf16 random -> y on chip p [1, 1, B 32, 4096] fp32 = P_p + P_q of p's
rows (lanes send the peer-owned blocks, reduce own + landed in fp32 on their TRISC, write y). y vs the host fp32 sum
(bitwise + max |diff|) and vs the traced ttnn.reduce_scatter (max |diff|); traced timing. ARGS: blocks_local ([8, 32]),
lanes, reducers (None: the lanes reduce; a core list), nb (16), replays (50), record, tag."""

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
REDUCERS = [tuple(int(v) for v in xy) for xy in A["reducers"]] if A.get("reducers") else None
POST = bool(A.get("post", False))  # PMK H6: the reducers write y into 64 members' POST 21 (+ FS_REDUCED); stub members copy out
REPLAYS = int(A.get("replays", 50))
OUT = os.environ.get("GLM53_OUT_DIR", "/tmp")
TAG = A.get("tag", "7a-fs-rsr-unit")
REC = bool(A.get("record", False))
DRAM = ttnn.DRAM_MEMORY_CONFIG
ROWS_OUT = []


def row(**kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test="fs_rsr_unit", tag=TAG, mock=MOCK,
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
        members = None
        if POST:
            g = mesh.compute_with_storage_grid_size()
            taken = set(LANES) | set(REDUCERS or [])
            members = [(x, y) for y in range(int(g.y)) for x in range(int(g.x)) if (x, y) not in taken][:64]
            assert len(members) == 64, len(members)
        rsr = FS.FabricStreamRSR(mesh, bl, lanes=LANES, nb=NB, reducers=REDUCERS, members=members)
        fn = lambda: rsr(x)  # noqa: E731
        tr = Traced(mesh)
        tr.warm(fn)
        us = timed(tr, fn)
        nbytes = rows * 4096 * 2
        res = dict(blocks_local=bl, rows_local=rows, lanes=len(LANES), reducers=len(REDUCERS or []), post=POST, nb=NB, us=us, us_ttnn_rs=us_ref,
                   recv_gb_s=None if us is None else round(nbytes / us / 1e3, 2),
                   ttnn_recv_gb_s=None if us_ref is None else round(nbytes / us_ref / 1e3, 2), replays=REPLAYS,
                   note=("send + fp32 reduce -> 64 members' POST 21 + FS_REDUCED (H6) -> stub members copy to y DRAM"
                         if POST else "send + fp32 lane reduce + y fp32 DRAM write, complete (ttnn: bf16 reduce)"))  # fmt: skip
        if tr.enabled:
            ys = [ttnn.to_torch(t).float().reshape(rows, 4096) for t in ttnn.get_device_tensors(rsr.y)]
            want = [ph[p, 0, p * rows:(p + 1) * rows].float() + ph[1 - p, 0, p * rows:(p + 1) * rows].float() for p in range(n)]
            res["y_bitwise_fp32_sum"] = [bool(torch.equal(ys[p], want[p])) for p in range(n)]
            res["y_maxabs_fp32_sum"] = [float((ys[p] - want[p]).abs().max()) for p in range(n)]
            if refs is not None:
                res["y_vs_ttnn_maxabs"] = [float((ys[p] - refs[p]).abs().max()) for p in range(n)]
            res["status"] = "pass" if all(m <= 1e-6 * float(want[p].abs().max()) for p, m in enumerate(res["y_maxabs_fp32_sum"])) else "fail"
            fails += res["status"] == "fail"
        tr.release()
        row(**res)
        rsr.release()
        ttnn.deallocate(x)
    if REC and not MOCK:
        with open(os.path.join(OUT, "perf.jsonl"), "a") as fh:
            for r in ROWS_OUT:
                fh.write(json.dumps(r, default=str) + "\n")
    return fails


RESULT = None if main() == 0 else "fail"
