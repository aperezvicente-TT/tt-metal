# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a unit: the PMK layer's P2 -> P3 on the TP2 pair in ONE program (tt/pf_ag.PfAg): the H1 stand-in fills each
chip's local x̂ blocks into their holders' XS (CB 35) -> the AG (FabricStreamC1: lanes write every local block into the
peer's XS, fused FS_ARRIVE) -> H2 (member G waits FS_ARRIVE >= 8 on its XS) -> the qkv projection (PfStream, 64
members, weights streamed once). x̂ [2048, 4096] bf16 split by rows over the pair; W [4096, 12288] -> bfp8.
Criterion: y [2048, 12288] on EVERY chip PCC >= 0.9999 vs x̂ @ Wq (fp32). Against the traced op path ttnn.all_gather +
ttnn.linear. ARGS: n (12288), kb (16), nb (4), nw (32), park (4), replays (50), record, tag."""

from __future__ import annotations

import importlib
import json
import os
import time

import torch

import ttnn
import models.autoports.zai_org_glm_5_3_flash.tt.fabric_stream as FS
import models.autoports.zai_org_glm_5_3_flash.tt.pf_stream as PF
import models.autoports.zai_org_glm_5_3_flash.tt.pf_ag as PA
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced

FS = importlib.reload(FS)
PF = importlib.reload(PF)
PA = importlib.reload(PA)
A = globals().get("ARGS", {}) or {}
mesh = globals().get("MESH")
MOCK = os.environ.get("GLM53_MOCK") == "1"
KB, NB, NW, PARK = int(A.get("kb", 16)), int(A.get("nb", 4)), int(A.get("nw", 32)), int(A.get("park", 4))
REPLAYS = int(A.get("replays", 50))
OUT = os.environ.get("GLM53_OUT_DIR", "/tmp")
TAG = A.get("tag", "7a-pf-ag-unit")
REC = bool(A.get("record", False))
DRAM = ttnn.DRAM_MEMORY_CONFIG
M, K, N = 2048, 4096, int(A.get("n", 12288))
PEAK = 297e12


def row(**kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test="pf_ag_unit", tag=TAG, mock=MOCK,
             ndev=int(mesh.get_num_devices()), **kw)  # fmt: skip
    print("ROW", json.dumps(r, default=str), flush=True)
    if REC and not MOCK:
        with open(os.path.join(OUT, "perf.jsonl"), "a") as fh:
            fh.write(json.dumps(r, default=str) + "\n")
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


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    n = int(mesh.get_num_devices())
    assert n == 2, "the pair"
    g = mesh.compute_with_storage_grid_size()
    assert (int(g.x), int(g.y)) == (11, 10), (g.x, g.y)
    members = [(x, y) for y in range(8) for x in range(3, 11)]
    streamers = [(x, y) for y in range(8) for x in range(3)]
    lanes = [(0, 8), (1, 8)]
    gen = torch.Generator().manual_seed(2424)
    xh = torch.randn(1, 1, M, K, generator=gen).to(torch.bfloat16)
    wh = (torch.randn(K, N, generator=gen) / K**0.5).to(torch.bfloat16)
    x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2))  # fmt: skip
    w_tt = ttnn.from_torch(wh.reshape(1, 1, K, N), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                           memory_config=DRAM, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip

    def ref_fn():
        return ttnn.linear(ttnn.all_gather(x, dim=2, cluster_axis=None, memory_config=DRAM), w_tt, memory_config=DRAM,
                           dtype=ttnn.bfloat16)  # fmt: skip

    tr = Traced(mesh)
    tr.warm(ref_fn)
    us_ref = timed(tr, ref_fn)
    wq = ttnn.to_torch(ttnn.get_device_tensors(w_tt)[0]).float().reshape(K, N) if tr.enabled else None
    tr.release()
    pa = PA.PfAg(mesh, wh.float(), members=members, streamers=streamers, lanes=lanes, nb=NB, nw=NW, kb=KB, park=PARK)
    fn = lambda: pa(x)  # noqa: E731
    tr = Traced(mesh)
    tr.warm(fn)
    us = timed(tr, fn)
    res = dict(m=M, k=K, n=N, kb=KB, nb=NB, nw=NW, park=PARK, streamers=pa.mm.ns, members=len(members), lanes=len(lanes),
               us=us, us_ttnn_ag_linear=us_ref, replays=REPLAYS,
               math_pct=None if us is None else round(2.0 * M * K * N / (PEAK * us * 1e-6) * 100, 1),
               note="H1 stub fill -> AG (C1 into the members' XS) -> H2 -> qkv PfStream; one program")  # fmt: skip
    if tr.enabled:
        want = xh.reshape(M, K).float() @ wq
        ys = [ttnn.to_torch(t).float().reshape(M, N) for t in ttnn.get_device_tensors(pa.y)]
        res["pcc"] = [round(pcc(y, want), 7) for y in ys]
        res["status"] = "pass" if min(res["pcc"]) >= 0.9999 else "fail"
    tr.release()
    row(**res)
    pa.release()
    for t in (x, w_tt):
        ttnn.deallocate(t)
    return 0 if res.get("status", "pass") == "pass" else 1


RESULT = None if main() == 0 else "fail"
