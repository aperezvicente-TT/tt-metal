# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a PfStream unit (program 10:33Z): ONE traced PfStreamPhase body (tt/pf_stream.PfStreamMM), qkv-shaped and
o_proj-shaped, vs the traced ttnn.linear of the same x / W. Each chip of the mesh runs the whole phase (per-chip unit,
no fabric). x [1, 1, 2048, K] bf16 random = 64 stationary row tiles on 64 members; W [K, N] bf16 random -> bfp8.
Criterion: PCC vs the torch fp32 reference with the dequantized bfp8 W >= 0.9999 on every chip. Reported: us (traced
replays), us_ttnn, MATH % = 2 M K N / (297 TFLOPS x us) (prefill_megakernel.md §3), the members' L1 footprint.
ARGS: shapes ([["qkv", 4096, 12288], ["o_proj", 4096, 4096]]), kb (8), nb (8), nw (32), park (4), probe (true: the
in-program wait counters of the last replay), replays (50), record, tag."""

from __future__ import annotations

import importlib
import json
import os
import time

import torch

import ttnn
import models.autoports.zai_org_glm_5_3_flash.tt.pf_stream as PF
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced

PF = importlib.reload(PF)
A = globals().get("ARGS", {}) or {}
mesh = globals().get("MESH")
MOCK = os.environ.get("GLM53_MOCK") == "1"
SHAPES = [(str(s[0]), int(s[1]), int(s[2])) for s in A.get("shapes", [["qkv", 4096, 12288], ["o_proj", 4096, 4096]])]
KB, NB, NW, PARK = int(A.get("kb", 8)), int(A.get("nb", 8)), int(A.get("nw", 32)), int(A.get("park", 4))
PROBE = bool(A.get("probe", True))
FUSED = bool(A.get("fused", True))  # + one program running every shape as consecutive phases (QBASE / SLOT offsets)
REPLAYS = int(A.get("replays", 50))
OUT = os.environ.get("GLM53_OUT_DIR", "/tmp")
TAG = A.get("tag", "7a-pfs-unit")
REC = bool(A.get("record", False))
DRAM = ttnn.DRAM_MEMORY_CONFIG
M = 2048
PEAK = 297e12  # HiFi2 grid peak per chip (prefill_megakernel.md §3)
ROWS_OUT = []


def row(**kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test="pfs_unit", tag=TAG, mock=MOCK,
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


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def placement():
    """members: logical x 3..10, y 0..7 (one multicast rectangle; spans the p300a's non-Tensix virtual columns 8 / 9);
    streamers: logical x 0..2 of rows 0..7, then rows 8..9 (46 cores)"""
    g = mesh.compute_with_storage_grid_size()
    assert (int(g.x), int(g.y)) == (11, 10), (g.x, g.y)
    members = [(x, y) for y in range(8) for x in range(3, 11)]
    streamers = [(x, y) for y in range(8) for x in range(3)] + [(x, y) for y in (8, 9) for x in range(11)]
    return members, streamers


def main():
    members, streamers = placement()
    fails = 0
    keep = []  # per shape: x (device), x host, W host, dequantized W, us_ttnn
    for name, k, n in SHAPES:
        gen = torch.Generator().manual_seed(7 + k + n)
        xh = torch.randn(1, 1, M, k, generator=gen).to(torch.bfloat16)
        wh = (torch.randn(k, n, generator=gen) / k**0.5).to(torch.bfloat16)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                            mesh_mapper=rep)  # fmt: skip
        w_tt = ttnn.from_torch(wh.reshape(1, 1, k, n), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                               memory_config=DRAM, mesh_mapper=rep)  # fmt: skip
        ref_out = {}

        def ref_fn():
            ref_out["y"] = ttnn.linear(x, w_tt, memory_config=DRAM, dtype=ttnn.bfloat16)
            return ref_out["y"]

        tr = Traced(mesh)
        tr.warm(ref_fn)
        us_ref = timed(tr, ref_fn)
        y_ttnn = [ttnn.to_torch(t).float().reshape(M, n) for t in ttnn.get_device_tensors(ref_out["y"])] if tr.enabled else None
        wq = ttnn.to_torch(ttnn.get_device_tensors(w_tt)[0]).float().reshape(k, n) if tr.enabled else None
        tr.release()
        pfs = PF.PfStreamMM(mesh, wh.float(), members=members, streamers=streamers, nb=NB, nw=NW, kb=KB, park=PARK,
                            probe=PROBE)  # fmt: skip
        keep.append((name, x, xh, wh, wq, us_ref))
        fn = lambda: pfs(x)  # noqa: E731
        tr = Traced(mesh)
        tr.warm(fn)
        us = timed(tr, fn)
        flops = 2.0 * M * k * n
        res = dict(shape=name, m=M, k=k, n=n, kb=KB, nb=NB, nw=NW, park=PARK, streamers=pfs.ns, members=len(members),
                   blocks=pfs.nt // pfs.nb * pfs.nbk, l1=pfs.footprint(), us=us, us_ttnn=us_ref, replays=REPLAYS,
                   math_pct=None if us is None else round(flops / (PEAK * us * 1e-6) * 100, 1),
                   math_pct_ttnn=None if us_ref is None else round(flops / (PEAK * us_ref * 1e-6) * 100, 1))  # fmt: skip
        if tr.enabled:
            want = xh.reshape(M, k).float() @ wq
            ys = [ttnn.to_torch(t).float().reshape(M, n) for t in ttnn.get_device_tensors(pfs.y)]
            res["pcc"] = [round(pcc(y, want), 7) for y in ys]
            res["pcc_ttnn"] = [round(pcc(y, want), 7) for y in y_ttnn]
            res["maxabs_vs_ttnn"] = [float((y - t).abs().max()) for y, t in zip(ys, y_ttnn)]
            res["ref_absmax"] = float(want.abs().max())
            res["status"] = "pass" if min(res["pcc"]) >= 0.9999 else "fail"
            res["probe_us"] = pfs.probes()
            fails += res["status"] == "fail"
        tr.release()
        row(**res)
        pfs.release()
        ttnn.deallocate(w_tt)
    if FUSED and len(keep) > 1:
        sync = PF.PfStreamMM.make_sync(mesh, streamers, slots=len(keep))
        mms = [PF.PfStreamMM(mesh, wh.float(), members=members, streamers=streamers, nb=NB, nw=NW, kb=KB, park=PARK,
                             probe=PROBE, sync=sync) for _, _, _, wh, _, _ in keep]  # fmt: skip
        xs = [x for _, x, _, _, _, _ in keep]
        fn = lambda: PF.fused_call(mms, xs)  # noqa: E731
        tr = Traced(mesh)
        tr.warm(fn)
        us = timed(tr, fn)
        flops = sum(2.0 * M * m.k * m.n for m in mms)
        us_ref = None if any(k[5] is None for k in keep) else round(sum(k[5] for k in keep), 1)
        res = dict(shape="+".join(k[0] for k in keep), fused=True, m=M, kb=KB, nb=NB, nw=NW, park=PARK,
                   blocks=[m.blocks for m in mms], us=us, us_ttnn_sum=us_ref, replays=REPLAYS,
                   math_pct=None if us is None else round(flops / (PEAK * us * 1e-6) * 100, 1))  # fmt: skip
        if tr.enabled:
            res["pcc"] = []
            for m, (_, _, xh, _, wq, _) in zip(mms, keep):
                want = xh.reshape(M, m.k).float() @ wq
                res["pcc"] += [round(pcc(ttnn.to_torch(t).float().reshape(M, m.n), want), 7) for t in ttnn.get_device_tensors(m.y)]
            res["status"] = "pass" if min(res["pcc"]) >= 0.9999 else "fail"
            fails += res["status"] == "fail"
            res["probe_us"] = [m.probes(slot=i) for i, m in enumerate(mms)]
        tr.release()
        row(**res)
        for m in mms:
            m.release()
        ttnn.deallocate(sync)
    for k in keep:
        ttnn.deallocate(k[1])
    if REC and not MOCK:
        with open(os.path.join(OUT, "perf.jsonl"), "a") as fh:
            for r in ROWS_OUT:
                fh.write(json.dumps(r, default=str) + "\n")
    return fails


RESULT = None if main() == 0 else "fail"
