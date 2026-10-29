# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, any mesh): Stage 16 P3 -- the pieces of the sparse_sdpa DSA prefill timed alone, device vs device,
each traced (compile launch, capture of k_calls calls, replays; us per call = replay / k_calls):

  sdpa   : ttnn.transformer.sparse_sdpa(q [1, H, R, 512] bf16 RM, kv [1, 1, T, 512] RM, ids [1, 1, R, W] uint32) for
           every (H, R) in `shapes` (H = heads per call, R = query rows: TP2 = (32, C); a head->sequence reshard
           at TP8 = (64, C / 8)), every k_chunk in `kchunks`, kv dtype bf16 (+ fp8_e4m3 if `fp8`); ids = each row's
           causal random 2,051 distinct tokens (+ sentinels), as the expand_rows output of a chunk at `start`.
  topk   : ttnn.experimental.topk_large_indices(scores [1, 1, R, P] bf16 RM, k=512) for every P in `pools`.
  expand : expand_rows (kernels/prefill/expand_rows.cpp) over R rows.

ARGS: shapes ([[32, 2048], [64, 256]]), t (8192), start (6144), kchunks ([128, 256]), fp8 (False), pools ([1024, 8192,
32768]), rows (1024: topk / expand rows), k_calls (4), replays (s16common.SOAK_REPLAYS), tag, record."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
SHAPES = [tuple(int(v) for v in s) for s in args.get("shapes", [[32, 2048], [64, 256]])]
T = int(args.get("t", 8192))
START = int(args.get("start", T - 2048))
KCHUNKS = [int(v) for v in args.get("kchunks", [128, 256])]
POOLS = [int(v) for v in args.get("pools", [1024, 8192, 32768])]
ROWS = int(args.get("rows", 1024))
K_CALLS = int(args.get("k_calls", 4))
tag = args.get("tag", "p3-sparse-bench")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
ckc = PF._ckc_fp32(mesh)
DRAM = ttnn.DRAM_MEMORY_CONFIG
W = PF.SPARSE_W


def dev(t_, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(t_, dtype=dtype, layout=layout, device=mesh, memory_config=DRAM, mesh_mapper=repl)


def timed(fn, free=True):
    """compile launch, capture of K_CALLS calls, the steady-state soak -> us per call (median), min; outputs freed
    (free=False: fn writes in place and returns None)"""
    out = fn()
    ttnn.synchronize_device(mesh)
    if free:
        ttnn.deallocate(out)
    if C.MOCK:
        return dict(median_us=None, traced=False)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = []
    try:
        for _ in range(K_CALLS):
            outs.append(fn())
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    st = S.replay_stats(mesh, tid, replays=args.get("replays"))  # >= 2,000 replays (program 03:20Z)
    ttnn.release_trace(mesh, tid)
    for o in outs if free else []:
        ttnn.deallocate(o)
    return dict(st, median_us=round(st["median_us"] / K_CALLS, 1), min_us=round(st["min_us"] / K_CALLS, 1),
                p90_us=round(st["p90_us"] / K_CALLS, 1), calls_per_replay=K_CALLS, traced=True)  # fmt: skip


def causal_ids(rows: int, start: int) -> torch.Tensor:
    """[1, 1, rows, W] uint32-as-int64: row r (position start + r) attends 2,051 distinct tokens <= its position (all of
    them while fewer), then sentinels"""
    g = torch.Generator().manual_seed(0)
    out = torch.full((rows, W), 0xFFFFFFFF, dtype=torch.int64)
    for r in range(rows):
        p = start + r
        n = min(p + 1, PF.TOPK_POOLS * PF.KPOOL + PF.KPOOL - 1)
        sel = torch.randperm(p + 1, generator=g)[:n] if p + 1 > n else torch.arange(p + 1)
        out[r, :n] = sel
    return out.reshape(1, 1, rows, W)


results = []
kv = {"bf16": dev(torch.randn(1, 1, T, 512) * 0.3, ttnn.bfloat16)}
if args.get("fp8"):
    kv["fp8_e4m3"] = dev(torch.randn(1, 1, T, 512) * 0.3, ttnn.float8_e4m3)
fmt = {"bf16": ttnn.transformer.SparseKVFormat.BF16, "fp8_e4m3": ttnn.transformer.SparseKVFormat.FP8_E4M3}
for h, r in SHAPES:
    ids = dev(causal_ids(r, START).to(torch.int32), ttnn.uint32)
    for kd, kvt in kv.items():
        q = dev(torch.randn(1, h, r, 512) * 0.3, ttnn.bfloat16)
        for kc in KCHUNKS:
            def fn(q=q, kvt=kvt, kd=kd, kc=kc, ids=ids):
                return ttnn.transformer.sparse_sdpa(q, kvt, ids, 512, kv_format=fmt[kd], scale=0.0625,
                                                    k_chunk_size=kc, compute_kernel_config=ckc)  # fmt: skip
            try:
                tm = timed(fn)
            except Exception as e:  # noqa: BLE001
                tm = dict(error=str(e)[:200])
            row = dict(test=f"p3 sparse_sdpa H {h} x R {r} rows, {W} ids per row, T {T}, kv {kd}, k_chunk {kc}",
                       milestone="P3", piece="sparse_sdpa", heads=h, rows=r, ids=W, t=T, kv=kd, k_chunk=kc, tp=TPN,
                       gather_gb=round(r * W * 512 * (2 if kd == "bf16" else 1) / 1e9, 3), tag=tag, **tm)  # fmt: skip
            S.log(row)
            results.append(row)
        ttnn.deallocate(q)
    ttnn.deallocate(ids)
for t_ in kv.values():
    ttnn.deallocate(t_)

for p in POOLS:
    sc = dev((torch.randn(1, 1, ROWS, p)).to(torch.bfloat16), ttnn.bfloat16)
    try:
        tm = timed(lambda sc=sc: ttnn.experimental.topk_large_indices(sc, k=PF.TOPK_POOLS))
    except Exception as e:  # noqa: BLE001
        tm = dict(error=str(e)[:200])
    row = dict(test=f"p3 topk_large_indices k 512 over {ROWS} rows x {p} pools", milestone="P3", piece="topk",
               rows=ROWS, pools=p, tp=TPN, tag=tag, **tm)  # fmt: skip
    S.log(row)
    results.append(row)
    ttnn.deallocate(sc)

pools = dev(torch.randint(0, START // PF.KPOOL, (1, 1, ROWS, PF.TOPK_POOLS), dtype=torch.int32), ttnn.uint32)
start = dev(torch.tensor([START], dtype=torch.int32), ttnn.int32)
base = dev(torch.zeros(1, 1, dtype=torch.int32), ttnn.int32)  # every row of the chunk (no per-chip offset)
out = ttnn.zeros((1, 1, ROWS, W), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM)

tm = timed(lambda: PF.expand_rows(pools, start, out, base) and None, free=False)
row = dict(test=f"p3 expand_rows {ROWS} rows x {W} ids", milestone="P3", piece="expand", rows=ROWS, tp=TPN, tag=tag,
           **tm)  # fmt: skip
S.log(row)
results.append(row)
for t_ in (pools, start, out, base):
    ttnn.deallocate(t_)
if rec:
    for r_ in results:
        C.record("perf", r_)
RESULT = results
