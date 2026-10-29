# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, host-p300 pair; 2026-10-23 23:50Z, the stage-1 rejection follow-up): is ttnn.experimental.topk_large_indices'
BOUNDED mode (valid_length_tensor) right? A synthetic plane per chip x [1, 1, R, P] bf16 ROW_MAJOR DRAM whose every row is
a random permutation of P DISTINCT finite bf16 values (bit patterns 0x0080 + j: no ties), the chip's word w (`words`,
chip k -> words[k]) as a [1] uint32 per chip (ShardTensorToMesh dim 0, the prefill_tp vis_cols_u form). Variants, each
ONE traced replay (capture, replay, read the persistent copy):
  bounded_stale  the bounded op on x with the columns >= w set LARGE (0x7F00 ~ 1.7e38): a right op never returns them;
  bounded_neg    the bounded op on x with the columns >= w = -inf;
  plain_neg      the plain op (no bound) on the -inf plane: the control (today's form).
Per chip: rows whose top-k SET (the first k ids) equals torch.topk(x[r, :w]) exactly, rows with an id >= w, and for the
first wrong rows whether the op's set is the true top-k of ANOTHER row (a row-addressing defect) or of a column window.
sign ("pos" | "mixed": a random half of the values negative).
ARGS: rows (1024), p (16384), k (512), words ([768, 1024]), seed (0), sign ("pos"), tag."""
import os
import time
import traceback

import torch

import ttnn

mesh, args = MESH, ARGS  # noqa: F821
TPN = int(mesh.get_num_devices())
R, P, K = int(args.get("rows", 1024)), int(args.get("p", 16384)), int(args.get("k", 512))
WS = [int(v) for v in args.get("words", [768, 1024])][:TPN]
tag = args.get("tag", f"topk-bound-op-tp{TPN}")
SENT = 0xFFFFFFFF
MOCK = os.environ.get("GLM53_MOCK", "") == "1"
t0 = time.time()
row, err = dict(rows=R, p=P, k=K, words=WS, tp=TPN, sign=args.get("sign", "pos")), None


def log(m):
    print(f"TOPKB {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {tag}: {m}", flush=True)


def chips(t_):
    return [ttnn.to_torch(d_).clone() for d_ in (ttnn.get_device_tensors(t_) if TPN > 1 else [t_])]


try:
    g = torch.Generator().manual_seed(int(args.get("seed", 0)))
    base = (torch.arange(P, dtype=torch.int32) + 0x0080).to(torch.int16)  # distinct finite positive bf16 bit patterns
    if args.get("sign", "pos") == "mixed":  # distinct magnitudes, a random half negative (the score plane's mix)
        neg_ = torch.randperm(P, generator=g)[: P // 2]
        b32 = base.to(torch.int32)
        b32[neg_] |= 0x8000
        base = b32.to(torch.int16)
    xb = torch.empty(TPN, 1, R, P, dtype=torch.int16)
    for k_ in range(TPN):
        for r in range(R):
            xb[k_, 0, r] = base[torch.randperm(P, generator=g)]
    x = xb.view(torch.bfloat16)
    stale, neg = x.clone(), x.clone()
    for k_, w in enumerate(WS):
        stale[k_, :, :, w:] = torch.tensor(0x7F00, dtype=torch.int16).view(torch.bfloat16)
        neg[k_, :, :, w:] = float("-inf")
    mp = ttnn.ShardTensorToMesh(mesh, dim=0) if TPN > 1 else None
    put = lambda v: ttnn.from_torch(v if TPN > 1 else v[0], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,  # noqa: E731
                                    mesh_mapper=mp, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    x_stale, x_neg = put(stale), put(neg)
    wt = ttnn.from_torch(torch.tensor(WS, dtype=torch.int32).reshape(TPN, 1) if TPN > 1 else torch.tensor(WS, dtype=torch.int32),
                         dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=mp,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    row["word_readback"] = [int(v.reshape(-1)[0]) & SENT for v in chips(wt)]
    VAR = dict(bounded_stale=lambda: ttnn.experimental.topk_large_indices(x_stale, k=K, valid_length_tensor=wt),
               bounded_neg=lambda: ttnn.experimental.topk_large_indices(x_neg, k=K, valid_length_tensor=wt),
               plain_neg=lambda: ttnn.experimental.topk_large_indices(x_neg, k=K))
    outs = {}
    for nm_, fn in VAR.items():  # the compile pass of every variant before the first capture + its persistent copy
        o_ = fn()
        outs[nm_] = ttnn.zeros(tuple(o_.shape), dtype=o_.dtype, layout=o_.layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(o_, outs[nm_])
        ttnn.deallocate(o_)
    ttnn.synchronize_device(mesh)
    n_pc = int(mesh.num_program_cache_entries())
    ref = []  # per chip: the exact top-k set of every row over [0, w)
    for k_, w in enumerate(WS):
        xs = x[k_, 0, :, :w].float()
        ref.append([set(v.tolist()) for v in torch.topk(xs, min(K, w), dim=-1).indices])
    res = {}
    for nm_, fn in VAR.items():
        def body():
            o_ = fn()
            ttnn.copy(o_, outs[nm_])
            ttnn.deallocate(o_)

        if MOCK:  # the mock cannot capture (create_trace_node aborts): the readback + compare path only
            body()
            us = None
        else:
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                body()
            finally:
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            t1 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            us = round((time.perf_counter() - t1) * 1e6, 1)
            ttnn.release_trace(mesh, tid)
        got = chips(outs[nm_])
        per = []
        for k_, w in enumerate(WS):
            gk = got[k_].to(torch.int64).reshape(R, -1) & SENT
            ok = past = 0
            bad = []
            for r in range(R):
                s_ = set(gk[r, :K].tolist())
                ok += int(s_ == ref[k_][r])
                past += int(any(v >= w for v in s_))
                if s_ != ref[k_][r] and len(bad) < 3:
                    other = [r2 for r2 in range(R) if ref[k_][r2] == s_][:2]  # the true set of ANOTHER row?
                    xs = x[k_, 0, r].float()
                    vals = xs[torch.tensor(sorted(v for v in s_ if v < P))]
                    bad.append(dict(row=r, n_wrong=len(s_ - ref[k_][r]), max_id=max(s_), min_id=min(s_), other_row=other,
                                    got_min_val=float(vals.min()) if vals.numel() else None,
                                    ref_min_val=float(torch.topk(xs[:w], min(K, w)).values.min())))  # fmt: skip
            per.append(dict(word=w, rows_exact=ok, rows_id_ge_w=past, wrong_examples=bad))
        res[nm_] = dict(replay_us=us, chips=per)
        log(f"{nm_}: {res[nm_]}")
    assert int(mesh.num_program_cache_entries()) == n_pc, "a program compiled inside a capture"
    row.update(variants=res, bounded_ok=all(c["rows_exact"] == R for v in ("bounded_stale", "bounded_neg") for c in res[v]["chips"]),
               plain_ok=all(c["rows_exact"] == R for c in res["plain_neg"]["chips"]), status="mock" if MOCK else "pass")  # fmt: skip
except Exception:  # noqa: BLE001
    err = traceback.format_exc()[-4000:]
    log(f"FAILED:\n{err}")
    row.update(status="fail", error=err)
RESULT = dict(test="topk_large_indices bounded mode (valid_length_tensor) vs torch.topk on a distinct-valued plane", tag=tag,
              wall_s=round(time.time() - t0, 1), **row)  # fmt: skip
log({k: v for k, v in RESULT.items() if k not in ("variants",)})
