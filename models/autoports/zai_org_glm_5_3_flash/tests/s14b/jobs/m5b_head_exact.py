# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the TP mesh / pair): Stage 14b — the head's device top-K (readers' scan, the select core's reader
merge, the in-program cross-chip candidate gather + merge: kernels/mk/topk.hpp, kernels/head/topk_select.cpp) exact on
the serving backend (m5common.serving): the decode trace replayed `steps` times on greedy rows (k = 1: word 0 = the
argmax), every row's K candidates (ids + bf16 order keys) == the host's top-K of the same step's gathered logits
(value descending, ties: the lower vocab id), identical on every chip; then the head alone (its own trace) timed.
ARGS: b (32), layers ([0, 1, 2]), native (true), steps (6), reps (20), shared_weights, rev, record, tag."""
import gc
import importlib
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402

for mod in (M5, G, SM):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 32))
NATIVE = bool(args.get("native", True))
STEPS, REPS = int(args.get("steps", 6)), int(args.get("reps", 20))
cfg = hf_config()
_L = args.get("layers", [0, 1, 2])
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
tag = args.get("tag", f"m5b-head-exact-B{B}")


def log(*a):
    print(f"M5B-HEADX {C.now()}", *a, flush=True)


def order_key(v: torch.Tensor) -> torch.Tensor:
    """topk.hpp's bf16 order key of bf16 values: (v & 0x8000) ? ~v & 0xffff : v | 0x8000"""
    u = v.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    return torch.where(u & 0x8000 != 0, ~u & 0xFFFF, u | 0x8000)


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=NATIVE, rev=args.get("rev"), log=log)
be = ctx[key]["backend"]
kv = be._kv
io = kv.extra["io"]
K = be.head.topk
sync = lambda: ttnn.synchronize_device(mesh)  # noqa: E731
be.reset_all(kv)
be.write(io.k, torch.ones(B, dtype=torch.int32))  # greedy rows: no draw swaps word 0
be.write(io.tokens, torch.tensor([1000 + 13 * u for u in range(B)]))
be.write(io.cur_pos, torch.tensor([300 + 5 * u for u in range(B)]))
sync()
chk = dict(rows=0, ids_exact=0, keys_exact=0, argmax_ok=0, chips_identical=True, ties_in_topk=0)
if SM.MOCK:  # mock cluster: kernels do not run (one eager step: the programs JIT-build), no checks / timing
    be.replay(kv.extra["decode_trace"])
    sync()
    STEPS = 0
for _ in range(STEPS):
    be.replay(kv.extra["decode_trace"])
    sync()
    lg = be.read(io.logits)[:B]
    dev = ttnn.get_device_tensors(be.head.topk_out) if be.tp > 1 else [be.head.topk_out]
    rows = [ttnn.to_torch(d).to(torch.int64).reshape(32, 64) & 0xFFFFFFFF for d in dev]
    chk["chips_identical"] &= all(torch.equal(rows[0][:B], r[:B]) for r in rows[1:])
    rows = rows[0]
    for r in range(B):
        v = lg[r].to(torch.bfloat16).float()
        _, idx = torch.sort(v, descending=True, stable=True)
        want = idx[:K]
        chk["rows"] += 1
        chk["ids_exact"] += int(torch.equal(rows[r, :K], want))
        chk["keys_exact"] += int(torch.equal(rows[r, 32 : 32 + K], order_key(v[want])))
        chk["argmax_ok"] += int(int(rows[r, 0]) == int(want[0]))
        chk["ties_in_topk"] += int(len(set(v[want].tolist())) < K)
ok = chk["ids_exact"] == chk["keys_exact"] == chk["argmax_ok"] == chk["rows"] and chk["chips_identical"]
ok = ok and (chk["rows"] > 0 or SM.MOCK)
# the head alone, its own trace (the residual / positions as the last step left them; positions rewound per batch)
head_us = 0.0
if not SM.MOCK:
    pos0 = be.read(io.cur_pos)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        be.head(kv.res, be.pos, pos_copies=be.pos_c)
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    try:
        be.write(io.cur_pos, pos0)
        sync()
        t0 = time.perf_counter()
        for _ in range(REPS):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        sync()
        head_us = (time.perf_counter() - t0) * 1e6 / REPS
    finally:
        ttnn.release_trace(mesh, tid)
        be.write(io.cur_pos, pos0)
        sync()
res = dict(B=B, K=K, tp=be.tp, layers=len(ids), steps=STEPS, checks=chk, head_us=round(head_us, 1), reps=REPS)
log("RESULT", res)
if args.get("record") and not SM.MOCK:
    C.record("tests", dict(test=f"m5b head device top-{K} exact on the serving backend (readers' scan + select merge + "
                                f"cross-chip gather), B={B} tp{be.tp}, {STEPS} traced steps: candidates == host top-K "
                                f"of the gathered logits (ties: the lower id)", status="pass" if ok else "fail",
                           milestone="M5b", tag=tag, **res))  # fmt: skip
    if not os.environ.get("TT_METAL_WATCHER"):
        C.record("perf", dict(test=f"m5b head alone (ONE execute_trace: final norm + lm_head shard + top-{K} + gather "
                                   f"+ position advance), B={B} tp{be.tp}", status="info", milestone="M5b",
                              traced=True, us=round(head_us, 1), reps=REPS, B=B, tp=be.tp, tag=tag))  # fmt: skip
RESULT = dict(ok=ok, head_us=round(head_us, 1), **chk)
