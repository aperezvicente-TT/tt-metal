# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the TP mesh / pair; lives outside the tree: runs on the server's tree): Stage 14b — the head alone
(its own trace: final norm + lm_head shard + top-K + gather + device sampling + position advance) timed with every
live row GREEDY vs every live row SAMPLING (k = 0 / p = 0.95 / temperature 0.8: the select core draws from all K
candidates; its exp / sums are float math on a core without an FPU). ARGS: b (32), layers, native (true), reps (30),
shared_weights, rev, record, tag."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.environ["TT_METAL_HOME"], "models/autoports/zai_org_glm_5_3_flash/tests/s14b"))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 32))
REPS = int(args.get("reps", 30))
cfg = hf_config()
_L = args.get("layers", [0, 1, 2])
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=bool(args.get("native", True)), rev=args.get("rev"),
                 log=lambda *a: print("HEAD-SAMP", *a, flush=True))  # fmt: skip
be = ctx[key]["backend"]
kv = be._kv
io = kv.extra["io"]
sync = lambda: ttnn.synchronize_device(mesh)  # noqa: E731


def greedy():
    """every row greedy on the device: StackBackend writes the sampler params page when temp is written (k, p, temp)"""
    be.write(io.k, torch.ones(B, dtype=torch.int32))
    be.write(io.p, torch.zeros(B))
    be.write(io.temp, torch.ones(B))


be.reset_all(kv)
greedy()
be.write(io.tokens, torch.tensor([1000 + 13 * u for u in range(B)]))
be.write(io.cur_pos, torch.tensor([300 + 5 * u for u in range(B)]))
be.replay(kv.extra["decode_trace"])  # a real residual / logits for the head
sync()
pos0 = be.read(io.cur_pos)
res = {}
if not SM.MOCK:
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        be.head(kv.res, be.pos, pos_copies=be.pos_c)
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    try:
        for mode in ("greedy", "sample"):
            if mode == "greedy":  # k, p, temp in this order: the device params page is written on temp
                greedy()
            else:
                be.write(io.k, torch.zeros(B, dtype=torch.int32))
                be.write(io.p, torch.full((B,), 0.95))
                be.write(io.temp, torch.full((B,), 0.8))
                be.write(io.seeds, torch.tensor([77 + u for u in range(B)]))
            be.write(io.cur_pos, pos0)
            sync()
            t0 = time.perf_counter()
            for _ in range(REPS):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            sync()
            res[mode] = round((time.perf_counter() - t0) * 1e6 / REPS, 1)
    finally:
        ttnn.release_trace(mesh, tid)
        greedy()  # every later job of this process sees greedy rows again
        be.write(io.cur_pos, pos0)
        sync()
out = dict(B=B, tp=be.tp, layers=len(ids), reps=REPS, head_us=res, code=os.environ.get("GLM53_CODE_SHA"))
print("HEAD-SAMP RESULT", out, flush=True)
if args.get("record") and res and not os.environ.get("TT_METAL_WATCHER"):
    C.record("perf", dict(test=f"m5b head alone (ONE execute_trace) with every live row greedy vs sampling (k 0, p 0.95, "
                               f"T 0.8), B={B} tp{be.tp}", status="info", milestone="M5b", traced=True,
                          tag=args.get("tag", f"m5b-head-sample-B{B}-tp{be.tp}"), **out))  # fmt: skip
RESULT = dict(ok=True, **out)
