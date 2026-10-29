# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; outside the tree: runs on the server's tree): Stage 14b — an unwatched replay soak of ONE decode trace
of the serving backend (program device duty 2026-10-21 03:20Z: every watched unit ends with a 30-s unwatched soak of
the same program). The trace of width `width` (None = the full width) is replayed back-to-back in batches of 100
(non-blocking, one synchronize per batch) for `seconds`, rows 0..w-1 live at distinct positions (rewound before each
batch, no position past the context), then the live rows' token rows are checked against a fresh 8-step run from the
same start (bitwise: the soak left no state behind). ARGS: b (32), layers, native (true), width, seconds (30),
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
SECS = float(args.get("seconds", 30))
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=bool(args.get("native", True)), rev=args.get("rev"),
                 log=lambda *a: print("SOAK", *a, flush=True))  # fmt: skip
be = ctx[key]["backend"]
kv = be._kv
io = kv.extra["io"]
W = int(args.get("width") or B)
trace = kv.extra["decode_trace"] if W == B else kv.extra["decode_traces"][W]
sync = lambda: ttnn.synchronize_device(mesh)  # noqa: E731
tok0 = torch.zeros(B, dtype=torch.int64)
tok0[:W] = torch.tensor([1000 + 13 * u for u in range(W)])
pos0 = torch.full((B,), -1, dtype=torch.int64)
pos0[:W] = torch.tensor([300 + 5 * u for u in range(W)])


def start():
    be.reset_all(kv)
    be.write(io.tokens, tok0)
    be.write(io.cur_pos, pos0)
    sync()


def rows(n):
    out = []
    for _ in range(n):
        be.replay(trace)
        sync()
        d = ttnn.get_device_tensors(be.head.topk_out)[0] if be.tp > 1 else be.head.topk_out
        out.append((ttnn.to_torch(d).to(torch.int64).reshape(32, 64) & 0xFFFFFFFF)[:W, 0].clone())
    return out


res = dict(B=B, width=W, layers=len(ids), seconds=SECS)
if SM.MOCK:
    start()
    rows(1)
    res.update(mock=True)
    ok = True
else:
    start()
    ref = rows(8)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < SECS:
        be.write(io.cur_pos, pos0)  # rewound per batch: positions stay < pos0 + 100
        for _ in range(100):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        sync()
        n += 100
    dt = time.perf_counter() - t0
    start()
    again = rows(8)
    ok = all(torch.equal(a, b) for a, b in zip(ref, again))
    res.update(replays=n, wall_s=round(dt, 1), us_per_replay=round(dt * 1e6 / n, 1), rows_bitwise_after=ok)
    start()
print("SOAK RESULT", res, flush=True)
if args.get("record") and not SM.MOCK and not os.environ.get("TT_METAL_WATCHER"):
    C.record("tests", dict(test=f"m5b unwatched replay soak of the width-{W} decode trace ({len(ids)} layers, B={B} "
                                f"server, tp{be.tp}): {res.get('replays')} replays in {res.get('wall_s')} s, the live rows' "
                                f"8-step tokens bitwise before / after", status="pass" if ok else "fail", milestone="M5b",
                           tag=args.get("tag", f"m5b-soak-w{W}"), **res))  # fmt: skip
RESULT = dict(ok=ok, **res)
