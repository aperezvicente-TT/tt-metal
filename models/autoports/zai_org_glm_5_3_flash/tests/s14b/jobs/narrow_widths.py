# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; mock or the mesh): Stage 14b — native-t width buckets (StackBackend GLM53_STACK_NATIVE_WIDTHS, e.g.
"8,16"): for every width w the live rows 0..w-1 decode `steps` traced steps on the width-w trace and on the full-width
trace (rows >= w idle in both) from the same state: the live rows' logits / top-K rows must agree (bitwise reported,
top-1 agreement counted); then the step time per width (back-to-back replays + the blocking token-out read).
ARGS: b (32), layers ("all"), steps (6), reps (20), shared_weights, record, tag, rev."""
import gc
import importlib
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.join(os.environ["TT_METAL_HOME"], "models/autoports/zai_org_glm_5_3_flash/tests/s14b"))
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
STEPS, REPS = int(args.get("steps", 6)), int(args.get("reps", 20))
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]


def log(*a):
    print(f"M5B-NARROW {C.now()}", *a, flush=True)


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=True, rev=args.get("rev"), log=log)
be = ctx[key]["backend"]
kv = be._kv
io = kv.extra["io"]
traces = kv.extra.get("decode_traces", {})
widths = list(getattr(be, "narrow_widths", ())) if args.get("narrow", True) else list(be.native_widths)
log("widths", widths, "traces", sorted(traces), be.trace_report())
assert widths and all(w in traces for w in widths), (widths, sorted(traces))


def run(trace, w, n):
    be._kv = kv
    be.reset_all(kv)
    tok = torch.zeros(B, dtype=torch.int64)
    tok[:w] = torch.tensor([1000 + 13 * u for u in range(w)])
    pos = torch.full((B,), -1, dtype=torch.int64)
    pos[:w] = torch.tensor([7 * u for u in range(w)])
    be.write(io.tokens, tok)
    be.write(io.cur_pos, pos)
    ttnn.synchronize_device(mesh)
    lg, rows = [], []
    for _ in range(n):
        be.replay(trace)
        ttnn.synchronize_device(mesh)
        lg.append(be.read(io.logits)[:w].clone())
        d = ttnn.get_device_tensors(be.head.topk_out)[0] if be.tp > 1 else be.head.topk_out
        rows.append((ttnn.to_torch(d).to(torch.int64).reshape(32, 64) & 0xFFFFFFFF)[:w].clone())
    return lg, rows


def timing(trace, w):
    be.reset_all(kv)
    tok = torch.tensor([1000 + 7 * u for u in range(B)])
    pos = torch.full((B,), -1, dtype=torch.int64)
    pos[:w] = torch.tensor([512 + u for u in range(w)])
    be.write(io.tokens, tok)
    be.write(io.cur_pos, pos)
    be.replay(trace)
    ttnn.synchronize_device(mesh)
    be.write(io.cur_pos, pos)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(REPS):
        be.replay(trace)
    ttnn.synchronize_device(mesh)
    dev = (time.perf_counter() - t0) * 1e6 / REPS
    be.write(io.cur_pos, pos)
    ttnn.synchronize_device(mesh)
    lat = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        be.replay(trace)
        be.read(io.sampled)
        lat.append((time.perf_counter() - t0) * 1e6)
    lat.sort()
    return round(dev, 1), round(lat[len(lat) // 2], 1)


full = traces[B]
res = {}
for w in widths:
    a_lg, a_rows = run(traces[w], w, STEPS)
    b_lg, b_rows = run(full, w, STEPS)
    bit = all(torch.equal(x, y) for x, y in zip(a_lg, b_lg))
    top1 = [float((x[:, 0] == y[:, 0]).float().mean()) for x, y in zip(a_rows, b_rows)]
    mad = max(float((x.float() - y.float()).abs().max()) for x, y in zip(a_lg, b_lg))
    dev_w, tok_w = timing(traces[w], w)
    res[w] = dict(logits_bitwise=bit, top1_agree_min=round(min(top1), 4), max_abs_logit_diff=round(mad, 4),
                  device_us=dev_w, token_out_us=tok_w)  # fmt: skip
    log("WIDTH", w, res[w])
dev_f, tok_f = timing(full, 1)
res[B] = dict(device_us=dev_f, token_out_us=tok_f)
log("RESULT", res)
ok = all(r.get("top1_agree_min", 1.0) == 1.0 for r in res.values())
r = dict(test=f"m5b native-t width buckets {widths} vs the full width ({len(ids)} layers, B={B}): live rows' logits / "
              f"top-K on the width-w trace vs the {B}-row trace + step time per width", status="pass" if ok else "info",
         milestone="M5b", widths=widths, results={str(k): v for k, v in res.items()}, tag=args.get("tag", "m5b-narrow"))  # fmt: skip
if args.get("record") and not os.environ.get("TT_METAL_WATCHER"):
    C.record("tests", r)
    for w, v in res.items():
        C.record("perf", dict(test=f"m5b decode step native width {w} (ONE execute_trace, {len(ids)} layers, B={B} "
                                   f"server, rows 0..{w - 1} live)", status="info", milestone="M5b", traced=True,
                              width=w, device_us=v["device_us"], token_out_us=v["token_out_us"], reps=REPS,
                              tag=args.get("tag", "m5b-narrow")))  # fmt: skip
RESULT = dict(ok=ok, results={str(k): v for k, v in res.items()})
