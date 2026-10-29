# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; mock or the mesh): Stage 14b — native-t width buckets (StackBackend GLM53_STACK_NATIVE_WIDTHS, e.g.
"8,16"; narrow: GLM53_STACK_NARROW_WIDTHS): for every width w the live rows 0..w-1 decode `steps` traced steps on
the width-w trace and on the full-width trace (rows >= w idle in both) from the same state: the live rows' logits
must be bitwise equal (the production rule; top-1 agreement reported); then the step time per width with ONE live row
and with every row of the width live (back-to-back replays + the blocking token-out read, both loops from the same
state; the 32-row trace with 1 and with 32 live rows). Failed criteria are recorded "fail"; with ARGS accepted (the
ruling that accepts width-dependent numerics) "record" + outcome fail.
ARGS: b (32), layers ("all"), steps (6), reps (20), narrow (false | true | "all"), shared_weights, record, tag, rev,
accepted."""
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
STEPS, REPS = int(args.get("steps", 6)), int(args.get("reps", 20))
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]


def log(*a):
    print(f"M5B-WIDTHS {C.now()}", *a, flush=True)


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=True, rev=args.get("rev"), log=log)
be = ctx[key]["backend"]
kv = be._kv
io = kv.extra["io"]
traces = kv.extra.get("decode_traces", {})
SEL = args.get("narrow", False)  # False: the native widths; True: the narrow widths (w <= 4); "all": both
nat, nar = list(be.native_widths), list(getattr(be, "narrow_widths", ()))
widths = nat if SEL is False else nar if SEL is True else sorted(set(nat) | set(nar))
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


def timing(trace, live):
    """rows 0..live-1 live (position 512 + u); the rest idle (position -1). Both loops start from the same state
    (reset_all + the same tokens / positions), so the pipelined loop and the blocking loop time the same work."""
    tok = torch.tensor([1000 + 7 * u for u in range(B)])
    pos = torch.full((B,), -1, dtype=torch.int64)
    pos[:live] = torch.tensor([512 + u for u in range(live)])

    def start():
        be.reset_all(kv)
        be.write(io.tokens, tok)
        be.write(io.cur_pos, pos)
        ttnn.synchronize_device(mesh)

    start()
    be.replay(trace)  # warm
    ttnn.synchronize_device(mesh)
    start()
    t0 = time.perf_counter()
    for _ in range(REPS):
        be.replay(trace)
    ttnn.synchronize_device(mesh)
    dev = (time.perf_counter() - t0) * 1e6 / REPS
    start()
    lat = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        be.replay(trace)
        be.read(io.sampled)
        lat.append((time.perf_counter() - t0) * 1e6)
    mean = sum(lat) / len(lat)
    lat.sort()
    # device = mean of the pipelined loop; token-out = median (and mean) of the blocking loop
    return dict(device_us=round(dev, 1), token_out_us=round(lat[len(lat) // 2], 1), token_out_mean_us=round(mean, 1),
                live_rows=live)  # fmt: skip


full = traces[B]
res = {}
for w in widths:
    a_lg, a_rows = run(traces[w], w, STEPS)
    b_lg, b_rows = run(full, w, STEPS)
    bit = all(torch.equal(x, y) for x, y in zip(a_lg, b_lg))
    top1 = [float((x[:, 0] == y[:, 0]).float().mean()) for x, y in zip(a_rows, b_rows)]
    mad = max(float((x.float() - y.float()).abs().max()) for x, y in zip(a_lg, b_lg))
    res[w] = dict(logits_bitwise=bit, top1_agree_min=round(min(top1), 4), max_abs_logit_diff=round(mad, 4),
                  timing={str(n): timing(traces[w], n) for n in sorted({1, w})})  # fmt: skip
    log("WIDTH", w, res[w])
res[B] = dict(timing={str(n): timing(full, n) for n in (1, B)})
log("RESULT", res)
# the production rule: a user's tokens must not depend on how many users are live -> bitwise
ok = all(r.get("logits_bitwise", True) for r in res.values())
bitwise = all(r.get("logits_bitwise", True) for r in res.values())
accepted = args.get("accepted")
status = "pass" if ok else "record" if accepted else "fail"  # record = a failed criterion a cited ruling accepts
r = dict(test=f"m5b width buckets {widths} vs the full width ({len(ids)} layers, B={B}): live rows' logits / "
              f"top-K on the width-w trace vs the {B}-row trace + step time per width (1 live row and w live rows)",
         status=status, milestone="M5b", widths=widths, criterion="logits bitwise vs the full-width trace on every width (production rule)",
         
         criterion_met=ok, outcome="pass" if ok else "fail", logits_bitwise_all=bitwise, accepted=accepted,
         nkmax=os.environ.get("GLM53_MLA_ATT_NKMAX"), results={str(k): v for k, v in res.items()},
         tag=args.get("tag", "m5b-widths"))  # fmt: skip
if args.get("record") and not os.environ.get("TT_METAL_WATCHER"):
    C.record("tests", r)
    for w, v in res.items():
        for n, t in v["timing"].items():
            C.record("perf", dict(test=f"m5b decode step {'narrow' if w in nar else 'native'} width {w} (ONE execute_trace, {len(ids)} layers, B={B} "
                                       f"server, {n} live row{'s' if int(n) > 1 else ''})", status="info",
                                  milestone="M5b", traced=True, width=w, live_rows=int(n), reps=REPS,
                                  device_us=t["device_us"], token_out_us=t["token_out_us"],
                                  token_out_mean_us=t["token_out_mean_us"],
                                  stat="device = mean of back-to-back replays; token-out = median (mean also given) of "
                                       "replay + the blocking sampled read", tag=args.get("tag", "m5b-widths")))  # fmt: skip
RESULT = dict(ok=ok or bool(accepted), results={str(k): v for k, v in res.items()})
