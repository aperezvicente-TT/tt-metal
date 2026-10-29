# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; program 2026-10-22 10:51Z / 11:42Z / 12:35Z, the row-isolation item): a live row's numerics and the
step cost must not depend on the IDLE rows (batch invariance), proven on the serving backend's captured decode traces
with the routed experts' live-row page ON (ReaderExperts MKX_LIVE) and OFF (StackBackend.moe_live_all = every row live
= the pre-fix routing on the SAME programs: the A/B of one build). Per mode:
  idle   per captured trace (full width + native widths): FRESH = reset_all, row 0 alone (token 1000 at position 7,
         every other row idle at -1, token 0) for `steps` replays; DIRTY = reset_all, a burst on rows 1.. (distinct
         tokens / positions, row 0 idle) for `burst` replays, then rows 1.. idle WITH their stale sampled tokens and row 0
         alone as in FRESH. Row 0's logits + sampled token per step BITWISE FRESH vs DIRTY; the 1-live step time fresh
         vs after the burst.
  abort  (9's abortmix, survivors vs solo) on the full-width trace: SOLO = row 0 alone for s1 + s2 replays; MIX = rows
         0..3 live (row 0 as in SOLO, rows 1..3 distinct) for s1 replays, then rows 1..3 aborted (position -1, stale
         tokens) and row 0 alone for s2. Row 0 bitwise per phase: phase 1 = coupling to LIVE co-rows (the union /
         grouping of the routed experts, not what MKX_LIVE removes), phase 2 = after the aborts.
Criterion (MKX_LIVE on): idle bitwise on every trace; phase 2 of abort bitwise when phase 1 is.
ARGS: b (32), layers ("all"), native (true), rev, steps (8), burst (6), reps (200), s1 (6), s2 (6), widths (null = every
captured trace), modes (["off", "on"]), record (true), tag."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
B, STEPS, BURST = int(args.get("b", 32)), int(args.get("steps", 8)), int(args.get("burst", 6))
REPS, S1, S2 = int(args.get("reps", 200)), int(args.get("s1", 6)), int(args.get("s2", 6))
TP = int(mesh.get_num_devices())


def log(*a):
    print(f"S14B-IDLE {C.now()}", *a, flush=True)


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=bool(args.get("native", True)), rev=args.get("rev"),
                 log=log)  # fmt: skip
be = ctx[key]["backend"]
kv = be._kv
io = kv.extra["io"]
traces = dict(kv.extra.get("decode_traces", {}))
if not traces and kv.extra.get("decode_trace") is not None:
    traces = {B: kv.extra["decode_trace"]}
sel = args.get("widths")
widths = sorted(traces) if sel is None else [int(w) for w in sel if int(w) in traces]
has_live = getattr(be, "_live_page", None) is not None
log("traces", sorted(traces), "testing", widths, "live page (MKX_LIVE programs)", has_live)
assert has_live, "the served build has no MKX_LIVE programs (GLM53_MOE_LIVE=0 or a chunked step): nothing to A/B"


def head_row(r=0):
    d = ttnn.get_device_tensors(be.head.topk_out)[0] if be.tp > 1 else be.head.topk_out
    return (ttnn.to_torch(d).to(torch.int64).reshape(32, 64) & 0xFFFFFFFF)[r].clone()


def start(tok, pos):
    be.write(io.tokens, tok)
    be.write(io.cur_pos, pos)  # also the live-row page (the mode's)
    ttnn.synchronize_device(mesh)


def prepare(trace, dirty: bool):
    be._kv = kv
    be.reset_all(kv)
    tok = torch.zeros(B, dtype=torch.int64)
    if dirty:  # rows 1.. live (row 0 idle), distinct tokens / positions -> stale sampled tokens left behind
        tok[1:] = torch.tensor([3000 + 17 * u for u in range(1, B)])
        pos = torch.full((B,), -1, dtype=torch.int64)
        pos[1:] = torch.tensor([300 + 5 * u for u in range(1, B)])
        start(tok, pos)
        for _ in range(BURST):
            be.replay(traces[B])  # the burst runs on the full width (every row 1.. live)
        ttnn.synchronize_device(mesh)
        tok = be.read(io.tokens).to(torch.int64).clone()  # the idle rows keep their stale tokens
    tok[0] = 1000
    pos = torch.full((B,), -1, dtype=torch.int64)
    pos[0] = 7
    start(tok, pos)
    return int((tok[1:] != 0).sum())


def steps(trace, n):
    lg, rows = [], []
    for _ in range(n):
        be.replay(trace)
        ttnn.synchronize_device(mesh)
        lg.append(be.read(io.logits)[0].clone())
        rows.append(head_row(0))
    return lg, rows


def first_diff(a, b):
    return next((s for s in range(min(len(a[0]), len(b[0]))) if not torch.equal(a[0][s], b[0][s])
                 or not torch.equal(a[1][s], b[1][s])), None)  # fmt: skip


def maxabs(a, b):
    return round(max((float((x.float() - y.float()).abs().max()) for x, y in zip(a[0], b[0])), default=0.0), 5)


def timing(trace, dirty: bool):
    prepare(trace, dirty)
    be.replay(trace)  # warm
    ttnn.synchronize_device(mesh)
    prepare(trace, dirty)
    t0 = time.perf_counter()
    for _ in range(REPS):
        be.replay(trace)
    ttnn.synchronize_device(mesh)
    return round((time.perf_counter() - t0) * 1e6 / REPS, 1)


def abort_mix(trace):
    """9's abortmix at the device: row 0 SOLO for S1 + S2 vs row 0 with rows 1..3 live for S1, then rows 1..3 aborted"""
    be._kv = kv
    be.reset_all(kv)
    tok = torch.zeros(B, dtype=torch.int64)
    tok[0] = 1000
    pos = torch.full((B,), -1, dtype=torch.int64)
    pos[0] = 7
    start(tok, pos)
    solo = steps(trace, S1 + S2)
    be.reset_all(kv)
    tok[1:4] = torch.tensor([3000, 3017, 3034])
    pos[1:4] = torch.tensor([300, 305, 310])
    start(tok, pos)
    m1 = steps(trace, S1)
    cur = be.read(io.cur_pos).to(torch.int64).reshape(-1)[:B].clone()  # row 0 advanced in-trace; tokens untouched
    cur[1:4] = -1
    be.write(io.cur_pos, cur)
    ttnn.synchronize_device(mesh)
    m2 = steps(trace, S2)
    s1_, s2_ = (solo[0][:S1], solo[1][:S1]), (solo[0][S1:], solo[1][S1:])
    return dict(phase1_bitwise=first_diff(s1_, m1) is None, phase1_first_diff=first_diff(s1_, m1),
                phase1_max_abs=maxabs(s1_, m1), phase1_tokens_equal=all(torch.equal(a, b) for a, b in zip(s1_[1], m1[1])),
                phase2_bitwise=first_diff(s2_, m2) is None, phase2_first_diff=first_diff(s2_, m2),
                phase2_max_abs=maxabs(s2_, m2), phase2_tokens_equal=all(torch.equal(a, b) for a, b in zip(s2_[1], m2[1])))  # fmt: skip


res = {}
for mode in args.get("modes", ["off", "on"]):
    be.moe_live_all = mode == "off"  # off: every row live on the page = the pre-fix routing (same programs)
    r = res[mode] = {}
    for w in widths:
        prepare(traces[w], False)
        f = steps(traces[w], STEPS)
        stale = prepare(traces[w], True)
        d = steps(traces[w], STEPS)
        tf, td = timing(traces[w], False), timing(traces[w], True)
        r[str(w)] = dict(bitwise=first_diff(f, d) is None, first_diff_step=first_diff(f, d), max_abs_logit=maxabs(f, d),
                         stale_idle_tokens=stale, step_us_fresh=tf, step_us_after_burst=td,
                         step_delta_pct=round(100 * (td - tf) / tf, 2))  # fmt: skip
        log(mode, "WIDTH", w, r[str(w)])
    r["abort"] = abort_mix(traces[B])
    log(mode, "ABORT", r["abort"])
be.moe_live_all = False
be.reset_all(kv)
start(torch.zeros(B, dtype=torch.int64), torch.full((B,), -1, dtype=torch.int64))
on = res.get("on", {})
ok = bool(on) and all(v["bitwise"] for k, v in on.items() if k != "abort") and (
    on["abort"]["phase2_bitwise"] or not on["abort"]["phase1_bitwise"])  # fmt: skip
row = dict(test=f"row isolation: a lone live row with fresh vs stale (after a burst) idle rows on every captured decode trace + "
                f"abortmix (survivor vs solo), routed-expert live-row page OFF / ON in one build, tp{TP} layers {len(ids)}",
           milestone="M5c", status="pass" if ok else "fail",
           criterion="MKX_LIVE on: row 0 logits + sampled token BITWISE fresh vs stale idle rows on every trace; the "
                     "survivor after the aborts bitwise vs solo when the co-live phase is; step time fresh ~= after burst",
           results=res, steps=STEPS, burst=BURST, reps=REPS, s1=S1, s2=S2, rev=args.get("rev"),
           tag=args.get("tag", f"m5c-row-isolation-tp{TP}"))  # fmt: skip
log("VERDICT", "PASS" if ok else "FAIL", {m: {k: (v.get("bitwise", v.get("phase2_bitwise")), v.get("step_delta_pct"))
                                             for k, v in r.items()} for m, r in res.items()})  # fmt: skip
if args.get("record", True) and not C.MOCK:
    C.record("pcc", row)
RESULT = dict(ok=ok, results=res)
