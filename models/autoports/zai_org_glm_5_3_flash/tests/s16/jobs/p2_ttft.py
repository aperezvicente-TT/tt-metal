# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 P2 — TTFT of the batched TP prefill through StackBackend.prefill_chunk (the
generator contract's entry point): per request the slot's KDA state into the per-layer scratch, the chunk's device
inputs, ONE trace replay per chunk (embed + every layer + the head on the last valid row: device top-K / sampler),
the scratch back into the slot, the first token read. Every bucket compiled (eager) then captured before the first
timed request; timed = traced replays only.

Rows: per prompt length the wall of the whole prefill_chunk sequence (host inputs + replays + reads = TTFT at B=1) and
the last chunk's trace replayed alone: the steady-state median of >= `replays` (s16common.SOAK_REPLAYS, 2,000) replays
(program 03:20Z device duty; the row cites the count); `reps` timed requests per prompt length.

ARGS: layers (default all 45), b (1), buckets ([128, 512, 2048]), ns ([128, 2048]), reps (5), tag, milestone ("P2"),
record (True), check_token (True: the full model's first token vs the CPU golden's argmax at n - 1), max_seq (4096),
dsa (False: GLM53_PF_DSA=1, DSA sparse chunks past position 2,050), sparse_ts ([]: GLM53_PF_SPARSE_T windows),
warmup ("decode": the generator's order; "prefill": prefill traces only), leak_debug (False: TRACE_LEAK lines)."""
import gc
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.generator import plan_prefill  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

GREEDY = SamplingParams()

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = list(range(45)) if args.get("layers", "all") == "all" else [int(v) for v in args["layers"]]
B = int(args.get("b", 1))
BUCKETS = tuple(int(v) for v in args.get("buckets", [128, 512, 2048]))
NS = [int(v) for v in args.get("ns", [128, 2048])]
REPS = int(args.get("reps", 5))
tag = args.get("tag", "p2-ttft")
ms = args.get("milestone", "P2")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())

os.environ["GLM53_PF_MHC"] = args.get("mhc", "comp")
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "ops")
os.environ["GLM53_PF_DSA"] = "1" if args.get("dsa", False) else "0"
if args.get("leak_debug"):
    os.environ["GLM53_TRACE_LEAK_DEBUG"] = "1"
os.environ["GLM53_PF_SPARSE_T"] = ",".join(str(int(v)) for v in args.get("sparse_ts", []))
S.set_attn(args)
ent = S.build(mesh, ctx, LAYERS, b=B, buckets=BUCKETS, max_seq=int(args.get("max_seq", 4096)))
pf, kv, io, backend = ent["pf"], ent["kv"], ent["io"], ent["backend"]
pf.moe_impl = args.get("moe", pf.moe_impl)
S.set_moe_v2(pf, bool(args.get("moe_v2", False)))
pf.kda_scan = args.get("kda_scan", pf.kda_scan)
pf.sparse_ts = tuple(int(v) for v in args.get("sparse_ts", []))
backend.tpp = pf
if not ent.get("captured"):
    t0 = time.time()
    pf.compile(BUCKETS)
    t_compile = time.time() - t0
    t0 = time.time()
    # the generator's warmup order (generator.warmup): every bucket compiled, the decode step compiled, the decode trace,
    # then the prefill traces (StackBackend.capture_decode captures them after the decode trace)
    if args.get("warmup", "decode") == "decode":
        backend.prepare_decode(io, kv, [B])
        backend.capture_decode(io, kv, B)
    else:  # "prefill": the prefill traces alone (contexts the dense decode stack cannot hold, e.g. 8,192 > MkMla's 6,144)
        # one eager request per bucket first (no trace yet: prefill_chunk runs the forward eagerly), so the request path's
        # lazily-allocated host-read buffers exist before any capture (19:36Z: a live FP32 buffer at the first replay)
        for c_ in BUCKETS:
            backend.prefill_chunk(torch.zeros(c_, dtype=torch.int64), kv_cache=kv, slot=0, chunk_start=0, n_valid=c_,
                                  logits_rows="last", io=io, sample=GREEDY)  # fmt: skip
        ttnn.synchronize_device(mesh)
        pf.capture(BUCKETS)
    ent["captured"] = True
    S.log(f"compiled {BUCKETS} in {t_compile:.1f} s, captured in {time.time() - t0:.1f} s, traces {pf.traces}")
prompt = S.golden_prompt(32768) if max(NS) > 4096 else S.golden_prompt(4096)
results, soaks = [], {}
for n in NS:
    toks = S.golden_prompt(n) if n in (128, 512, 2048, 4096) else prompt[:n]
    if toks.numel() < n:  # past the longest golden prompt (32,768): the prompt repeated (a timing row; the first token
        toks = prompt.repeat(-(-n // prompt.numel()))[:n]  # is not checked against any golden)
    walls, replay_us = [], []
    first = None
    eager0 = pf.counters["eager"]
    for rep in range(REPS + 1):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        plans = plan_prefill(n, 0, backend.prefill_chunk_tokens)
        for p in plans:  # the last chunk samples the first token on device (greedy), as the serving path does
            chunk = toks[p.offset : p.offset + p.n_valid]
            last_p = p is plans[-1]  # the generator's protocol: nothing asked of a non-final chunk
            out = backend.prefill_chunk(chunk, kv_cache=kv, slot=0, chunk_start=p.start, n_valid=p.n_valid,
                                        logits_rows="last" if last_p else "none", io=io,
                                        sample=GREEDY if last_p else None)  # fmt: skip
        first = int(out)
        wall = (time.perf_counter() - t0) * 1e3
        if rep:  # rep 0 = the first request after capture (warm caches), not timed
            walls.append(wall)
    # the replay alone (same inputs as the last chunk): >= SOAK_REPLAYS steady-state replays of its trace (program
    # 03:20Z device duty), once per key; the mock (no traces) times REPS eager forwards
    plans_n = plan_prefill(n, 0, backend.prefill_chunk_tokens)  # chunks at the largest bucket
    last = plans_n[-1]
    c, t_win = pf.plan(last.start, last.n_valid)
    rs = {}
    if (c, t_win) in pf.traces and not C.MOCK:
        if (c, t_win) not in soaks:
            soaks[(c, t_win)] = S.replay_stats(mesh, pf.traces[(c, t_win)], replays=args.get("replays"))
        rs = dict(soaks[(c, t_win)])
    if not rs:  # no soak (replays=0) or the mock: REPS blocking (synchronized) replays
        for _ in range(REPS):
            t0 = time.perf_counter()
            pf.replay(c, blocking=True, t=t_win)
            replay_us.append((time.perf_counter() - t0) * 1e6)
        rs = C.stats(replay_us)
    st = C.stats([w * 1e3 for w in walls])
    row = dict(test=f"p2 TTFT B={B} prompt {n} tp{TPN} layers {len(LAYERS)} (batched TP prefill, traced per bucket)",
               milestone=(ms if n == int(args.get("gate_n", 2048)) else f"{ms}-n{n}") if len(LAYERS) == 45
               else f"{ms}-prefix", status="info", traced=pf.counters["eager"] == eager0, B=B, tp=TPN,
               n=n, layers=len(LAYERS), buckets=list(BUCKETS),
               chunks=[(p.start, p.n_valid, *pf.plan(p.start, p.n_valid)) for p in plans_n],
               dsa=pf.dsa, sparse_ts=list(pf.sparse_ts),
               ms=round(st["median_us"] / 1e3, 2), ms_min=round(st["min_us"] / 1e3, 2),
               replay_ms=round(rs["median_us"] / 1e3, 2), replay_key=[c, t_win], replays=rs.get("replays", REPS),
               tok_per_s=round(n / (st["median_us"] / 1e6), 1), first_token=first, requests=REPS,
               counters=dict(pf.counters), tag=tag,
               dram_gib_per_chip=backend.trace_report().get("dram_gib_per_chip"))  # fmt: skip
    if len(LAYERS) == 45 and args.get("check_token", True) and n in (128, 512, 2048, 4096):
        g = S.golden_logits(n)
        row["golden_argmax"] = int(torch.argmax(g["last_logits"]))
        row["golden_top5"] = [int(v) for v in g["prompt_top32_indices"][n - 1, :5]]
        row["first_in_golden_top5"] = first in row["golden_top5"]
    S.log(row)
    if rec:
        C.record("perf", row)
    results.append(row)
RESULT = results
