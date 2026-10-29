# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 milestone rows of the FULL model from ONE build (a full-model load per process):
tt/generator.py on tt/stack_model.py's StackBackend with GLM53_STACK_PREFILL=tp (the batched TP prefill, one traced
replay per chunk), warmed up in the generator's order (every prefill bucket compiled, decode compiled, the decode trace,
then the prefill traces).
  1. TF (P2 acceptance): the Stage 6 harness (readiness_check.run_teacher_forcing) -- prefill of the golden prompt then
     forced decode; bar top-5 >= 0.98 (ref = the CPU fp32 golden's top-32, make_prefill_tf_refpt.py).
  2. TTFT through the serving entry point: Glm53Generator.generate(prompt, max_new_tokens=1) (slot 0: page rows,
     chunked prefill replays, the first token sampled on device in the last replay, read back) -- wall per prompt
     length, median of reps; the last chunk's replay alone; the first token vs the CPU golden's argmax.
     Milestones: n = 2,048 -> "P2" (bar 0.5 s); n = 8,192 with dsa -> "P3" (bar 2.5 s); others "P2-n<n>".
  3. Throughput: 2,048-token prompts back to back (each a fresh request in slot 0): tokens / s (bar 8,000).

ARGS: layers ("all"), ref, b (1), max_seq (2560; 8192 with the 8,192 row), buckets ("128,512,2048"), dsa (False),
sparse_ts ([]), ns ([128, 512, 2048]), reps (3), tput_n (8: prompts of the throughput row), tf (True), tag, record,
deadline (ISO UTC, optional: the mesh slot's hard stop -- a section that cannot finish before it is skipped),
budget_min (optional: the same, as minutes from the job's start)."""
import datetime as dt
import gc
import os
import sys
import time
from pathlib import Path

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.generator import plan_prefill  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

C = S.C
WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
from readiness_check.run_teacher_forcing import run_teacher_forcing  # noqa: E402

MODEL_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
cfg = hf_config()
LAYERS = list(range(cfg.num_hidden_layers)) if args.get("layers", "all") == "all" else [int(v) for v in args["layers"]]
FULL = len(LAYERS) == cfg.num_hidden_layers
B = int(args.get("b", 1))
MAXSEQ = int(args.get("max_seq", 2560))
REF = args.get("ref", "/opt/glm53/work-s16/refpt/prefill2048_full.refpt")
NS = [int(v) for v in args.get("ns", [128, 512, 2048])]
REPS = int(args.get("reps", 3))
DSA = bool(args.get("dsa", False))
tag = args.get("tag", "p2-full")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
DEADLINE = (
    dt.datetime.fromisoformat(args["deadline"].replace("Z", "+00:00")).timestamp() if args.get("deadline") else None
)
if args.get("budget_min"):  # minutes from the job's start (the mesh slot opens right before it)
    _budget = time.time() + 60 * float(args["budget_min"])
    DEADLINE = min(DEADLINE, _budget) if DEADLINE is not None else _budget


def time_left() -> float:
    return float("inf") if DEADLINE is None else DEADLINE - time.time()


# stack_prefill (tp): "decode" = the prompt through the decode stack (prefill-as-decode) -- the A/B that tells a TP
# prefill issue from a decode / reference one in the TF row (TTFT rows then measure the decode-step prefill)
os.environ["GLM53_STACK_PREFILL"] = str(args.get("stack_prefill", "tp"))
DECODE_PF = os.environ["GLM53_STACK_PREFILL"] != "tp"
os.environ["GLM53_PREFILL_BUCKETS"] = args.get("buckets", "128,512,2048")
os.environ["GLM53_PF_MOE"] = args.get("moe", "ep")
os.environ["GLM53_PF_MHC"] = args.get("mhc", "comp")
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "ops")
os.environ["GLM53_PF_MOE_V2"] = "1" if args.get("moe_v2", False) else "0"
os.environ["GLM53_PF_KDA_SCAN"] = args.get("kda_scan", "op")
os.environ["GLM53_PF_DSA"] = "1" if DSA else "0"
os.environ["GLM53_PF_SPARSE_T"] = ",".join(str(int(v)) for v in args.get("sparse_ts", []))
S.set_attn(args)

key = (
    "s16-full",
    os.environ["GLM53_STACK_PREFILL"],
    tuple(LAYERS),
    B,
    TPN,
    MAXSEQ,
    os.environ["GLM53_PREFILL_BUCKETS"],
    os.environ["GLM53_PF_DSA"],
    os.environ["GLM53_PF_SPARSE_T"],
    os.environ["GLM53_PF_MOE"],
    os.environ["GLM53_PF_MHC"],
    os.environ["GLM53_PF_MHC_POST"],
    os.environ["GLM53_PF_MOE_V2"],
    S._attn_knobs(),
    os.environ.get("GLM53_PF_REP_BUCKETS", ""),
    os.environ.get("GLM53_PF_MM_NQ", ""),
    os.environ.get("GLM53_PF_MM_KB", ""),
)  # fmt: skip  # every knob the build reads: a change rebuilds
for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] in ("s16-full", "s16-tf", "s16-backend") and k != key]:
    old = ctx.pop(k)  # one model per process: the other builds' device memory first
    if "pf" in old:
        old["pf"].release()
    old["backend"].close()
gc.collect()
if key not in ctx:
    t0 = time.time()
    backend = SM.build_model_backend(None, mesh, max_batch=B, layers=LAYERS, hf_config=cfg,
                                     weight_cache=ctx.setdefault("weight_cache", {}))  # fmt: skip
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=MAXSEQ,
                            stop_token_ids=())  # fmt: skip
    t1 = time.time()
    gen.warmup()
    ctx[key] = dict(
        backend=backend,
        gen=gen,
        build_s=round(t1 - t0, 1),
        warmup_s=round(time.time() - t1, 1),
        eager0=backend.tpp.counters["eager"] if backend.tpp is not None else 0,
    )  # the compile forwards (one per trace key)
    S.log(f"full backend layers={len(LAYERS)} built in {t1 - t0:.1f}s, warmup {time.time() - t1:.1f}s "
          f"(prefill traces {sorted(backend.tpp.traces) if backend.tpp is not None else 'none: decode-stack prefill'})")  # fmt: skip
ent = ctx[key]
backend, gen = ent["backend"], ent["gen"]
pf = backend.tpp
results = []
dram = backend.trace_report().get("dram_gib_per_chip")

# ---- 1. prefill -> decode teacher forcing (args tf_first: before the TTFT rows (default) or after the throughput)


def run_tf():
    if args.get("tf", True) and time_left() < 90:
        S.log(f"TF skipped: {time_left():.0f} s to the deadline")
    if args.get("tf", True) and time_left() >= 90:
        c0, r0 = dict(gen.counters), dict(backend.trace_report())
        t0 = time.time()
        per_entry = run_teacher_forcing(model_dir=MODEL_DIR, reference_path=Path(REF), mesh_device=mesh,
                                        build_kwargs=dict(generator=gen))  # fmt: skip
        wall = time.time() - t0
        c1, r1 = dict(gen.counters), backend.trace_report()
        d = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
        s = per_entry[0]
        tp_chunks = r1.get("prefill_chunks_tp", 0) - r0.get("prefill_chunks_tp", 0)
        traced = d.get("eager_decode_steps", 0) == 0 and r1.get("decode_eager", 0) == r0.get("decode_eager", 0) and \
            d.get("decode_trace_replays", 0) > 0 and (DECODE_PF or (tp_chunks > 0 and pf.counters["eager"] == ent["eager0"]))  # fmt: skip
        ok = traced and s["top5"] >= 0.98
        r = dict(test=f"p2 prefill-then-decode TF (Stage 6 harness) {'full model' if FULL else f'prefix {LAYERS}'}: "
                      f"{'the prompt through the DECODE stack (prefill-as-decode A/B)' if DECODE_PF else 'batched TP prefill'} "
                      f"then forced decode, B={B} tp{TPN}", status="pass" if ok else "fail",
                 milestone="P2" if FULL else "P2-prefix", B=B, tp=TPN, layers=len(LAYERS), reference=REF, top1=s["top1"],
                 top5=s["top5"], top100=s["top100"],
                 matches=[s["matches_top1"], s["matches_top5"], s["matches_top100"], s["total"]], ttft_ms=s.get("ttft_ms"),
                 prefill_chunks_tp=tp_chunks, traced=traced, counters_delta=d,
                 tpp=dict(pf.counters) if pf is not None else None, stack_prefill=os.environ["GLM53_STACK_PREFILL"], dsa=DSA,
                 wall_s=round(wall, 1), tag=tag)  # fmt: skip
        S.log(r)
        if rec:
            C.record("pcc", r)
        results.append(r)


if args.get("tf_first", True) or DECODE_PF:
    run_tf()
if DECODE_PF:  # the A/B row only: the TTFT / throughput rows below measure the TP prefill -> none
    NS = []
    args = {**args, "tput_n": 0, "tf_first": True}

# ---- 2. TTFT through Glm53Generator.generate(prompt, 1)
long_prompt = S.golden_prompt(32768) if max(NS, default=0) > 4096 else S.golden_prompt(4096)
greedy = SamplingParams()
for n in NS:
    if time_left() < 45:
        S.log(f"TTFT n={n} skipped: {time_left():.0f} s to the deadline")
        continue
    toks = S.golden_prompt(n) if n in (128, 512, 2048, 4096) else long_prompt[:n]
    walls, first = [], None
    eager0 = pf.counters["eager"]
    if args.get("host_timing"):  # GLM53_PF_HOST_TIMING: per-step host ms of the timed requests (rep 0 excluded below)
        os.environ["GLM53_PF_HOST_TIMING"] = "1"
    for rep in range(REPS + 1):
        if rep == 1:
            backend.counters.pop("pf_host_ms", None)
            pf.counters.pop("host_ms", None)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        out = gen.generate([int(v) for v in toks], 1, sampling=greedy, stop_on_eos=False)
        wall = (time.perf_counter() - t0) * 1e3
        first = int(out[0])
        if rep:  # rep 0: the first request of this length (warm host caches), not timed
            walls.append(wall)
    plans_n = plan_prefill(n, 0, backend.prefill_chunk_tokens)  # chunks at the largest bucket
    last = plans_n[-1]
    c, t_win = pf.plan(last.start, last.n_valid)
    # the last chunk's trace alone: >= 2,000 steady-state replays (program 03:20Z device duty), once per key
    if (c, t_win) in pf.traces and not C.MOCK:
        soak_by_key = globals().setdefault("_soaks", {})
        if (c, t_win) not in soak_by_key:
            soak_by_key[(c, t_win)] = S.replay_stats(mesh, pf.traces[(c, t_win)], replays=args.get("replays"))
        rst = soak_by_key[(c, t_win)]
    else:
        rst = {}
    if not rst:  # the mock, or replays=0 (no soak in a mesh window, program 02:52Z): REPS blocking replays
        rs = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            pf.replay(c, blocking=True, t=t_win)
            rs.append((time.perf_counter() - t0) * 1e6)
        rst = C.stats(rs)
    st = C.stats([w * 1e3 for w in walls])
    if FULL and n == 2048:
        ms = "P2"
    elif FULL and DSA and n == 8192:
        ms = "P3"
    else:
        ms = f"P2-n{n}" if FULL else "P2-prefix"
    row = dict(test=f"TTFT B={B} prompt {n} tp{TPN} layers {len(LAYERS)} (Glm53Generator.generate, batched TP prefill "
                    f"traced per bucket{', DSA sparse chunks' if DSA else ''}, first token sampled on device)",
               milestone=ms, status="info", traced=pf.counters["eager"] == eager0, B=B, tp=TPN, n=n, layers=len(LAYERS),
               chunks=[(p.start, p.n_valid, *pf.plan(p.start, p.n_valid)) for p in plans_n],
               ms=round(st["median_us"] / 1e3, 2), ms_min=round(st["min_us"] / 1e3, 2),
               last_replay_ms=round(rst["median_us"] / 1e3, 2), tok_per_s=round(n / (st["median_us"] / 1e6), 1),
               first_token=first, requests=REPS, replays=rst.get("replays", REPS), replay_key=[c, t_win], dsa=DSA,
               sparse_ts=list(pf.sparse_ts), tag=tag, dram_gib_per_chip=dram,
               build_s=ent.get("build_s"), warmup_s=ent.get("warmup_s"))  # fmt: skip
    if args.get("host_timing"):
        acc, acc2 = dict(backend.counters.get("pf_host_ms", {})), dict(pf.counters.get("host_ms", {}))
        k_ = max(1, int(acc.pop("n", 1)))
        row["host_ms_per_request"] = {k: round(v / k_, 2) for k, v in list(acc.items()) + list(acc2.items())}
        os.environ["GLM53_PF_HOST_TIMING"] = "0"
    if not row["traced"]:
        row["status"] = "fail"  # an eager forward ran inside a measured request
    if FULL and n in (128, 512, 2048, 4096):
        g = S.golden_logits(n)
        row["golden_argmax"] = int(torch.argmax(g["last_logits"]))
        row["golden_top5"] = [int(v) for v in g["prompt_top32_indices"][n - 1, :5]]
        row["first_in_golden_top5"] = first in row["golden_top5"]
    S.log(row)
    if rec:
        C.record("perf", row)
    results.append(row)

# ---- 3. throughput: back-to-back 2,048-token requests
TN = int(args.get("tput_n", 8))
if TN and 2048 in NS and time_left() < 45:
    S.log(f"throughput skipped: {time_left():.0f} s to the deadline")
elif TN and 2048 in NS:
    toks = [int(v) for v in S.golden_prompt(2048)]
    eager_t = pf.counters["eager"]
    t0 = time.perf_counter()
    for _ in range(TN):
        gen.generate(toks, 1, sampling=greedy, stop_on_eos=False)
    wall = time.perf_counter() - t0
    row = dict(test=f"prefill throughput: {TN} x 2,048-token requests back to back, B={B} tp{TPN} layers {len(LAYERS)}",
               milestone="P2-tput" if FULL else "P2-prefix", status="info", traced=pf.counters["eager"] == eager_t, tp=TPN, requests=TN,
               tokens=TN * 2048, wall_s=round(wall, 3), tok_per_s=round(TN * 2048 / wall, 1),
               s_per_32x2048=round(wall / TN * 32, 3), tag=tag)  # fmt: skip
    S.log(row)
    if rec:
        C.record("perf", row)
    results.append(row)
if not args.get("tf_first", True):
    run_tf()
RESULT = results
