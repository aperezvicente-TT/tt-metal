# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, the TP mesh / pair / chip 3): Stage 14b M5b latency — the decode step of tt/stack_model.py's
StackBackend as the serving contract runs it (ONE execute_trace per step: embed -> stack programs / sparse_dsa
programs -> head with device top-K + the in-trace position advance), timed on the captured decode trace with B active
rows at a mid-context position:

  device: N back-to-back replays (non-blocking) then one synchronize -> the steady per-step time;
  token-out: per step one replay + the blocking read of the sampled token rows (what a serving loop waits for).

Input regime: every row decodes at positions pos .. pos + rewind - 1 from a DEFINED state.
history "fill" (default): reset_all, then pos replays from position 0 with fixed tokens write the latent history of
positions 0..pos-1 (untimed, once); every timed chunk of <= rewind replays starts from reset_all (KDA / conv states
zero) + the same tokens at position pos, so each chunk sees the same history and rewrites its own rows before reading
them; history "zero" skips the fill. Positions stay inside supported_context. The token feedback is the model's own
(in-trace) output from that defined start. device = mean of back-to-back replays (one sync per chunk); token-out =
replay + the blocking sampled read per step (median, mean and p90 of the same loop).
history "text": every chunk starts from reset_all + a teacher-forced fill of positions 0..pos-1 with reference text
(ARGS ref: a .refpt; prompt + continuation, row u lagged by 7 u tokens so the B rows carry distinct tokens and route
like distinct users), then decodes self-fed from position pos: every chunk sees the same full history (KDA states and
latent rows). The token-out loop also reports the median of each chunk's first and last 100 steps.
ARGS: b (1), layers ("all" or ids), dense / sparse / dsa (ids), reps (50), pos (512), rewind (500), history ("fill" |
"text" | "zero"), ref, loops ("both" | "device" | "token"), native (one launch of B rows per kind run), shared_weights,
tag, record."""
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
import models.autoports.zai_org_glm_5_3_flash.tt.mk_head as MKH  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402

for mod in (M5, MKS, MKH, G, SM):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 1))
REPS = int(args.get("reps", 50))
POS = int(args.get("pos", 512))
tag = args.get("tag", f"m5c-step-B{B}")
TPN = int(mesh.get_num_devices())
cfg = hf_config()
if args.get("layers") is not None:
    ids = list(range(cfg.num_hidden_layers)) if args["layers"] == "all" else [int(v) for v in args["layers"]]
else:
    ids = sorted([int(v) for v in args.get("dense", [0, 1, 2])] + [int(v) for v in args.get("sparse", [])]
                 + [int(v) for v in args.get("dsa", [])])  # fmt: skip
assert TPN > 1, "the TP backend (a mesh); chip 3 times its stacks with m5_sweep"


def log(*a):
    print(f"M5C-STEP {C.now()}", *a, flush=True)


NATIVE = bool(args.get("native", False))
key = ("m5c_step", B, tuple(ids), TPN, NATIVE, args.get("rev"))
if args.get("serving"):  # the gate batch's shared backend (m5common.serving)
    key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=NATIVE,
                     rev=args.get("rev"), log=log)  # fmt: skip
else:
    M5.prune_ctx(ctx, [key])
if key not in ctx:
    t0 = time.time()
    M5.keep_decoders(ctx, [])
    backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg, native_t=NATIVE,
                                     weight_cache=ctx.setdefault("s14b_weights", {}) if args.get("shared_weights")
                                     else None)  # fmt: skip
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=2048,
                            stop_token_ids=())  # fmt: skip
    t1 = time.time()
    gen.warmup()
    ctx[key] = dict(backend=backend, gen=gen)
    log(f"backend {backend.trace_report()} built in {t1 - t0:.1f}s, warmup {time.time() - t1:.1f}s")
backend, gen = ctx[key]["backend"], ctx[key]["gen"]
kv = backend._kv
io = kv.extra["io"]
trace = kv.extra["decode_trace"]
toks = torch.tensor([1000 + 7 * u for u in range(B)], dtype=torch.int64)
RW = int(args.get("rewind", 500))
HIST = args.get("history", "fill")
LOOPS = args.get("loops", "both")
ctx_max = int(getattr(backend, "supported_context", 1 << 20))
assert POS + RW <= ctx_max, f"positions {POS}..{POS + RW - 1} exceed supported_context {ctx_max}"
text = None
if HIST == "text":
    _e = torch.load(args.get("ref", "/opt/glm53/kit/goldens_gpu_fp32_aime24/"
                                    "readiness_aime24_chat.refpt"), weights_only=True)["entries"][0]  # fmt: skip
    text = _e["prompt_tokens"][0].tolist() + _e["generated_tokens"][0].tolist()


def start(p):
    backend.reset_all(kv)  # KDA / conv states zero (history "fill": the DSA latent rows below p are the fill's)
    if text is not None:  # the full history again: teacher-forced reference text, row u lagged by 7 u tokens
        for q in range(p):
            backend.write(io.tokens, torch.tensor([text[(q + 7 * u) % len(text)] for u in range(B)], dtype=torch.int64))
            backend.write(io.cur_pos, torch.full((B,), q, dtype=torch.int64))
            backend.replay(trace)
        backend.write(io.tokens, torch.tensor([text[(p + 7 * u) % len(text)] for u in range(B)], dtype=torch.int64))
    else:
        backend.write(io.tokens, toks)
    backend.write(io.cur_pos, torch.full((B,), p, dtype=torch.int64))
    ttnn.synchronize_device(mesh)


t_fill = time.perf_counter()
if HIST == "fill" and POS > 0:
    start(0)
    for _ in range(POS):
        backend.replay(trace)
    ttnn.synchronize_device(mesh)
t_fill = time.perf_counter() - t_fill
start(POS)
backend.replay(trace)  # warm
ttnn.synchronize_device(mesh)
chunks = [min(RW, REPS - c) for c in range(0, REPS, RW)]
dev_us = None
if LOOPS in ("both", "device"):
    dev_s = 0.0
    for n in chunks:
        start(POS)
        t0 = time.perf_counter()
        for _ in range(n):
            backend.replay(trace)
        ttnn.synchronize_device(mesh)
        dev_s += time.perf_counter() - t0
    dev_us = dev_s * 1e6 / REPS
lat, first, last = [], [], []
out = backend.read(io.sampled)
if LOOPS in ("both", "token"):
    for n in chunks:
        start(POS)
        cl = []
        for _ in range(n):
            t0 = time.perf_counter()
            backend.replay(trace)
            out = backend.read(io.sampled)
            cl.append((time.perf_counter() - t0) * 1e6)
        lat += cl
        first += cl[:100]
        last += cl[-100:]
med = lambda v: round(sorted(v)[len(v) // 2], 1) if v else None  # noqa: E731
tok_mean = sum(lat) / len(lat) if lat else None
tok_us = med(lat)
names = [f"L{e['L']}" for e in backend.seq]
# the goal's bars apply to the full model only (a prefix row carries none); B=32 re-based from 25 to 38 ms
full = len(names) == cfg.num_hidden_layers
bar = (10000.0 if B == 1 else 38000.0 if B == 32 else None) if full and tok_us is not None else None
prow = dict(test=f"m5b decode step (ONE execute_trace: embed + {len(names)} layers + head/top-K) B={B} tp{TPN}"
                 + (" native-t" if NATIVE else ""), native_t=NATIVE, chunk=backend.chunk,
            status="info", milestone="M5b", traced=True, B=B, tp=TPN, layers=len(names), device_us=round(dev_us, 1) if dev_us is not None else None,
            token_out_us=tok_us, token_out_mean_us=round(tok_mean, 1) if tok_mean is not None else None,
            token_out_chunk_first100_us=med(first), token_out_chunk_last100_us=med(last), loops=LOOPS,
            token_out_p90_us=round(sorted(lat)[int(len(lat) * 0.9)], 1) if lat else None, reps=REPS, pos=POS,
            pos_range=[POS, POS + max(chunks) - 1], rewind_every=RW, history=HIST, fill_s=round(t_fill, 1),
            supported_context=ctx_max, full_model=full, bar_us=bar, meets_bar=(tok_us <= bar) if bar else None,
            sampled_rows=int(out.numel()), dram_gib_per_chip=backend.trace_report().get("dram_gib_per_chip"), tag=tag,
            note="device = mean of back-to-back replays (one sync per chunk of <= rewind_every); token-out = median "
                 "(mean, p90 also given) of replay + the blocking sampled-token read; every chunk starts from reset_all "
                 "+ the same tokens at pos (history: the fill's latent rows below pos)")  # fmt: skip
log("PERF", prow)
if args.get("record") and not os.environ.get("TT_METAL_WATCHER"):
    C.record("perf", prow)
RESULT = dict(ok=True, device_us=prow["device_us"], token_out_us=prow["token_out_us"])
