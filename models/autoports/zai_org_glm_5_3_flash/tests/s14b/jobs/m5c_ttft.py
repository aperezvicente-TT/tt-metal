# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, the mesh): Stage 14b — time to first token through the serving generator on StackBackend, for the
record (program 16:29Z / 16:41Z): the stack PREFILLS AS DECODE (one prompt token per traced decode step), so TTFT
~ prompt length x step time. Prompt = real text (the AIME24 chat prompt repeated, cut to the length). One user (B=1
backend) per length in ARGS lengths; prompts longer than the backend's supported context are reported, not run.
ARGS: b (1), lengths ([128, 2048, 8192]), layers ("all"), shared_weights, tag, record."""
import gc
import importlib
import os
import sys
import time

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402

for mod in (M5, G, SM):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 1))
LENS = [int(v) for v in args.get("lengths", [128, 2048, 8192])]
tag = args.get("tag", "m5c-ttft")
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]


def log(*a):
    print(f"M5C-TTFT {C.now()}", *a, flush=True)


tok = load_tokenizer()
text = ("Find the number of ordered pairs (m, n) of positive integers such that m^2 n = 20^20. Show every step of "
        "the reasoning, check each case carefully, and give the final answer in a box. ")  # fmt: skip
base = tok.encode(text)
key = ("m5c_ttft", B, tuple(ids), int(mesh.get_num_devices()), args.get("rev"))
M5.prune_ctx(ctx, [key])
if key not in ctx:
    M5.keep_decoders(ctx, [])
    backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg,
                                     weight_cache=ctx.setdefault("s14b_weights", {}) if args.get("shared_weights")
                                     else None)  # fmt: skip
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=tok, max_batch=B,
                            max_seq_len=min(max(LENS) + 16, backend.supported_context), stop_token_ids=())  # fmt: skip
    gen.warmup()
    ctx[key] = dict(backend=backend, gen=gen)
backend, gen = ctx[key]["backend"], ctx[key]["gen"]
rows = []
for n in LENS:
    if n + 1 > min(gen.max_seq_len, backend.supported_context):
        r = dict(test=f"m5c TTFT prefill-as-decode {n}-token prompt B={B} (45-layer stack)", status="info",
                 milestone="M5c", prompt_tokens=n, ttft_s=None,
                 note=f"not run: prompt + 1 > supported context {backend.supported_context} (sparse_dsa dense-exact)",
                 tag=tag)  # fmt: skip
        log("PERF", r)
        rows.append(r)
        continue
    prompt = (base * (n // len(base) + 1))[:n]
    gen.reset()
    t0 = time.perf_counter()
    out = gen.generate(prompt, 1, enable_trace=True)
    ttft = time.perf_counter() - t0
    r = dict(test=f"m5c TTFT prefill-as-decode {n}-token prompt B={B} ({len(ids)}-layer stack, one traced decode step "
                  f"per prompt token)", status="info", milestone="M5c", traced=True, prompt_tokens=n,
             ttft_s=round(ttft, 3), per_token_ms=round(1e3 * ttft / n, 2), first_token=int(out[0]) if out else None,
             path="prefill as decode (no sharded op-chain prefill in the stack)", target_s=1.0 if n == 2048 else None,
             tag=tag)  # fmt: skip
    log("PERF", r)
    rows.append(r)
    if args.get("record") and not os.environ.get("TT_METAL_WATCHER"):
        C.record("perf", r)
RESULT = dict(ok=True, rows=[(r["prompt_tokens"], r["ttft_s"]) for r in rows])
