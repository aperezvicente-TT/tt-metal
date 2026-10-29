# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, the mesh): Stage 14b -- why the int25 TP8 widths row gave narrow w1 top-1 agreement 0.0 against the
32-row trace while w1's own lagged TF passed (0.94 / 1.00 / 1.00). On the resident serving backend (m5common.serving):
  A  same-input replays (m5b_widths' method): reset_all, rows 0..w-1 = tokens 1000 + 13u at positions 7u, `steps`
     replays of the width-w trace and of the 32-row trace; per step and per DEVICE: top-K page row 0 (first 8 words),
     the logits argmax of row 0 (host read), the sampled-token row 0;
  B  the production rule on the generator: one greedy user (`n` prompt tokens + `new` steps) decoded alone (the
     generator picks the narrowest bucket) vs the same user with B - 1 other live users (the 32-row trace).
No record (diagnostic); the RESULT carries the per-step values.
ARGS: b (32), layers ("all"), rev, w (1), steps (4), n (64), new (16)."""
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
B, W, STEPS = int(args.get("b", 32)), int(args.get("w", 1)), int(args.get("steps", 4))
N, NEW = int(args.get("n", 64)), int(args.get("new", 16))
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
GOLDENS = os.environ.get("GLM53_GOLDENS", "/opt/glm53/kit/goldens")


def log(*a):
    print(f"NARROW-DIAG {C.now()}", *a, flush=True)


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=True, rev=args.get("rev"), log=log)
be, gen = ctx[key]["backend"], ctx[key]["gen"]
kv = be._kv
io = kv.extra["io"]
traces = kv.extra.get("decode_traces", {})
log("narrow", getattr(be, "narrow_widths", ()), "native", be.native_widths, "traces", sorted(traces))
assert W in traces and B in traces, (W, sorted(traces))


def per_device(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)] if be.tp > 1 else [ttnn.to_torch(t)]


def run(trace, w):
    be._kv = kv
    be.reset_all(kv)
    tok = torch.zeros(B, dtype=torch.int64)
    tok[:w] = torch.tensor([1000 + 13 * u for u in range(w)])
    pos = torch.full((B,), -1, dtype=torch.int64)
    pos[:w] = torch.tensor([7 * u for u in range(w)])
    be.write(io.tokens, tok)
    be.write(io.cur_pos, pos)
    ttnn.synchronize_device(mesh)
    out = []
    for _ in range(STEPS):
        be.replay(trace)
        ttnn.synchronize_device(mesh)
        lg = be.read(io.logits)[0]
        pages = [(p.to(torch.int64).reshape(-1) & 0xFFFFFFFF)[:8].tolist() for p in per_device(be.head.topk_out)]
        out.append(dict(argmax=int(torch.argmax(lg.float())), tok=int(be.read(io.tokens)[0]),
                        sampled=int(be.read(io.sampled)[0]), page0=pages))  # fmt: skip
    return out


a, b = run(traces[W], W), run(traces[B], W)
for s, (x, y) in enumerate(zip(a, b)):
    log(f"step {s}: w{W} argmax {x['argmax']} tok {x['tok']} sampled {x['sampled']} | w{B} argmax {y['argmax']} "
        f"tok {y['tok']} sampled {y['sampled']}")
    log(f"  w{W} topk page row 0 per device {x['page0']}")
    log(f"  w{B} topk page row 0 per device {y['page0']}")
# B: the generator, one user alone vs with B - 1 other live users
toks = [int(v) for v in torch.load(os.path.join(GOLDENS, "prompts", "128.pt"), map_location="cpu",
                                   weights_only=True).reshape(-1)[:N]]  # fmt: skip
greedy = SamplingParams()
gen.reset()
alone = [int(v) for v in gen.generate(toks, NEW, sampling=greedy, stop_on_eos=False)]
gen.reset()
others = [[int(v) for v in torch.load(os.path.join(GOLDENS, "prompts", "2048.pt"), map_location="cpu",
                                      weights_only=True).reshape(-1)[17 * u : 17 * u + N]] for u in range(1, B)]  # fmt: skip
batch = gen.generate_batch([toks] + others, NEW, sampling=[greedy] * B, stop_on_eos=False) \
    if hasattr(gen, "generate_batch") else None  # fmt: skip
with_others = [int(v) for v in batch[0]] if batch is not None else None
log("generator: alone", alone)
log("generator: with others", with_others)
gen.reset()
RESULT = dict(same_input=[dict(w=x, full=y) for x, y in zip(a, b)], alone=alone, with_others=with_others,
              alone_equals_with_others=(alone == with_others) if with_others is not None else None)
