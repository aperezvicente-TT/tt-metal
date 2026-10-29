# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py): Stage 14b — a numerics fingerprint of the serving backend (m5common.serving: the process's first
load may come from a Stage 12 DRAM image): fixed token / position rows, `steps` traced decode steps, sha256 of the
logits rows + the top-K rows of every step. Two processes (tensor load + image build, then resident restart) must print
the same fingerprint. ARGS: b (32), layers ("all"), native (true), steps (4), shared_weights, rev, tag."""
import gc
import hashlib
import importlib
import os
import sys

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
NATIVE = bool(args.get("native", True))
STEPS = int(args.get("steps", 4))
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]


def log(*a):
    print(f"M5B-FP {C.now()}", *a, flush=True)


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=NATIVE, rev=args.get("rev"), log=log)
be = ctx[key]["backend"]
kv = be._kv
io = kv.extra["io"]
be.reset_all(kv)
be.write(io.tokens, torch.tensor([1000 + 13 * u for u in range(B)]))
be.write(io.cur_pos, torch.tensor([7 * u for u in range(B)]))
ttnn.synchronize_device(mesh)
h = hashlib.sha256()
for _ in range(STEPS):
    be.replay(kv.extra["decode_trace"])
    ttnn.synchronize_device(mesh)
    h.update(be.read(io.logits)[:B].float().numpy().tobytes())
    d = ttnn.get_device_tensors(be.head.topk_out)[0] if be.tp > 1 else be.head.topk_out
    h.update((ttnn.to_torch(d).to(torch.int64).reshape(32, 64) & 0xFFFFFFFF)[:B].numpy().tobytes())
fp = h.hexdigest()[:16]
res = dict(B=B, layers=len(ids), steps=STEPS, fingerprint=fp, tag=args.get("tag"),
           weight_source=os.environ.get("GLM53_WEIGHT_SOURCE", "tensors"))  # fmt: skip
log("RESULT", res)
RESULT = dict(ok=True, **res)
