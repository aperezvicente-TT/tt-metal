# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py: a pair first (watcher), then the mesh): Stage 14b — gate of one shared fabric phase per KDA kind
(StackBackend, GLM53_STACK_SHARE_FABRIC=1, b9681a70ce3) against one phase per layer (=0): the same layers, the same
inputs, `steps` traced decode steps each; the sampled rows and the logits of every step must be BITWISE equal (sharing
only moves the lanes' sync buffers; every stack program of a kind then runs on one buffer, generation-counted).
ARGS: b (32), layers ([0..6]), native (true), steps (4), shared_weights, record, rev."""
import gc
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
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 32))
NATIVE = bool(args.get("native", True))
STEPS = int(args.get("steps", 4))
cfg = hf_config()
ids = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4, 5, 6])]


def log(*a):
    print(f"M5B-SHAREAB {C.now()}", *a, flush=True)


def run(share: str):
    os.environ["GLM53_STACK_SHARE_FABRIC"] = share
    M5.prune_ctx(ctx, [])
    M5.keep_decoders(ctx, [])
    be = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg, native_t=NATIVE,
                                weight_cache=ctx.setdefault("s14b_weights", {}) if args.get("shared_weights")
                                else None)  # fmt: skip
    gen = G.build_generator(None, mesh, backend=be, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=512,
                            stop_token_ids=())  # fmt: skip
    gen.warmup()
    ctx[("m5b_shareab", share)] = dict(backend=be, gen=gen)  # pruned (closed) by the next run
    kv = be._kv
    io = kv.extra["io"]
    be.reset_all(kv)
    be.write(io.tokens, torch.tensor([1000 + 13 * u for u in range(B)]))
    be.write(io.cur_pos, torch.tensor([7 * u for u in range(B)]))
    ttnn.synchronize_device(mesh)
    rows, logits = [], []
    for _ in range(STEPS):
        be.replay(kv.extra["decode_trace"])
        ttnn.synchronize_device(mesh)
        rows.append(be.read(io.sampled).clone())
        logits.append(be.read(io.logits).clone())
    phases = len({id(e["tp"].fabric) for e in be.seq if e["sparse"] != "dsa" and e["tp"].fabric is not None})
    return rows, logits, phases


try:
    r0 = run("0")
    r1 = run("1")
finally:
    os.environ.pop("GLM53_STACK_SHARE_FABRIC", None)
    M5.prune_ctx(ctx, [])
same_rows = all(torch.equal(a, b) for a, b in zip(r0[0], r1[0]))
same_logits = all(torch.equal(a, b) for a, b in zip(r0[1], r1[1]))
res = dict(B=B, native_t=NATIVE, layers=len(ids), steps=STEPS, kda_phases_per_layer=r0[2], kda_phases_shared=r1[2],
           sampled_bitwise=same_rows, logits_bitwise=same_logits, tp=int(mesh.get_num_devices()))  # fmt: skip
log("RESULT", res)
ok = same_rows and same_logits
r = dict(test=f"m5b one shared fabric phase per KDA kind vs one per layer ({len(ids)} layers, B={B}"
              f"{' native-t' if NATIVE else ''}, tp{res['tp']}, {STEPS} traced steps): bitwise sampled rows + logits",
         status="pass" if ok else "fail", milestone="M5b", **res)  # fmt: skip
if args.get("record"):
    C.record("tests", r)
RESULT = dict(ok=ok, **res)
