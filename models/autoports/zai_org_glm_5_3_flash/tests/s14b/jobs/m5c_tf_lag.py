# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the mesh / pair / chip 3): Stage 14b — teacher forcing with B rows at B DIFFERENT positions in every
step, on the bare decode trace of tt/stack_model.py's StackBackend (one execute_trace per step; the host writes the
step's token and position rows between replays, as the TF harness does). Row u runs the reference sequence (prompt P +
its greedy continuation G) starting u steps after row 0, so in one step the rows sit at B distinct positions (distinct
KDA states, latent pages, embeddings); every row is scored at every continuation position against the reference top-k
(the Stage 6 harness's TokenAccuracy rule) -> B x len(G) predictions. All B rows prefill the prompt IN the same
steps (lagged), which makes the 45-layer B=32 run ~len(P) + len(G) + B steps instead of B per-user prefills.

width (default b): run U = width lagged users on the width-w decode trace (a native / narrow bucket of the B-row
server; rows >= w idle), i.e. the trace a w-user batch is served on.
ARGS: b (32), layers ("all" or ids), ref, width, shared_weights, native, tag, record, rev."""
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
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402

for mod in (M5, MKH, G, SM):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 32))
REF = args["ref"]
tag = args.get("tag", f"m5c-tf-lag-B{B}")
TPN = int(mesh.get_num_devices())
cfg = hf_config()
ids = list(range(cfg.num_hidden_layers)) if args.get("layers", "all") == "all" else [int(v) for v in args["layers"]]


def log(*a):
    print(f"M5C-TFLAG {C.now()}", *a, flush=True)


ref = torch.load(REF, weights_only=True)["entries"][0]
P = ref["prompt_tokens"][0].tolist()
Gt = ref["generated_tokens"][0].tolist()
topk = ref["topk_tokens"]  # [len(G), 100]
seq = P + Gt
NP, NG = len(P), len(Gt)
key = ("m5c_tflag", B, tuple(ids), TPN, bool(args.get("native", False)), args.get("rev"))
if args.get("serving"):  # the gate batch's shared backend (m5common.serving)
    key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg,
                     native=bool(args.get("native", False)),
                     max_seq_len=max(2048, NP + NG + 8), rev=args.get("rev"),
                     log=log)  # fmt: skip
else:
    M5.prune_ctx(ctx, [key])
if key not in ctx:
    t0 = time.time()
    M5.keep_decoders(ctx, [])
    backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg,
                                     native_t=bool(args.get("native", False)),
                                     weight_cache=ctx.setdefault("s14b_weights", {}) if args.get("shared_weights")
                                     else None)  # fmt: skip
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B,
                            max_seq_len=max(512, NP + NG + 8), stop_token_ids=())  # fmt: skip
    t1 = time.time()
    gen.warmup()
    ctx[key] = dict(backend=backend, gen=gen)
    log(f"backend {backend.trace_report()} built in {t1 - t0:.1f}s, warmup {time.time() - t1:.1f}s")
backend, gen = ctx[key]["backend"], ctx[key]["gen"]
kv = backend._kv
io = kv.extra["io"]
W = int(args.get("width", B))  # users U = W, on the width-W trace
trace = kv.extra["decode_trace"] if W == B else kv.extra["decode_traces"][W]
backend.reset_all(kv)  # fresh KDA states (latent rows are overwritten position by position)
ttnn.synchronize_device(mesh)
preds = [[] for _ in range(W)]
n_steps = NP + NG + W - 1
t0 = time.perf_counter()
for s in range(n_steps):
    tok = torch.zeros(B, dtype=torch.int64)
    pos = torch.full((B,), -1, dtype=torch.int64)
    for u in range(W):
        q = s - u  # row u's position this step
        if 0 <= q < NP + NG - 1:  # feeds seq[q], predicts seq[q + 1]
            tok[u], pos[u] = seq[q], q
    backend.write(io.tokens, tok)
    backend.write(io.cur_pos, pos)
    backend.replay(trace)
    out = backend.read(io.sampled)
    for u in range(W):
        q = s - u
        if NP - 1 <= q < NP + NG - 1:  # a continuation position: prediction of G[q - NP + 1]
            preds[u].append(int(out[u]))
wall = time.perf_counter() - t0
m1 = m5 = m100 = tot = 0
per_user = []
for u in range(W):
    hits = [(p == int(topk[i, 0]), p in topk[i, :5].tolist(), p in topk[i, :100].tolist())
            for i, p in enumerate(preds[u][:NG])]  # fmt: skip
    a1, a5, a100 = (sum(h[j] for h in hits) for j in range(3))
    per_user.append(dict(u=u, n=len(hits), top5=a5))
    m1, m5, m100, tot = m1 + a1, m5 + a5, m100 + a100, tot + len(hits)
complete = all(p["n"] == NG for p in per_user)
acc = dict(top1=round(m1 / tot, 4), top5=round(m5 / tot, 4), top100=round(m100 / tot, 4), total=tot)
worst = min(per_user, key=lambda p: p["top5"] / max(p["n"], 1))
rows_identical = all(preds[u][:NG] == preds[0][:NG] for u in range(W))  # same context per position -> same argmax
names = [f"L{e['L']}" for e in backend.seq]
ok = complete and acc["top5"] >= 0.98 and acc["top100"] >= 1.0
on = f"the width-{W} trace of the B={B} server" if W != B else "the decode trace"
r = dict(test=f"m5c TF on {on} with {W} rows at {W} distinct positions per step (row u lagged u steps), "
              f"{len(names)} layers + head, vs {os.path.basename(REF)}, tp{TPN}", status="pass" if ok else "fail",
         milestone="M5b", B=B, width=W, users=W, tp=TPN, nkmax=os.environ.get("GLM53_MLA_ATT_NKMAX"), layers=len(names), reference=REF, **acc, rows_complete=complete,
         rows_agree=rows_identical, worst_row=dict(u=worst["u"], top5=round(worst["top5"] / max(worst["n"], 1), 4)),
         steps=n_steps, wall_s=round(wall, 1), dram_gib_per_chip=backend.trace_report().get("dram_gib_per_chip"),
         tag=tag)  # fmt: skip
log("ROW", r)
if args.get("record"):
    C.record("pcc", r)
RESULT = dict(ok=ok, **acc)
