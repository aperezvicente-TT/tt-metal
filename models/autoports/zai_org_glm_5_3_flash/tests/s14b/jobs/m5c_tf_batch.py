# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, chip 3 or the mesh): Stage 14b — teacher forcing with B DISTINCT users decoding together (the Stage 6
harness metric, the plugin's TokenAccuracy rule: the device's predicted token must be in the reference's top-k at that
position), on tt/generator.py's generate_batch over tt/stack_model.py's StackBackend.

The users are distinct sequences cut from one reference (readiness_v1 .refpt: prompt P, its greedy continuation G, the
top-100 at every continuation position): user u's prompt = P + G[:u] (161 + u tokens), it decodes 100 - u steps
teacher-forced with G[u:], and its step s is scored against the reference top-k of position u + s. Every user is at a
different position (distinct KDA states, latent rows, embeddings) in the same decode steps — B rows of real work per
step, not one scored row in a wide batch. ARGS: b (32), dense ([0, 1, 2]), sparse ([]), dsa ([]), ref, tag, record,
rev."""
import gc
import importlib
import os
import sys
import time

import torch

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
B = int(args.get("b", 32))
DENSE = [int(v) for v in args.get("dense", [0, 1, 2])]
SPARSE = [int(v) for v in args.get("sparse", [])]
DSA = [int(v) for v in args.get("dsa", [])]
REF = args.get("ref", "/opt/glm53/work-s14b/refpt/aime24_prefix_L0-2.refpt")
tag = args.get("tag", f"m5c-tf-batch-B{B}")
TPN = int(mesh.get_num_devices())
cfg = hf_config()
if args.get("layers") is not None:  # the model's own layer ids ("all" = the 45-layer model) -> the three kinds
    ids = list(range(cfg.num_hidden_layers)) if args["layers"] == "all" else [int(v) for v in args["layers"]]
    DENSE = [L for L in ids if cfg.layer_kind(L) == "kda_dense"]
    SPARSE = [L for L in ids if cfg.layer_kind(L) == "kda_moe"]
    DSA = [L for L in ids if cfg.layer_kind(L) == "mla_moe"]


def log(*a):
    print(f"M5C-TFB {C.now()}", *a, flush=True)


ref = torch.load(REF, weights_only=True)["entries"][0]
P = ref["prompt_tokens"][0].tolist()
Gt = ref["generated_tokens"][0].tolist()
topk = ref["topk_tokens"]  # [100, 100]
NG = len(Gt)
assert B <= NG // 2, (B, NG)
key = ("m5c_tfb", bool(args.get("native", False)), B, tuple(DENSE), tuple(SPARSE), tuple(DSA), TPN, args.get("rev"))
M5.prune_ctx(ctx, [key])
if key not in ctx:
    M5.set_env({"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3", "GLM53_MK_RING_KB_QKV": "700"})
    if TPN == 1:
        M5.keep_decoders(ctx, DENSE + SPARSE + DSA)
        env_s = {**M5.SPARSE_KNOBS, "GLM53_MKX_RING": "3", "GLM53_MKX_INFLIGHT": "2"}
        seq = M5.build_layers(mesh, cfg, ctx, DENSE, SPARSE, M5.DENSE_KNOBS, env_s, dsa=DSA)
        backend = SM.build_model_backend(
            None, mesh, max_batch=B, seq=seq, hf_config=cfg, native_t=bool(args.get("native", False))
        )
    else:
        M5.keep_decoders(ctx, [])
        backend = SM.build_model_backend(
            None,
            mesh,
            max_batch=B,
            layers=sorted(DENSE + SPARSE + DSA),
            hf_config=cfg,
            native_t=bool(args.get("native", False)),
            weight_cache=ctx.setdefault("s14b_weights", {}) if args.get("shared_weights") else None,
        )
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=512,
                            stop_token_ids=())  # fmt: skip
    gen.warmup()
    ctx[key] = dict(backend=backend, gen=gen)
backend, gen = ctx[key]["backend"], ctx[key]["gen"]
prompts = [P + Gt[:u] for u in range(B)]
steps = [NG - u for u in range(B)]
preds = [[] for _ in range(B)]


def next_input(user, step, predicted):
    preds[user].append(int(predicted))
    return Gt[user + step]  # teacher forcing: the reference's own continuation


c0 = dict(gen.counters)
gen.reset()
t0 = time.time()
gen.generate_batch(prompts, steps, next_input=next_input, enable_trace=True, stop_on_eos=False)
wall = time.time() - t0
d = {k: gen.counters.get(k, 0) - c0.get(k, 0) for k in gen.counters}
per_user, m1, m5, m100, tot = [], 0, 0, 0, 0
for u in range(B):
    hits = [(p == int(topk[u + s, 0]), p in topk[u + s, :5].tolist(), p in topk[u + s, :100].tolist())
            for s, p in enumerate(preds[u][: steps[u]])]  # fmt: skip
    a1, a5, a100 = (sum(h[i] for h in hits) for i in range(3))
    per_user.append(dict(u=u, n=len(hits), top1=a1, top5=a5, top100=a100))
    m1, m5, m100, tot = m1 + a1, m5 + a5, m100 + a100, tot + len(hits)
complete = all(p["n"] == steps[p["u"]] for p in per_user)
traced = d.get("eager_decode_steps", 0) == 0 and d.get("decode_trace_replays", 0) > 0
acc = dict(top1=round(m1 / tot, 4), top5=round(m5 / tot, 4), top100=round(m100 / tot, 4), total=tot)
worst = min(per_user, key=lambda p: p["top5"] / max(p["n"], 1))
names = [f"L{e['L']}" for e in backend.seq]
ok = complete and traced and acc["top5"] >= 0.98 and acc["top100"] >= 1.0
r = dict(test=f"m5c TF with {B} DISTINCT users decoding together (generate_batch, users at positions 160..{160 + B - 1}), "
              f"prefix {names} vs the same prefix's CPU fp32 reference, tp{TPN}", status="pass" if ok else "fail",
         milestone="M5c", B=B, tp=TPN, prefix=names, reference=REF, **acc, users_complete=complete,
         worst_user=dict(u=worst["u"], top5=round(worst["top5"] / max(worst["n"], 1), 4), n=worst["n"]),
         decode_traced_no_eager=traced, counters_delta=d, dram=M5.dram_state(mesh), tag=tag,
         note="prefix model: scores the device numerics + the batched serving path, not the model's accuracy")  # fmt: skip
prow = dict(test=f"m5c TF batch B={B} tp{TPN} prefix {len(names)} layers", status="info", milestone="M5c", traced=True,
            B=B, tp=TPN, wall_s=round(wall, 1), decode_replays=d.get("decode_trace_replays"), tag=tag,
            note=f"wall of generate_batch incl. prefill-as-decode of {B} prompts (161..{160 + B} tokens)")  # fmt: skip
log("ROW", r)
log("PERF", prow)
if args.get("record"):
    C.record("pcc", r)
    C.record("perf", prow) if not os.environ.get("TT_METAL_WATCHER") else None
RESULT = dict(ok=ok, **acc)
