# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 9's STRICT API rank row through the TP prefill (operator 2026-10-22 08:38Z): the
AIME24 readiness prompt (161 ids) + its first t golden continuation tokens, t = 0 .. 31 (161 .. 192 ids: the 512 bucket,
sequence-parallel at TP8), each a FRESH prefill through StackBackend.prefill_chunk (one traced replay, last-row
logits) -> the golden token's rank in the device logits (golden = the GPU fp32 continuation logits' argmax = the
refpt's greedy token t) and the golden margin (top1 - top2). Summary per variant: top-1 / top-5 rates over the
positions with margin >= `min_margin` (0.5): bar top-1 >= 0.85, top-5 >= 0.98. Variants (env overlays, e.g.
{"GLM53_PF_REP_BUCKETS": "128,512"}) rebuild the TPPrefill on the same backend and repeat.

construction: "m5" (14b's served construction, M5.serving; the job builds it unless an earlier job did) | "s16"
(an earlier p2_full / p2_tf_ab build of these layers). ARGS: layers ("all"), ts (0..31), variants ([{}]),
construction ("m5"), b (32), native (true), rev ("t25b"), env, min_margin (0.5), tag, record (True)."""
import os
import sys
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

C = S.C
AIME = "/opt/glm53/kit/goldens_gpu_fp32_aime24"
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
cfg = hf_config()
LAYERS = list(range(cfg.num_hidden_layers)) if args.get("layers", "all") == "all" else [int(v) for v in args["layers"]]
TS = [int(v) for v in args.get("ts", list(range(32)))]
VARIANTS = list(args.get("variants", [{}]))
CONS = args.get("construction", "m5")
MINM = float(args.get("min_margin", 0.5))
tag = args.get("tag", "p2-strict-rank")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
VOCAB = int(cfg.vocab_size)


def set_env(variant):
    for k, v in (args.get("env") or {}).items():
        os.environ[str(k)] = str(v)
    os.environ["GLM53_PF_PROBE_LAYERS"] = ""
    for k, v in variant.items():
        os.environ[str(k)] = str(v)


set_env(VARIANTS[0])
if CONS == "m5":
    keys = [k for k in ctx if isinstance(k, tuple) and k and k[0] == "m5_serving" and list(k[2]) == LAYERS]
    if not keys:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "s14b"))
        import m5common as M5  # noqa: E402

        keys = [M5.serving(ctx, mesh, SM, G, B=int(args.get("b", 32)), ids=LAYERS, cfg=cfg,
                           native=bool(args.get("native", True)), rev=args.get("rev", "t25b"), log=S.log)]  # fmt: skip
else:
    keys = [k for k in ctx if isinstance(k, tuple) and k and k[0] == "s16-full" and list(k[2]) == LAYERS]
    assert keys, "construction s16 needs an earlier p2_full / p2_tf_ab build of these layers"
backend, gen = ctx[keys[-1]]["backend"], ctx[keys[-1]]["gen"]
assert getattr(backend, "tpp", None) is not None, "needs the TP prefill attached (GLM53_STACK_PREFILL=tp)"

aref = torch.load(f"{AIME}/readiness_aime24_chat.refpt", map_location="cpu", weights_only=False)["entries"][0]
aime = aref["prompt_tokens"].reshape(-1).long()
agen = aref["generated_tokens"].reshape(-1).long()
alog = torch.load(f"{AIME}/logits/161.pt", map_location="cpu", weights_only=True, mmap=True)["continuation_logits"]

rows = []
for vi, variant in enumerate(VARIANTS):
    if vi:  # this variant's TPPrefill on the same backend / KV cache (the first one: as built / as left)
        set_env(variant)
        t1 = time.time()
        b_ = gen._own_binding()
        backend.tpp.release()
        backend.tpp = None
        backend.prepare_prefill(b_.kv_cache)
        backend.tpp.capture()
        gen.reset()
        ttnn.synchronize_device(mesh)
        S.log(f"TPPrefill rebuilt for {variant} in {time.time() - t1:.1f} s")
    pf = backend.tpp
    b_ = gen._own_binding()
    kv, io = b_.kv_cache, b_.io
    eager0 = pf.counters["eager"]
    per = []
    for t in TS:
        toks = torch.cat([aime, agen[:t]])
        n = int(toks.numel())
        gen.reset()
        lg = backend.prefill_chunk(toks, kv_cache=kv, slot=0, chunk_start=0, n_valid=n, logits_rows="last",
                                   io=io).float().reshape(-1)[:VOCAB]  # fmt: skip
        gl = alog[t].float()
        g1 = int(gl.argmax())
        v2 = gl.topk(2).values
        rank = int((lg > lg[g1]).sum()) + 1
        per.append(dict(t=t, n=n, bucket=list(pf.plan(0, n)), mode=pf.mode(pf.plan(0, n)[0]), golden=g1,
                        golden_ref=int(agen[t]), margin=round(float(v2[0] - v2[1]), 4), rank=rank,
                        dev_top1=int(lg.argmax())))  # fmt: skip
    gen.reset()
    ok_ = [p for p in per if p["margin"] >= MINM]
    top1 = sum(p["rank"] == 1 for p in ok_) / max(1, len(ok_))
    top5 = sum(p["rank"] <= 5 for p in ok_) / max(1, len(ok_))
    row = dict(test=f"strict API rank row through the TP prefill (Stage 9's metric): AIME + golden[:t], t = {TS[0]} .. "
                    f"{TS[-1]}, golden token's rank in the device logits, positions with margin >= {MINM}, {CONS} "
                    f"construction, tp{TPN} layers {len(LAYERS)}", milestone="P2",
               status=("pass" if top1 >= 0.85 and top5 >= 0.98 else "fail") if len(LAYERS) == cfg.num_hidden_layers
               else "info", variant=variant, construction=CONS, traced=pf.counters["eager"] == eager0, tp=TPN,
               layers=len(LAYERS), top1=round(top1, 4), top5=round(top5, 4), n_scored=len(ok_), n=len(per),
               misses=[p for p in per if p["rank"] > 1 and p["margin"] >= MINM], per=per, tag=tag)  # fmt: skip
    S.log({k: v for k, v in row.items() if k != "per"})
    rows.append(row)
    if rec:
        C.record("pcc", row)
RESULT = rows
