# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job ZERO of 14b's 06:07Z turn (program 2026-10-22 05:52Z): Glm53Generator.prefill_forward on identity rows,
slot 0, a fresh prefill per length, first token (device greedy) and its RANK in the fp32 golden top-32:
  golden_ids(2048)[:n] for n in ns (96, 97, 130, 161): golden = goldens/logits/2048.prefill.pt prompt_top32_indices[n-1];
  the AIME prompt (refpt, 161 ids) + its golden generated[:1] / [:3] (162 / 164 ids): golden = refpt topk_tokens[1] / [3].
Needs the TP prefill attached (served env GLM53_STACK_PREFILL=tp) -- asserted. No soak.
ARGS: b (32), layers ("all"), native (true), rev, ns, refpt, record (true), tag."""
import datetime as dt
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "s14b"))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
GOLDENS = os.environ.get("GLM53_GOLDENS", "/opt/glm53/kit/goldens")
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
B, REV = int(args.get("b", 32)), args.get("rev", "t25b")
NS = [int(v) for v in args.get("ns", [96, 97, 130, 161])]
TP = int(mesh.get_num_devices())


def log(*a):
    print(f"S14B-JOB0 {dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}", *a, flush=True)


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=bool(args.get("native", True)), rev=REV, log=log)
be, gen = ctx[key]["backend"], ctx[key]["gen"]
assert getattr(be, "tpp", None) is not None, "job zero needs the TP prefill attached (GLM53_STACK_PREFILL=tp at build)"
b = gen._own_binding()
greedy = SamplingParams()
g2048 = [int(v) for v in torch.load(os.path.join(GOLDENS, "prompts", "2048.pt"), map_location="cpu",
                                    weights_only=True).reshape(-1)]  # fmt: skip
top32 = torch.load(os.path.join(GOLDENS, "logits", "2048.prefill.pt"), map_location="cpu",
                   weights_only=True)["prompt_top32_indices"]  # fmt: skip
ref = torch.load(args.get("refpt", "/opt/glm53/kit/goldens_gpu_fp32_aime24/"
                                   "readiness_aime24_chat.refpt"), map_location="cpu", weights_only=False)["entries"][0]  # fmt: skip
aime = [int(v) for v in ref["prompt_tokens"].reshape(-1).tolist()]
gen_t = [int(v) for v in ref["generated_tokens"].reshape(-1).tolist()]
cases = [(f"golden2048[:{n}]", g2048[:n], [int(v) for v in top32[n - 1].tolist()]) for n in NS]
cases += [("aime161", aime, [int(v) for v in ref["topk_tokens"][0].tolist()][:32]),
          ("aime162", aime + gen_t[:1], [int(v) for v in ref["topk_tokens"][1].tolist()][:32]),
          ("aime164", aime + gen_t[:3], [int(v) for v in ref["topk_tokens"][3].tolist()][:32])]
out = {}
for lab, toks, gold in cases:
    gen.reset()
    first = gen.prefill_forward(torch.tensor([toks], dtype=torch.int64), page_table=b.identity_row(0).reshape(1, -1),
                                kv_cache=b.kv_cache, prompt_lens=[len(toks)], empty_slots=[0],
                                sampling_params=[greedy])  # fmt: skip
    t = int(first[0])
    rank = gold.index(t) + 1 if t in gold else None
    out[lab] = dict(n=len(toks), first=t, golden_top1=gold[0], rank_in_top32=rank, bucket=list(be.tpp.plan(0, len(toks))))
    log(lab, out[lab])
gen.reset()
bad = [k for k, v in out.items() if v["rank_in_top32"] is None or v["rank_in_top32"] > 5]
row = dict(test=f"job zero: TP prefill first token rank vs the fp32 golden top-32, Glm53Generator.prefill_forward on identity "
                f"rows, slot 0, fresh prefill per length, tp{TP} layers {len(ids)}", milestone="M5c-triage",
           status="pass" if not bad else "info", outside_top5=bad, results=out, rev=REV,
           tag=args.get("tag", f"m5c-job0-rank-tp{TP}"))  # fmt: skip
log("JOB0 outside top-5:", bad)
if args.get("record", True) and not C.MOCK:
    C.record("pcc", row)
RESULT = dict(outside_top5=bad, ranks={k: v["rank_in_top32"] for k, v in out.items()})
