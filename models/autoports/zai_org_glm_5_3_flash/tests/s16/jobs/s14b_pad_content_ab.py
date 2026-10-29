# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 14b's mesh server): the TP8 padded-bucket defect (14b 06:07Z job zero: the TP prefill's first token is
outside the fp32 golden top-5 on PADDED buckets only). Does the PADDING CONTENT reach the valid rows? Per prompt, a fresh
prefill (Glm53Generator.prefill_forward, identity rows, slot 0, last-position logits) with the padding rows' token ids
= 0 (served), the valid ids repeated, random (2 seeds) -- GLM53_PF_PAD_TOKENS. Valid rows must not depend on the padding
(causal attention, KDA identity steps); a change = an op mixing rows across the bucket at this TP (e.g. a per-expert
capacity of the MoE routing: identical padding rows all route to the same experts). Row: per prompt and pad mode the
argmax, its golden rank (when a golden is given) and max |logit - logit(pad 0)|.
ARGS: b (32), layers ("all"), native (true), rev, ns ([97, 130]: golden2048[:n]), aime (true: the refpt's 161 ids),
modes (["zero", "repeat", "random:1", "random:2"]), refpt, record (true), tag."""
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

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
GOLDENS = os.environ.get("GLM53_GOLDENS", "/opt/glm53/kit/goldens")
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
FULL = len(ids) == cfg.num_hidden_layers
B, REV = int(args.get("b", 32)), args.get("rev", "t25")
NS = [int(v) for v in args.get("ns", [97, 130])]
MODES = list(args.get("modes", ["zero", "repeat", "random:1", "random:2"]))
TP = int(mesh.get_num_devices())


def log(*a):
    print(f"S14B-PADAB {dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}", *a, flush=True)


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=bool(args.get("native", True)), rev=REV, log=log)
be, gen = ctx[key]["backend"], ctx[key]["gen"]
assert getattr(be, "tpp", None) is not None, "needs the TP prefill attached (GLM53_STACK_PREFILL=tp at build)"
b = gen._own_binding()
g2048 = [int(v) for v in torch.load(os.path.join(GOLDENS, "prompts", "2048.pt"), map_location="cpu",
                                    weights_only=True).reshape(-1)]  # fmt: skip
top32 = torch.load(os.path.join(GOLDENS, "logits", "2048.prefill.pt"), map_location="cpu",
                   weights_only=True)["prompt_top32_indices"] if FULL else None  # fmt: skip
cases = [(f"golden2048[:{n}]", g2048[:n], [int(v) for v in top32[n - 1].tolist()] if FULL else None) for n in NS]
if args.get("aime", True):
    ref = torch.load(args.get("refpt", "/opt/glm53/kit/goldens_gpu_fp32_aime24/"
                                       "readiness_aime24_chat.refpt"), map_location="cpu", weights_only=False)["entries"][0]  # fmt: skip
    cases.append(("aime161", [int(v) for v in ref["prompt_tokens"].reshape(-1).tolist()],
                  [int(v) for v in ref["topk_tokens"][0].tolist()][:32] if FULL else None))
saved = os.environ.get("GLM53_PF_PAD_TOKENS")
out = {}
try:
    for lab, toks, gold in cases:
        z = None
        out[lab] = dict(n=len(toks), bucket=list(be.tpp.plan(0, len(toks))))
        for m in MODES:
            os.environ["GLM53_PF_PAD_TOKENS"] = m
            gen.reset()
            lg = gen.prefill_forward(torch.tensor([toks], dtype=torch.int64), page_table=b.identity_row(0).reshape(1, -1),
                                     kv_cache=b.kv_cache, prompt_lens=[len(toks)], empty_slots=[0])  # fmt: skip
            lg = torch.as_tensor(lg).reshape(-1).float()
            z = lg if z is None else z
            am = int(torch.argmax(lg))
            out[lab][m] = dict(argmax=am, rank=(gold.index(am) + 1 if gold and am in gold else None) if gold else "n/a",
                               max_abs_vs_zero=round(float((lg - z).abs().max()), 4))  # fmt: skip
            log(lab, m, out[lab][m])
finally:
    if saved is None:
        os.environ.pop("GLM53_PF_PAD_TOKENS", None)
    else:
        os.environ["GLM53_PF_PAD_TOKENS"] = saved
    gen.reset()
leak = {lab: max(v[m]["max_abs_vs_zero"] for m in MODES) for lab, v in out.items()}
row = dict(test=f"TP prefill padded buckets: valid-row logits vs the padding CONTENT (pad ids 0 / repeated / random), "
                f"tp{TP} layers {len(ids)}", milestone="M5c-triage", status="info", max_abs_vs_pad0=leak, results=out,
           rev=REV, tag=args.get("tag", f"m5c-pad-content-ab-tp{TP}"))  # fmt: skip
log("LEAK max|dlogit| vs pad 0:", leak)
if args.get("record", True) and not C.MOCK:
    C.record("pcc", row)
RESULT = dict(leak=leak)
