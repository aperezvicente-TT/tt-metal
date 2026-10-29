# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 14b's mesh server; program 2026-10-22 07:19Z / 07:20Z gate row): the TP prefill vs the decode-step
prefill on the 8 API TF prompts (the AIME prompt + its fp32 golden continuation[:k], k = 0..7 -> 161..168 ids) and on
golden2048[:n] (n = 96 / 97 / 130 / 161); per prompt a fresh prefill (Glm53Generator.prefill_forward, identity row,
slot 0, device greedy first token), its rank in the fp32 golden top-32 and the golden top1-top2 margin.
Phase TP: the resident backend with Stage 16's TP prefill attached (attached here if the build did not). Phase DS: the
TP prefill detached (GLM53_STACK_PREFILL unset, decode-only re-warm) -> the decode-step prefill. Gate criterion: a
TP-prefill first token outside the golden top-5 at margin >= 0.5 that the decode-step prefill gets right (rank 1) =
FAIL; misses at margin < 0.5 or inside the top-5 pass. Run it LAST in a process: phase DS leaves the backend decode-only.
ARGS: b (32), layers ("all"), native (true), rev, ns ([96, 97, 130, 161]), ks (8), knobs, buckets, record (true), tag."""
import datetime as dt
import gc
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
S16 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(S16), "s14b"))
sys.path.insert(0, S16)

AIME_LOGITS = "/opt/glm53/kit/goldens_gpu_fp32_aime24/logits/161.pt"
AIME_PROMPT = "/opt/glm53/kit/goldens_gpu_fp32_aime24/teacher_forcing/161.pt"
GOLDENS = os.environ.get("GLM53_GOLDENS", "/opt/glm53/kit/goldens")


def build_cases(ns, ks, tf=()):
    """[(label, ids, golden top-32 ids, margin)] -- host only (torch), testable without a device. tf: [(label, the
    teacher_forcing .pt (prompt_ids, targets), the fp32 logits .pt (continuation_logits), [k ...])]: prompt +
    targets[:k], golden = continuation_logits[k] (program 12:35Z golden-length diagnostic: tf/512 token 5)"""
    a = torch.load(AIME_LOGITS, map_location="cpu", weights_only=False)
    p = torch.load(AIME_PROMPT, map_location="cpu", weights_only=False)
    prompt = [int(v) for v in p["prompt_ids"].reshape(-1).tolist()]
    cont = [int(v) for v in a["continuation_ids"].reshape(-1).tolist()]
    cases = []
    for k in range(int(ks)):
        v, i = torch.topk(a["continuation_logits"][k].float(), 32)
        cases.append((f"aime{161 + k}", prompt + cont[:k], [int(x) for x in i], round(float(v[0] - v[1]), 3)))
    g = torch.load(os.path.join(GOLDENS, "logits", "2048.prefill.pt"), map_location="cpu", weights_only=True)
    ids2048 = [int(v) for v in torch.load(os.path.join(GOLDENS, "prompts", "2048.pt"), map_location="cpu",
                                          weights_only=True).reshape(-1)]  # fmt: skip
    for n in ns:
        vals, idx = g["prompt_top32_values"][n - 1].float(), g["prompt_top32_indices"][n - 1]
        cases.append((f"golden2048[:{n}]", ids2048[:n], [int(x) for x in idx], round(float(vals[0] - vals[1]), 3)))
    for lab, tfp, lgp, kk in tf:
        t = torch.load(tfp, map_location="cpu", weights_only=False)
        lgt = torch.load(lgp, map_location="cpu", weights_only=False)["continuation_logits"]
        pr = [int(v) for v in t["prompt_ids"].reshape(-1).tolist()]
        tg = [int(v) for v in t["targets"].reshape(-1).tolist()]
        for k in kk:
            v, i = torch.topk(lgt[int(k)].float(), 32)
            cases.append((f"{lab}+{k}", pr + tg[: int(k)], [int(x) for x in i], round(float(v[0] - v[1]), 3)))
    return cases


def verdict(rows):
    """FAIL rows: the GOLDEN token's rank in the TP-prefill device logits > 5 at an fp32 margin >= 0.5 where the
    decode-step prefill has it at rank 1 (the 07:20Z rule on the 08:20Z metric of record)"""
    bad = []
    for lab, r in rows.items():
        tp, ds, m = r.get("golden_rank_tp"), r.get("golden_rank_ds"), r["margin"]
        if tp is not None and tp > 5 and m >= 0.5 and ds == 1:
            bad.append(lab)
    return bad


if __name__ == "__main__" and "MESH" not in globals():  # host self-test: python s14b_prefill_rank_ab.py
    cs = build_cases([96, 97, 130, 161], 8)
    for c in cs:
        print(c[0], len(c[1]), c[2][:3], c[3])
    print(verdict({"x": dict(golden_rank_tp=10, golden_rank_ds=1, margin=3.8), "y": dict(golden_rank_tp=2, golden_rank_ds=1, margin=3.3),
                   "z": dict(golden_rank_tp=40, golden_rank_ds=4, margin=1.7), "w": dict(golden_rank_tp=12, golden_rank_ds=1, margin=0.09)}))
    sys.exit(0)

import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402
import svc_attach  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
B, REV = int(args.get("b", 32)), args.get("rev", "t26")
TP = int(mesh.get_num_devices())
greedy = SamplingParams()


def log(*a):
    print(f"S14B-RANKAB {dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}", *a, flush=True)


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=bool(args.get("native", True)), rev=REV, log=log)
att = svc_attach.Attached(ctx, knobs=args.get("knobs"), buckets=str(args.get("buckets", "128,512,2048")))
be, gen = att.be, att.gen
if getattr(be, "tpp", None) is None:
    att.attach()
cases = build_cases([int(v) for v in args.get("ns", [96, 97, 130, 161])], int(args.get("ks", 8)),
                    [tuple(v) for v in args.get("tf", [])])  # fmt: skip


def first_tokens(phase):
    """per prompt: the device argmax of the last-position logits, its rank in the golden top-32 (the 07:20Z metric) and
    the GOLDEN token's rank in the device logits (the gate of record, program 08:20Z)"""
    b = gen._own_binding()
    out = {}
    for lab, toks, gold, m in cases:
        gen.reset()
        lg = gen.prefill_forward(torch.tensor([toks], dtype=torch.int64), page_table=b.identity_row(0).reshape(1, -1),
                                 kv_cache=b.kv_cache, prompt_lens=[len(toks)], empty_slots=[0])  # fmt: skip
        lg = torch.as_tensor(lg).reshape(-1).float()
        t = int(torch.argmax(lg))
        g_rank = int((lg > lg[gold[0]]).sum()) + 1  # the golden token's rank in the device distribution
        out[lab] = (t, gold.index(t) + 1 if t in gold else None, g_rank, round(float(lg[gold[0]] - lg[gold[1]]), 4))
        log(phase, lab, len(toks), "argmax", t, "rank-in-golden", out[lab][1], "golden", gold[0], "GOLDEN-RANK-IN-DEVICE",
            g_rank, "margin", m, "device gap golden1-golden2", out[lab][3])
    gen.reset()
    return out


rows = {lab: dict(n=len(toks), golden=gold[0], margin=m) for lab, toks, gold, m in cases}
for lab, (t, r, gr, gap) in first_tokens("TP").items():
    rows[lab].update(first_tp=t, rank_tp=r, golden_rank_tp=gr, device_gap_tp=gap)
# phase DS: detach the TP prefill -> the decode-step prefill (the decode trace per prompt token)
att._rewarm_release()
os.environ.pop("GLM53_STACK_PREFILL", None)
att.binding.prepared_prefill = att.binding.prepared_decode = False
gc.collect()
gen.warmup()
assert getattr(be, "tpp", None) is None, "the TP prefill is still attached"
for lab, (t, r, gr, gap) in first_tokens("DS").items():
    rows[lab].update(first_ds=t, rank_ds=r, golden_rank_ds=gr, device_gap_ds=gap)
bad = verdict(rows)
row = dict(test=f"TP prefill vs decode-step prefill: first-token rank in the fp32 golden top-32 + top1-top2 margin, the 8 API "
                f"TF prompts (AIME 161..168) + golden2048[:96/97/130/161], tp{TP} layers {len(ids)}", milestone="M5c",
           status="fail" if bad else "pass", criterion="the golden token's rank in the TP-prefill device logits > 5 at "
           "fp32 margin >= 0.5 where the decode-step prefill has it at rank 1 = FAIL (07:20Z rule, 08:20Z metric)", fails=bad, results=rows, rev=REV,
           knobs=att.knobs, tag=args.get("tag", f"m5c-prefill-rank-ab-tp{TP}"))  # fmt: skip
log("VERDICT", "FAIL" if bad else "PASS", bad)
if args.get("record", True) and not C.MOCK:
    C.record("pcc", row)
RESULT = dict(verdict="fail" if bad else "pass", fails=bad, rows=rows)
