# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 14b's pair or mesh server): Stage 14b triage of the served int24svc2 32k+1 failures (program
2026-10-22 05:23Z: prompts of 161 / 65 tokens fail, 31 / 1 pass). On the resident serving backend (m5common.serving,
built decode-only: its prefill = the decode-step path, plain int24's 1.0 / 1.0) each plugin prompt (the exact ids of
9's pfdump) is run three ways, greedy `new` tokens each:
  ref   the decode-step prefill (no TP prefill attached) on the generator-owned identity row;
  tp    Stage 16's TP prefill attached (svc_attach) on the identity row;
  tpv   the TP prefill with the plugin's own page row (vLLM blocks, 0-padded -> vllm_page_rows: -1 past the allocation);
  tp_loop  the TP prefill with GLM53_PF_KDA_SCAN=loop (Stage 1's per-chunk ops, not the recurrent_chunk_scan op).
Row: per length, the first token of each run and the first index where tp / tpv differ from ref. Lengths: the given
prompts plus their first n-1 and n+1 tokens (n+1 appends ref's first generated token).
ARGS: b (32), layers ([0..7]), native (true), rev, prompts (path of {n: {ids, page_table}}), ns ([161, 65]), new (16),
slot (0), knobs, buckets, record (true), tag."""
import datetime as dt
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
S16 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(S16), "s14b"))
sys.path.insert(0, S16)
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402
import svc_attach  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm import vllm_page_rows  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
cfg = hf_config()
_L = args.get("layers", list(range(8)))
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
B, NATIVE, REV = int(args.get("b", 32)), bool(args.get("native", True)), args.get("rev", "p25")
NEW, SLOT = int(args.get("new", 16)), int(args.get("slot", 0))
TP = int(mesh.get_num_devices())
PROMPTS = {int(k): v for k, v in json.load(open(args["prompts"])).items()}
NS = [int(v) for v in args.get("ns", [161, 65])]
greedy = SamplingParams()


def log(*a):
    print(f"S14B-PFLEN {dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}", *a, flush=True)


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if int(x) != int(y):
            return i
    return None


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=NATIVE, rev=REV, log=log)
be, gen = ctx[key]["backend"], ctx[key]["gen"]
assert getattr(be, "tpp", None) is None, "the reference needs a decode-only backend (GLM53_STACK_PREFILL unset)"
cases = []  # (label, ids, plugin page row or None)
for n in NS:
    p = PROMPTS[n]
    cases += [(f"{n - 1}", p["ids"][: n - 1], None), (f"{n}", p["ids"], p["page_table"]), (f"{n + 1}", None, None)]


def run_ref(toks):
    gen.reset()
    return [int(v) for v in gen.generate(toks, NEW + 1, sampling=greedy, stop_on_eos=False)]


def run_caller(toks, row):
    b = gen._own_binding()
    gen.reset()
    pt = torch.full((1, int(b.blocks_per_seq)), 0, dtype=torch.int64)
    pt[0, : len(row)] = torch.tensor(row)
    pt = vllm_page_rows(pt).to(torch.int64)
    tok = torch.tensor([toks], dtype=torch.int64)
    first = gen.prefill_forward(tok, page_table=pt, kv_cache=b.kv_cache, prompt_lens=[len(toks)], empty_slots=[SLOT],
                                sampling_params=[greedy])  # fmt: skip
    seq = [int(first[0])]
    for i in range(NEW):
        o = gen.decode_forward(torch.tensor([-1]), torch.tensor([len(toks) + i]), page_table=pt, kv_cache=b.kv_cache,
                               slots=[SLOT], sampling_params=[greedy])  # fmt: skip
        seq.append(int(torch.as_tensor(o).reshape(-1)[0]))
    gen.reset()
    gen._ensure_identity_rows(b, list(range(B)))
    return seq


# reference pass (decode-step prefill); the n + 1 cases append ref's first generated token of the n case
ref = {}
for i, (lab, toks, row) in enumerate(cases):
    if toks is None:
        base = cases[i - 1]
        toks = base[1] + [ref[base[0]][0]]
        cases[i] = (lab, toks, row)
    ref[lab] = run_ref(toks)
    log("ref", lab, len(toks), ref[lab])
att = svc_attach.Attached(ctx, knobs=args.get("knobs"), buckets=str(args.get("buckets", "128,512,2048")))
pf = None
res = {}
try:
    pf = att.attach()
    for lab, toks, row in cases:
        tp = run_ref(toks)
        out = dict(n=len(toks), ref=ref[lab], tp=tp, first_div_tp=first_diff(tp, ref[lab]),
                   bucket=list(pf.plan(0, len(toks))))  # fmt: skip
        if row:
            tpv = run_caller(toks, [int(v) for v in row if int(v) > 0])
            out.update(tpv=tpv, first_div_tpv=first_diff(tpv, ref[lab]), page_row=[int(v) for v in row if int(v) > 0])
        res[lab] = out
        log("tp", lab, out)
finally:
    pf = None
    log(f"restored: decode-only warm-up {att.restore()} s")
# the same TP prefill with Stage 1's per-chunk KDA op loop instead of the recurrent_chunk_scan op (GLM53_PF_KDA_SCAN)
if args.get("scan_ab", True):
    kn = dict(args.get("knobs") or {})
    kn["GLM53_PF_KDA_SCAN"] = "loop"
    att2 = svc_attach.Attached(ctx, knobs=kn, buckets=str(args.get("buckets", "128,512,2048")))
    try:
        pf = att2.attach()
        for lab, toks, row in cases:
            tl = run_ref(toks)
            res[lab].update(tp_loop=tl, first_div_tp_loop=first_diff(tl, ref[lab]))
            log("tp_loop", lab, res[lab]["first_div_tp_loop"], tl)
    finally:
        pf = None
        log(f"restored: decode-only warm-up {att2.restore()} s")
row_ = dict(test=f"served-prefill triage (32k+1): TP prefill vs the decode-step prefill on the plugin prompts, greedy {NEW} "
                 f"steps, tp{TP} layers {len(ids)}", milestone="M5c-triage",
            status="info", results=res, rev=REV, tag=args.get("tag", f"m5c-pf-len-ab-tp{TP}"))  # fmt: skip
log(json.dumps({k: (v["n"], v["first_div_tp"], v.get("first_div_tpv"), v.get("first_div_tp_loop")) for k, v in res.items()}))
if args.get("record", True) and not C.MOCK:
    C.record("pcc", row_)
RESULT = {k: (v["n"], v["first_div_tp"], v.get("first_div_tpv"), v.get("first_div_tp_loop")) for k, v in res.items()}
