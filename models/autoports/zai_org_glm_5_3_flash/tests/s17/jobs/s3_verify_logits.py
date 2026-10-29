# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, a pair): the verify rows' LOGITS vs plain decode's logits at the same positions (design §6 "next
verify row 0 == plain decode"). The plain reference runs the backend's narrowest plain trace (narrow="1":
the width-1 trace a lone user decodes on; "none": the full width), the prompt teacher-forced with the MTP cache kept
(s17common.mtp_prefill). Spec: the same prefill, enter, then per step the verify input rows are forced to plain's
continuation (every draft accepted, n = K) and the step runs as TWO traces: T1 = verify + accept, T2 = the drafter
(catch-up + chained rows + gather) -- T1 then T2 is exactly one step; the verify head's K+1 logits rows are read
between them (the drafter's heads reuse the same logits rows; a verify replayed twice is NOT idempotent: with
sel = 0 row 0 is both the committed source and candidate 0, so the step cannot be re-run for a read). Row j at position
c + j must equal plain's logits at c + j: bitwise (recorded per row with max |diff| and the token). Host reads / writes
between synchronized replays only.
Width invariance across backends (a B=4 backend has no narrow width-1 trace, and a B=32 + narrow-1 backend
with a width-4 verify does not fit L1): ref_out=<path> runs ONLY the plain reference on a backend of width b (b=1:
the t = 1 programs a lone user decodes on; no speculative step) and saves tokens + logits; a later job with
ref_in=<path> compares its verify rows against that file instead of its own plain trace.
ARGS: layers, b, k, narrow, prompt (or prompt_len: the first N tokens of goldens/prompts/2048.pt), steps (8), ref_out,
ref_in, profile, env ({knob: value} for this backend's build), tag."""
import importlib
import json
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
B, K = int(args.get("b", 4)), int(args.get("k", 3))
_L = str(args.get("layers", "0,1,2,3,4"))
NARROW = str(args.get("narrow", "1"))
STEPS = int(args.get("steps", 8))
WS = "/opt/glm53/kit"  # (host-8chip and host-p300)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.spec_mtp import read_rows, write_rows  # noqa: E402

cfg = hf_config()
V = int(cfg.vocab_size)
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L.split(",")]
REF_OUT, REF_IN = args.get("ref_out"), args.get("ref_in")
XENV = {k: str(v) for k, v in (args.get("env") or {}).items()}  # knobs for this backend (e.g. GLM53_MLA_ATT_LANES_T4)
key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=0 if REF_OUT else K, narrow=NARROW,
                         profile=str(args.get("profile", "auto")), extra_env=XENV or None)  # fmt: skip
sp, kv = be.spec, be._kv
io = kv.extra["io"]
pt = (torch.arange(B * kv.blocks_per_seq) % kv.num_blocks).reshape(B, -1)
be.write(io.page_table, pt)
wts = kv.extra.get("decode_traces", {})
w0 = min([w for w in be.native_widths + be.narrow_widths if w in wts] or [B])
plain = wts.get(w0, kv.extra.get("decode_trace"))
PROMPTS = {q["id"]: q for q in json.load(open(os.path.join(WS, "doc/prework/p5_mtp_acceptance/prompts.json")))["prompts"]}
if args.get("prompt_len"):  # long contexts (the MLA lane split differs by width past ~4 tiles):
    # the first prompt_len tokens of the 2,048-token golden prompt
    prompt = [int(v) for v in torch.load(os.path.join(WS, "goldens/prompts/2048.pt"), map_location="cpu",
                                         weights_only=False).reshape(-1)[: int(args["prompt_len"])]]  # fmt: skip
else:
    prompt = [int(v) for v in PROMPTS[str(args.get("prompt", "aime24_04"))]["ids"]]
G = (K + 1) * STEPS + 2

# plain reference: tokens and logits per position
L = len(prompt)
if REF_OUT:  # the plain trace of THIS backend only (no speculative step on it)
    be.reset_all(kv)
    for p, x in enumerate(prompt):
        tk = torch.zeros(B, dtype=torch.int64)
        tk[0] = x
        pos = torch.full((B,), -1, dtype=torch.int64)
        pos[0] = p
        be.write(io.tokens, tk)
        be.write(io.cur_pos, pos)
        be.replay(plain)
    ttnn.synchronize_device(mesh)
    g0 = int(be.read(io.tokens)[0])
elif REF_IN and not C.MOCK:
    R = torch.load(str(REF_IN), weights_only=False)
    assert R["prompt"] == prompt and R["layers"] == _L, "the reference file is for another prompt / prefix"
else:
    g0 = S.mtp_prefill(be, sp, io, prompt, plain, pt[0], B)
if REF_IN and not C.MOCK:
    ref, ref_lg, w_ref, G = R["tokens"], R["logits"], R["width"], min(G, len(R["tokens"]) - 1)
    g0 = ref[0]
else:
    ref, ref_lg, w_ref = [g0], {}, w0
    for i in range(G):
        be.replay(plain)
        ttnn.synchronize_device(mesh)
        ref_lg[L + i] = be.read(io.logits)[0].float()[:V].clone()
        ref.append(int(be.read(io.tokens)[0]))
if REF_OUT:
    if not C.MOCK:
        torch.save(dict(prompt=prompt, layers=_L, width=w0, B=B, tokens=ref, logits=ref_lg, env=XENV,
                        code=os.environ.get("GLM53_CODE_SHA")), str(REF_OUT))  # fmt: skip
    S.record("tests", dict(test="s3-verify-logits-ref", milestone="S3", status="info", B=B, layers=_L, plain_width=w0,
                           tokens=len(ref), file=str(REF_OUT), env=XENV, tag=args.get("tag")))  # fmt: skip
    RESULT = dict(ref=str(REF_OUT), width=w0)

if not REF_OUT:
    # spec: T1 (verify + accept) / T2 (drafter) per step, the verify logits read between them
    S.mtp_prefill(be, sp, io, prompt, plain, pt[0], B)
    sp.enter(pt[0])
    ttnn.synchronize_device(mesh)
    vh = sp.heads["verify"]
    tid, rows, toks = None, [], [g0]
    try:
        for s_ in range(STEPS if not C.MOCK else 1):
            Lc = len(toks)
            tin = read_rows(mesh, sp.tin).reshape(32, -1).clone()
            for j in range(1, K + 1):
                tin[j, 0] = ref[Lc - 1 + j]
            write_rows(mesh, sp.tin, tin)
            ttnn.synchronize_device(mesh)
            if C.MOCK:
                sp._verify()
                sp._ctl_accept(False)
                sp._draft()
                break
            if tid is None:
                tid = []
                for fn in ((lambda: (sp._verify(), sp._ctl_accept(False))), sp._draft):
                    t_ = ttnn.begin_trace_capture(mesh, cq_id=0)
                    try:
                        fn()
                    finally:
                        ttnn.end_trace_capture(mesh, t_, cq_id=0)
                    tid.append(t_)
            c = int(read_rows(mesh, sp.pos_v).reshape(-1)[0])  # (this step's verify rows: positions c .. c + K)
            ttnn.execute_trace(mesh, tid[0], cq_id=0, blocking=True)  # T1: verify + accept
            lg = vh.gather(vh.outs).float()[: K + 1, :V]
            for j in range(K + 1):
                r = ref_lg.get(c + j)
                rows.append(dict(step=s_, j=j, pos=c + j, token=int(torch.argmax(lg[j])),
                                 ref_token=None if r is None else int(torch.argmax(r)),
                                 max_abs=None if r is None else float((lg[j] - r).abs().max()),
                                 bitwise=None if r is None else bool(torch.equal(lg[j], r))))  # fmt: skip
            ttnn.execute_trace(mesh, tid[1], cq_id=0, blocking=True)  # T2: the drafter (completes the step)
            rec = sp.read()
            toks += rec["committed"]
            if rec["n"] != K:  # (plain's continuation not fully accepted: identity broke at this step -> recorded, stop)
                rows.append(dict(step=s_, n=rec["n"], note="forced continuation not fully accepted"))
                break
    finally:
        for t_ in tid or []:
            ttnn.release_trace(mesh, t_)
    cmp_ = [r for r in rows if r.get("bitwise") is not None]
    ok = bool(cmp_) and all(r["token"] == r["ref_token"] for r in cmp_) and not any("n" in r for r in rows)
    S.record("tests", dict(test=f"s3-verify-logits-k{K}", milestone="S3", status="pass" if (ok or C.MOCK) else "fail", K=K,
                           B=B, layers=_L, plain_width=w_ref, verify_width=K + 1, narrow=NARROW, rows=len(cmp_),
                           tokens_equal=sum(r["token"] == r["ref_token"] for r in cmp_),
                           bitwise=sum(bool(r["bitwise"]) for r in cmp_),
                           max_abs=max((r["max_abs"] for r in cmp_), default=None), detail=rows[:64],
                           tokens_identity=toks == ref[: len(toks)], env=XENV,
                           ref_env=(R.get("env") if REF_IN and not C.MOCK else None), tag=args.get("tag")))  # fmt: skip
    sp.leave()
    RESULT = dict(ok=ok, rows=len(cmp_), bitwise=sum(bool(r["bitwise"]) for r in cmp_))
