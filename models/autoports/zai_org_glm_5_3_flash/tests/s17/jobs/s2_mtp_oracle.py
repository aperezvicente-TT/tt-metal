# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; a pair or the mesh — S2): Stage 17 — the MTP layer program on device vs the CPU oracle on the SAME
hidden states (tests/s17/mtp_oracle.py: float64, the tile cache's exact weight values).

A prompt is teacher-forced into slot 0 with the backend's plain decode trace; after each plain step at position p the
speculative BOOTSTRAP replay (tt/spec_mtp.py: the MTP catch-up row 0 = (h_p, E[a_0]) at MTP position p, then K-1
chained rows) drafts d'1..d'K on device. The job reads h_p (the head's bf16 h row 0), a_0, the drafts and the MTP
block's raw outputs (the chain inputs) and runs the oracle row by row on exactly those inputs: draft tokens must be
equal (bitwise ids; the top-1 / top-2 margin of the oracle is recorded per row). Per draft the job also splits the
error: head_exact = the device's draft == the CPU head (shared_head.norm + lm_head, the oracle's own) applied to the
DEVICE's MTP output for that depth (the head is exact on its input); x_rel = |x_dev - x_oracle| / |x_oracle| of the MTP
output; dlogit = the logit deviation that hidden-state difference induces on the two candidates. Criterion (design.md
§8, the near-tie rule): the device head's logits are bf16 (quantum 0.0625-0.125 at |logit| 8-32, ties -> lowest id),
so exact equality is unreachable where the oracle's top-2 gap is within ~2 quanta: pass = every draft that differs
from the float64 oracle is EXPLAINED and the median x_rel over all compared rows is < 0.05. With dbg_ids=1 (the
device's routed experts recorded per MTP row) a mismatch is explained when either the routing is the oracle's and the
oracle's top-2 gap < `tie` (0.25 logits), or the routing differs and the oracle re-run with the DEVICE's experts gives
the device's token with its output deviation back under `x_rel_ok` (0.05; the router gaps of such rows are recorded,
not thresholded); without dbg_ids only the logit near-tie rule applies. Strict equality, head exactness and the per-mismatch split are recorded. The decisive functional check is
jobs/s2_golden_accept.py (the drafter's acceptance on real states == the CPU oracle's). Then the BOOT trace is timed
back-to-back (>= 2,000 replays: idempotent — same positions, same token rows) -> us per drafted token.
ARGS: layers ("0,1,2,3,4"), b (32), k (3), narrow ("none"), n (positions, 24), reps (2000), timing (1; never with
dbg_ids), dbg_ids (0), prompt ("aime24"), tag."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib  # noqa: E402

import s17common as S  # noqa: E402

S = importlib.reload(S)  # a long-lived server caches modules: this job's helpers
from mtp_oracle import MtpOracle  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B, K, N = int(args.get("b", 32)), int(args.get("k", 3)), int(args.get("n", 24))
REPS = int(args.get("reps", 2000))
_L = str(args.get("layers", "0,1,2,3,4"))


def log(*a):
    print(f"S17-S2 {C.now()}", *a, flush=True)


from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

cfg = hf_config()
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L.split(",")]
DBG = int(args.get("dbg_ids", 0))  # diagnosis: the device's routed experts per MTP row (SpecMTP.dbg_ids)
key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=K, narrow=str(args.get("narrow", "none")), profile=str(args.get("profile", "auto")), log=log,
                         extra_env={"GLM53_SPEC_DBG_IDS": "1"} if DBG else None)  # fmt: skip
sp = be.spec
kv = be._kv
io = kv.extra["io"]
MOCK = C.MOCK


def dev0(t):
    return ttnn.get_device_tensors(t)[0] if be.tp > 1 else t


def prompt_tokens(n: int) -> list:
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    tok = load_tokenizer()
    msgs = [{"role": "user", "content": "Find the number of ordered pairs (x, y) of positive integers such that "
                                         "x + 2y = 100 and x is divisible by 3. Show your reasoning step by step."}]
    ids_ = tok.apply_chat_template(msgs, add_generation_prompt=True) if hasattr(tok, "apply_chat_template") else []
    ids_ = list(ids_) if ids_ else list(range(1000, 1000 + n))
    return [int(v) for v in ids_][:n]


toks = prompt_tokens(N)
be.reset_all(kv)
pt = (torch.arange(B * kv.blocks_per_seq) % kv.num_blocks).reshape(B, -1)
be.write(io.page_table, pt)
wts = kv.extra.get("decode_traces", {})
w0 = min([w for w in be.native_widths + be.narrow_widths if w in wts] or [B])
plain = wts.get(w0, kv.extra.get("decode_trace"))
log("plain trace width", w0, "prompt", len(toks), "K", K)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import mk_head as MKH  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource  # noqa: E402

src = FdTileCacheSource(CACHE_DIR, MANIFEST)
orc = None if MOCK else MtpOracle(src, cfg, MKH.lm_head_host(src), MKH.embed_host(src))
rows, mism = [], 0


def route(o: dict, dev: list, rerun, r: torch.Tensor) -> dict:
    """the device's routed experts vs the oracle's; when they differ, the oracle row re-run with the DEVICE's experts
    (x_rel_forced: the MTP output deviation once the routing is the same)"""
    out = dict(dev_experts=dev, experts_equal=dev == o["experts"], router_gap=round(o["router_gap"], 6))
    if dev != o["experts"]:
        of = rerun()
        out.update(x_rel_forced=float((r.to(torch.float64).reshape(-1) - of["x"]).norm() / of["x"].norm()),
                   token_forced=of["token"])  # fmt: skip
    return out


def split(o: dict, r: torch.Tensor, dev_tok: int) -> dict:
    """the error split of one draft: the CPU head on the device's own MTP output (r: fp32 [4096], stream 0) vs the
    device's token, and the hidden-state deviation from the oracle's x"""
    from mtp_oracle import VOCAB

    x_ref, r64 = o["x"], r.to(torch.float64).reshape(-1)
    lg_h = orc._rms(r64, orc.w_sh).to(torch.float32) @ orc.lm[:, :VOCAB]
    tok_h = int(torch.nonzero(lg_h == lg_h.max())[0])
    ref_tok, lg = o["token"], o["logits"]
    cand = [ref_tok, int(dev_tok)]
    return dict(x_rel=float((r64 - x_ref).norm() / x_ref.norm()), head_tok=tok_h, head_exact=tok_h == int(dev_tok),
                dlogit=float((lg_h[cand] - lg[cand]).abs().max()), dev_gap=float(lg_h[int(dev_tok)] - lg_h[ref_tok]))
for p, x in enumerate(toks):
    tk = torch.zeros(B, dtype=torch.int64)
    tk[0] = x
    pos = torch.full((B,), -1, dtype=torch.int64)
    pos[0] = p
    be.write(io.tokens, tk)
    be.write(io.cur_pos, pos)
    be.replay(plain)
    ttnn.synchronize_device(mesh)
    h_p = ttnn.to_torch(dev0(be.head.h)).float().reshape(-1, 4096)[0].clone()
    sp.enter(pt[0])
    ttnn.synchronize_device(mesh)
    rec = sp.read()
    if MOCK:
        continue
    a0, drafts = rec["a"][0], rec["drafts"]
    assert rec["c"] == p, (rec["c"], p)
    r_m = ttnn.to_torch(dev0(sp.R_m)).float().reshape(sp.w, -1, 4096)[0, 0].clone()  # row 0, stream 0
    r_c = [ttnn.to_torch(dev0(t_)).float().reshape(1, -1, 4096)[0, 0].clone() for t_ in sp.R_c]
    dev_ex = ([sorted(int(v) for v in ttnn.to_torch(dev0(t_)).to(torch.int64).reshape(-1, 8)[0])
               for t_ in sp.dbg_ids] if DBG and sp.dbg_ids is not None else None)  # fmt: skip
    o = orc.row(a0, h_p, p)
    if dev_ex is not None:
        o["route"] = route(o, dev_ex[0], lambda: orc.row(a0, h_p, p, force_experts=dev_ex[0]), r_m)
    got = [dict(depth=1, dev=drafts[0], ref=o["token"], margin=o["margin"], experts=o["experts"], **o.get("route", {}),
                **split(o, r_m, drafts[0]))]  # fmt: skip
    chain_in = [(drafts[0], r_m)] + [(drafts[i], r_c[i - 1]) for i in range(1, K - 1)]
    for i, (tok_i, hid) in enumerate(chain_in[: K - 1]):
        oc = orc.row(tok_i, hid, p + 1 + i)
        if dev_ex is not None:
            oc["route"] = route(oc, dev_ex[i + 1],
                                lambda: orc.row(tok_i, hid, p + 1 + i, force_experts=dev_ex[i + 1]), r_c[i])  # fmt: skip
        got.append(dict(depth=i + 2, dev=drafts[i + 1], ref=oc["token"], margin=oc["margin"], **oc.get("route", {}),
                        **split(oc, r_c[i], drafts[i + 1])))  # fmt: skip
    mism += sum(g["dev"] != g["ref"] for g in got)
    rows.append(dict(p=p, x=x, a0=a0, got=got))
    log(f"p {p} a0 {a0} drafts {drafts} ref {[g['ref'] for g in got]} margins {[round(g['margin'], 3) for g in got]} "
        f"head {[g['head_tok'] for g in got]} x_rel {[round(g['x_rel'], 5) for g in got]}")
n_cmp = sum(len(r["got"]) for r in rows)
allg = [g for r in rows for g in r["got"]]
max_dlogit = max((g["dlogit"] for g in allg), default=0.0)
bad = [g for g in allg if g["dev"] != g["ref"]]
TIE = float(args.get("tie", 0.25))
XREL_OK = float(args.get("x_rel_ok", 0.05))
if DBG:  # mechanistic: same routing -> a logit near-tie; different routing -> the oracle given the DEVICE's experts
    # reproduces the device's token and its output deviation falls back to the baseline (< x_rel_ok)
    near = [g for g in bad if (g.get("experts_equal") and g["margin"] < TIE)
            or (g.get("experts_equal") is False and g.get("token_forced") == g["dev"]
                and (g.get("x_rel_forced") or 1.0) < XREL_OK)]
else:
    near = [g for g in bad if g["margin"] < TIE]
heads_exact = all(g["head_exact"] for g in allg)
x_rel_max = max((g["x_rel"] for g in allg), default=0.0)
xs = sorted(g["x_rel"] for g in allg)
x_rel_med = xs[len(xs) // 2] if xs else 0.0
status = "pass" if (MOCK or (n_cmp > 0 and len(near) == len(bad) and x_rel_med < 0.05)) else "fail"
S.record("pcc", dict(test="s2-mtp-draft-vs-oracle", milestone="S2", status=status, K=K, B=B, layers=_L, n_pos=len(toks),
                     compared=n_cmp, mismatches=mism, min_margin=min((g["margin"] for r in rows for g in r["got"]),
                                                                     default=None),
                     mismatch_margins=[g["margin"] for r in rows for g in r["got"] if g["dev"] != g["ref"]],
                     strict_equal=mism == 0, heads_exact=heads_exact, head_exact_count=sum(g["head_exact"] for g in allg),
                     criterion=("explained mismatches (design.md §8): routing equal + oracle top-2 gap < tie, or routing "
                                "differs and the oracle given the device's experts gives the device token with its "
                                "deviation back under x_rel_ok" if DBG else "near-tie rule (design.md §8; routing not "
                                "recorded)"),
                     tie=TIE, x_rel_ok=XREL_OK if DBG else None, mismatches_explained=len(near),
                     max_router_gap_routing_differs=max((g["router_gap"] for g in allg
                                                         if g.get("experts_equal") is False), default=None),
                     max_dlogit=round(max_dlogit, 4), x_rel_max=round(x_rel_max, 5), x_rel_median=round(x_rel_med, 5),
                     mismatch_detail=[{k: g.get(k) for k in ("depth", "dev", "ref", "margin", "head_tok", "dlogit", "dev_gap",
                                                             "x_rel", "experts_equal", "router_gap", "x_rel_forced",
                                                             "token_forced")} for g in bad],
                     routing=None if not DBG else dict(
                         rows=len(allg), experts_differ=sum(1 for g in allg if g.get("experts_equal") is False),
                         x_rel_median_routing_equal=(sorted(g["x_rel"] for g in allg if g.get("experts_equal"))
                                                     or [None])[len([g for g in allg if g.get("experts_equal")]) // 2],
                         differ=[{k: g.get(k) for k in ("depth", "x_rel", "x_rel_forced", "router_gap", "dev", "ref",
                                                         "token_forced")} for g in allg if g.get("experts_equal") is False],
                         x_rel_over_0p05_routing_equal=[{k: g.get(k) for k in ("depth", "x_rel", "router_gap")}
                                                        for g in allg if g.get("experts_equal") and g["x_rel"] > 0.05]),
                     tag=args.get("tag")))  # fmt: skip
# timing: the BOOT trace back-to-back (idempotent), then one blocking read
tr = sp.capture()["boot"]
us = None
TIMING = int(args.get("timing", 1)) and not DBG  # (no timing row from a debug build: its boot trace carries the copies)
if not MOCK and TIMING:
    for _ in range(20):
        ttnn.execute_trace(mesh, tr, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(REPS):
        ttnn.execute_trace(mesh, tr, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t0) / REPS * 1e6
    # the plain step for reference (positions advance by one per replay: rows reset every 400)
    t_plain = []
    for rnd in range(max(1, REPS // 400)):
        pos = torch.full((B,), -1, dtype=torch.int64)
        pos[0] = 100
        be.write(io.cur_pos, pos)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(400):
            ttnn.execute_trace(mesh, plain, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        t_plain.append((time.perf_counter() - t0) / 400 * 1e6)
    S.record("perf", dict(test="s2-mtp-draft-us", milestone="S2", status="pass", K=K, B=B, layers=_L, reps=REPS,
                          boot_us=round(us, 1), us_per_drafted_token=round(us / K, 1),
                          plain_step_us=round(sorted(t_plain)[len(t_plain) // 2], 1), plain_width=w0, tp=be.tp,
                          tag=args.get("tag")))  # fmt: skip
RESULT = dict(status=status, compared=n_cmp, mismatches=mism, boot_us=us)
