# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; a pair with a model prefix, or the mesh with all 45 layers — S3): Stage 17 speculative decoding in
the persistent step (tt/spec_mtp.py) vs plain greedy decode of the same backend.

Per prompt (doc/prework/p5_mtp_acceptance/prompts.json, P5's AIME24 set): the prompt is teacher-forced into slot 0 with
the backend's plain decode trace (the narrowest width trace covering slot 0); then
  plain: G greedy tokens with that trace (token / position fed back on device), the logits of every step kept;
  spec:  the same prefill, the bootstrap replay, then speculative-step replays until G tokens are committed; at the
         first R rejections (n < K) the run leaves speculation (KDA row sel -> row 0, positions), runs ONE plain step
         and compares its logits / token with plain's at that position (the ROLLBACK proof: decode after a rejection
         == plain decode from that position), then re-enters (bootstrap) and continues.
Gates: every generated token equal (greedy identity, hard); every rollback step's token equal and its logits bitwise
(max |diff| recorded); acceptance per draft depth and tokens per step (P5: 0.933 / 0.867 / 0.671, 3.17 tokens / verify
at K = 3). Timing (unwatched runs, timing=1): rounds of `reps_round` back-to-back spec-step replays (no host work: the
device feeds tokens / positions; one record read per round) and plain replays from the same prefill -> us / step,
tokens / step, tok/s; >= 2,000 replays per row.
FORCED drafts (forced=1; a model prefix, whose MTP drafts are never accepted, still exercises every n): before each
spec-step replay the host writes the verify input rows 1..K (the draft tokens ACCEPT compares) = plain's continuation
with ONE corruption at depth n_i + 1, n_i cycling 0..K -> the step must accept exactly n_i (`forced_n_ok`), commit
plain's tokens, and the rollbacks happen after partial acceptance (n > 0: the row-sel KDA hand-off). A host write
between synchronized replays (no eager op, no dispatch inside a trace).
DSA (env GLM53_STACK_DSA=1; the DSA-capable verify, mk_mla rev 56): prompts "g4096:N" = the first N tokens of
goldens/prompts/4096.pt with max_seq_len (4096) -> the verify rows run the sparse path past 2,051 tokens; every spec
step's slot-0 index state (each sparse_dsa layer's closed pooled keys + the open pool's committed ring rows) is
compared BITWISE with plain's after the same committed position (`s3-dsa-state-k{K}`; bitwise needs the verify and the
plain decode on the same width: B = K + 1, narrow none).
ARGS: layers ("0,1,2,3,4" | "all"), b (32), k (3), narrow ("1"), prompts ("aime24_00"), gen (64), rollbacks (3),
forced (0), timing (0), reps (2000), reps_round (400), profile ("auto"; "asis" + rev: the integrator's backend as its
env job built it), env ({knob: value}), max_seq_len (2048), rev, dsa_check (1), dsa_tol (0: bitwise), prefill ("decode" | "tp"), tag."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib  # noqa: E402

import s17common as S  # noqa: E402

S = importlib.reload(S)  # a long-lived server caches modules: this job's helpers

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B, K = int(args.get("b", 32)), int(args.get("k", 3))
G, NROLL = int(args.get("gen", 64)), int(args.get("rollbacks", 3))
_L = str(args.get("layers", "0,1,2,3,4"))


def log(*a):
    print(f"S17-S3 {C.now()}", *a, flush=True)


from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

cfg = hf_config()
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L.split(",")]
XENV = {k: str(v) for k, v in (args.get("env") or {}).items()}  # knobs for this backend's build (e.g. the served set)
key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=K, narrow=str(args.get("narrow", "1")), profile=str(args.get("profile", "auto")),
                         log=log, extra_env=XENV or None, max_seq_len=int(args.get("max_seq_len", 2048)), rev=args.get("rev"))  # fmt: skip
sp = be.spec
kv = be._kv
io = kv.extra["io"]
MOCK = C.MOCK
SHARD = kv.extra.get("dsa_shard")  # GLM53_DSA_SHARD (+ GLM53_SPEC_ON_SHARD=1, Stage 14 dsa/spec_on_shard_design.md)
if SHARD is None:
    pt = (torch.arange(B * kv.blocks_per_seq) % kv.num_blocks).reshape(B, -1)
else:  # owner-aware PHYSICAL rows: column j on chip o = (j // bpu) % C, a distinct local block >= 1 per (slot, column)
    _C, _nbl, _bpu = SHARD
    _j = torch.arange(kv.blocks_per_seq)
    _lb = (_j // (_bpu * _C)) * _bpu + _j % _bpu  # the column's local block within its slot (0-based)
    _per = int(_lb.max()) + 1
    pt = torch.stack([(1 + s_ * _per + _lb) * _C + (_j // _bpu) % _C for s_ in range(B)])
    assert int(pt.max()) // _C < _nbl, ("sharded pool: local blocks", int(pt.max()) // _C, _nbl)
be.write(io.page_table, pt)
wts = kv.extra.get("decode_traces", {})
w0 = min([w for w in be.native_widths + be.narrow_widths if w in wts] or [B])
plain = wts.get(w0, kv.extra.get("decode_trace"))


# the main sparse_dsa layers with a DSA state (GLM53_STACK_DSA=1); dsa_check 0: skip the bitwise state compare (it
# needs the plain decode and the verify on the same width: B = K + 1, or GLM53_SPEC_VERIFY_W = the plain width) ->
# dsa_tol > 0: |spec - plain| <= dsa_tol per layer instead (a coarse screen only: it hides sub-tolerance state errors)
DSA = sorted(kv.extra.get("dsa", {})) if int(args.get("dsa_check", 1)) else []
DSA_TOL = float(args.get("dsa_tol", 0))


def dsa_state(P: int) -> dict:
    """slot 0's index state after committed positions 0 .. P - 1, per sparse_dsa layer: the closed pooled keys (rows
    < P / 4) and the open pool's committed ring rows (rows < P % 4); the device copy of chip 0"""
    from models.autoports.zai_org_glm_5_3_flash.tt.mk_mla import DsaIndexer

    out = {}
    for i in DSA:
        st = kv.extra["dsa"][i]
        if SHARD is not None:  # every chip's LOCAL closed pools (n_closed_local) + every chip's (replicated) ring rows
            from models.autoports.zai_org_glm_5_3_flash.tt.dsa_shard import ShardMap

            sm = ShardMap(SHARD[0])
            parts = ttnn.get_device_tensors(st["pools"]), ttnn.get_device_tensors(st["ring"])
            pk = torch.cat([ttnn.to_torch(x).reshape(-1, DsaIndexer.DIM)[: st["tp"]][: sm.n_closed_local(P - 1, r)]
                            for r, x in enumerate(parts[0])])  # fmt: skip
            rg = torch.cat([ttnn.to_torch(x)[0, 0, : P % 4] for x in parts[1]])
            out[i] = (pk.clone(), rg.clone())
            continue
        host = lambda x: ttnn.to_torch(ttnn.get_device_tensors(x)[0] if be.tp > 1 else x)  # noqa: E731
        pk = host(st["pools"]).reshape(-1, DsaIndexer.DIM)[: st["tp"]][: P // 4]
        rg = host(st["ring"])[0, 0, : P % 4]
        out[i] = (pk.clone(), rg.clone())
    return out


def dsa_maxdiff(a: tuple, b: tuple) -> float:
    """max |diff| over a layer's closed pooled keys and committed ring rows"""
    return max([float((x.float() - y.float()).abs().max()) for x, y in zip(a, b) if x.numel()] or [0.0])


def dsa_cmp(a: dict, b: dict) -> list:
    """layers whose state differs (empty = bitwise equal; dsa_tol > 0: max |diff| <= dsa_tol)"""
    if DSA_TOL > 0:
        return [i for i in DSA if dsa_maxdiff(a[i], b[i]) > DSA_TOL]
    return [i for i in DSA if not (torch.equal(a[i][0], b[i][0]) and torch.equal(a[i][1], b[i][1]))]


def dsa_where(a: dict, b: dict) -> dict:
    """per differing layer: the pool rows / ring rows that differ (spec a vs plain b) and the max |diff|"""
    out = {}
    for i in dsa_cmp(a, b):
        (pa, ra), (pb, rb) = a[i], b[i]
        prow = [int(q) for q in torch.nonzero(((pa.float() - pb.float()).abs() > DSA_TOL).any(-1)).reshape(-1)]
        rrow = [int(q) for q in torch.nonzero(((ra.float() - rb.float()).abs() > DSA_TOL).any(-1)).reshape(-1)]
        out[i] = dict(pools=prow[:16], ring=rrow, pmax=float((pa.float() - pb.float()).abs().max()) if prow else 0.0,
                      rmax=float((ra.float() - rb.float()).abs().max()) if rrow else 0.0)  # fmt: skip
    return out


def tok0() -> int:
    return int(be.read(io.tokens)[0])


PREFILL = str(args.get("prefill", "decode"))  # "tp": the served chunked TP prefill (long prompts on a mesh turn)


def prefill(prompt: list) -> int:
    """teacher-force the prompt into slot 0 -> the first generated token. "decode" (default): plain decode steps with the
    MTP cache kept complete (s17common.mtp_prefill: a bootstrap replay after every prompt token); "tp": the SERVED
    chunked TP prefill of slot 0 (backend.prefill_chunk over generator.plan_prefill, traced replays, greedy first token
    on device) -- minutes instead of ~1 h for a 2,200-4,096-token prompt in a watched mesh turn. The MTP layer's own
    latent cache is then not written for the prompt positions (weaker drafts, identity unaffected); the plain reference
    and the spec run take the same prefill, so both start from the same state."""
    if PREFILL != "tp" or C.MOCK:
        return S.mtp_prefill(be, sp, io, prompt, plain, pt[0], B)
    from models.autoports.zai_org_glm_5_3_flash.tt import generator as GEN
    from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams

    assert getattr(be, "tpp", None) is not None, "prefill 'tp': a GLM53_STACK_PREFILL=tp backend"
    be.reset_all(be._kv)
    sp.set_page_table(pt[0])
    seq = torch.tensor(prompt, dtype=torch.int64)
    plans = GEN.plan_prefill(len(prompt), 0, be.prefill_chunk_tokens)
    first = None
    for p_ in plans:
        out = be.prefill_chunk(seq[p_.offset : p_.offset + p_.n_valid], kv_cache=be._kv, slot=0, chunk_start=p_.start,
                               n_valid=p_.n_valid, logits_rows="last", io=io,
                               sample=SamplingParams() if p_ is plans[-1] else None)  # fmt: skip
        if p_ is plans[-1]:
            first = int(out)
    tk = torch.zeros(B, dtype=torch.int64)
    pos = torch.full((B,), -1, dtype=torch.int64)
    tk[0], pos[0] = first, len(prompt)
    be.write(io.tokens, tk)
    be.write(io.cur_pos, pos)
    ttnn.synchronize_device(mesh)
    return first


def plain_logits() -> torch.Tensor:
    return be.read(io.logits)[0].float().clone()


def run_plain(prompt: list, n_gen: int):
    g0 = prefill(prompt)
    toks, lg, ds = [g0], {}, {}
    for i in range(n_gen - 1):
        be.replay(plain)
        ttnn.synchronize_device(mesh)
        lg[len(prompt) + i] = plain_logits()  # the step at position P + i (input toks[i]) -> toks[i + 1]
        toks.append(tok0())
        if DSA:
            ds[len(prompt) + i + 1] = dsa_state(len(prompt) + i + 1)
    return toks, lg, ds


FORCED = int(args.get("forced", 0))
# leave_full 1: also leave (and check the next plain step) after FULLY accepted steps (n = K: the hand-off of KDA row K
# at position c + K + 1) -- the plugin leaves at any n whenever a second request arrives
LEAVE_FULL = int(args.get("leave_full", 0))


def force_drafts(ref: list, L: int, n_i: int):
    """verify input rows 1..K = ref[L - 1 + j] (j = 1..K), the draft at depth n_i + 1 corrupted (ref + 1). Token ids
    are checked in range: the device embedding does not bounds-check its token rows (an out-of-range id reads outside
    the table -> non-finite rows -> state corruption)"""
    from models.autoports.zai_org_glm_5_3_flash.tt.spec_mtp import read_rows, write_rows

    ttnn.synchronize_device(mesh)
    rows = read_rows(mesh, sp.tin).reshape(32, -1).clone()
    for j in range(1, K + 1):
        t = ref[min(L - 1 + j, len(ref) - 1)]
        assert isinstance(t, int) and 0 <= t < int(cfg.vocab_size), ("forced draft not a token id", type(t), t)
        rows[j, 0] = (t + 1) % int(cfg.vocab_size) if j == n_i + 1 else t
    write_rows(mesh, sp.tin, rows)


def run_spec(prompt: list, ref_logits: dict, forced_ref: list | None = None, ref_dsa: dict | None = None):
    g0 = prefill(prompt)
    toks, steps, rolls, dchk = [g0], [], [], []
    sp.enter(pt[0])
    while len(toks) < G and not MOCK:
        if forced_ref is not None:
            n_i = len(steps) % (K + 1)
            force_drafts(forced_ref, len(toks), n_i)
        sp.step()
        ttnn.synchronize_device(mesh)
        rec = sp.read()
        steps.append(dict(n=rec["n"], c=rec["c"], **({"target": n_i} if forced_ref is not None else {})))
        toks += rec["committed"]
        P = rec["c"] + rec["n"] + 1  # committed through c + n (ACCEPT moved the rings)
        if ref_dsa and P in ref_dsa:
            st_ = dsa_state(P)
            dchk.append(dict(P=P, n=rec["n"], c=rec["c"], differ=dsa_cmp(st_, ref_dsa[P]), where=dsa_where(st_, ref_dsa[P]),
                             maxd=max(dsa_maxdiff(st_[i], ref_dsa[P][i]) for i in DSA)))  # fmt: skip
        if (rec["n"] < K or LEAVE_FULL) and len(rolls) < NROLL and len(toks) < G:
            out = sp.leave()  # KDA row sel -> 0, plain pos = c + n + 1 (= the position of the bonus token)
            p_next = out["c"] + out["n"] + 1
            be.replay(plain)
            ttnn.synchronize_device(mesh)
            lg = plain_logits()
            ref_p = ref_logits.get(p_next)  # (not `ref`: that is the forced-draft token list)
            t_ = tok0()
            rolls.append(dict(step=len(steps), n=out["n"], pos=p_next, sel=out["n"], token=t_,
                              ref_token=None if ref_p is None else int(torch.argmax(ref_p)),
                              max_abs=None if ref_p is None else float((lg - ref_p).abs().max()),
                              bitwise=None if ref_p is None else bool(torch.equal(lg, ref_p))))  # fmt: skip
            toks.append(t_)
            sp.enter(pt[0])
    if MOCK:
        sp.step()
        steps.append(sp.read())
    return toks[:G], steps, rolls, dchk


def accept_stats(steps: list) -> dict:
    ns = [s["n"] for s in steps]
    per = []
    for j in range(1, K + 1):
        reach = sum(1 for n in ns if n >= j - 1)
        per.append(round(sum(1 for n in ns if n >= j) / reach, 4) if reach else None)
    return dict(steps=len(ns), tokens_per_step=round(sum(n + 1 for n in ns) / max(1, len(ns)), 4), per_depth=per)


all_ok = True
prompt_ids = S.prompt_ids  # (P5's AIME24 ids, or "g4096:N")

for pid in str(args.get("prompts", "aime24_00")).split(","):
    prompt = prompt_ids(pid)
    t0 = time.time()
    ref_full, ref_lg, ref_ds = run_plain(prompt, G + K + 1 if FORCED else G)  # (forced drafts read K past the last token)
    ref = ref_full[:G]
    t1 = time.time()
    got, steps, rolls, dchk = run_spec(prompt, ref_lg, ref_full if FORCED else None, ref_ds)
    t2 = time.time()
    if MOCK:
        S.record("tests", dict(test="s3-mock", milestone="S3", status="pass", K=K, prompt=pid, tag=args.get("tag")))
        continue
    first_diff = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
    ident = first_diff is None and len(got) == len(ref)
    # at a divergence: the DEVICE plain decode's top-3 / margin at that position (a near-tie of the device's own logits
    # vs a defect of the spec path) + both token lists
    diff_at = None
    if first_diff is not None and first_diff >= 1 and ref_lg.get(len(prompt) + first_diff - 1) is not None:
        tv = torch.topk(ref_lg[len(prompt) + first_diff - 1].float(), 3)
        diff_at = dict(pos=len(prompt) + first_diff - 1, ref_token=ref[first_diff], spec_token=got[first_diff],
                       plain_top3=[int(v) for v in tv.indices], plain_vals=[round(float(v), 4) for v in tv.values],
                       plain_margin=round(float(tv.values[0] - tv.values[1]), 4))  # fmt: skip
        log(pid, "divergence", diff_at)
    st = accept_stats(steps)
    # a rollback compares the post-leave plain step at p_next with the plain run's logits at p_next: valid only while
    # the spec run's tokens equal the reference's up to that step's input (generated index p_next - len(prompt))
    for r in rolls:
        r["comparable"] = first_diff is None or r["pos"] - len(prompt) < first_diff
    crolls = [r for r in rolls if r["comparable"]]
    roll_ok = bool(crolls) and all(r["token"] == r["ref_token"] for r in crolls)
    forced_ok = all(s_["n"] == s_["target"] for s_ in steps) if FORCED else None
    all_ok &= ident and (roll_ok or not crolls) and forced_ok is not False
    ident = ident and forced_ok is not False
    S.record("tests", dict(test=f"s3-identity-k{K}", milestone="S3", status="pass" if ident else "fail", K=K, B=B,
                           layers=_L, prompt=pid, prompt_len=len(prompt), gen=len(got), first_diff=first_diff,
                           plain_width=w0, verify_width=int(sp.wv), tp=be.tp,
                           dsa_layers=len(kv.extra.get("dsa", {})), max_seq_len=int(args.get("max_seq_len", 2048)),
                           forced=bool(FORCED), forced_n_ok=forced_ok, n_hist=[s_["n"] for s_ in steps][:64],
                           ref_head=ref[:16], spec_head=got[:16], diff_at=diff_at,
                           **({"ref_toks": ref, "spec_toks": got} if first_diff is not None else {}),
                           accept=st, wall_plain_s=round(t1 - t0, 1),
                           wall_spec_s=round(t2 - t1, 1), env=XENV, tag=args.get("tag")))  # fmt: skip
    S.record("tests", dict(test=f"s3-rollback-k{K}", milestone="S3",
                           status="pass" if roll_ok else ("fail" if crolls else ("no-comparable-rollback" if rolls
                                                                               else "no-rejection")),
                           K=K, B=B, layers=_L, prompt=pid, rollbacks=rolls, comparable=len(crolls),
                           plain_width=w0, verify_width=int(sp.wv),
                           all_bitwise=all(r["bitwise"] for r in crolls) if crolls else None,
                           env=XENV, tag=args.get("tag")))  # fmt: skip
    if DSA:  # the DSA-capable verify (mk_mla rev 56): the index state after every spec step == plain's, bitwise
        # the state after committed positions < P compares with the plain run's only while the tokens agree up to P - 1
        for d in dchk:
            d["comparable"] = first_diff is None or d["P"] - 1 - len(prompt) < first_diff
        cdchk = [d for d in dchk if d["comparable"]]
        bad = [d for d in cdchk if d["differ"]]
        dsa_ok = bool(cdchk) and not bad
        all_ok &= dsa_ok
        S.record("tests", dict(test=f"s3-dsa-state-k{K}", milestone="S3", status="pass" if dsa_ok else "fail", K=K, B=B,
                               layers=_L, dsa_layers=DSA, prompt=pid, prompt_len=len(prompt), checked=len(cdchk),
                               not_comparable=len(dchk) - len(cdchk),
                               tol=DSA_TOL, max_diff=max(d["maxd"] for d in cdchk) if cdchk else None,
                               differ=bad[:8], n_seen=sorted({d["n"] for d in dchk}),
                               ring_moves=sum(1 for d in cdchk if (d["P"]) // 4 > d["c"] // 4),
                               sparse_rows=all(d["c"] + 1 > 2051 for d in cdchk), env=XENV, tag=args.get("tag")))  # fmt: skip
        log(pid, "dsa state", "ok" if dsa_ok else f"DIFFER {bad[:4]}", "checked", len(cdchk))
    S.record("perf", dict(test=f"s3-accept-k{K}", milestone="S3", status="info", K=K, layers=_L, prompt=pid, **st,
                          p5_greedy_per_depth=[0.933, 0.867, 0.671][:K], p5_tokens_per_verify={1: 1.889, 2: 2.658, 3: 3.174}.get(K),  # (MTP K <= 3; None for the t = 8 verify)
                          tag=args.get("tag")))  # fmt: skip
    log(pid, "identity", ident, "first_diff", first_diff, "accept", st, "rollbacks", rolls)

# ---- timing (unwatched runs): back-to-back replays, rounds re-prefilled (positions stay inside max_seq_len)
if int(args.get("timing", 0)) and not MOCK:
    reps, rr = int(args.get("reps", 2000)), int(args.get("reps_round", 400))
    prompt = prompt_ids(str(args.get("prompts", "aime24_00")).split(",")[0])
    spec_us, plain_us, tps = [], [], []
    tr = sp.capture()["step"]
    for _ in range(max(1, reps // rr)):
        prefill(prompt)
        sp.enter(pt[0])
        ttnn.synchronize_device(mesh)
        c0 = sp.read()["c"]
        t0 = time.perf_counter()
        for _ in range(rr):
            ttnn.execute_trace(mesh, tr, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        spec_us.append((time.perf_counter() - t0) / rr * 1e6)
        rec = sp.read()
        tps.append((rec["c"] + rec["n"] + 1 - (c0 + 1)) / rr)
        sp.leave()
        prefill(prompt)
        t0 = time.perf_counter()
        for _ in range(rr):
            ttnn.execute_trace(mesh, plain, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        plain_us.append((time.perf_counter() - t0) / rr * 1e6)
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    s_us, p_us, tp_ = med(spec_us), med(plain_us), sum(tps) / len(tps)
    S.record("perf", dict(test=f"s3-step-k{K}", milestone="S3", status="pass", K=K, B=B, layers=_L, tp=be.tp,
                          reps=rr * len(spec_us), spec_step_us=round(s_us, 1), plain_step_us=round(p_us, 1),
                          plain_width=w0, tokens_per_step=round(tp_, 3), spec_tok_s=round(tp_ / s_us * 1e6, 1),
                          plain_tok_s=round(1e6 / p_us, 1), speedup=round(tp_ / s_us * p_us, 3),
                          rounds_spec_us=[round(v, 1) for v in spec_us], rounds_tokens_per_step=[round(v, 3) for v in tps],
                          env=XENV, tag=args.get("tag")))  # fmt: skip
RESULT = dict(ok=all_ok)
