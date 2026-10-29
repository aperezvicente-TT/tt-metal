# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; a pair with a model prefix, or the mesh): Stage 17 GLM53_SPEC_ASYNC as the device sees it.

The adapter (tt/generator_vllm.py) enqueues the next speculative step right after reading a lone step's record, so the
device runs it during the runner's host turn; a call that deviates from that step's block leaves speculation and the
row sits out the next plain replays (its tokens answered from the held record) until the runner reaches the device's
position. Per prompt (P5's AIME24 ids), with a simulated host turn of H ms between steps (time.sleep):
  identity   the committed stream with the record read BEFORE the next step is enqueued (sync) == the stream with the
             next step enqueued right after the record (async), for every H in host_ms
  idle gap   after the spec steps, leave; slot 0 sits out `idle` plain replays (position -1), then resumes at its own
             position with its next input rewritten -> its next `gen2` plain tokens == the same continuation without
             the idle replays (the held path's device part)
  timing     per-step wall (record to record) in both orders: sync ~ the device step + H, async ~ max(the device step,
             H) + the record read
ARGS: layers ("0,1,2,3,4"), b (32), k (3), prompts ("aime24_00"), gen (48), gen2 (16), idle (3), host_ms ("0,5,12.5"),
profile ("auto"), env ({}), narrow ("1"), max_seq_len (2048), rev, tag."""
import os
import statistics
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
G, G2, IDLE = int(args.get("gen", 48)), int(args.get("gen2", 16)), int(args.get("idle", 3))
HOST_MS = [float(v) for v in str(args.get("host_ms", "0,5,12.5")).split(",") if v != ""]
_L = str(args.get("layers", "0,1,2,3,4"))


def log(*a):
    print(f"S17-ASYNC {C.now()}", *a, flush=True)


from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

cfg = hf_config()
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L.split(",")]
XENV = {k: str(v) for k, v in (args.get("env") or {}).items()}
key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=K, narrow=str(args.get("narrow", "1")),
                         profile=str(args.get("profile", "auto")), log=log, extra_env=XENV or None,
                         max_seq_len=int(args.get("max_seq_len", 2048)), rev=args.get("rev"))  # fmt: skip
sp = be.spec
kv = be._kv
io = kv.extra["io"]
assert kv.extra.get("dsa_shard") is None, "s17_async_overlap: the replicated pool (the shard's page rows: s3_identity)"
pt = (torch.arange(B * kv.blocks_per_seq) % kv.num_blocks).reshape(B, -1)
be.write(io.page_table, pt)
wts = kv.extra.get("decode_traces", {})
w0 = min([w for w in be.native_widths + be.narrow_widths if w in wts] or [B])
plain = wts.get(w0, kv.extra.get("decode_trace"))


def tok0() -> int:
    return int(be.read(io.tokens)[0])


def prefill(prompt: list) -> int:
    return S.mtp_prefill(be, sp, io, prompt, plain, pt[0], B)


def run_spec(prompt: list, order: str, host_s: float):
    """G committed tokens by spec steps in `order` ("sync" | "async") with a host turn of host_s between steps ->
    (tokens, per-step wall seconds record to record, the last record)"""
    toks = [prefill(prompt)]
    sp.enter(pt[0])
    ttnn.synchronize_device(mesh)
    walls, rec, t_prev = [], None, None
    if order == "async":
        sp.step()  # (the adapter enqueues the first step at the bootstrap's verify)
    while len(toks) < G:
        if order == "sync":
            sp.step()
        rec = sp.read()  # blocks until the step's record lands
        if order == "async" and len(toks) + rec["n"] + 1 < G:
            sp.step()  # the next step's inputs (bonus, drafts) are resident: enqueue it before the host turn
        t = time.perf_counter()
        if t_prev is not None:
            walls.append(t - t_prev)
        t_prev = t
        toks += rec["committed"]
        if host_s > 0:
            time.sleep(host_s)  # the runner's turn: accept walk, output processing, schedule, propose
    ttnn.synchronize_device(mesh)
    return toks[:G], walls, rec


def continuation(n: int, idle: int) -> list:
    """after a spec run: leave; `idle` plain replays with slot 0 out (position -1, its next input then rewritten);
    then n plain tokens of slot 0"""
    out = sp.leave()
    p_next, bonus = out["c"] + out["n"] + 1, out["a"][out["n"]]
    if idle:
        pos = torch.full((B,), -1, dtype=torch.int64)
        be.write(io.cur_pos, pos)
        for _ in range(idle):
            be.replay(plain)
        ttnn.synchronize_device(mesh)
        tk = torch.zeros(B, dtype=torch.int64)
        tk[0] = bonus
        be.write(io.tokens, tk)  # (the fused sampler rewrote every token row during the idle replays)
        pos[0] = p_next
        be.write(io.cur_pos, pos)
    toks = []
    for _ in range(n):
        be.replay(plain)
        ttnn.synchronize_device(mesh)
        toks.append(tok0())
    return toks


all_ok = True
for pid in str(args.get("prompts", "aime24_00")).split(","):
    prompt = S.prompt_ids(pid)
    if C.MOCK:
        S.record("tests", dict(test="s17-async-mock", milestone="S4", status="pass", K=K, prompt=pid, tag=args.get("tag")))
        continue
    cells, ident, ref = [], True, None
    for h in HOST_MS:
        row = dict(host_ms=h)
        for order in ("sync", "async"):
            toks, walls, _ = run_spec(prompt, order, h / 1000.0)
            ref = toks if ref is None else ref
            same = toks == ref
            ident &= same
            w = sorted(walls)[len(walls) // 10: max(len(walls) // 10 + 1, len(walls) * 9 // 10)] or walls
            row[order] = dict(us_per_step_median=round(statistics.median(walls) * 1e6, 1) if walls else None,
                              us_per_step_trimmed_mean=round(sum(w) / len(w) * 1e6, 1) if w else None,
                              steps=len(walls), equal_to_first=same)  # fmt: skip
            log(pid, f"host {h} ms {order}: {row[order]}")
        cells.append(row)
    # the held path's device part: the same continuation with and without the idle replays after the leave
    run_spec(prompt, "sync", 0.0)
    cont_ref = continuation(G2, 0)
    run_spec(prompt, "async", 0.0)
    cont_gap = continuation(G2, IDLE)
    gap_ok = cont_gap == cont_ref
    all_ok &= ident and gap_ok
    S.record("tests", dict(test=f"s17-async-identity-k{K}", milestone="S4", status="pass" if ident else "fail", K=K, B=B,
                           layers=_L, prompt=pid, gen=G, host_ms=HOST_MS, plain_width=w0, verify_width=int(sp.wv),
                           tp=be.tp, drafter=getattr(sp, "drafter", None), head=ref[:16], env=XENV,
                           tag=args.get("tag")))  # fmt: skip
    S.record("tests", dict(test=f"s17-async-idle-gap-k{K}", milestone="S4", status="pass" if gap_ok else "fail", K=K,
                           B=B, layers=_L, prompt=pid, idle=IDLE, gen2=G2, ref=cont_ref, got=cont_gap, env=XENV,
                           tag=args.get("tag")))  # fmt: skip
    S.record("perf", dict(test=f"s17-async-overlap-k{K}", milestone="S4", status="pass" if ident else "fail", K=K, B=B,
                          layers=_L, prompt=pid, tp=be.tp, drafter=getattr(sp, "drafter", None), cells=cells,
                          note="per-step wall record to record with a simulated host turn (time.sleep) between steps; "
                               "sync = record read before the next step is enqueued, async = enqueued right after it",
                          env=XENV, tag=args.get("tag")))  # fmt: skip
log("done", "ok" if all_ok else "FAIL")
