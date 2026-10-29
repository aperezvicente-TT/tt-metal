# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, the TP mesh / pair): Stage 14b — where the decode step's time goes. The serving backend's step
(m5common.serving) split into its programs: the embed, every stack run (one program per chip over a run of KDA layers
of one kind), every sparse_dsa layer (MkMla + MkMoE; plus its MkMla alone, which rewrites the same latent row: the MoE
share = the pair - the MLA), the head (final norm + lm_head shard + device top-K + gather / sampling + position
advance). Each piece is its own captured trace; a step = the pieces replayed in step order, each followed by a
synchronize (traced replays only; the synchronize round trip measured on its own and subtracted). The tokens are the
head's own fed-back outputs (real routing, no MoE drift). Compared with the full step trace timed the same way.
ARGS: b (32), layers ("all"), native (true), reps (10), pos (512), width (None: the full width; a native width bucket:
its stacks / dsa calls on the w-row views, rows 0..w-1 live), shared_weights, rev, record, tag."""
import gc
import importlib
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.join(os.environ["TT_METAL_HOME"], "models/autoports/zai_org_glm_5_3_flash/tests/s14b"))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402

for mod in (M5, G, SM):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 32))
NATIVE = bool(args.get("native", True))
REPS, POS = int(args.get("reps", 10)), int(args.get("pos", 512))
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
WID = args.get("width")
tag = args.get("tag", f"m5b-breakdown-B{B}" + (f"-w{WID}" if WID else ""))


def log(*a):
    print(f"M5B-BRK {C.now()}", *a, flush=True)


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=NATIVE, rev=args.get("rev"), log=log)
be = ctx[key]["backend"]
kv = be._kv
io = kv.extra["io"]
assert be.n_chunks == 1, ("one row chunk (native B=32 or B <= 4)", be.n_chunks)
sync = lambda: ttnn.synchronize_device(mesh)  # noqa: E731


W = int(WID) if WID else be.chunk
NARROW = W in getattr(be, "narrow_widths", ())
assert W == be.chunk or W in be.native_widths or NARROW, (W, be.native_widths)
if NARROW:  # a narrow width: its own stacked residual / stacks, t = w embed / head views
    RES, MASKS, STACKS, SEQ = kv.extra["nres"][W], kv.extra["nmasks"][W], kv.extra["nstacks"][W], be.nseq[W]
else:
    RES = kv.res[0] if W == be.chunk else kv.extra["wres"][W]
    MASKS = kv.masks[0] if W == be.chunk else kv.extra["wmasks"][W]
    STACKS = kv.stacks[0] if W == be.chunk else kv.extra["wstacks"][W]
    SEQ = be.seq if W == be.chunk else be.wseq[W]
KNOBS = getattr(SM, "_narrow_knobs", SM.set_knobs) if NARROW else SM.set_knobs


def dsa(e, i, moe=True):
    if be.fl is not None:
        SM.set_knobs("dsa", be.tp, W)
    if W != be.chunk and not NARROW:
        for o in (getattr(e["mla"], "mk", None), getattr(e["moe"], "mk", None)):
            if o is not None:
                SM._seed_width(o, W, be.chunk)
    s1 = e["mla"](RES, kv.extra["latent"][i], kv.extra["pt_c"][0], be.pos_c[0], t=W, keep=("s1",),
                  stack=True)["s1"]  # fmt: skip
    if moe:
        e["moe"](s1, t=W, out=RES)
    ttnn.deallocate(s1)


def run(r, stk):
    if be.fl is not None:
        KNOBS(SEQ[r[0]]["sparse"], be.tp, W)
    pool = kv.extra.get("pool")
    if W != be.chunk:
        pool.alias = True
    try:
        stk([kv.states[0][i][1] for i in r], MASKS, [kv.states[0][i][0] for i in r])
    finally:
        if W != be.chunk:
            pool.alias = False


# (name, kind, layers, fn, counts in the step sum)
if NARROW:
    pieces = [("embed", "embed", 0, lambda: be._narrow_io(W)[0](be.head.topk_out, [RES]), True)]
else:
    pieces = [("embed", "embed", 0, lambda: be.embed(be.head.topk_out, kv.res), True)]
for r, stk in STACKS:
    e = SEQ[r[0]]
    Ls = [be.seq[i]["L"] for i in r]
    if stk is None:
        pieces.append((f"dsa_mla L{Ls[0]}", "dsa_mla", 1, lambda e=e, i=r[0]: dsa(e, i, moe=False), False))
        pieces.append((f"dsa L{Ls[0]}", "sparse_dsa", 1, lambda e=e, i=r[0]: dsa(e, i), True))
    else:
        k = "sparse_kda" if e["sparse"] else "dense_kda"
        pieces.append((f"{k} L{Ls[0]}-L{Ls[-1]}", k, len(r), lambda r=r, stk=stk: run(r, stk), True))
if NARROW:
    pieces.append(("head", "head", 0, lambda: be._narrow_io(W)[1]([RES], be.pos, pos_copies=be.pos_c), True))
elif W != be.chunk and hasattr(be, "_width_head") and os.environ.get("GLM53_STACK_WIDTH_HEAD", "1") == "1":
    pieces.append(("head", "head", 0, lambda: be._width_head(W)([RES], be.pos, pos_copies=be.pos_c), True))
else:
    pieces.append(("head", "head", 0, lambda: be.head(kv.res, be.pos, pos_copies=be.pos_c), True))


def state():
    be.reset_all(kv)
    pos = torch.full((B,), POS, dtype=torch.int64) + torch.arange(B)
    pos[W:] = -1  # rows >= W idle
    be.write(io.tokens, torch.tensor([1000 + 7 * u for u in range(B)]))
    be.write(io.cur_pos, pos)
    sync()


def measure():
    """the pieces' traces, the step = the pieces in order (each + a synchronize), and the full step trace alike"""
    tids = []
    try:
        for _, _, _, fn, _ in pieces:
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                fn()
            finally:
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            tids.append(tid)
        state()
        rt = []  # the synchronize round trip alone
        for _ in range(50):
            t0 = time.perf_counter()
            sync()
            rt.append((time.perf_counter() - t0) * 1e6)
        sync_us = med(rt)
        per = {p[0]: [] for p in pieces}
        for _ in range(REPS):
            for (name, *_), tid in zip(pieces, tids):
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                sync()
                per[name].append((time.perf_counter() - t0) * 1e6 - sync_us)
        state()
        full = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, kv.extra["decode_traces"][W] if W != be.chunk else kv.extra["decode_trace"], cq_id=0, blocking=False)
            sync()
            full.append((time.perf_counter() - t0) * 1e6 - sync_us)
    finally:
        for tid in tids:
            ttnn.release_trace(mesh, tid)
        state()
    return sync_us, per, full


med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
state()
if SM.MOCK:  # mock cluster: no trace capture -> each piece once, eagerly (the structure only, no timing)
    for _, _, _, fn, _ in pieces:
        fn()
    sync()
    state()
    log("RESULT mock", [p[0] for p in pieces])
    RESULT = dict(ok=True, mock=True, pieces=[p[0] for p in pieces])
else:
    sync_us, per, full = measure()
    rows = {p[0]: round(med(per[p[0]]), 1) for p in pieces}
    kinds = {}
    for name, kind, nl, _, counted in pieces:
        k = kinds.setdefault(kind, dict(us=0.0, layers=0, pieces=0))
        k["us"] += rows[name]
        k["layers"] += nl
        k["pieces"] += 1
    for k in kinds.values():
        k["us"] = round(k["us"], 1)
        if k["layers"]:
            k["us_per_layer"] = round(k["us"] / k["layers"], 1)
    res = dict(B=B, width=W, layers=len(ids), reps=REPS, sync_us=round(sync_us, 1), full_step_us=round(med(full), 1),
               piece_sum_us=round(sum(rows[p[0]] for p in pieces if p[4]), 1), kinds=kinds, pieces=rows)  # fmt: skip
    log("RESULT", res)
    if args.get("record") and not os.environ.get("TT_METAL_WATCHER"):
        C.record("perf", dict(test=f"m5b decode step breakdown (each program its own trace, replayed in step order "
                                   f"with a synchronize after each; {len(ids)} layers, B={B}, tp{be.tp})",
                              status="info", milestone="M5b", traced=True, tag=tag, **res))  # fmt: skip
    RESULT = dict(ok=True, **{k: v for k, v in res.items() if k != "pieces"})
