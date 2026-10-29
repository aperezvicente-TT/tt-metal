# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, a pair or one chip): the DFlash2 drafter device prototype (tt/dflash2.py; Apache-2.0 drafter only)
on golden taps vs the GPU reference (doc/ttnn/17-mtp-spec-decode/scripts/dflash2/dflash2_golden.py, host-p300 GPU 0).

Two captured traces, every row a replay: ctx (the sequence's taps -> each layer's context K / V) and block (one draft
block [bonus, mask x K] -> final hidden + lm_head top-16). Per golden anchor the host writes the block inputs (embedding
rows, RoPE rows, mask) between synchronized replays, replays the block trace, reads the hidden / candidates and runs the
candidate selector (host, fp32). Gates (vs the GPU's bf16 drafter with fp32 attention / selector): the K drafts equal
per anchor (`draft_eq`), the plain mask-row argmax equal, the top-16 sets, hidden PCC / max |diff|; the ctx K of layer 0
(64 positions) PCC. Timing: >= `reps` back-to-back block replays (no host work between them) -> us per block.
layout "tp" (cache = tests/s17/dflash2_convert.py output for the mesh's chip count): tt/dflash2_tp.py, the sharded
bfloat8_b image layout (compare against a bfp8 golden: dflash2_golden.py --bfp8).
ARGS: drafter, taps, golden (paths), k (7), reps (2000), soak_s (0: the unwatched soak), layout ("replicated" | "tp"),
cache, tag."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib  # noqa: E402

import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
K = int(args.get("k", 7))
D2 = "/opt/glm53/scratch-big/17/dflash2"


def log(*a):
    print(f"S17-DF2 {C.now()}", *a, flush=True)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


from models.autoports.zai_org_glm_5_3_flash.tt import dflash2 as DF  # noqa: E402

DF = importlib.reload(DF)
LAYOUT = str(args.get("layout", "replicated"))
if LAYOUT == "tp":
    from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_tp as DFT  # noqa: E402

    DFT = importlib.reload(DFT)
# a standing server's earlier jobs' stack backends (their persistent L1 buffers lower the floor under this job's
# programs' static CBs): closed first (the weight cache stays in DRAM)
S.M5.prune_ctx(ctx, [k_ for k_ in ctx if isinstance(k_, tuple) and k_ and k_[0] == "dflash2"])
G = torch.load(args.get("golden", f"{D2}/golden_aime24_00_k7.pt"), weights_only=False)
T = torch.load(args.get("taps", f"{D2}/taps/taps_aime24_00.pt"), weights_only=False)
taps = T["taps"]
n = int(taps.shape[1])
t0 = time.time()
key = ("dflash2", n, LAYOUT)
if key not in ctx:
    for k_ in [k_ for k_ in ctx if isinstance(k_, tuple) and k_[0] == "dflash2"]:
        ctx.pop(k_)
    if LAYOUT == "tp":
        ctx[key] = dict(d=DFT.DFlash2TP(mesh, args.get("drafter", f"{D2}/drafter"), args["cache"], n, log=log))
    else:
        ctx[key] = dict(d=DF.DFlash2TT(mesh, args.get("drafter", f"{D2}/drafter"), n, log=log))
D = ctx[key]["d"]
log(f"drafter on device in {time.time() - t0:.1f}s (cap {D.cap}, taps {n} positions)")


def capture(fn):
    """capture an already-warm program -> trace id ('mock': the function)"""
    if C.MOCK:
        return ("mock", fn)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    return tid


def replay(tr, blocking=False):
    if isinstance(tr, tuple):
        tr[1]()
    else:
        ttnn.execute_trace(mesh, tr, cq_id=0, blocking=blocking)


D.ctx(taps)
a0 = G["anchors"][0]
D.set_block(a0["bonus"], a0["L"], a0["L"] + 1, K)
if "traces" not in ctx[key]:
    # every program compiled before the FIRST capture (a program first compiled after a capture keeps a live buffer);
    # the block twice: its first call allocates D.out, the second compiles the output copies (d0 std1 f1: the copies
    # were first compiled inside the capture -> TT_FATAL)
    D.ctx_program()
    D.block_program()
    D.block_program()
    ttnn.synchronize_device(mesh)
    n_keys = mesh.num_program_cache_entries()
    if C.MOCK:  # the capture-key gate: one more call of each program adds no program-cache entry
        D.ctx_program()
        D.block_program()
        new_keys = mesh.num_program_cache_entries() - n_keys
        log("capture keys: warm", n_keys, "new on a further call", new_keys)
        assert new_keys == 0, new_keys
    ctx[key]["traces"] = (capture(D.ctx_program), capture(D.block_program))
tr_ctx, tr_blk = ctx[key]["traces"]
replay(tr_ctx)
ttnn.synchronize_device(mesh)
if C.MOCK:
    S.record("tests", dict(test="dflash2-proto-mock", milestone="DF2", status="pass", K=K, n=n, tag=args.get("tag")))
    RESULT = dict(ok=True)
else:
    k0 = ttnn.to_torch(ttnn.get_device_tensors(D.Kc[0])[0] if D.rep is not None else D.Kc[0]).float()[0, :, :64]
    ctx_pcc = pcc(k0, G["ctx_k0"][: k0.shape[0]])  # (tp: chip 0 holds kv heads 0 .. nkv / tp - 1)
    rows = []
    for a in G["anchors"]:
        D.set_block(a["bonus"], a["L"], a["L"] + 1, K)
        replay(tr_blk)
        ttnn.synchronize_device(mesh)
        r = D.read_block(K)
        drafts = D.select(r, a["bonus"])
        plain = [int(v) for v in r["cand"][:, 0]]
        overlap = [len(set(r["cand"][s].tolist()) & set(a["cand"][s].tolist())) for s in range(K)]
        rows.append(dict(L=a["L"], draft_eq=drafts == a["drafts"], plain_eq=plain == a["plain"], top16_overlap=overlap,
                         hid_pcc=round(pcc(r["hid"], a["hid"]), 6), hid_max=round(float((r["hid"] - a["hid"]).abs().max()), 4),
                         unary_max=round(float((r["unary"] - a["unary"]).abs().max()), 4), drafts=drafts, ref=a["drafts"],
                         accepted_vs_truth=next((j for j, (x, y) in enumerate(zip(drafts, a["truth"])) if x != y), K)))  # fmt: skip
        log("anchor", a["L"], "drafts", drafts, "ref", a["drafts"], "hid pcc", rows[-1]["hid_pcc"])
    # timing: back-to-back block replays (inputs unchanged), then the ctx trace
    reps = int(args.get("reps", 2000))
    ttnn.synchronize_device(mesh)
    t1 = time.perf_counter()
    for _ in range(reps):
        replay(tr_blk)
    ttnn.synchronize_device(mesh)
    blk_us = (time.perf_counter() - t1) / reps * 1e6
    t1 = time.perf_counter()
    for _ in range(20):
        replay(tr_ctx)
    ttnn.synchronize_device(mesh)
    ctx_us = (time.perf_counter() - t1) / 20 * 1e6
    # the unwatched soak (soak_s > 0): block replays for soak_s seconds; the outputs must then equal the timed run's
    soak_s, soak_n, soak_det = float(args.get("soak_s", 0)), 0, None
    if soak_s > 0:
        replay(tr_blk)
        ttnn.synchronize_device(mesh)
        ref = D.read_block(K)
        t1 = time.perf_counter()
        while time.perf_counter() - t1 < soak_s:
            for _ in range(100):
                replay(tr_blk)
            ttnn.synchronize_device(mesh)
            soak_n += 100
        r2 = D.read_block(K)
        soak_det = all(torch.equal(ref[k_], r2[k_]) for k_ in ref)
        log("soak", soak_n, "block replays in", round(time.perf_counter() - t1, 1), "s, deterministic", soak_det)
    eq = sum(r["draft_eq"] for r in rows)
    ok = ctx_pcc > 0.999 and eq >= len(rows) - 2 and min(r["hid_pcc"] for r in rows) > 0.99 and soak_det is not False
    S.record("tests", dict(test="dflash2-proto-golden", milestone="DF2", status="pass" if ok else "fail", K=K, n=n,
                           tp=int(mesh.get_num_devices()), layout=LAYOUT, golden_weights=G.get("weights", "bfloat16"), anchors=len(rows), draft_eq=eq,
                           plain_eq=sum(r["plain_eq"] for r in rows), ctx_k0_pcc=round(ctx_pcc, 6),
                           min_hid_pcc=min(r["hid_pcc"] for r in rows), rows=rows, tag=args.get("tag")))  # fmt: skip
    S.record("perf", dict(test="dflash2-proto-block-us", milestone="DF2", status="info", K=K, replays=reps,
                          us_per_block=round(blk_us, 1), ctx_us=round(ctx_us, 1), ctx_rows=D.cap,
                          tp=int(mesh.get_num_devices()), layout=LAYOUT if LAYOUT == "tp" else "replicated (every chip the whole drafter)",
                          soak_s=soak_s, soak_replays=soak_n, soak_deterministic=soak_det,
                          note="prototype of stock ttnn ops; the selector walk on the host is outside the timing",
                          tag=args.get("tag")))  # fmt: skip
    log("draft_eq", eq, "/", len(rows), "ctx pcc", ctx_pcc, "block us", round(blk_us, 1), "ctx us", round(ctx_us, 1))
    RESULT = dict(ok=ok, draft_eq=eq, anchors=len(rows), block_us=blk_us)
