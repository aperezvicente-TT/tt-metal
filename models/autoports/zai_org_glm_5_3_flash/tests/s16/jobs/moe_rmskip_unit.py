# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): W8 unit 1 (program 2026-10-25T13:44:19Z) -- the EP layers' rm combine without the
other-chip pairs' reads and without the zero-fill (GLM53_PF_MOE_RM_SKIP: moe_combine_rm_reader.cpp RM_SKIP_ZERO_W,
EpMoe.rm_skip). One EP layer (default L3 sparse_dsa) in the W6 hybrid's MoE form (GLM53_PF_MOE_FFN=xffn,
GLM53_PF_MOE_XFFN_EP=1): its real FFN input (the golden streams -> mHC pre -> RMSNorm -> router, as moe_ep_combine_ab)
through EpMoe with rm_skip off (A) and on (B). Each arm: the first launch is the compile, then a captured trace and
`replays` replays (s16common.replay_stats); the routed partial y of the LAST replay of each arm is read per chip and
compared A vs B BITWISE (pass = every chip equal). No eager output is used: both sides of the comparison are traced
replays. Rows: perf.jsonl (median us per arm, the delta, bitwise per chip).

ARGS: layers ([0..4]), layer (3), n (2048), max_seq (n + 512), replays (100), tag, record (True)."""
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
L = int(args.get("layer", 3))
N = int(args.get("n", 2048))
REPLAYS = int(args.get("replays", 100))
tag = args.get("tag", "w8u1-rmskip")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
for k_, v_ in (("GLM53_PF_MOE_FFN", "xffn"), ("GLM53_PF_MOE_XFFN_EP", "1"), ("GLM53_PF_MHC", "comp"),
               ("GLM53_PF_MHC_POST", "fused")):  # fmt: skip
    os.environ.setdefault(k_, v_)
row = dict(test=f"W8 unit 1: EP rm combine skip + no zero-fill, L{L} (n={N}) tp{TPN}", milestone="W8", layer=L, n=N,
           tp=TPN, tag=tag, ffn=os.environ["GLM53_PF_MOE_FFN"], xffn_ep=os.environ["GLM53_PF_MOE_XFFN_EP"])  # fmt: skip
try:
    S.set_attn({"mla_form": "absorbed", "mm_nq": 4, "mm_kb": 640, **args})
    ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,), max_seq=int(args.get("max_seq", N + 512)))
    pf = ent["pf"]
    c = pf.bucket_for(N)
    sp = pf.mode(c) == "sp"
    pl = pf.layers[LAYERS.index(L)]
    xsl = getattr(pl, "xsl", None)
    assert xsl is None, "an EP layer (the skip only removes reads of other-chip pairs)"
    per_chip = lambda t_: [ttnn.to_torch(p).clone() for p in (ttnn.get_device_tensors(t_) if TPN > 1 else [t_])]  # noqa: E731

    x_in, _ = S.golden_layer(L, N)
    pad = torch.zeros(c, 4, 4096)
    pad[:N] = x_in.reshape(N, 4, 4096).float()
    X = pf.streams_to_device(pad, sp)
    hh, post, comb, xs = pl.hc["ffn"].pre_streams(X)
    x32 = pl.rms(hh, "ffn", PF.FP32)
    idx, scores = pl.moe.router(x32, decode=False)
    hb = pf.gather(ttnn.typecast(x32, PF.BF16), sp)
    sel = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), PF.FP32)
    if sp:
        idx = ttnn.typecast(pf.gather(ttnn.typecast(idx, PF.FP32), sp), ttnn.uint16)
        sel = pf.gather(sel, sp)

    eps, outs, arms = {}, {}, {}
    for name, skip in (("A_fill", False), ("B_skip", True)):
        ep = PF.EpMoe(pf, pl.moe, xsl=None)
        assert ep.ffn == "xffn" and ep.combine == "rm", (ep.ffn, ep.combine)
        ep.rm_skip = skip
        eps[name] = ep
        y = ep(hb, idx, sel)  # the compile launch (outside any trace); its output is not compared
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(y)
        arms[name] = dict(rm_skip=skip)
        if C.MOCK:
            continue
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            y = ep(hb, idx, sel)
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        rs = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} {name}")
        outs[name] = per_chip(y)
        arms[name].update(median_us=rs.get("median_us"), replays=rs.get("replays"),
                          finite=all(bool(torch.isfinite(t_.float()).all()) for t_ in outs[name]))  # fmt: skip
        ttnn.deallocate(y)
        ttnn.release_trace(mesh, tid)
    row["arms"] = arms
    if C.MOCK:
        row["status"] = "mock"
    else:
        eq = [bool(torch.equal(a_, b_)) for a_, b_ in zip(outs["A_fill"], outs["B_skip"])]
        row.update(bitwise_per_chip=eq, bitwise=all(eq),
                   delta_us=round(arms["B_skip"]["median_us"] - arms["A_fill"]["median_us"], 1),
                   status="pass" if all(eq) and arms["B_skip"]["finite"] else "fail")  # fmt: skip
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    row.update(status="fail", error=f"{type(e).__name__}: {e}"[:2000])
S.log(row)
if rec and not C.MOCK:
    C.record("perf", row)
RESULT = row
