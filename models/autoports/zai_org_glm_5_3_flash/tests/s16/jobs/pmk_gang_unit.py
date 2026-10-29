# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, pair or 2x4): W7 step 5 on a DSA (EP) layer (program 2026-10-25 20:12:14Z: the gang applies to the
11 DSA layers; bring-up on L3) -- program G (tt/pmk_gang.PmkGang: the EP plan + 14's gang FFN + the RM_SKIP combine
in ONE program; another chip's pairs masked in-program by the plan's sentinel dest) vs the served EP path (EpMoe:
GLM53_PF_MOE_FFN=xffn + XFFN_EP + VALID: the plan programs, the xffn program, host w x local, the rm combine) on the
layer's REAL FFN input (golden streams -> mHC pre -> RMSNorm -> router, as 14's moe_gang_unit). Each arm: one compile
launch, one capture, replays (s16common.replay_stats); the routed partial y of the first replay per chip compared (PCC,
scale, rel; the W-slice sum order differs: not bitwise), determinism over `det` replays of G. The inputs of G (x_ext,
ids, the normalised weights) are built once before its capture (setup, outside the measured trace).
ARGS: layer (3), build_layers ([3, 4]: the stack needs one non-DSA layer), n (2048), max_seq (n + 512), replays (30),
det (20), fid (LoFi), gangs (None = pmk_gang.gang_sets on the grid; else [[[x, y]] * 8 per gang, ring order]), tag,
record."""
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_gang as PG  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
L = int(args.get("layer", 3))
LAYERS = [int(v) for v in args.get("build_layers", [3, 4])]
N = int(args.get("n", 2048))
REPLAYS = int(args.get("replays", 30))
DET = int(args.get("det", 20))
FID = str(args.get("fid", "LoFi"))
tag = args.get("tag", f"pmk-gang-L{L}")
rec = bool(args.get("record", False))
TPN = int(mesh.get_num_devices())
SERVED = {"GLM53_PF_MOE_FFN": "xffn", "GLM53_PF_MOE_XFFN_EP": "1", "GLM53_PF_MOE_XFFN_VALID": "1",
          "GLM53_PF_MHC": "comp", "GLM53_PF_MHC_POST": "fused"}  # fmt: skip
# served_ffn = "mfs" (program 2026-10-26 06:09:28Z): the served arm on moe_fused_swiglu (bf16 DEST accumulation:
# the routed rows x ~1.04, the 9 g0-base prep regression) instead of xffn-EP -- the TP8 cost of xffn-EP on the DSA
# layers' MoE, and G (fp32 DEST + L1-accumulated partials) against the biased mode; a measurement arm: G vs served is
# expected to read scale ~0.96, so its status gates only G's own finite / determinism
SERVED_FFN = str(args.get("served_ffn", "xffn-ep"))
assert SERVED_FFN in ("xffn-ep", "mfs"), SERVED_FFN
if SERVED_FFN == "mfs":
    SERVED.update(GLM53_PF_MOE_FFN="", GLM53_PF_MOE_XFFN_EP="0", GLM53_PF_MOE_XFFN_VALID="0")
for k_, v_ in SERVED.items():
    os.environ[k_] = v_
row = dict(test=f"W7 step 5: program G (EP plan + gang FFN + RM_SKIP combine, ONE program) vs the served EP {SERVED_FFN} path, "
           f"L{L} (n={N}) tp{TPN}", milestone="W7-S5", layer=L, n=N, tp=TPN, fid=FID, tag=tag, served_ffn=SERVED_FFN)  # fmt: skip


def _cmp(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return dict(pcc=round(C.pcc(a.float(), b.float()), 7), scale=round(float((a * b).sum() / (b * b).sum().clamp_min(1e-300)), 6),
                rel=float((a - b).norm() / b.norm().clamp_min(1e-30)), maxabs=float((a - b).abs().max()),
                finite=bool(torch.isfinite(a).all()))  # fmt: skip


per_chip = lambda t_: [ttnn.to_torch(p).clone() for p in (ttnn.get_device_tensors(t_) if TPN > 1 else [t_])]  # noqa: E731
try:
    S.set_attn({"mla_form": "absorbed", "mm_nq": 4, "mm_kb": 640, **args})
    ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,), max_seq=int(args.get("max_seq", N + 512)))
    pf = ent["pf"]
    c = pf.bucket_for(N)
    sp = pf.mode(c) == "sp"
    pl = pf.layers[LAYERS.index(L)]
    assert getattr(pl, "xsl", None) is None, f"L{L}: an EP layer (sparse_dsa) is required"
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
    ep = PF.EpMoe(pf, pl.moe, xsl=None)
    row["served_ffn_ep"] = getattr(ep, "ffn", None)  # the mode EpMoe actually BUILT (xffn / mfs)
    assert ep.ffn == ("xffn" if SERVED_FFN == "xffn-ep" else "mfs") and not ep.xsl, (ep.ffn, ep.xsl, SERVED_FFN)
    moe = pl.moe
    k, h = int(moe.top_k), int(moe.hidden)
    # ---- G's inputs (setup, once): x_ext [C, H] RM, ids [1, K C] uint32 (pair p = j C + t), w = sel / sum x scaling
    x_ext = ttnn.to_layout(ttnn.reshape(hb, (c, h)), ttnn.ROW_MAJOR_LAYOUT, memory_config=PF.DRAM)
    ids_t = ttnn.transpose(ttnn.typecast(idx, ttnn.uint32), -2, -1)
    ids_rm = ttnn.to_layout(ttnn.reshape(ids_t, (1, k * c)), ttnn.ROW_MAJOR_LAYOUT, memory_config=PF.DRAM)
    w = ttnn.multiply(sel, ttnn.reciprocal(ttnn.add(ttnn.sum(sel, dim=-1, keepdim=True), 1e-20)))
    wt = ttnn.multiply(w, moe.scaling)
    grid = mesh.compute_with_storage_grid_size()
    gangs = args.get("gangs") or PG.gang_sets(int(grid.x), int(grid.y))
    gk = PG.PmkGang(mesh, e_loc=ep.e_loc, top_k=k, hidden=h, limit=float(moe.limit), gate_up=ep.gate_up, down=ep.down,
                    gangs=gangs, lo=[ep.e_loc * r for r in range(TPN)], fid=FID)  # fmt: skip
    row.update(gangs=len(gk.gangs), p=gk.p, e_loc=ep.e_loc, w_slice=gk.w_slice, blocks=list(gk.blocks(c)),
               g_cb_bytes=gk.l1_bytes(c))  # fmt: skip
    S.log(f"{tag}: G gangs {len(gk.gangs)} P {gk.p} E {ep.e_loc} W/G {gk.w_slice} blocks {gk.blocks(c)} CB {gk.l1_bytes(c)} B")
    arms, outs = {}, {}
    for name, fn in (("served", lambda: ep(hb, idx, sel)), ("g", lambda: gk(x_ext, ids_rm, wt, c))):
        y = fn()  # the compile launch (outside any trace)
        ttnn.synchronize_device(mesh)
        if name == "served":
            ttnn.deallocate(y)
        arms[name] = {}
        if C.MOCK:
            fn()  # the mock runs no trace: the path once more for its allocations
            continue
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            y = fn()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        outs[name] = per_chip(y)
        if name == "g" and DET:
            same = True
            for _ in range(DET):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                same = same and all(torch.equal(a_, b_) for a_, b_ in zip(per_chip(y), outs["g"]))
            arms[name]["determinism"] = {str(DET): bool(same)}
        rs = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} {name}")
        arms[name].update(median_us=rs.get("median_us"), replays=rs.get("replays"),
                          finite=all(bool(torch.isfinite(t_.float()).all()) for t_ in outs[name]))  # fmt: skip
        if name == "served":
            ttnn.deallocate(y)
        ttnn.release_trace(mesh, tid)
    row["arms"] = arms
    if C.MOCK:
        row["status"] = "mock"
    else:
        row["g_vs_served"] = [_cmp(g_, s_) for g_, s_ in zip(outs["g"], outs["served"])]
        row["delta_us"] = round(arms["g"]["median_us"] - arms["served"]["median_us"], 1)
        # PCC is scale-blind (memory pcc-scale-blind): the scale <g, s> / <s, s> must sit within 0.5 % of 1 too (the
        # first gangw read PCC 0.99990 at scale 0.975: the combine at LoFi)
        ok = (min(x_["pcc"] for x_ in row["g_vs_served"]) >= 0.9999 and arms["g"]["finite"]
              and all(abs(x_["scale"] - 1.0) <= 0.005 for x_ in row["g_vs_served"])
              and all(arms["g"].get("determinism", {"0": True}).values()))  # fmt: skip
        if SERVED_FFN == "mfs":  # the measurement arm (see SERVED_FFN): G's own finite / determinism only
            ok = arms["g"]["finite"] and all(arms["g"].get("determinism", {"0": True}).values()) and row["served_ffn_ep"] == "mfs"
        row["status"] = "pass" if ok else "fail"
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    row.update(status="fail", error=f"{type(e).__name__}: {e}"[:2000])
for k_ in SERVED:
    os.environ.pop(k_, None)
S.log(f"{tag}: {row}")
if rec and not C.MOCK:
    C.record("pcc", row)
RESULT = [row]
print("PMK_GANG_UNIT", row, flush=True)
