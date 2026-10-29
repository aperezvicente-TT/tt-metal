# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 EP MoE v2 (bounded dispatch / combine kernels, GLM53_PF_MOE_V2=1) vs v1 (the
embedding dispatch over CAP rows + the O(C k) combine) on the same inputs (device vs device): the golden layer input's
FFN rows (mhc pre of the CPU golden streams -> RMSNorm -> the layer's router), the routed partial of this chip's
experts summed over the mesh; both timed as traced replays.

ARGS: layers ([0, 1, 2, 3, 4]), check ([4, 3]), n (2048), reps (5), tag, record."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
CHECK = [int(v) for v in args.get("check", [4, 3])]
N = int(args.get("n", 2048))
REPS = int(args.get("reps", 5))
tag = args.get("tag", "moe-v2-check")
TPN = int(mesh.get_num_devices())
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "ops"
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,))
pf = ent["pf"]
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"


def timed(fn):
    out = fn()
    ttnn.synchronize_device(mesh)
    if C.MOCK:
        return out, []
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        out = fn()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ts.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(mesh, tid)
    return out, ts


def summed(t_):
    parts = ttnn.get_device_tensors(t_) if TPN > 1 else [t_]
    return sum(ttnn.to_torch(p).float() for p in parts).reshape(-1, 4096)


results = []
for L in CHECK:
    li = LAYERS.index(L)
    pl = pf.layers[li]
    if pl.kind == "dense_kda":
        continue
    x_in, _ = S.golden_layer(L, N)
    pad = torch.zeros(c, 4, 4096)
    pad[:N] = x_in.reshape(N, 4, 4096).float()
    X = pf.streams_to_device(pad, sp)
    h, post, comb, xs = pl.hc["ffn"].pre_streams(X)  # the FFN mHC pre of the golden streams: realistic rows
    x32 = pl.rms(h, "ffn", PF.FP32)
    idx, scores = pl.moe.router(x32, decode=False)
    hb = pf.gather(ttnn.typecast(x32, PF.BF16), sp)
    k = int(idx.shape[-1])
    sel = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), PF.FP32)
    if sp:
        idx = ttnn.typecast(pf.gather(ttnn.typecast(idx, PF.FP32), sp), ttnn.uint16)
        sel = pf.gather(sel, sp)
    outs = {}
    for v in ("0", "1"):
        os.environ["GLM53_PF_MOE_V2"] = v
        ep = PF.EpMoe(pf, pl.moe)
        y, ts = timed(lambda: ep(hb, idx, sel))
        outs[v] = (summed(y), C.stats(ts).get("median_us"))
        ttnn.deallocate(y)
    os.environ["GLM53_PF_MOE_V2"] = "0"
    a, b = outs["0"][0], outs["1"][0]
    row = dict(test=f"EP MoE v2 (bounded dispatch / combine) vs v1, L{L} {pl.kind} (n={N}) tp{TPN}", milestone="P2",
               status="info", traced=not C.MOCK, layer=L, n=N, tp=TPN, pcc=round(C.pcc(b, a), 6),
               max_abs=float((a - b).abs().max()), rel_l2=float((a - b).norm() / max(a.norm(), 1e-30)),
               v1_us=outs["0"][1], v2_us=outs["1"][1], tag=tag)  # fmt: skip
    if not C.MOCK:
        row["status"] = "pass" if row["pcc"] >= 0.9999 else "fail"
    S.log(row)
    if args.get("record", True) and not C.MOCK:
        C.record("pcc", row)
    results.append(row)
    for t_ in [X, h, x32, hb, idx, sel, scores] + list(xs):
        ttnn.deallocate(t_)
RESULT = results
