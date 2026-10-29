# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; run WATCHED): the EP MoE v2 hang diagnosis -- v1 output first (reference), then each v2 mode
("dispatch": bounded dispatch + v1 combine; "combine": v1 dispatch + bounded combine; "both") eager, then captured and
replayed `reps` times with a log line after every replay (a hang names its mode and step); every output vs v1.

ARGS: layers ([0, 1, 2, 3, 4]), layer (4), n (2048), modes (["dispatch", "combine", "both"]), reps (2), xd_zero
(False), tag, record."""
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
L = int(args.get("layer", 4))
N = int(args.get("n", 2048))
REPS = int(args.get("reps", 2))
MODES = list(args.get("modes", ["dispatch", "combine", "both"]))
tag = args.get("tag", "moe-v2-split")
TPN = int(mesh.get_num_devices())
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "ops"
os.environ["GLM53_PF_MOE_V2_XDZERO"] = "1" if args.get("xd_zero") else "0"
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,))
pf = ent["pf"]
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
pl = pf.layers[LAYERS.index(L)]


def summed(t_):
    parts = ttnn.get_device_tensors(t_) if TPN > 1 else [t_]
    return sum(ttnn.to_torch(p).float() for p in parts).reshape(-1, 4096)


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

S.set_moe_v2(pf, False)
y = PF.EpMoe(pf, pl.moe)(hb, idx, sel)
ttnn.synchronize_device(mesh)
ref = summed(y)
ttnn.deallocate(y)
S.log(f"v1 reference: finite {bool(torch.isfinite(ref).all())}")
results = []
for mode in MODES:
    S.set_moe_v2(pf, mode)
    ep = PF.EpMoe(pf, pl.moe)
    y = ep(hb, idx, sel)
    ttnn.synchronize_device(mesh)
    e = summed(y)
    ttnn.deallocate(y)
    fin = torch.isfinite(e)
    S.log(f"{mode}: eager done, non-finite {int((~fin).sum())}, pcc vs v1 {C.pcc(torch.where(fin, e, 0.0), ref):.6f}")
    row = dict(test=f"EP MoE v2 split check L{L} mode={mode} (n={N}) tp{TPN}", milestone="P2", layer=L, mode=mode,
               tp=TPN, eager_nonfinite=int((~fin).sum()), eager_pcc=round(C.pcc(torch.where(fin, e, 0.0), ref), 6),
               xd_zero=bool(args.get("xd_zero")), tag=tag)  # fmt: skip
    if not C.MOCK:
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            y = ep(hb, idx, sel)
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        S.log(f"{mode}: captured")
        for i in range(REPS):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            S.log(f"{mode}: replay {i} done")
        r = summed(y)
        ttnn.deallocate(y)
        ttnn.release_trace(mesh, tid)
        rf = torch.isfinite(r)
        row.update(replay_nonfinite=int((~rf).sum()), replay_pcc=round(C.pcc(torch.where(rf, r, 0.0), ref), 6))
    row["status"] = "pass" if row.get("replay_pcc", row["eager_pcc"]) >= 0.9999 and not row["eager_nonfinite"] \
        and not row.get("replay_nonfinite", 0) else ("info" if C.MOCK else "fail")  # fmt: skip
    S.log(row)
    if args.get("record", True) and not C.MOCK:
        C.record("pcc", row)
    results.append(row)
S.set_moe_v2(pf, False)
RESULT = results
