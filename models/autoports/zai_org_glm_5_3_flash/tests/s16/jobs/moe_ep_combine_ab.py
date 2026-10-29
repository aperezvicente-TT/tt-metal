# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 EP MoE dispatch / combine A/B on one EP layer (default L3 sparse_dsa: the served
stack has no XSL slices for the DSA layers, so its prefill MoE runs EP -- moe_fused_swiglu + the v2 EP combine, 1.84 ms
of combine per TP8 chip at 2,048 rows vs 0.44 ms for L4's XSL rm combine). The layer's real FFN input (the CPU golden's
streams -> mHC pre -> RMSNorm -> router, as moe_v2_split_check) through EpMoe in each variant
"name:v2mode:combine" (v2mode = S.set_moe_v2: "1" both halves, "dispatch", "combine", "0"; combine =
GLM53_PF_MOE_COMBINE: "" (default), "rm", "v1"); each eager vs the first variant (summed over chips), then captured and
replayed (s16common.replay_stats). Rows: perf.jsonl (status info, PCC vs the first variant).

ARGS: layers ([0..4]), layer (3), n (512), max_seq (n + 512: reuses a same-process build of that size), variants
(["v2:1:", "v2d_rm:dispatch:rm", "v2d_v1:dispatch:v1", "v1_rm:0:rm"]), replays (200), tag, record (True)."""
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
N = int(args.get("n", 512))
VARIANTS = [
    v.split(":") for v in args.get("variants", ["v2:1:", "v2d_rm:dispatch:rm", "v2d_v1:dispatch:v1", "v1_rm:0:rm"])
]
tag = args.get("tag", "moe-ep-combine-ab")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
S.set_attn({"mla_form": "absorbed", "mm_nq": 4, "mm_kb": 640, **args})
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,), max_seq=int(args.get("max_seq", N + 512)))
pf = ent["pf"]
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
pl = pf.layers[LAYERS.index(L)]
xsl = getattr(pl, "xsl", None)


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

keep = os.environ.get("GLM53_PF_MOE_COMBINE")
ref, rows = None, []
for name, v2mode, combine in VARIANTS:
    os.environ["GLM53_PF_MOE_COMBINE"] = combine
    S.set_moe_v2(pf, v2mode if v2mode in ("dispatch", "combine") else v2mode == "1")
    ep = PF.EpMoe(pf, pl.moe, xsl=xsl)
    cfg = dict(
        ffn=ep.ffn, combine=ep.combine, v2_dispatch=ep.v2_dispatch, v2_combine=ep.v2_combine, xsl=xsl is not None
    )
    y = ep(hb, idx, sel)
    ttnn.synchronize_device(mesh)
    e = summed(y)
    ttnn.deallocate(y)
    if ref is None:
        ref = e
    pcc = round(C.pcc(e, ref), 6)
    row = dict(test=f"EP MoE combine A/B L{L} {name} (n={N}) tp{TPN}", milestone="P2", status="info", layer=L, n=N,
               variant=name, cfg=cfg, finite=bool(torch.isfinite(e).all()), pcc_vs_first=pcc, tp=TPN, tag=tag)  # fmt: skip
    if not C.MOCK:
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            y = ep(hb, idx, sel)
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        rs = S.replay_stats(mesh, tid, replays=int(args.get("replays", 200)), label=name)
        r = summed(y)
        ttnn.deallocate(y)
        ttnn.release_trace(mesh, tid)
        row.update(
            replay_pcc_vs_first=round(C.pcc(r, ref), 6), median_us=rs.get("median_us"), replays=rs.get("replays")
        )
    S.log(row)
    rows.append(row)
    if rec and not C.MOCK:
        C.record("perf", row)
if keep is None:
    os.environ.pop("GLM53_PF_MOE_COMBINE", None)
else:
    os.environ["GLM53_PF_MOE_COMBINE"] = keep
S.set_moe_v2(pf, True)
RESULT = rows
