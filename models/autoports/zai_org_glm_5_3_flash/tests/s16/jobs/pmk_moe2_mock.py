# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 2x4 MOCK or any mesh): the PMK MoE v2 phase set (tt/pmk_moe2.py: levers 2 / 3 / 5) built and
launched ONCE on synthetic XSL expert slices (TP8: 256 wide, TP2: 1,024) at C rows with random routing, plus the P13
shared-expert PfStream pair (7a's PfStreamMM) built on the same mesh -- in the mock the kernels JIT-build, the CBs
allocate (cb_guard + the v2 ledger) and the programs validate; on a device the launch is the compile launch and the
output is compared with EpMoe (xffn forced) on the same inputs (PCC).
ARGS: experts (288), width (256), c (2048), k (8), seed (0), blocks ("" = the budget's choice), mcast (4), shared (1),
ids_u16 (1), p13 (1), fid (""), tag, record."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.pmk_moe as PM  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.pmk_moe2 as M2  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import _nd_dram, gate_up_tile_perm  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
E, W, CR, K = int(args.get("experts", 288)), int(args.get("width", 256)), int(args.get("c", 2048)), int(args.get("k", 8))
H, LIMIT = 4096, 10.0
tag = args.get("tag", "pmk-moe2-mock")
TPN = int(mesh.get_num_devices())
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
g = torch.Generator().manual_seed(int(args.get("seed", 0)))
os.environ["GLM53_PF_MOE_FID"] = str(args.get("fid", ""))
lv = M2.Levers(x_l1=bool(int(args.get("x_l1", 1 if CR <= 2048 else 0))), mcast=int(args.get("mcast", 4)), shared=bool(int(args.get("shared", 1))),
               ids_u16=bool(int(args.get("ids_u16", 1))), out_bfp8=bool(int(args.get("out_bfp8", 0))))  # fmt: skip

perm = gate_up_tile_perm(2 * W // 32)
gus, dns = [], []
t0 = time.time()
for e in range(E):
    gu = torch.cat([torch.randn(H, W, generator=g), torch.randn(H, W, generator=g)], dim=1) * H**-0.5
    gu = gu.reshape(H, 2 * W // 32, 32)[:, perm, :].reshape(H, 2 * W)
    dn = torch.randn(W, H, generator=g) * W**-0.5
    gus.append(ttnn.from_torch(gu, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=_nd_dram(H, 64), mesh_mapper=repl))  # fmt: skip
    dns.append(ttnn.from_torch(dn, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=_nd_dram(W, 64), mesh_mapper=repl))  # fmt: skip
S.log(f"weights on device in {time.time() - t0:.1f} s")

x_h = torch.randn(1, 1, CR, H, generator=g) * 0.5
idx_h = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(CR)]).reshape(1, 1, CR, K)
sel_h = torch.rand(1, 1, CR, K, generator=g) + 0.05
dev = lambda t_, dt: ttnn.from_torch(t_, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=PF.DRAM, mesh_mapper=repl)  # noqa: E731
x, idx, sel = dev(x_h, ttnn.bfloat16), dev(idx_h.to(torch.int32), ttnn.uint16), dev(sel_h, ttnn.float32)

blocks = tuple(int(v) for v in str(args.get("blocks", "")).split(",")) if args.get("blocks") else None
lv = M2.fit_levers(M2.Geometry(c=CR, w=W, p=110, h_split=4), lv)  # the TP2 pair drops lever 2 standalone
pm = M2.PmkMoe2(mesh, levers=lv, num_experts=E, top_k=K, hidden=H, limit=LIMIT, gate_up=gus, down=dns, blocks=blocks,
                fid=str(args.get("fid", "")))  # fmt: skip
geo = pm.geometry(CR)
mg, kc, nc = pm.blocks(CR)
plan = pm.plan(CR)
model = M2.streaming_model2(geo, mg, lv, pm.fid)
S.log(f"grid P={pm.p} W_T={pm.w_t} levers={lv} rects={len(pm.rects)} blocks={(mg, kc, nc)} CB moe={plan['moe']} "
      f"xs={plan['xs']} total={plan['total']} B member_fits={plan['member_fits']} groups={[(m, t) for m, t, _ in plan['groups']]}")  # fmt: skip
S.log(f"streaming model v2: { {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in model.items()} }")

x_ext, ids_rm, wt = PM.inputs_from_op_path(x, idx, sel, 1.0)
if lv.ids_u16:
    ids_rm, wt = M2.tail_layout_inputs(idx, sel, 1.0)
t0 = time.time()
y = pm(x_ext, ids_rm, wt, CR, x_tile=x)  # the compile launch (mock: JIT build + validation only)
ttnn.synchronize_device(mesh)
S.log(f"pmk_moe2 program launched in {time.time() - t0:.1f} s (calls {pm.calls})")
row = dict(test=f"pmk_moe2 mock build W={W} C={CR} E={E} tp{TPN} P={pm.p} G={lv.mcast} bfp8={int(lv.out_bfp8)}", milestone="PMK", status="pass" if C.MOCK else "info",
           blocks=[mg, kc, nc], cb_moe=plan["moe"], cb_total=plan["total"], member_fits=plan["member_fits"], rects=len(pm.rects),
           levers=dict(x_l1=lv.x_l1, mcast=lv.mcast, shared=lv.shared, ids_u16=lv.ids_u16, out_bfp8=lv.out_bfp8), fid=pm.fid, mock=C.MOCK, tag=tag,
           model={k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in model.items()})  # fmt: skip
if int(args.get("p13", 1)):
    # P13: the shared slice [H, 2 W_s] bfp8 gate_up + [W_s, H] down as two PfStream programs on 64 members (an exact
    # NoC rectangle) with the 2 x NST streamers off the rectangle
    t0 = time.time()
    members = [(x_, y_) for y_ in range(0, 8) for x_ in range(0, 8)]
    streamers = [(x_, y_) for y_ in range(8, 10) for x_ in range(0, 8)] + [(8, y_) for y_ in range(0, 8)]
    ws = W  # the shared slice width = the XSL width at this TP
    p13 = M2.SharedP13(mesh, torch.randn(H, 2 * ws, generator=g) * H**-0.5, torch.randn(ws, H, generator=g) * ws**-0.5,
                       members=members, streamers=streamers, nw=32 if 2 * ws // 32 >= 32 else 2 * ws // 32)  # fmt: skip
    xg = ttnn.from_torch(torch.randn(1, 1, 64 * 32, H, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=PF.DRAM, mesh_mapper=repl)  # fmt: skip
    xa = ttnn.from_torch(torch.randn(1, 1, 64 * 32, ws, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=PF.DRAM, mesh_mapper=repl)  # fmt: skip
    p13.gu(xg)
    p13.dn(xa)
    ttnn.synchronize_device(mesh)
    row["p13"] = dict(built=True, footprint=p13.footprint(), wall_s=round(time.time() - t0, 1))
    S.log(f"P13 PfStream pair built + launched in {time.time() - t0:.1f} s: {p13.footprint()}")
if not C.MOCK:
    import types

    class _Tpp:
        def __init__(self):
            self.mesh, self.ndev, self.repl = mesh, TPN, repl

    moe = types.SimpleNamespace(num_experts=E, top_k=K, hidden=H, intermediate=W, limit=LIMIT, scaling=1.0, gate_up=[None] * E, down=[None] * E)  # fmt: skip
    y_ref = PF.EpMoe(_Tpp(), moe, xsl=(TPN, gus, dns))(x, idx, sel)
    ttnn.synchronize_device(mesh)
    a = ttnn.to_torch(ttnn.get_device_tensors(y)[0] if TPN > 1 else y).float()
    b = ttnn.to_torch(ttnn.get_device_tensors(y_ref)[0] if TPN > 1 else y_ref).float()
    row.update(pcc=C.pcc(a, b), bitwise=bool(torch.equal(a, b)), maxabs=float((a - b).abs().max()))
    row["status"] = "pass" if row["bitwise"] or row["pcc"] >= 0.9999 else "fail"
S.log(row)
if args.get("record", True):
    C.record("tests", row)
RESULT = row
