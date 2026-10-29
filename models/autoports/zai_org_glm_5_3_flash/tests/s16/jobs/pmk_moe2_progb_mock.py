# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 2x4 MOCK or any mesh): PROGRAM B (pmk/moe2 README sec 12): the PMK MoE v2 phase set with the expert
slices streamed by the 24 bank-adjacent cores (exp_streamer.hpp, contiguous per bank) and multicast to G-core
rectangles of the other 86 cores (PMK_MCAST_EXT); otherwise the v2 phase set (tt/pmk_moe2.py: levers 2 / 3 / 5) built and
launched ONCE on synthetic XSL expert slices (TP8: 256 wide, TP2: 1,024) at C rows with random routing, plus the P13
shared-expert PfStream pair (7a's PfStreamMM) built on the same mesh -- in the mock the kernels JIT-build, the CBs
allocate (cb_guard + the v2 ledger) and the programs validate; on a device the launch is the compile launch and the
output is compared with EpMoe (xffn forced) on the same inputs (PCC).
ARGS: experts (288), width (256), c (2048), k (8), seed (0), blocks ("" = the budget's choice), mcast (4), shared (1),
ids_u16 (1), p13 (1), fid (""), prefetch (0: PMK_STREAM_PREFETCH), probe (0: the streamer / member probe rows), tag,
record."""
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
tag = args.get("tag", "pmk-moe2-progb-mock")
TPN = int(mesh.get_num_devices())
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
g = torch.Generator().manual_seed(int(args.get("seed", 0)))
os.environ["GLM53_PF_MOE_FID"] = str(args.get("fid", ""))
lv = M2.Levers(x_l1=bool(int(args.get("x_l1", 1 if CR <= 2048 else 0))), mcast=int(args.get("mcast", 4)), shared=bool(int(args.get("shared", 1))),
               ids_u16=bool(int(args.get("ids_u16", 1))), out_bfp8=bool(int(args.get("out_bfp8", 0))), yw_db=bool(int(args.get("yw_db", 0))), yw_coal=int(args.get("yw_coal", 1)),
               ring_slots=int(args.get("ring_slots", 2)))  # fmt: skip

perm = gate_up_tile_perm(2 * W // 32)
gus, dns, gus_h, dns_h = [], [], [], []
t0 = time.time()
for e in range(E):
    gu = torch.cat([torch.randn(H, W, generator=g), torch.randn(H, W, generator=g)], dim=1) * H**-0.5
    gu = gu.reshape(H, 2 * W // 32, 32)[:, perm, :].reshape(H, 2 * W)
    dn = torch.randn(W, H, generator=g) * W**-0.5
    gus_h.append(gu)
    dns_h.append(dn)
    gus.append(ttnn.from_torch(gu, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=_nd_dram(H, 64), mesh_mapper=repl))  # fmt: skip
    dns.append(ttnn.from_torch(dn, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=_nd_dram(W, 64), mesh_mapper=repl))  # fmt: skip
S.log(f"weights on device in {time.time() - t0:.1f} s")

x_h = torch.randn(1, 1, CR, H, generator=g) * 0.5
idx_h = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(CR)]).reshape(1, 1, CR, K)
sel_h = torch.rand(1, 1, CR, K, generator=g) + 0.05
dev = lambda t_, dt: ttnn.from_torch(t_, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=PF.DRAM, mesh_mapper=repl)  # noqa: E731
x, idx, sel = dev(x_h, ttnn.bfloat16), dev(idx_h.to(torch.int32), ttnn.uint16), dev(sel_h, ttnn.float32)

blocks = tuple(int(v) for v in str(args.get("blocks", "")).split(",")) if args.get("blocks") else None
lv = M2.fit_levers(M2.Geometry(c=CR, w=W, p=86, h_split=4), lv)
import faulthandler  # noqa: E402

faulthandler.dump_traceback_later(float(args.get("construct_s", 180)), repeat=True)  # the construction watchdog
t0 = time.time()
pm = M2.ProgramB(mesh, g_size=int(args.get("mcast", 2)), levers=lv, num_experts=E, top_k=K, hidden=H, limit=LIMIT, gate_up=gus, down=dns, blocks=blocks,
                fid=str(args.get("fid", "")), prefetch=bool(int(args.get("prefetch", 0))), probe=bool(int(args.get("probe", 0))),
                full_sync=bool(int(args.get("full_sync", 0))), rect_mode=str(args.get("rect_mode", "cover")), gmax=int(args.get("gmax", 4)),
                gu_rt=bool(int(args.get("gu_rt", 0))), gu_ct=int(args.get("gu_ct", 0)), mcast_linked=bool(int(args.get("mcast_linked", 0))), mcast_unicast=bool(int(args.get("mcast_unicast", 0))), mcast_dual=bool(int(args.get("mcast_dual", 0))),
                lpt=int(args.get("lpt", 0)), gu_split=bool(int(args.get("gu_split", 0))), h_split=int(args.get("h_split", 4)),
                inplace=bool(int(args.get("inplace", 1))), lpt_w=M2.lpt_weights_from(args.get("lpt_w")), comb_split=bool(int(args.get("comb_split", 0))), yw_brisc=bool(int(args.get("yw_brisc", 0))),
                selfstream=bool(int(args.get("selfstream", 0))))  # fmt: skip
S.log(f"program B constructed in {time.time() - t0:.1f} s")
t0 = time.time()
pages = pm.load_streams(gus_h, dns_h)
S.log(f"streams loaded in {time.time() - t0:.1f} s")
S.log(f"expert streams packed: {pages} pages ({pages * 576 / 1e6:.0f} MB), streamers {pm.streamers} rects {len(pm.rects)} "
      f"sizes {pm.rect_sizes} members {sum(pm.rect_sizes)} of {pm.p} compute cores (rect_mode {pm.rect_mode}, gmax {pm.gmax}, "
      f"gu_rt {pm.gu_rt} ct {pm.gu_ct}); rects {pm.rects}")
geo = pm.geometry(CR)
mg, kc, nc = pm.blocks(CR)
plan = pm.plan(CR)
model = M2.streaming_model2(geo, mg, lv, pm.fid)
SELF = isinstance(pm, M2.SelfStreamB)  # the TP8 form of record: no streamers / rectangles / phase table
pb = M2.program_b_plan(CR, pm.g_size, W, landing_bfp8=True, n_ffn=None if SELF else sum(pm.rect_sizes))
S.log(f"program B plan: {pb}")
S.log(f"grid P={pm.p} W_T={pm.w_t} levers={lv} rects={len(pm.rects)} blocks={(mg, kc, nc)} CB moe={plan['moe']} "
      f"xs={plan['xs']} total={plan['total']} B member_fits={plan['member_fits']} groups={[(m, t) for m, t, _ in plan['groups']]}")  # fmt: skip
S.log(f"streaming model v2: { {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in model.items()} }")

x_ext, ids_rm, wt = PM.inputs_from_op_path(x, idx, sel, 1.0)
if lv.ids_u16:
    ids_rm, wt = M2.tail_layout_inputs(idx, sel, 1.0)
t0 = time.time()
pt = None if SELF else pm.phase_parts(CR, x_tile=x, ids_rm=ids_rm, wt=wt, x_ext=x_ext)
S.log("self-stream form: no phase table" if SELF else f"phase table: { {k: v['what'][:60] for k, v in pt['phases'].items()} } sems={pt['sems']} cbs receiver="
      f"{[(i, f, pg, n, b) for i, f, pg, n, b in pt['cbs']['receiver']]} streamer={pt['cbs']['streamer']}")
y = pm(x_ext, ids_rm, wt, CR, x_tile=x)  # the compile launch (mock: JIT build + validation only)
ttnn.synchronize_device(mesh)
faulthandler.cancel_dump_traceback_later()
S.log(f"pmk_moe2 program launched in {time.time() - t0:.1f} s (calls {pm.calls})")
row = dict(members=sum(pm.rect_sizes), rect_sizes=pm.rect_sizes, progb=dict(n_ffn=pb["n_ffn"], n_pure=pb["n_pure"], roles={k: (v["total"], v["fits"]) for k, v in pb["roles"].items()},
                      dram_us=round(pb["dram_us"]), floor_us=round(pb["span_floor_us"]), math=round(pb["math_ceiling"], 3)),
           test=f"pmk_moe2 PROGRAM-B mock build W={W} C={CR} E={E} tp{TPN} P={pm.p} G={pm.g_size} bfp8={int(lv.out_bfp8)} prefetch={int(pm.prefetch)} probe={int(pm.probe)} full_sync={int(pm.full_sync)} {pm.rect_mode} sizes={pm.rect_sizes} gu_rt={int(pm.gu_rt)} ct={pm.gu_ct} yw_db={int(lv.yw_db)} linked={int(pm.mcast_linked)} unicast={int(pm.mcast_unicast)} dual={int(pm.mcast_dual)} coal={lv.yw_coal} lpt={pm.lpt} slots={lv.ring_slots} split={int(pm.lv.gu_split)} hs={pm.h_split} inplace={int(pm.inplace)}", milestone="PMK", status="pass" if C.MOCK else "info",
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
