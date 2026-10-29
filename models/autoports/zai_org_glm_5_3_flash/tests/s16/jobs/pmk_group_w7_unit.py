# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; pair or 2x4; program 2026-10-26 20:33:37Z (A), rulings H1-H8 20:36:28Z): THE GROUP FORM's W7 half --
W7-A(k) (the SPLIT W7 program, pmk_assembly A15M_SCOPE: A15 + P11 + P12 + P13, program B not in it) per chunk k of n
consecutive c-row chunks of ONE request, each writing its MoE-half outputs (P11x's x_ext rows, the router tail's ids /
weights, P13's y_s) at row offset k c of the group's buffers (runtime args only, H3), then program B over n c rows ONCE
(its own program, H6), vs the SAME split programs at n = 1 (chunk k at row 0 of a c-row group, B over c rows after each).

Criterion (H2): BITWISE per row -- every chunk's x_ext / ids / weights / y_s row block and B's y rows [k c, (k + 1) c) of the
group equal the n = 1 run's (B is per token: every (pair, row) keeps its K-chunk order and fidelity). The KDA state carries
chunk 0 -> chunk 1 through the layer's scratch in both runs (the same A order). Both runs TRACED (one capture each, the
compile launches before), `det` replays of each with the group / reference buffers POISONED (NaN / 0xFFFF) before every
replay (host writes outside the traces): any row a program did not write shows. H4: full chunks only (n_valid = c).

ARGS: layer (4, a sparse_kda layer), rows (2048 = c), n (2), det (20), b_cfg (program B knobs, as pmk_fuse3_unit), pfs_park
(4), tag, record."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe2 as M2  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PSTK  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16, EpMoe  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
L = int(args.get("layer", 4))
N = int(args.get("rows", 2048))
NG = int(args.get("n", 2))
DET = int(args.get("det", 20))
tag = args.get("tag", f"pmk-group-w7-L{L}")
NDEV = int(mesh.get_num_devices())
MOCK = bool(C.MOCK)
assert NG >= 2, NG
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ""
os.environ.setdefault("PMK_XEXT", "1")  # P11x writes x_ext (the group's B reads it)
os.environ.setdefault("GLM53_PMK_ARGS", "table")
S.set_attn(args)
t0 = time.time()
ent = S.build(mesh, ctx, [L], b=1, buckets=(N,), max_seq=N * NG + 512)
pf = ent["pf"]
pl = pf.layers[0]
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
assert sp and c == N, (sp, c, N)
ci = pf.chunk_inputs(c)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)  # H4: full chunks only
rows0, _ = S.golden_layer(L, N)
rows0 = rows0.reshape(N, 4, 4096).float()
ins = [rows0] + [torch.roll(rows0, shifts=(k * 997) % N, dims=0) for k in range(1, NG)]  # n different chunks
x0 = [pf.streams_to_device(r_, sp) for r_ in ins]  # pristine (outside every trace)
xp = [pf.streams_to_device(r_, sp) for r_ in ins]  # chunk k's residual (its address is in its programs' args)
kda = pl.kda
state_p = kda.allocate_state(1)
ttnn.copy(
    ttnn.ones(tuple(state_p.slot_mask.shape), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh), state_p.slot_mask
)
kw = dict(probe=False, pfs_park=int(args.get("pfs_park", 4)))
if NDEV == 8:
    kw["tp8_resident"] = False
S.log(f"{tag}: pf ready in {time.time() - t0:.1f} s (tp{NDEV}, c={c}, n={NG})")

# program B (one object: B(c) for the reference, B(n c) for the group) -- pmk_fuse3_unit's construction
ctl = ASM.PmkProgramA(pf, 0, c, sp, xp[0], state_p, ci.valid_end, a34_merge=False, a2_fuse=False, o_rects=0, **kw)
ep = getattr(pl, "epmoe", None) or EpMoe(pf, pl.moe, xsl=getattr(pl, "xsl", None))
pl.epmoe = ep
bcfg = PSTK.b_config({}, tp=NDEV)
if not int(bcfg.get("out_bfp8", 0)):
    bcfg.update(yw_db=0, yw_coal=1, comb_split=0, yw_brisc=0)
bcfg["ids_u16"] = 1  # B reads the router tail's [C, 8] uint16 ids / bf16 weights
for k_, v_ in (args.get("b_cfg") or {}).items():
    bcfg[str(k_)] = tuple(int(x) for x in (v_.split(",") if isinstance(v_, str) else v_)) if k_ == "blocks" else v_
G_ = int(bcfg["g"])
lv_ = M2.Levers(x_l1=bool(bcfg["x_l1"]), mcast=G_, shared=True, ids_u16=True, out_bfp8=bool(bcfg["out_bfp8"]),
                yw_db=bool(bcfg["yw_db"]), yw_coal=int(bcfg["yw_coal"]), ring_slots=int(bcfg["ring_slots"]),
                gu_split=bool(bcfg["gu_split"]), comb_split=bool(bcfg["comb_split"]), yw_brisc=bool(bcfg["yw_brisc"]),
                **({k_: bool(int(bcfg.get(k_, 0))) for k_ in ("valid", "snake", "xs_rows") if k_ in M2.Levers.__dataclass_fields__}))  # fmt: skip
W_ = int(ep.down[0].shape[-2])
lv_ = M2.fit_levers(M2.Geometry(c=NG * c, w=W_, p=110, h_split=int(bcfg["h_split"])), lv_)  # fitted at the group's rows
assert not lv_.x_l1, "the group's B reads x_ext (PMK_XEXT, x_l1 0)"
strm_ = None
rc_ = ctl.engine.place.role_cores()
if NDEV == 8:
    strm_ = M2.adj_pmk_streamers(mesh, members=list(ctl.members), avoid=list(rc_["lane"]) + list(rc_["tail"]))
b_self_kw = {}
if int(
    bcfg.get("selfstream", 0)
):  # the served W7's B7g self-stream form (pmk_fuse3_unit's construction): roots + MM-only
    # first (the routing plan runs on the first plan_cores), then the members; no streamers
    b_self_kw = dict(selfstream=True, cores=[tuple(c_) for r_ in ("root", "mm_only", "member") for c_ in rc_[r_]])
    if int(bcfg.get("plan_cores", 0) or 0):
        b_self_kw["plan_cores"] = int(bcfg["plan_cores"])
    if bcfg.get("ffn_fid"):
        b_self_kw["ffn_fid"] = str(bcfg["ffn_fid"])
    if "blocks" not in (args.get("b_cfg") or {}):
        bcfg["blocks"] = (4, 8, 8)
    strm_ = None
b = M2.ProgramB(mesh, g_size=G_, levers=lv_, streamers=strm_, num_experts=ep.e_all, top_k=int(pl.moe.top_k), hidden=int(pl.moe.hidden),
                limit=float(pl.moe.limit), gate_up=ep.gate_up, down=ep.down, blocks=bcfg["blocks"], fid=str(bcfg["fid"]),
                probe=False, prefetch=bool(bcfg["prefetch"]), full_sync=bool(bcfg["full_sync"]), rect_mode=str(bcfg["rect"]),
                gmax=int(bcfg["gmax"]), mcast_linked=bool(bcfg["mcast_linked"]), mcast_unicast=bool(bcfg["mcast_unicast"]),
                mcast_dual=bool(bcfg["mcast_dual"]), lpt=int(bcfg["lpt"]), gu_split=bool(bcfg["gu_split"]),
                h_split=int(bcfg["h_split"]), inplace=bool(bcfg["inplace"]), lpt_w=None, comb_split=bool(bcfg["comb_split"]),
                yw_brisc=bool(bcfg["yw_brisc"]), compute_only=list(ctl.members), **b_self_kw)  # fmt: skip
K_, H_ = int(pl.moe.top_k), int(pl.moe.hidden)
dram = ttnn.DRAM_MEMORY_CONFIG


def grp_bufs(rows: int) -> dict:
    """the group's buffers = program B(rows)'s inputs (H1: in serving these alias the per-chunk transients they replace)"""
    rep_ = ttnn.ReplicateTensorToMesh(mesh) if NDEV > 1 else None
    # x_ext: rows [rows, rows + 32) = program B's gather PAD row block (zero; P11x writes rows [0, rows) only)
    return dict(xext=ttnn.from_torch(torch.zeros(rows + 32, H_), dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                     memory_config=dram, mesh_mapper=rep_),
                rt_ids=ttnn.empty((1, rows * K_), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, mesh, dram),
                rt_wt=ttnn.empty((1, rows * K_), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh, dram),
                y_s=b.scratch(rows)["y_s"])  # fmt: skip


G1, G2 = grp_bufs(c), grp_bufs(NG * c)
bw = dict(b=b, x_ext=None, ids_rm=None, wt=None, x_tile=None, y_s=None)
split = dict(a15_fuse=True, b_p11=True, b_p13=True, b_p12=True, b_p15=False, b_split=True, b_window=bw)
t1 = time.time()
A1, A2 = [], []
for k in range(NG):  # the reference: chunk k at row 0 of the c-row group; the group: chunk k at row k c
    A1.append(ASM.PmkProgramA(pf, 0, c, sp, xp[k], state_p, ci.valid_end, grp=dict(row=0, **G1),
                              reuse=(A1[0] if k else None), **split, **kw))  # fmt: skip
for k in range(NG):
    A2.append(ASM.PmkProgramA(pf, 0, c, sp, xp[k], state_p, ci.valid_end, grp=dict(row=k * c, **G2),
                              reuse=A1[0], **split, **kw))  # fmt: skip
assert all(tuple(a_.scopes[0]) == tuple(ASM.A15M_SCOPE) for a_ in A1 + A2), [a_.scopes for a_ in A1 + A2]
# 14's B1 (21:00:25Z): the chunk-k WRITE-ADDRESS check on the host, every geometry (mock included): P13 down's output tile
# rows, P11x's x_ext rows, the router tail's ids / weights byte offset = chunk k's block of the group buffers, nothing past
WCHK = []
for k, a_ in enumerate(A2):
    exp_tiles = (k * c // 32, (k + 1) * c // 32)
    got = dict(p13_tiles=tuple(getattr(a_.sh_dn, "out_tile_rows", (None, None))), xext_rows=tuple(a_.report.get("grp_xext_rows") or ()),
               rtail_off=a_.report.get("grp_rtail_byte_off"))  # fmt: skip
    exp = dict(p13_tiles=exp_tiles, xext_rows=(k * c, (k + 1) * c), rtail_off=k * c * K_ * 2)
    WCHK.append(dict(k=k, got=got, exp=exp, ok=got == exp))
assert all(w_["ok"] for w_ in WCHK), ("the group's chunk write addresses", WCHK)
S.log(f"{tag}: chunk write-address check PASS {WCHK}")
S.log(f"{tag}: {2 * NG} W7-A programs built in {time.time() - t1:.1f} s; scope {list(ASM.A15M_SCOPE)}; "
      f"grp reports {[a_.report.get('grp') for a_ in A1 + A2]}; pfs geometry {A2[-1].report.get('pfs_geometry')}")  # fmt: skip

# persistent copies of the reference's per-chunk outputs (allocated before any capture)
REF = [dict(y=ttnn.empty((1, 1, c, H_), BF16, ttnn.TILE_LAYOUT, mesh, dram),
            xext=ttnn.empty((c + 32, H_), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh, dram),
            rt_ids=ttnn.empty((1, c * K_), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, mesh, dram),
            rt_wt=ttnn.empty((1, c * K_), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh, dram),
            y_s=ttnn.empty(tuple(G1["y_s"].shape), BF16, G1["y_s"].layout, mesh, dram)) for _ in range(NG)]  # fmt: skip
_zeros = {}


def zero(t_):
    key = (tuple(t_.shape), t_.dtype, t_.layout)
    if key not in _zeros:
        _zeros[key] = ttnn.zeros(tuple(t_.shape), dtype=t_.dtype, layout=t_.layout, device=mesh, memory_config=dram)
    ttnn.copy(_zeros[key], t_)


def launch(prog):
    prog.layer.engine.programs = [set(s_) for s_ in prog.scopes]
    prog.layer.order = [set(s_) for s_ in prog.scopes]
    prog(prog.x, ci)


def reset():
    for k in range(NG):
        ttnn.copy(x0[k], xp[k])
    zero(state_p.S)
    zero(state_p.C)


def run_ref():  # n = 1: A(k), then B(c) on the c-row group, per chunk; the outputs copied out per chunk
    reset()
    for k in range(NG):
        launch(A1[k])
        y_ = b(G1["xext"], G1["rt_ids"], G1["rt_wt"], c)
        ttnn.copy(y_, REF[k]["y"])
        for nm_ in ("xext", "rt_ids", "rt_wt", "y_s"):
            ttnn.copy(G1[nm_], REF[k][nm_])


def run_grp():  # n = NG: A(0) .. A(n - 1) into the n c-row group, then B(n c) ONCE
    reset()
    for k in range(NG):
        launch(A2[k])
    b(G2["xext"], G2["rt_ids"], G2["rt_wt"], NG * c)


def host(t_):
    return [ttnn.to_torch(d_).clone() for d_ in (ttnn.get_device_tensors(t_) if NDEV > 1 else [t_])]


def poison():  # host writes outside the traces: every buffer a run must rewrite
    for t_ in (
        [G1[k_] for k_ in ("xext", "rt_ids", "rt_wt", "y_s")]
        + [G2[k_] for k_ in ("xext", "rt_ids", "rt_wt", "y_s")]
        + [b.scratch(c)["y"], b.scratch(NG * c)["y"]]
        + [r_[k_] for r_ in REF for k_ in r_]
    ):
        sh_ = tuple(int(v) for v in t_.shape)
        if t_.dtype == ttnn.uint16:
            v_ = torch.full(sh_, 0xFFFF, dtype=torch.int32)
        else:
            v_ = torch.full(sh_, float("nan"))
            if (
                t_.layout == ttnn.ROW_MAJOR_LAYOUT
                and len(sh_) == 2
                and int(sh_[0]) % 32 == 0
                and int(sh_[-1]) == H_
                and any(t_ is b_["xext"] for b_ in (G1, G2))
            ):  # keep B's gather pad rows zero
                v_[-32:] = 0.0
        rep_ = ttnn.ReplicateTensorToMesh(mesh) if NDEV > 1 else None
        h_ = ttnn.from_torch(v_, dtype=t_.dtype, layout=t_.layout, device=mesh, memory_config=dram, mesh_mapper=rep_)
        ttnn.copy(h_, t_)
        ttnn.deallocate(h_)


def compare() -> dict:
    """per chip: chunk k's group rows vs the n = 1 reference's, BITWISE, for B's y and the four W7-A outputs"""
    y2 = host(b.scratch(NG * c)["y"])
    g2 = {nm_: host(G2[nm_]) for nm_ in ("xext", "rt_ids", "rt_wt", "y_s")}
    out = dict(bitwise=True, rows={})
    for k in range(NG):
        rk = {nm_: host(REF[k][nm_]) for nm_ in ("y", "xext", "rt_ids", "rt_wt", "y_s")}
        res = {}
        for d_ in range(len(y2)):
            yg = y2[d_].reshape(NG * c, H_)[k * c : (k + 1) * c]
            yr = rk["y"][d_].reshape(c, H_)
            ok = {"y": torch.equal(yg, yr) and bool(torch.isfinite(yr.float()).all())}
            ok["xext"] = torch.equal(
                g2["xext"][d_].reshape(-1, H_)[k * c : (k + 1) * c], rk["xext"][d_].reshape(-1, H_)[:c]
            )
            for nm_ in ("rt_ids", "rt_wt"):
                ok[nm_] = torch.equal(
                    g2[nm_][d_].reshape(-1)[k * c * K_ : (k + 1) * c * K_], rk[nm_][d_].reshape(-1)[: c * K_]
                )
            ok["y_s"] = torch.equal(
                g2["y_s"][d_].reshape(-1, H_)[k * c : (k + 1) * c], rk["y_s"][d_].reshape(-1, H_)[:c]
            )
            if not ok["y"]:
                diff_ = (yg.float() - yr.float()).abs()
                ok["y_maxabs"] = float(torch.nan_to_num(diff_, nan=1e30).max())
                ok["y_rows_diff"] = int((torch.nan_to_num(diff_, nan=1e30) > 0).any(-1).sum())
            res[d_] = ok
            out["bitwise"] = out["bitwise"] and all(v_ for k_, v_ in ok.items() if isinstance(v_, bool))
        out["rows"][k] = res
    return out


row = dict(test=f"the GROUP form's W7 half: W7-A(k) (A15 + P11 + P12 + P13, split) x {NG} chunks at row offsets k c + program "
                f"B over {NG} x {c} rows vs the same programs at n = 1 (B over c rows per chunk), L{L} tp{NDEV}, traced; "
                f"criterion BITWISE per row (H2) + {DET} poisoned replays", milestone="16-GROUP-A", layer=L, tp=NDEV, c=c,
           n=NG, tag=tag, mock=MOCK, scope=list(ASM.A15M_SCOPE), grp=[a_.report.get("grp") for a_ in A1 + A2])  # fmt: skip
try:
    S.log(f"{tag}: compile launches (ref, then group)")
    run_ref()
    ttnn.synchronize_device(mesh)
    run_grp()
    ttnn.synchronize_device(mesh)
    if MOCK:
        row["status"] = "mock"
    else:
        tids = {}
        for nm_, fn_ in (("ref", run_ref), ("grp", run_grp)):
            t_ = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                fn_()
            finally:
                ttnn.end_trace_capture(mesh, t_, cq_id=0)
            tids[nm_] = t_
        dets, ok_all, times = [], True, {"ref": [], "grp": []}
        for i in range(max(1, DET)):
            poison()
            ttnn.synchronize_device(mesh)
            for nm_ in ("ref", "grp"):
                ts_ = time.perf_counter()
                ttnn.execute_trace(mesh, tids[nm_], cq_id=0, blocking=True)
                times[nm_].append((time.perf_counter() - ts_) * 1e6)
            cmp_ = compare()
            ok_all = ok_all and cmp_["bitwise"]
            if i == 0 or not cmp_["bitwise"]:
                dets.append(dict(replay=i, **cmp_))
            if not cmp_["bitwise"] and len(dets) > 3:
                break
        for t_ in tids.values():
            ttnn.release_trace(mesh, t_)
        med = lambda v: round(sorted(v)[len(v) // 2], 1) if v else None  # noqa: E731
        row.update(bitwise=ok_all, det_replays=DET, first=dets[0] if dets else None, fails=dets[1:],
                   us_ref_median=med(times["ref"]), us_grp_median=med(times["grp"]),
                   status="pass" if ok_all else "fail")  # fmt: skip
        S.log(f"{tag}: bitwise {ok_all} over {DET} poisoned replays; ref {row['us_ref_median']} us vs group "
              f"{row['us_grp_median']} us (host wall per traced replay); first {str(row['first'])[:600]}")  # fmt: skip
        if bool(args.get("record", True)):
            C.record("pcc", dict(test=row["test"], milestone=row["milestone"], layer=L, tp=NDEV, c=c, n=NG, bitwise=ok_all,
                                 det=DET, status=row["status"], tag=tag))  # fmt: skip
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    row.update(status="fail", error=f"{type(e).__name__}: {e}"[:1500])
RESULT = row
print("PMK_GROUP_W7_UNIT", row, flush=True)
