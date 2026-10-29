# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, any mesh): Stage 16 -- the prefill EpMoe on SYNTHETIC XSL expert slices of a chosen width, so a pair
exercises the TP8 geometry: TP8's slice of I = 2,048 is 256 = 8 tiles -> moe_fused_swiglu on an 8 x 8 grid (EpMoe
mfs_grid), never run on device before 14b's TP8 service would.
Random bfloat4_b gate_up [4096, 2 W] (tile-interleaved g|u,
as the permuted expert files) / down [W, 4096] per expert (the same on every chip), random rows x [C, 4096] bf16, random
routing (top-k distinct experts per row, scores > 0). EpMoe(xsl=) runs every expert locally; the routed output is
compared with the host fp32 product of the same block-float weights (read back from the device) on `check_rows` rows,
then the trace is soaked (>= 2,000 replays, program 03:20Z).

ARGS: experts (288), width (256: the per-chip slice of the intermediate), c (2048 rows), k (8), check_rows (64), seed
(0), moe_v2 (True), replays, parts (False: also trace + soak the prefixes up to the plan / dispatch / swiglu -> the
per-phase times as differences), grids ([]: moe_fused_swiglu core grids [x, y] to time, one trace + soak each, after
the main row), upto (None: the whole MoE; "plan" / "dispatch" / "swiglu" / "untilize": the main trace stops there -- no
output, no PCC -- the hang bisect's soak of one prefix), combine ("v1" | "rm": GLM53_PF_MOE_COMBINE), crm_split (2:
GLM53_PF_MOE_CRM_SPLIT), plan ("v2" | "mc": GLM53_PF_MOE_PLAN), plan_check (False: True = the v2 (one core) and mc
(multi-core) plans of the same routing compared table by table, bitwise, for the XSL layout and the EP layout of the
mesh (E / TP local experts per chip, lo = rank E / TP), then each plan's traced time after the main row), probe (False: xffn only -- the
in-program probe counters of a traced replay (GLM53_PF_MOE_XFFN_PROBE, kernels/prefill/moe_xffn_probe.hpp) into the
row; the mock compiles the probe program only), probe_replays (200), tag, record."""
import os
import sys
import types

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import _nd_dram, gate_up_tile_perm  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
E = int(args.get("experts", 288))
W = int(args.get("width", 256))
CR = int(args.get("c", 2048))
K = int(args.get("k", 8))
NCHK = int(args.get("check_rows", 64))
H = 4096
LIMIT = 10.0
tag = args.get("tag", "moe-xsl-bench")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
os.environ["GLM53_PF_MOE_V2"] = "1" if args.get("moe_v2", True) else "0"
# v2_dispatch / v2_combine: each half alone (EpMoe's GLM53_PF_MOE_V2_DISPATCH / _COMBINE; moe_v2 False to use them)
for half in ("dispatch", "combine"):
    if f"v2_{half}" in args:
        os.environ[f"GLM53_PF_MOE_V2_{half.upper()}"] = "1" if args[f"v2_{half}"] else "0"
if "xgather" in args:  # GLM53_PF_MOE_XGATHER: moe_fused_swiglu gathers x through the disp table (no dispatch)
    os.environ["GLM53_PF_MOE_XGATHER"] = "1" if args["xgather"] else "0"
os.environ["GLM53_PF_MOE_COMBINE"] = str(args.get("combine", ""))
os.environ["GLM53_PF_MOE_CRM_SPLIT"] = str(int(args.get("crm_split", 2)))
os.environ["GLM53_PF_MOE_PLAN"] = str(args.get("plan", "mc2"))
os.environ["GLM53_PF_MOE_FFN"] = str(args.get("ffn", ""))  # "xffn": one expert per core (XSL)
os.environ["GLM53_PF_MOE_XFFN_MG"] = str(int(args.get("xffn_mg", 4)))
os.environ["GLM53_PF_MOE_FID"] = str(args.get("moe_fid", ""))  # "" = HiFi2 (EpMoe's default); "LoFi"
os.environ["GLM53_PF_MOE_XFFN_FULLSYNC"] = "1" if args.get("xffn_fullsync") else "0"
os.environ["GLM53_PF_MOE_XFFN_DIAG"] = str(args.get("xffn_diag", ""))  # timing-only variants (outputs wrong)
os.environ["GLM53_PF_MOE_XFFN_XNOC1"] = "1" if args.get("xffn_xnoc1", True) else "0"
os.environ["GLM53_PF_MOE_XFFN_BLOCKS"] = str(args.get("xffn_blocks", ""))  # "mg,kc,nc" (default: the widest that fits)
os.environ["GLM53_PF_MOE_XFFN_XSPLIT"] = "1" if args.get("xffn_xsplit") else "0"  # x gather on both RISCs
os.environ["GLM53_PF_MOE_XFFN_SNAKE"] = "1" if args.get("xffn_snake") else "0"  # cost-class unit dealing
os.environ["GLM53_PF_MOE_XFFN_VALID"] = "1" if args.get("xffn_valid") else "0"  # valid rows only, x_ext unpadded
g = torch.Generator().manual_seed(int(args.get("seed", 0)))

WEIGHTS = args.get("weights", "synthetic")
ROUTING = args.get("routing", "uniform")
perm = gate_up_tile_perm(2 * W // 32)
gus, dns = [], []
if WEIGHTS == "real":  # the served XSL slices of a real layer: chip k = TP8 slice k (fused_decoder.load_xsl_experts)
    from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST
    from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoderConfig, load_xsl_experts
    from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

    XTP = int(args.get("xsl_tp", 8))
    assert W * XTP == 2048, ("real weights: width = I / xsl_tp", W, XTP)
    gus, dns = load_xsl_experts(FdTileCacheSource(CACHE_DIR, MANIFEST), int(args.get("layer", 4)), E, mesh,
                                FusedDecoderConfig().fused_experts_cache_dir, tp=XTP, slices=list(range(TPN)))  # fmt: skip
# synthetic slices: gate / up [H, W] -> the interleaved [H, 2W] file layout; down [W, H]
for e in range(E if WEIGHTS != "real" else 0):
    gate = torch.randn(H, W, generator=g) * H**-0.5
    up = torch.randn(H, W, generator=g) * H**-0.5
    gu = torch.cat([gate, up], dim=1).reshape(H, 2 * W // 32, 32)[:, perm, :].reshape(H, 2 * W)
    dn = torch.randn(W, H, generator=g) * W**-0.5
    gus.append(ttnn.from_torch(gu, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                               memory_config=_nd_dram(H, 64), mesh_mapper=repl))  # fmt: skip
    dns.append(ttnn.from_torch(dn, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                               memory_config=_nd_dram(W, 64), mesh_mapper=repl))  # fmt: skip


class _Tpp:  # what EpMoe reads of a TPPrefill
    def __init__(self):
        self.mesh, self.ndev, self.repl = mesh, TPN, repl


moe = types.SimpleNamespace(num_experts=E, top_k=K, hidden=H, intermediate=W, limit=LIMIT, scaling=1.0,
                            gate_up=[None] * E, down=[None] * E)  # fmt: skip
ep = PF.EpMoe(_Tpp(), moe, xsl=(TPN, gus, dns))
x_h = torch.randn(1, 1, CR, H, generator=g) * 0.5
if ROUTING == "zipf":  # skewed expert popularity (p_e ~ 1 / rank^s over a random expert order): load-imbalance stress
    pop = (1.0 / (torch.arange(E, dtype=torch.float32) + 1) ** float(args.get("zipf_s", 1.0)))[
        torch.randperm(E, generator=g)
    ]
    idx_h = torch.stack([torch.multinomial(pop, K, replacement=False, generator=g) for _ in range(CR)]).reshape(
        1, 1, CR, K
    )
else:
    idx_h = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(CR)]).reshape(1, 1, CR, K)
sel_h = torch.rand(1, 1, CR, K, generator=g) + 0.05
x = ttnn.from_torch(x_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=PF.DRAM,
                    mesh_mapper=repl)  # fmt: skip
idx = ttnn.from_torch(idx_h.to(torch.int32), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=mesh,
                      memory_config=PF.DRAM, mesh_mapper=repl)  # fmt: skip
sel = ttnn.from_torch(sel_h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=PF.DRAM,
                      mesh_mapper=repl)  # fmt: skip
Y = ttnn.from_torch(torch.zeros(1, 1, CR, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                    memory_config=PF.DRAM, mesh_mapper=repl)  # fmt: skip


UPTO = args.get("upto")
PLAN_CHECK = bool(args.get("plan_check", False))
PLAN_MODES = ["v2"] + [m_ for m_ in args.get("plan_modes", ["mc", "mc2"])]
plan_rows, plan_cfgs, plan_bufs = [], [], {}
if PLAN_CHECK:  # before any capture: every plan program compiles here
    n_ = CR * K
    ids_t = ttnn.transpose(ttnn.typecast(idx, ttnn.uint32), -2, -1)
    ids_rm = ttnn.to_layout(ttnn.reshape(ids_t, (1, n_)), ttnn.ROW_MAJOR_LAYOUT, memory_config=PF.DRAM)
    ttnn.deallocate(ids_t)
    keep = (ep.e_loc, ep.lo, ep.ep, ep.plan_mode, ep.v2, ep.bufs)
    plan_cfgs.append(("xsl", E, ep.lo))
    if TPN > 1:
        lo_h = torch.zeros(TPN, 32, dtype=torch.int32)
        lo_h[:, 0] = torch.arange(TPN, dtype=torch.int32) * (E // TPN)
        lo_ep = ttnn.from_torch(lo_h, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                memory_config=PF.DRAM, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))  # fmt: skip
        plan_cfgs.append(("ep", E // TPN, lo_ep))

    def use(e_loc, lo_t):
        ep.e_loc, ep.lo, ep.ep, ep.v2, ep.bufs = e_loc, lo_t, -(-e_loc // 32) * 32, True, {}

    host = lambda t_: [
        ttnn.to_torch(d_).clone() for d_ in (ttnn.get_device_tensors(t_) if TPN > 1 else [t_])
    ]  # noqa: E731
    for name, e_loc, lo_t in plan_cfgs:
        use(e_loc, lo_t)
        tabs = {}
        for mode in PLAN_MODES:
            ep.plan_mode = mode
            bb = ep.plan(ids_rm, CR)
            ttnn.synchronize_device(mesh)
            tabs[mode] = {k_: host(bb[k_]) for k_ in ("disp", "dest", "cnt", "start", "meta")}
        plan_bufs[name] = ep.bufs  # the timing captures reuse them: no allocation (write) inside a capture
        diffs = {f"{m_}.{k_}": sum(int((a_ != b_).sum()) for a_, b_ in zip(tabs["v2"][k_], tabs[m_][k_]))
                 for m_ in PLAN_MODES[1:] for k_ in tabs["v2"]}  # fmt: skip
        plan_rows.append(dict(cfg=name, e_loc=e_loc, bitwise=not any(diffs.values()), diffs=diffs,
                              rows_used=[int(m_.reshape(-1)[0]) for m_ in tabs["v2"]["meta"]]))  # fmt: skip
        S.log(plan_rows[-1])
    ep.e_loc, ep.lo, ep.ep, ep.plan_mode, ep.v2, ep.bufs = keep


def run():
    y = ep(x, idx, sel, upto=UPTO)
    if y is not None:
        ttnn.copy(y, Y)
        ttnn.deallocate(y)


MHZ = 1350.0  # the RISC wall clock (AICLK), as Stage 14b's probes


def xffn_probe(replays: int) -> dict:
    """GLM53_PF_MOE_XFFN_PROBE: the probe program compiled (eager), captured, replayed `replays` times; the counters of
    the last replay (kernels/prefill/moe_xffn_probe.hpp), per chip: per counter the median / max over the P cores in
    us, the span, the bytes moved and the DRAM bandwidth they imply over the slowest core's span"""
    os.environ["GLM53_PF_MOE_XFFN_PROBE"] = "1"
    try:
        run()
        ttnn.synchronize_device(mesh)
        t_ = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            run()
        finally:
            ttnn.end_trace_capture(mesh, t_, cq_id=0)
        st = S.replay_stats(mesh, t_, replays=replays, label="xffn probe")
        ttnn.release_trace(mesh, t_)
    finally:
        os.environ["GLM53_PF_MOE_XFFN_PROBE"] = "0"
    out = dict(replays=st.get("replays"), median_us_with_probe=st.get("median_us"), chips=[])
    rnames = {0: "gu_read", 1: "dn_read", 2: "gu_wait", 3: "dn_wait", 6: "tables"}
    wnames = {0: "x_read", 1: "x_wait", 2: "y_wait", 3: "y_write", 4: "ctl_wait", 5: "disp_read"}
    for chip in ep.xffn_probe_read():
        r, w = chip["reader"].float(), chip["writer"].float()
        us = lambda col: dict(
            med=round(float(col.median()) / MHZ, 1), max=round(float(col.max()) / MHZ, 1)
        )  # noqa: E731
        span = torch.maximum(r[:, 13], w[:, 13])
        kb = float(r[:, 5].sum() + w[:, 6].sum())
        d = dict(span_us=us(span), reader={n: us(r[:, i]) for i, n in rnames.items()},
                 writer={n: us(w[:, i]) for i, n in wnames.items()},
                 units=dict(min=int(r[:, 4].min()), max=int(r[:, 4].max()), total=int(r[:, 4].sum())),
                 read_mb=round(kb / 1024, 1), write_mb=round(float(w[:, 7].sum()) / 1024, 1),
                 weights_mb=round(float(r[:, 5].sum()) / 1024, 1), x_mb=round(float(w[:, 6].sum()) / 1024, 1),
                 read_gbps=round(kb * 1024 / (float(span.max()) / MHZ * 1e-6) / 1e9, 1),
                 start_skew_us=round(float(((r[:, 14] - r[:, 14].min()) % 2**32).max()) / MHZ, 1),
                 # the share of the span each RISC spends with DRAM traffic in flight / waiting on the compute
                 reader_read_frac=round(float(((r[:, 0] + r[:, 1]) / span).mean()), 3),
                 reader_wait_frac=round(float(((r[:, 2] + r[:, 3]) / span).mean()), 3),
                 writer_xread_frac=round(float((w[:, 0] / span).mean()), 3),
                 writer_ywait_frac=round(float((w[:, 2] / span).mean()), 3))  # fmt: skip
        out["chips"].append(d)
    S.log(f"xffn probe: {out}")
    return out


run()  # compile launch
ttnn.synchronize_device(mesh)
tid = None
if not C.MOCK:
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        run()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
if UPTO:  # the bisect: the prefix's soak only
    row = dict(test=f"moe xsl bench prefix up to {UPTO}: E {E} x width {W} (grid {ep.mfs_grid.x}x{ep.mfs_grid.y}), "
                    f"C {CR}, top-{K} tp{TPN} (hang bisect soak)", milestone="P2", upto=UPTO, experts=E, width=W,
               rows=CR, k=K, v2_dispatch=ep.v2_dispatch, v2_combine=ep.v2_combine, traced=tid is not None, tag=tag,
               **S.replay_stats(mesh, tid, replays=args.get("replays")))  # fmt: skip
    if tid is not None:
        ttnn.release_trace(mesh, tid)
    S.log(row)
    if rec and tid is not None:
        C.record("perf", row)
    for t_ in gus + dns + [x, idx, sel, Y]:
        ttnn.deallocate(t_)
    RESULT = [row]
else:  # the whole MoE: PCC vs the host reference, the soak, parts / grids
    yd = ttnn.to_torch(ttnn.get_device_tensors(Y)[0] if TPN > 1 else Y).float().reshape(CR, H)

    # host fp32 reference on check rows from the device's own block-float weights (chip 0's copy)
    dev0 = lambda t_: ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()  # noqa: E731
    rows = torch.linspace(0, CR - 1, NCHK).long()
    wnorm = sel_h[0, 0] / sel_h[0, 0].sum(-1, keepdim=True)
    need = sorted({int(e) for r in rows.tolist() for e in idx_h[0, 0, r].tolist()})
    inv = torch.argsort(torch.tensor(perm))
    wts = {}
    for e in need:
        gu = dev0(gus[e]).reshape(H, 2 * W // 32, 32)[:, inv, :].reshape(H, 2 * W)
        wts[e] = (gu[:, :W], gu[:, W:], dev0(dns[e]))
    xb = x_h[0, 0].to(torch.bfloat16).float()
    ref = torch.zeros(NCHK, H)
    for i, r in enumerate(rows.tolist()):
        for j in range(K):
            e = int(idx_h[0, 0, r, j])
            gt, ut, dt = wts[e]
            a = xb[r] @ gt
            b = xb[r] @ ut
            hsw = torch.nn.functional.silu(a.clamp(max=LIMIT)) * b.clamp(-LIMIT, LIMIT)
            ref[i] += wnorm[r, j] * (hsw.to(torch.bfloat16).float() @ dt)
    got = yd[rows]
    p = C.pcc(got, ref)
    row = dict(test=f"moe xsl bench: EpMoe on synthetic XSL slices E {E} x width {W} ({W // 32} tiles, grid "
                    f"{ep.mfs_grid.x}x{ep.mfs_grid.y}), C {CR} rows, top-{K}, vs host fp32 on {NCHK} rows tp{TPN}",
               milestone="P2", experts=E, width=W,
               grid=[ep.mfs_grid.x, ep.mfs_grid.y] if ep.ffn != "xffn" else [int(mesh.compute_with_storage_grid_size().x),
                                                                            int(mesh.compute_with_storage_grid_size().y)],
               weights=(f"real L{args.get('layer', 4)} TP{args.get('xsl_tp', 8)} XSL slices" if WEIGHTS == "real"
                        else "synthetic bfloat4_b"), routing=ROUTING + (f" s={args.get('zipf_s', 1.0)}" if ROUTING == "zipf" else ""),
               rows=CR, k=K, pcc=round(p, 6),
               rel_l2=round(float((got - ref).norm() / ref.norm()), 6), status="pass" if p >= 0.99 else "fail",
               moe_v2=os.environ["GLM53_PF_MOE_V2"] == "1", v2_dispatch=ep.v2_dispatch, v2_combine=ep.v2_combine,
               xgather=ep.xgather, combine=ep.combine, ffn=ep.ffn, moe_fid=os.environ["GLM53_PF_MOE_FID"] or "HiFi2",
               xffn_bases=ep._xffn_full_height_shards() if ep.ffn == "xffn" else None, crm_split=ep.crm_split if ep.combine == "rm" else None,
               traced=tid is not None, tag=tag,
               **S.replay_stats(mesh, tid, replays=args.get("replays")))  # fmt: skip
    if args.get("parts") and tid is not None:  # the same programs, stopped early (EpMoe upto): nothing new compiles
        pre = {}
        for u in ("plan", "dispatch", "swiglu", "untilize"):
            t_ = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                ep(x, idx, sel, upto=u)
            finally:
                ttnn.end_trace_capture(mesh, t_, cq_id=0)
            pre[u] = S.replay_stats(mesh, t_, replays=args.get("replays"))["median_us"]
            ttnn.release_trace(mesh, t_)
        pre["all"] = row["median_us"]
        order = ["plan", "dispatch", "swiglu", "untilize", "all"]
        row["prefix_us"] = pre
        if ep.ffn == "xffn":  # xffn returns after its FFN for any upto past the plan: name the phases by what runs
            order = ["plan", "dispatch", "all"]
            row["prefix_us"] = {"plan": pre["plan"], "ffn": pre["dispatch"], "all": pre["all"]}
            row["phase_us"] = {"plan": pre["plan"], "ffn": round(pre["dispatch"] - pre["plan"], 1),
                               "combine": round(pre["all"] - pre["dispatch"], 1)}  # fmt: skip
        else:
            row["phase_us"] = {k_: round(pre[k_] - (pre[order[i - 1]] if i else 0.0), 1) for i, k_ in enumerate(order)}
    if tid is not None:
        ttnn.release_trace(mesh, tid)
        tid = None
    if PLAN_CHECK:
        keep = (ep.e_loc, ep.lo, ep.ep, ep.plan_mode, ep.v2, ep.bufs)
        for pr, (name, e_loc, lo_t) in zip(plan_rows, plan_cfgs):
            for mode in PLAN_MODES if not C.MOCK else ():
                use(e_loc, lo_t)
                ep.bufs = plan_bufs[name]
                ep.plan_mode = mode
                t_ = ttnn.begin_trace_capture(mesh, cq_id=0)
                try:
                    ep.plan(ids_rm, CR)
                finally:
                    ttnn.end_trace_capture(mesh, t_, cq_id=0)
                pr[f"{mode}_us"] = S.replay_stats(
                    mesh, t_, replays=args.get("replays"), label=f"plan {name} {mode}"
                ).get("median_us")
                ttnn.release_trace(mesh, t_)
        ep.e_loc, ep.lo, ep.ep, ep.plan_mode, ep.v2, ep.bufs = keep
        row["plan_check"] = plan_rows
        if not all(pr["bitwise"] for pr in plan_rows):
            row["status"] = "fail"
    row["plan"] = ep.plan_mode
    if ep.ffn == "xffn":
        row["xffn_blocks"] = list(getattr(ep, "xffn_cfg", ()))  # (mg, kc, nc) the FFN ran with
        row.update(xffn_xsplit=ep.xffn_xsplit, xffn_snake=ep.xffn_snake, xffn_valid=ep.xffn_valid)
    if args.get("probe") and ep.ffn == "xffn":  # in-program probe counters of a traced replay
        if C.MOCK:  # the probe program compiles (no counters on a mock)
            os.environ["GLM53_PF_MOE_XFFN_PROBE"] = "1"
            run()
            os.environ["GLM53_PF_MOE_XFFN_PROBE"] = "0"
        else:
            row["probe"] = xffn_probe(args.get("probe_replays", 200))
    if os.environ["GLM53_PF_MOE_XFFN_DIAG"]:  # a timing diagnostic: wrong outputs by design, not a pass / fail row
        row.update(status="record", xffn_diag=os.environ["GLM53_PF_MOE_XFFN_DIAG"])
    S.log(row)
    if rec:
        C.record("pcc", row)
    rows = [row]
    for gx, gy in (
        args.get("grids", []) if not C.MOCK else []
    ):  # each grid compiles before its own capture (no live trace)
        ep.mfs_grid = ttnn.CoreCoord(int(gx), int(gy))
        try:
            run()
            ttnn.synchronize_device(mesh)
            t_ = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                run()
            finally:
                ttnn.end_trace_capture(mesh, t_, cq_id=0)
            st = S.replay_stats(mesh, t_, replays=args.get("replays"))
            ttnn.release_trace(mesh, t_)
            g_row = dict(test=f"moe xsl bench grid {gx}x{gy}: E {E} x width {W}, C {CR}, top-{K} tp{TPN}", milestone="P2",
                         experts=E, width=W, grid=[int(gx), int(gy)], rows=CR, k=K, v2_combine=ep.v2_combine, tag=tag,
                         **st)  # fmt: skip
        except Exception as e:  # noqa: BLE001  (an invalid geometry fails at validation: recorded, next grid)
            g_row = dict(test=f"moe xsl bench grid {gx}x{gy}", grid=[int(gx), int(gy)], error=str(e)[:160], tag=tag)
        S.log(g_row)
        if rec and "median_us" in g_row:
            C.record("perf", g_row)
        rows.append(g_row)
    for t_ in gus + dns + [x, idx, sel, Y]:
        ttnn.deallocate(t_)
    RESULT = rows
