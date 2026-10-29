# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, pair or 2x4): G v2a (pmk/w7/gv2_design.md sec 4) -- a sparse_dsa layer's MoE half as ONE program:
A5 (P10) + P11 (the AG of hn[1] into the holders' XS) + P12 (the router; P11x writes x_ext) + P13 (the shared expert) +
program G (P14: tt/pmk_gang.PmkGang via pmk_b_window.register_g, 12 gangs inside the member / MM-only roles; the RM_SKIP
combine adds P13's y_s) vs the op chain on the program's OWN hn[1] (pmk_fuse3_unit's b_chain recipe: RMSNorm -> router
-> gathers -> the served EP path EpMoe (plan + xffn + host w x local + rm combine) + the shared expert op). The residual
X = the layer's golden input streams, the attention landing ys[0] = zeros (the unit isolates the MoE half). Both traced
(one compile launch each, one capture each); criteria: x_ext (P11x) bitwise vs the chain's gathered rows, the router's
row match (as W7's P12: >= 0.999), G's y vs routed + shared PCC >= 0.9999 and |scale - 1| <= 0.005 per chip, finite,
determinism over `det` replays; + the phase probe (GLM53_PMK_PHASE_PROBE) timeline of the G v2a program.
ARGS: layer (3), build_layers ([3, 4]), rows (2048), replays (20), det (20), fid (LoFi), phase_probe (1), tag, record."""
import os
import sys
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_b_window as PBW  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_gang as PG  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMK  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16, FP32, swiglu_prefill  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
L = int(args.get("layer", 3))
LAYERS = [int(v) for v in args.get("build_layers", [3, 4])]
N = int(args.get("rows", 2048))
REPLAYS = int(args.get("replays", 20))
DET = int(args.get("det", 20))
FID = str(args.get("fid", "LoFi"))
PHASE_PROBE = bool(int(args.get("phase_probe", 1)))
P16 = bool(int(args.get("p16", 0)))  # M1 (program 23:49:25Z): + P16 (the mHC post, FFN) in-program after P15 (G16)
P15 = bool(
    int(args.get("p15", 0)) or P16
)  # G v2b (program 01:52:14Z): the RS of G's y in-program as P15 (PMK_P15_STREAM via env)
POISON = bool(int(args.get("poison", 0)))  # the poison rule (01:56:05Z): NaN over G's y / P13's y_s / P15's y15 before each launch
tag = args.get("tag", f"pmk-gv2-L{L}")
rec = bool(args.get("record", False))
NDEV = int(mesh.get_num_devices())
ENV = {"GLM53_PF_MHC": "comp", "GLM53_PF_MHC_POST": "fused", "GLM53_PF_PACKED": "", "GLM53_PF_PROBE_LAYERS": "",
       "GLM53_PF_MOE_FFN": "xffn", "GLM53_PF_MOE_XFFN_EP": "1", "GLM53_PF_MOE_XFFN_VALID": "1"}  # fmt: skip
for k_, v_ in ENV.items():
    os.environ[k_] = v_
for k_, v_ in (args.get("env") or {}).items():
    os.environ[str(k_)] = str(v_)
row = dict(test=f"G v2a: L{L} MoE half (P10 + P11 + P12 + P13 + program G) in ONE program vs the op chain on its own hn[1] "
           f"(n={N}) tp{NDEV}", milestone="W7-GV2", layer=L, n=N, tp=NDEV, fid=FID, tag=tag)  # fmt: skip
per_chip = lambda t_: [ttnn.to_torch(p).clone() for p in (ttnn.get_device_tensors(t_) if NDEV > 1 else [t_])]  # noqa: E731


def _cmp(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return dict(pcc=round(C.pcc(a.float(), b.float()), 7), scale=round(float((a * b).sum() / (b * b).sum().clamp_min(1e-300)), 6),
                rel=float((a - b).norm() / b.norm().clamp_min(1e-30)), maxabs=float((a - b).abs().max()),
                finite=bool(torch.isfinite(a).all()))  # fmt: skip


def phase_spans(prog, mhz: float = 1350.0) -> list:
    """pmk_fuse3_unit.phase_spans: per chip, per label, max / median duration and END (us from the earliest start)"""
    rc = prog.engine.place.role_cores()
    out = []
    for t_ in (ttnn.get_device_tensors(PP["rows"]) if NDEV > 1 else [PP["rows"]]):
        rows_ = ttnn.to_torch(t_).to(torch.int64) & 0xFFFFFFFF
        dur, ends, t0_ = {}, {}, None
        for (role, risc), labels in prog.engine.probe_labels.items():
            ri = 0 if risc == "dm0" else 1
            for xy in rc.get(role, []):
                w = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
                m = rows_[((int(w.y) & 31) * 32 + (int(w.x) & 31)) * 2 + ri].tolist()
                n = min(len(labels), 32)
                if m[0] == 0 or any(m[i] == 0 for i in range(n)):
                    continue
                t0_ = m[0] if t0_ is None else min(t0_, m[0])
                for i in range(1, n):
                    dur.setdefault(labels[i], []).append((m[i] - m[i - 1]) & 0xFFFFFFFF)
                    ends.setdefault(labels[i], []).append(m[i])
        summ = {}
        for pid, ds in dur.items():
            ds = sorted(ds)
            summ[pid] = dict(max_us=round(ds[-1] / mhz, 1), med_us=round(ds[len(ds) // 2] / mhz, 1), n=len(ds),
                             end_us=round(((max(ends[pid]) - t0_) & 0xFFFFFFFF) / mhz, 1) if t0_ is not None else None)  # fmt: skip
        out.append(summ)
    return out


try:
    S.set_attn({"mla_form": "absorbed", "mm_nq": 4, "mm_kb": 640, **args})
    t0 = time.time()
    ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,), max_seq=N + 512)
    pf = ent["pf"]
    li = LAYERS.index(L)
    pl = pf.layers[li]
    assert getattr(pl, "xsl", None) is None, f"L{L}: an EP layer (sparse_dsa) is required"
    c = pf.bucket_for(N)
    sp = pf.mode(c) == "sp"
    assert sp and c == N, (sp, c, N)
    ci = pf.chunk_inputs(c)
    pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
    rows, _ = S.golden_layer(L, N)
    rows = rows.reshape(N, 4, 4096).float()
    x0 = pf.streams_to_device(rows, sp)  # pristine
    x_p = pf.streams_to_device(rows, sp)  # the residual the program is built on (its address is in its args)
    PP = None
    if PHASE_PROBE:
        _g = mesh.compute_with_storage_grid_size()
        _crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(int(_g.x) - 1, int(_g.y) - 1))])
        _rep = ttnn.ReplicateTensorToMesh(mesh) if NDEV > 1 else None
        PP = dict(rows=ttnn.from_torch(torch.zeros(2048, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                       device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=_rep),
                  scr=ttnn.from_torch(torch.zeros(int(_g.x) * int(_g.y), 64, dtype=torch.int32), dtype=ttnn.uint32,
                                      layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=_rep,
                                      memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                                                      ttnn.ShardSpec(_crs, [1, 64], ttnn.ShardOrientation.ROW_MAJOR))))  # fmt: skip
        os.environ["GLM53_PMK_PHASE_PROBE"] = f"{PP['rows'].buffer_address()}:{PP['scr'].buffer_address()}"
    ep = PF.EpMoe(pf, pl.moe, xsl=None)  # the served EP path (the reference) and G's local expert tensors
    moe = pl.moe
    k, h = int(moe.top_k), int(moe.hidden)
    gangs = PBW.w7_gang_sets(PMK.Placement())
    def _l1_view():  # (2026-10-26) the L1 allocator before G's program: PMK lays its CBs (and windows) up to the fixed
        # 1,330,000 B ceiling assuming every ttnn L1 buffer sits above it -- more persistent L1 lowers that floor
        mv_ = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
        return {a_: int(getattr(mv_, a_)) for a_ in dir(mv_) if not a_.startswith("_") and isinstance(getattr(mv_, a_), int)}
    row["l1_before_g"] = _l1_view()
    gk = PG.PmkGang(mesh, e_loc=ep.e_loc, top_k=k, hidden=h, limit=float(moe.limit), gate_up=ep.gate_up, down=ep.down,
                    gangs=gangs, lo=[ep.e_loc * r for r in range(NDEV)], fid=FID, ys=True, wt_bf16=True, ids_u16=True)  # fmt: skip
    def _wck():  # (2026-10-26) host reads of L3's shared expert + router weights (what P12 / P13 upload): sum / absmax / finite
        out_ = {}
        for nm_, t_ in (("sh_gu", pl.shared.gate_up), ("sh_dn", pl.shared.down), ("rt_w", pl.moe.router.weights[320]),
                        ("rt_b", pl.moe.router.biases[320])):
            hs_ = ASM._host_shards(t_, NDEV)
            out_[nm_] = [[round(float(h_.double().sum()), 3), round(float(h_.abs().max()), 4), bool(torch.isfinite(h_).all())] for h_ in hs_]
        return out_
    if args.get("sync_before_build"):  # (program 09:12:16Z (1)) every upload / layout op in flight lands before the
        # PMK build's host reads of the router / shared weights
        ttnn.synchronize_device(mesh)
        row["sync_before_build"] = True
    row["wck_build"] = _wck()
    def _mem_view():  # (2026-10-26 w7lv5) the allocator per chip: DRAM / L1 bytes allocated (per bank x banks)
        out_ = {}
        for nm_, bt_ in (("dram", ttnn.BufferType.DRAM), ("l1", ttnn.BufferType.L1)):
            mv_ = ttnn.get_memory_view(mesh, bt_)
            out_[nm_] = int(mv_.total_bytes_allocated_per_bank) * int(mv_.num_banks)
        return out_

    MEM = {"before_program": _mem_view()}  # (PmkGang's objects are built before: their tables come with its parts)
    prog = ASM.PmkProgramA(pf, li, c, sp, x_p, None, ci.valid_end, attn=False, g_window=dict(g=gk, p15=P15, p16=P16), probe=False,
                           pfs_park=int(args.get("pfs_park", 3)))  # fmt: skip
    y15_t = (prog.rsr15.y if prog.rsr15 is not None else prog.rs8_15.y) if P15 else None
    if args.get("l1_report"):  # (cf3h-k B1, 2026-10-26) the G program's windowed declaration per role + every declared CB
        e_ = prog.engine
        e_.scope = set(prog.scopes[0])
        try:
            for role_ in ("member", "root", "mm_only"):
                wl_ = e_._win_layout(role_)
                cen_ = e_._census(e_.active_phases())[role_]
                if wl_ is None:
                    S.log(f"{tag}: L1 {role_}: unwindowed cb_end {cen_['cb_end']} slots {[(list(i), b, n[:30]) for i, b, _, n in cen_['slots']][:60]}")
                    continue
                S.log(f"{tag}: L1 {role_}: cb_end {cen_['cb_end']} bytes {wl_['bytes']} span {wl_['span']} pad {wl_['pad']} need {wl_['need']} "
                      f"hosted {wl_['hosted']} windows {wl_['windows']}")  # fmt: skip
                for us_, b_ in wl_["decl"]:
                    S.log(f"{tag}:   {role_} decl {[u.index for u in us_]} {int(b_)} {us_[0].note[:50]}")
                for w_, tab_ in wl_["tabs"].items():
                    S.log(f"{tag}:   {role_} window {w_ + 1}: {[(t[0], t[2]) for t in tab_]}")
        finally:
            e_.scope = None
    # (2026-10-26 wg20, THE G v2a NaN) G v2a has no P1: P10's post rebuilds its coefficients from DRAM posts[0] / combs[0]
    # (pmk_pcoef_dram.cpp), which P1 writes in the fused layer -- here they were ttnn.empty, so P10 mixed the residual with
    # whatever bytes the allocator left there (layout-dependent: probe / XSL / a previous W7 job only moved them; wg19's hb
    # = inf rows in the failing layout, 896 zero rows in the 'passing' one). pcoef_seed 1 (default): restore them in-trace
    # every replay as x is -- comb = the identity, post = 0 (with ys[0] = 0: x_new = x, hn[1] = the FFN pre of the golden
    # residual); 0 = the old harness (the A/B)
    SEEDS = []
    if int(args.get("pcoef_seed", 1)):
        _rp0 = ttnn.ReplicateTensorToMesh(mesh) if NDEV > 1 else None
        sp_r = int(prog.posts[0].shape[-2])
        seed_ = lambda t_: ttnn.from_torch(t_, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=_rp0)  # fmt: skip
        SEEDS = [(seed_(torch.zeros(1, 1, sp_r, int(prog.posts[0].shape[-1]))), prog.posts[0]),
                 (seed_(torch.eye(4).reshape(1, 1, 1, 16).repeat(1, 1, sp_r, 1)), prog.combs[0])]  # fmt: skip
        assert int(prog.combs[0].shape[-1]) == 16, prog.combs[0].shape
    row["pcoef_seed"] = "comb=I post=0 (in-trace)" if SEEDS else "none (P10 reads uninitialized posts[0] / combs[0])"

    def _dram_map():  # (2026-10-26) every DRAM tensor the G program holds (one attribute level deep): name, address, bytes
        seen_, out_ = set(), []
        def add_(nm_, t_):
            if isinstance(t_, ttnn.Tensor) and t_.storage_type() == ttnn.StorageType.DEVICE:
                try:
                    if t_.memory_config().buffer_type != ttnn.BufferType.DRAM:
                        return
                    a_ = int(t_.buffer_address())
                except Exception:  # noqa: BLE001
                    return
                if a_ in seen_:
                    return
                seen_.add(a_)
                dt_, lay_ = str(t_.dtype).split(".")[-1], str(t_.layout).split(".")[-1]
                sh_ = [int(v) for v in (t_.padded_shape if hasattr(t_, "padded_shape") else t_.shape)]
                vol_ = 1
                for v in sh_:
                    vol_ *= v
                if "TILE" in lay_:  # per-bank footprint of an interleaved buffer: ceil(pages / 8) x aligned page
                    pg_, npg_ = {"BFLOAT8_B": 1088, "BFLOAT4_B": 576, "BFLOAT2_B": 320, "FLOAT32": 4096, "UINT32": 4096,
                                 "INT32": 4096}.get(dt_, 2048), vol_ // 1024  # fmt: skip
                else:
                    pg_, npg_ = sh_[-1] * {"FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT8": 1}.get(dt_, 2), vol_ // max(1, sh_[-1])
                bank_ = -(-npg_ // 8) * (-(-pg_ // 64) * 64)
                out_.append((a_, a_ + bank_, nm_, sh_, dt_, lay_))
        roots_ = [("prog", prog), ("gk", gk), ("pl", pl), ("pf", pf), ("x_p", x_p)]
        roots_ += [(f"pl.{k_}", v_) for k_, v_ in vars(pl).items() if hasattr(v_, "__dict__") and not isinstance(v_, ttnn.Tensor)]
        for o_nm, o_ in roots_:
            if isinstance(o_, (ttnn.Tensor, list, tuple)):
                for i_, x_ in enumerate(o_ if isinstance(o_, (list, tuple)) else [o_]):
                    add_(f"{o_nm}[{i_}]", x_)
                continue
            for k_, v_ in (vars(o_) if hasattr(o_, "__dict__") else {}).items():
                if isinstance(v_, (list, tuple)):
                    for i_, x_ in enumerate(v_[:64]):
                        add_(f"{o_nm}.{k_}[{i_}]", x_)
                elif isinstance(v_, dict):
                    for kk_, x_ in list(v_.items())[:64]:
                        if isinstance(x_, dict):
                            for kkk_, y_ in x_.items():
                                add_(f"{o_nm}.{k_}[{kk_}].{kkk_}", y_)
                        else:
                            add_(f"{o_nm}.{k_}[{kk_}]", x_)
                elif hasattr(v_, "__dict__") and not isinstance(v_, ttnn.Tensor):
                    for kk_, x_ in vars(v_).items():
                        add_(f"{o_nm}.{k_}.{kk_}", x_)
                else:
                    add_(f"{o_nm}.{k_}", v_)
        return sorted(out_)
    if args.get("dram_map"):
        dm_ = _dram_map()
        ysa_ = int(prog.sh_dn.y.buffer_address())
        i_ = next((j_ for j_, e_ in enumerate(dm_) if e_[0] == ysa_), None)
        row["dram_map_n"] = len(dm_)
        row["ys_neighbours"] = dm_[max(0, i_ - 3): i_ + 4] if i_ is not None else None
        nbr_ = {}
        for nm_, t_ in (("sh_gu.weight", prog.sh_gu.weight), ("sh_gu.y", prog.sh_gu.y), ("sh_dn.weight", prog.sh_dn.weight),
                        ("sh_dn.y", prog.sh_dn.y), ("sync", prog.sync)):  # fmt: skip
            try:
                a_ = int(t_.buffer_address())
            except Exception:  # noqa: BLE001
                continue
            j_ = next((j for j, e_ in enumerate(dm_) if e_[0] == a_), None)
            nbr_[nm_] = dict(addr=a_, prev=dm_[j_ - 1] if j_ else None, this=dm_[j_] if j_ is not None else None,
                             next=dm_[j_ + 1] if j_ is not None and j_ + 1 < len(dm_) else None)  # fmt: skip
        row["p13_neighbours"] = nbr_
        row["p13_lay"] = {k_: {kk_: vv_ for kk_, vv_ in m_.lay.items() if kk_ != "m0"} for k_, m_ in (("gu", prog.sh_gu), ("dn", prog.sh_dn))}
        row["map_overlaps"] = [(dm_[j][2], dm_[j + 1][2]) for j in range(len(dm_) - 1) if dm_[j][1] > dm_[j + 1][0]][:16]
        S.log(f"{tag}: dram map {dm_}")
    row["rs8"] = {k_: [list(c_) for c_ in v_] if isinstance(v_, list) else v_ for k_, v_ in (prog.report.get("rs8") or {}).items()}
    row.update(
        p16=P16,
        p16_report=prog.report.get("p16"),
        p15=P15,
        p15_stream=bool(getattr(prog, "p15_stream", False)),
        poison=POISON,
        p12_rects=prog.report.get("p12_rects"),
    )  # the BUILT value (w7lv3: a leaked env set it to 8 unseen)
    MEM["after_build"] = _mem_view()  # before the unit's poison sources (PZ: out_rm's alone is ~170 MB per chip)
    PZ = {}
    if POISON:  # the NaN sources, before any capture
        _rp = ttnn.ReplicateTensorToMesh(mesh) if NDEV > 1 else None
        # (2026-10-26) + G's FFN output rows out_rm (ttnn.empty: an unwritten row that the combine reads would carry the
        # previous allocation's bytes -- the XSL / G-after-W7 NaN): poison_rm 0 leaves it out (the old rule)
        rm_ = ((("out_rm", gk.scratch(c)["out_rm"]),) if int(args.get("poison_rm", 1)) else ())
        for k_, t_ in (("y", gk.scratch(c)["y"]), ("y_s", prog.sh_dn.y)) + ((("y15", y15_t),) if P15 else ()) + rm_:
            PZ[k_] = (ttnn.from_torch(torch.full(tuple(t_.shape), float("nan")), dtype=t_.dtype, layout=t_.layout, device=mesh,
                                      memory_config=t_.memory_config(), mesh_mapper=_rp), t_)  # fmt: skip
    row.update(scopes=[list(s_) for s_ in prog.scopes], gangs=len(gk.gangs), g_cb_bytes=gk.l1_bytes(c),
               g_window={k_: v_ for k_, v_ in (prog.report.get("g_window") or {}).items() if k_ in ("roles", "need", "hub_role", "n")})  # fmt: skip
    S.log(f"{tag}: built in {time.time() - t0:.1f} s; scopes {prog.scopes}; G {len(gk.gangs)} gangs, CB {gk.l1_bytes(c)} B; "
          f"g_window {row['g_window']}")
    y_g = gk.scratch(c)["y"]

    def run_g():  # the member / MM-only / root DM composites read their args from the program's DRAM table (P10..P13 + G
        # exceed 256 launch args on the member BRISC: 391), as the W7 fused unit's B program does
        ttnn.copy(x0, x_p)
        for src_, dst_ in SEEDS:
            ttnn.copy(src_, dst_)
        for src_, dst_ in PZ.values():
            ttnn.copy(src_, dst_)
        old_ = os.environ.get("GLM53_PMK_ARGS")
        os.environ["GLM53_PMK_ARGS"] = "table"
        try:
            prog(x_p, ci)
        finally:
            if old_ is None:
                os.environ.pop("GLM53_PMK_ARGS", None)
            else:
                os.environ["GLM53_PMK_ARGS"] = old_

    def chain():  # the op chain on the program's own hn[1] (pmk_fuse3_unit b_chain's recipe) -> (routed y, y_s, hb, idx16)
        h1 = prog.hns[1]
        pl._normed["ffn"] = h1
        x32 = pl.rms(h1, "ffn", FP32)
        idx, scores = pl.moe.router(x32, decode=False)
        sel_l = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), FP32)
        ids_f, sel = pf.gather(ttnn.typecast(idx, FP32), sp), pf.gather(sel_l, sp)
        idx16 = ttnn.typecast(ids_f, ttnn.uint16)
        hb = pf.gather(ttnn.typecast(x32, BF16), sp)
        y_r = ep(hb, idx16, sel)
        y_s = swiglu_prefill(pl.shared, hb)
        return y_r, y_s, hb, idx16

    if args.get("dump_src"):  # (host bisect, 2026-10-25 gv2w2 watcher NOC error on core (0, 0)) the generated composites
        os.environ["GLM53_PMK_KEEP_SRC"] = "1"
    if args.get("l1_scrub"):  # (2026-10-26, the program's (c)) fill ~all worker L1 (the CB / window region included) with
        # NaN or zeros through one maximal L1-sharded tensor, then free it, BEFORE G's first launch: G reading L1 it never
        # initialises shows as NaN after a nan scrub (XSL 0) and as a pass after a zero scrub (XSL 1)
        g_ = mesh.compute_with_storage_grid_size()
        ncore = int(g_.x) * int(g_.y)
        free_ = int(ttnn.get_memory_view(mesh, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank)
        rows_ = (free_ // (512 * 2)) // 32 * 32
        crs_ = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(int(g_.x) - 1, int(g_.y) - 1))])
        val_ = float("nan") if args["l1_scrub"] == "nan" else 0.0
        scr_ = ttnn.from_torch(torch.full((ncore * rows_, 512), val_), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                               memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                                               ttnn.ShardSpec(crs_, [rows_, 512], ttnn.ShardOrientation.ROW_MAJOR)),
                               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if NDEV > 1 else None)  # fmt: skip
        ttnn.synchronize_device(mesh)
        row["l1_scrub"] = dict(mode=args["l1_scrub"], bytes_per_core=rows_ * 1024, addr=int(scr_.buffer_address()), free=free_)
        ttnn.deallocate(scr_)
    run_g()  # compile launch
    ttnn.synchronize_device(mesh)
    MEM["after_compile_launch"] = _mem_view()
    row["mem"] = MEM
    row["mem_program_delta"] = {k_: MEM["after_build"][k_] - MEM["before_program"][k_] for k_ in ("dram", "l1")}
    if args.get("dram_map"):  # (2026-10-26 wg17) after the compile launch (the arg tables are filled): the ALLOCATOR's
        # DRAM buffers (chip 0, every live one: ttnn reports), named by a deep walk of prog / gk / pl / pf, and every
        # DRAM-address-like word of the program's arg-table rows that lies in NO live buffer (a stale / foreign address)
        import bisect as _bs

        from ttnn._ttnn import reports as _R

        names_, seen_o = {}, set()

        def walk_(o_, nm_, d_):
            if id(o_) in seen_o or d_ > 5:
                return
            seen_o.add(id(o_))
            if isinstance(o_, ttnn.Tensor):
                try:
                    if o_.storage_type() == ttnn.StorageType.DEVICE and o_.memory_config().buffer_type == ttnn.BufferType.DRAM:
                        names_.setdefault(int(o_.buffer_address()), nm_)
                except Exception:  # noqa: BLE001
                    pass
                return
            if isinstance(o_, dict):
                for k_, v_ in list(o_.items())[:512]:
                    walk_(v_, f"{nm_}[{k_}]", d_ + 1)
            elif isinstance(o_, (list, tuple)):
                for i_, v_ in enumerate(o_[:512]):
                    walk_(v_, f"{nm_}[{i_}]", d_ + 1)
            elif hasattr(o_, "__dict__") and not isinstance(o_, type):
                for k_, v_ in list(vars(o_).items()):
                    walk_(v_, f"{nm_}.{k_}", d_ + 1)

        for nm_, o_ in (("prog", prog), ("gk", gk), ("x_p", x_p), ("PZ", PZ), ("pl", pl), ("pf", pf)):
            walk_(o_, nm_, 0)
        bl_ = [b_ for b_ in _R.get_buffers(mesh) if b_.buffer_type == ttnn.BufferType.DRAM]
        d0_ = min(int(b_.device_id) for b_ in bl_)
        live_ = sorted((int(b_.address), int(b_.address) + int(b_.max_size_per_bank)) for b_ in bl_ if int(b_.device_id) == d0_)
        st_ = [a_ for a_, _ in live_]
        row["alloc_n"] = len(live_)
        row["alloc_named"] = sum(1 for a_, _ in live_ if a_ in names_)
        S.log(f"{tag}: ALLOC " + repr([(a_, e_, names_.get(a_, "?")) for a_, e_ in live_]))

        def owner_(v_):
            j_ = _bs.bisect_right(st_, v_) - 1
            return j_ if j_ >= 0 and v_ < live_[j_][1] else None

        stale_, used_ = [], {}
        for key_, tb_ in (getattr(prog.engine, "arg_tables", None) or {}).items():
            for rk_, rows_ in (tb_.get("host") or {}).items():
                for ri_, vals_ in rows_.items():
                    for vi_, v_ in enumerate(vals_):
                        v_ = int(v_) & 0xFFFFFFFF
                        if (16 << 20) <= v_ < (1 << 32):
                            j_ = owner_(v_)
                            if j_ is None:
                                stale_.append((rk_, ri_, vi_, v_))
                            else:
                                k2_ = names_.get(live_[j_][0], str(live_[j_][0]))
                                used_[k2_] = used_.get(k2_, 0) + 1
        row["arg_tables"] = [dict(key=str(k_), nrows=tb_.get("nrows"), addr=tb_.get("addr")) for k_, tb_ in (getattr(prog.engine, "arg_tables", None) or {}).items()]
        row["arg_dram_words_stale_n"] = len(stale_)
        row["arg_dram_words_stale"] = sorted(set(v_ for *_, v_ in stale_))[:48]
        row["arg_dram_words_stale_where"] = stale_[:24]
        row["arg_dram_buffers_used"] = used_

    def _static_ck():  # (2026-10-26 wg18) the program's never-written DRAM inputs, per chip: arg table rows + P12 / P13's
        # streamed weights, the router bias, G's expert address table (a change across replays = a stray write into them)
        out_ = {}
        for k_, tb_ in (getattr(prog.engine, "arg_tables", None) or {}).items():
            for i_, d_ in enumerate(ttnn.get_device_tensors(tb_["t"]) if NDEV > 1 else [tb_["t"]]):
                out_[f"argtab{i_}"] = ttnn.to_torch(d_).to(torch.int64).reshape(-1, int(prog.engine.arg_page) // 4) & 0xFFFFFFFF
        for nm_, t_ in (("rt_mm.weight", prog.rt_mm.weight), ("sh_gu.weight", prog.sh_gu.weight), ("sh_dn.weight", prog.sh_dn.weight),
                        ("rt_bias", prog.rt_bias), ("tbl", gk.tbl)):  # fmt: skip
            for i_, d_ in enumerate(ttnn.get_device_tensors(t_) if NDEV > 1 else [t_]):
                out_[f"{nm_}#{i_}"] = ttnn.to_torch(d_).float()
        return out_

    def _ck_diff(a_, b_):
        res_ = {}
        for k_ in a_:
            x_, y_ = a_[k_], b_[k_]
            if x_.shape != y_.shape or not torch.equal(x_.nan_to_num(0), y_.nan_to_num(0)) or not torch.equal(x_.isnan(), y_.isnan()):
                if k_.startswith("argtab"):
                    w_ = (x_ != y_).nonzero().tolist()
                    res_[k_] = dict(n=len(w_), rows=sorted(set(r_ for r_, _ in w_))[:40],
                                    first=[(r_, c_, int(x_[r_, c_]), int(y_[r_, c_])) for r_, c_ in w_[:16]])  # fmt: skip
                else:
                    d_ = (x_.reshape(-1) != y_.reshape(-1)) | (x_.reshape(-1).isnan() != y_.reshape(-1).isnan())
                    ix_ = d_.nonzero().flatten()
                    res_[k_] = dict(n=int(d_.sum()), first_idx=ix_[:8].tolist(), n_elems=int(d_.numel()))
        return res_

    CK0 = None
    if args.get("static_ck"):
        CK0 = _static_ck()
        exp_ = {}
        for k_, tb_ in (getattr(prog.engine, "arg_tables", None) or {}).items():
            for rk_, rows_ in (tb_.get("host") or {}).items():
                full_ = torch.zeros(int(tb_["nrows"]), int(prog.engine.arg_page) // 4, dtype=torch.int64)
                for ri_, v_ in rows_.items():
                    full_[ri_, : len(v_)] = torch.tensor([int(u_) & 0xFFFFFFFF for u_ in v_], dtype=torch.int64)
                exp_[f"argtab{rk_}"] = full_
        row["static_ck_vs_host"] = _ck_diff(exp_, {k_: CK0[k_] for k_ in exp_ if k_ in CK0}) if exp_ and all(k_ in CK0 for k_ in exp_) else "n/a"
    if args.get("dump_src"):
        import json as _json

        d_ = str(args["dump_src"])
        os.makedirs(d_, exist_ok=True)
        for (tg_, role_, risc_, rank_), src_ in (getattr(prog.engine, "sources", None) or {}).items():
            with open(os.path.join(d_, f"{tg_}_{role_}_{risc_}_r{rank_}.cpp"), "w") as f_:
                f_.write(src_)
        rc_ = prog.engine.place.role_cores()
        lay_ = {f"{r_}/{k_}": [(p_, c0_, r0_, pt_.get("name")) for p_, c0_, r0_, pt_ in v_] for (r_, k_), v_ in prog.engine.layouts.items()}
        with open(os.path.join(d_, "roles.json"), "w") as f_:
            _json.dump(dict(role_cores={r_: [list(c_) for c_ in v_] for r_, v_ in rc_.items()}, layouts=lay_,
                            sub_roles=getattr(prog.engine.place, "sub_roles", None) and str(prog.engine.place.sub_roles),
                            window_report=getattr(prog.engine, "window_report", None)), f_, default=str, indent=1)  # fmt: skip
        S.log(f"{tag}: sources + roles dumped to {d_}")
    outs_c = chain()  # the chain's compile launch (on the compile launch's hn[1])
    ttnn.synchronize_device(mesh)
    for t_ in outs_c:
        ttnn.deallocate(t_)
    old_rf = pf.reduce_fp32  # the served RS op's compile launch, BEFORE any capture (review 02:57Z: nothing compiles after one)
    pf.reduce_fp32 = 2
    try:
        y_w = pf.reduce(y_g, True)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(y_w)
    finally:
        pf.reduce_fp32 = old_rf
    if C.MOCK:
        run_g()
        row["status"] = "mock"
    else:
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            run_g()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        got = dict(y=per_chip(y_g), xext=per_chip(prog.xext), ids=per_chip(prog.rt_ids))
        # (wg20) bitwise identity ACROSS arms / processes / layouts: sha1 per chip of the first replay's G y, P13's y_s,
        # hn[1] (P10's output, this chip's rows) and the routes -- equal hashes = bitwise equal tensors
        import hashlib as _hl

        def _h(ts_):
            out_ = []
            for t_ in ts_:
                t_ = t_.contiguous()
                if t_.dtype not in (torch.float32, torch.int32, torch.int16, torch.uint8):  # bf16 / uint16 / uint32: raw bits
                    t_ = t_.view(torch.int16 if t_.element_size() == 2 else torch.int32)
                out_.append(_hl.sha1(t_.numpy().tobytes()).hexdigest()[:16])
            return out_

        row["hash_first"] = dict(g_y=_h(got["y"]), y_s=_h(per_chip(prog.sh_dn.y)), hn1=_h(per_chip(prog.hns[1])), ids=_h(got["ids"]),
                                 xext=_h(got["xext"]))  # fmt: skip
        if CK0 is not None:
            row["static_ck_after_replay1"] = _ck_diff(CK0, _static_ck())
        if P15:
            got["y15"] = per_chip(y15_t)
        same = True
        for _ in range(DET):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            same = same and all(torch.equal(a_, b_) for a_, b_ in zip(per_chip(y_g), got["y"]))
            if P15:
                same = same and all(torch.equal(a_, b_) for a_, b_ in zip(per_chip(y15_t), got["y15"]))
        row["determinism"] = {str(DET): bool(same)}
        if CK0 is not None:
            row["static_ck_after_det"] = _ck_diff(CK0, _static_ck())
        perf = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} G v2a")
        row["us_gv2"] = perf.get("median_us")
        if PP is not None:
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            row["phase_probe"] = phase_spans(prog)
            S.log(f"{tag}: phase probe (chip 0) {row['phase_probe'][0]}")
        ttnn.release_trace(mesh, tid)
        if args.get("split_time"):  # (program 06:43:29Z (5)) the replay minus the probed span: the in-trace restore
            # copies alone (x0 -> x_p, + the poison copies), their own trace; the program + its launch = us_gv2 - copies
            def copies():
                ttnn.copy(x0, x_p)
                for src_, dst_ in SEEDS:
                    ttnn.copy(src_, dst_)
                for src_, dst_ in PZ.values():
                    ttnn.copy(src_, dst_)
            tid_ = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                copies()
            finally:
                ttnn.end_trace_capture(mesh, tid_, cq_id=0)
            ttnn.execute_trace(mesh, tid_, cq_id=0, blocking=True)
            us_cp = S.replay_stats(mesh, tid_, replays=REPLAYS, label=f"{tag} split copies").get("median_us")
            ttnn.release_trace(mesh, tid_)
            span_ = max((v_.get("end_us") or 0) for v_ in (row.get("phase_probe") or [{}])[0].values() if isinstance(v_, dict)) if row.get("phase_probe") else None
            row["split_us"] = dict(copies=us_cp, prog_and_launch=round(row["us_gv2"] - us_cp, 1) if us_cp else None,
                                   probed_span_chip0=round(span_, 1) if span_ else None)  # fmt: skip
            S.log(f"{tag}: split {row['split_us']}")
        # the chain, traced, on the program's hn[1] of the last replay
        tid2 = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            y_r, y_s, hb, idx16 = chain()
        finally:
            ttnn.end_trace_capture(mesh, tid2, cq_id=0)
        ttnn.execute_trace(mesh, tid2, cq_id=0, blocking=True)
        ref_r, ref_s, ref_hb, ref_ids = per_chip(y_r), per_chip(y_s), per_chip(hb), per_chip(idx16)
        perf2 = S.replay_stats(mesh, tid2, replays=REPLAYS, label=f"{tag} chain")
        row["us_chain"] = perf2.get("median_us")
        ttnn.release_trace(mesh, tid2)
        # the SERVED RS op on G's y (pf.reduce, sp, fp32 sums and fp32 out): G v2a's next program, timed and traced; G v2b's
        # P15 rows must equal it bitwise at TP2 (both the exact two-term fp32 sum), PCC at TP8 (ttnn's order vs RS8's)
        old_rf = pf.reduce_fp32
        pf.reduce_fp32 = 2
        try:
            tid3 = ttnn.begin_trace_capture(mesh, cq_id=0)  # (compiled with the chain, before every capture)
            try:
                y_rs = pf.reduce(y_g, True)
            finally:
                ttnn.end_trace_capture(mesh, tid3, cq_id=0)
            ttnn.execute_trace(mesh, tid3, cq_id=0, blocking=True)
            got_rs = [t_.reshape(-1, h).float() for t_ in per_chip(y_rs)]
            perf3 = S.replay_stats(mesh, tid3, replays=REPLAYS, label=f"{tag} served RS")
            row["us_rs_op"] = perf3.get("median_us")
            ttnn.release_trace(mesh, tid3)
        finally:
            pf.reduce_fp32 = old_rf
        if P15:
            row["p15_vs_rs_op"] = [dict(bitwise=bool(torch.equal(a_.reshape(-1, h).float(), b_)), pcc=round(C.pcc(a_.reshape(-1, h).float(), b_), 7))
                                   for a_, b_ in zip(got["y15"], got_rs)]  # fmt: skip
        # x_ext (P11x) vs the chain's gathered rows; the routes; G's y vs routed + shared
        row["xext_vs_hb"] = [bool(torch.equal(a_.reshape(-1, h)[:N], b_.reshape(-1, h)[:N])) for a_, b_ in zip(got["xext"], ref_hb)]
        match = []
        for a_, r_ in zip(got["ids"], ref_ids):
            a2, r2 = a_.reshape(-1, k).to(torch.int64), r_.reshape(-1, k).to(torch.int64)
            match.append(round(float((a2 == r2).all(dim=-1).float().mean()), 6))
        row["ids_row_match"] = match
        # (G-after-W7 / XSL NaN, 2026-10-26) the routed ids P13 wrote: how many are not real experts (>= the expert count:
        # a pad / garbage router column won), and the first mismatched rows with both sides' ids
        n_exp = int(getattr(moe, "num_experts", 288))
        # (2026-10-26) G's other inputs vs the chain: P13's shared y_s, the in-program top-k weights, the EP geometry
        ys0 = per_chip(prog.sh_dn.y)
        row["ys_vs_chain"] = [_cmp(a_.float().reshape(-1, h), b_.float().reshape(-1, h)) for a_, b_ in zip(ys0, ref_s)]
        # (2026-10-26 wg19) WHERE y_s is wrong: per chip the 32-row tiles (= P13 members) and 32-column tiles (16 per
        # stream / bank) whose maxabs error is > 1, and the bad element count; ys_dump=<dir> saves y_s / the chain's / hb
        ysb_ = []
        for a_, b_ in zip(ys0, ref_s):
            d_ = (a_.float().reshape(-1, h) - b_.float().reshape(-1, h)).abs().nan_to_num(3e38)
            rt_ = d_.reshape(-1, 32, h).amax(dim=(1, 2))
            ct_ = d_.reshape(-1, h // 32, 32).amax(dim=(0, 2))
            ysb_.append(dict(bad_elems=int((d_ > 1.0).sum()), bad_rows=int((d_ > 1.0).any(dim=-1).sum()),
                             bad_row_tiles=(rt_ > 1.0).nonzero().flatten().tolist()[:64],
                             bad_col_tiles=(ct_ > 1.0).nonzero().flatten().tolist()[:128],
                             bad_rows_first=(d_ > 1.0).any(dim=-1).nonzero().flatten()[:24].tolist()))  # fmt: skip
        row["ys_bad"] = ysb_
        hbs_ = []  # (wg20) hn[1] (gathered) sanity: P10's output, the chain's and P13's input
        for t_ in ref_hb:
            t_ = t_.float().reshape(-1, h)
            hbs_.append(dict(finite=bool(torch.isfinite(t_).all()), zero_rows=int((t_ == 0).all(dim=-1).sum()),
                             rms_med=round(float(t_.nan_to_num(0, 0, 0).pow(2).mean(dim=-1).sqrt().median()), 5),
                             absmax=float(t_.nan_to_num(0, 0, 0).abs().max())))  # fmt: skip
        row["hn1_sanity"] = hbs_
        if args.get("ys_dump"):
            os.makedirs(str(args["ys_dump"]), exist_ok=True)
            torch.save(dict(ys=[t_.cpu() for t_ in ys0], ref=[t_.cpu() for t_ in ref_s], hb=[t_.cpu() for t_ in ref_hb],
                            xext=[t_.cpu() for t_ in got["xext"]], ids=[t_.cpu() for t_ in got["ids"]], ref_ids=[t_.cpu() for t_ in ref_ids],
                            wt=[t_.cpu() for t_ in per_chip(prog.rt_wt)], logits=[t_.cpu() for t_ in per_chip(prog.rt_mm.y)], tag=tag),
                       os.path.join(str(args["ys_dump"]), f"{tag}.pt"))  # fmt: skip
        wt0 = per_chip(prog.rt_wt)
        row["wt_finite"] = [bool(torch.isfinite(t_.float()).all()) for t_ in wt0]
        row["wt_absmax"] = [round(float(t_.float().abs().max()), 4) for t_ in wt0]
        row["ep_geom"] = dict(e_loc=int(ep.e_loc), n_gate_up=len(ep.gate_up), num_experts=int(getattr(moe, "num_experts", -1)),
                              top_k=int(moe.top_k), limit=float(moe.limit), ep=str(getattr(pl, "ep", None)))  # fmt: skip
        row["wck_end"] = _wck()
        row["wck_same"] = row["wck_end"] == row["wck_build"]
        row["ids_out_of_range"] = [int((a_.reshape(-1, k).to(torch.int64) >= n_exp).sum()) for a_ in got["ids"]]
        a0, r0 = got["ids"][0].reshape(-1, k).to(torch.int64), ref_ids[0].reshape(-1, k).to(torch.int64)
        bad_ = (a0 != r0).any(dim=-1).nonzero().flatten()[:6].tolist()
        row["ids_mismatch_rows"] = [dict(row=int(i_), pmk=a0[i_].tolist(), chain=r0[i_].tolist()) for i_ in bad_]
        if "out_rm" in PZ:  # the FFN rows never written in the replay (still NaN) and the routed pairs that point at them
            rm0 = ttnn.get_device_tensors(gk.scratch(c)["out_rm"])[0] if NDEV > 1 else gk.scratch(c)["out_rm"]
            rm0 = ttnn.to_torch(rm0).float().reshape(-1, h)
            unw = (~torch.isfinite(rm0)).any(dim=-1)
            row["out_rm_unwritten_rows"] = int(unw.sum())
            row["out_rm_rows"] = int(rm0.shape[0])
            row["out_rm_unwritten_first"] = unw.nonzero().flatten()[:12].tolist()
        g0 = got["y"][0].float().reshape(-1, h)
        nan_r = (~torch.isfinite(g0)).any(dim=-1)[: a0.shape[0]]
        row["g_nan_rows"] = int(nan_r.sum())
        if bool(nan_r.any()):  # which LOCAL experts of chip 0 the NaN rows use and the finite rows never do
            lo0, el = 0, int(ep.e_loc)
            loc = lambda r_: {int(e_) for e_ in a0[r_].tolist() if lo0 <= e_ < lo0 + el}  # noqa: E731
            in_nan = set().union(*[loc(r_) for r_ in nan_r.nonzero().flatten().tolist()])
            in_fin = set().union(*[loc(r_) for r_ in (~nan_r).nonzero().flatten().tolist()]) if bool((~nan_r).any()) else set()
            row["nan_only_experts"] = sorted(in_nan - in_fin)[:40]
            row["nan_rows_without_local"] = int(sum(1 for r_ in nan_r.nonzero().flatten().tolist() if not loc(r_)))
        def _tinfo(t_):
            mc_ = t_.memory_config()
            return dict(addr=int(t_.buffer_address()), shape=list(t_.shape), dtype=str(t_.dtype).split(".")[-1],
                        layout=str(mc_.memory_layout).split(".")[-1], buffer=str(mc_.buffer_type).split(".")[-1])
        def _wsum(t_):  # chip 0's shard of an expert weight tensor: sum / max / finite (the content, not the address)
            w_ = (ttnn.to_torch(ttnn.get_device_tensors(t_)[0]) if NDEV > 1 else ttnn.to_torch(t_)).float()
            return [round(float(w_.double().sum()), 3), round(float(w_.abs().max()), 4), bool(torch.isfinite(w_).all())]
        if args.get("weight_sums"):  # (2026-10-26) does the XSL L4 build overwrite L3's expert weights?
            row["expert_sums"] = {f"e{e_}": dict(gu=_wsum(ep.gate_up[e_]), dn=_wsum(ep.down[e_])) for e_ in args["weight_sums"]}
        el_ = len(ep.gate_up)
        row["expert_tensors"] = {f"gu{i_}": _tinfo(ep.gate_up[i_]) for i_ in (0, 1, el_ // 2, el_ - 1)} | \
            {f"dn{i_}": _tinfo(ep.down[i_]) for i_ in (0, el_ - 1)}  # fmt: skip
        ref_y = [(a_.float().reshape(-1, h) + b_.float().reshape(-1, h)) for a_, b_ in zip(ref_r, ref_s)]
        row["g_vs_chain"] = [_cmp(g_.float().reshape(-1, h), r_) for g_, r_ in zip(got["y"], ref_y)]
        ok = (all(row["xext_vs_hb"]) and min(match) >= 0.999 and all(abs(x_["scale"] - 1) <= 0.005 and x_["pcc"] >= 0.9999 and x_["finite"]
                                                                   for x_ in row["g_vs_chain"]) and same)  # fmt: skip
        if P15:  # P15's owned rows: the device order of the served fp32 RS (TP2: own + peer, exact = the RS op's sum;
            # TP8 RS8: stage 1 the board pair (exact fp32), stage 2 ((t0 + t1) + (t2 + t3)) over the row) of G's OWN y,
            # bitwise; and PCC / scale vs the same reduce of the chain's routed + shared
            rows_ = c // NDEV

            def red(ys):
                b2 = [y_.reshape(-1, h).float() for y_ in ys]
                out_ = []
                for o in range(NDEV):
                    R = slice(o * rows_, (o + 1) * rows_)
                    if NDEV == 2:
                        out_.append(b2[0][R] + b2[1][R])
                    else:
                        t = [b2[cc][R] + b2[4 + cc][R] for cc in range(4)]
                        out_.append((t[0] + t[1]) + (t[2] + t[3]))
                return out_
            self_ref, chain_ref = red(got["y"]), red(ref_y)
            # which side deviates from the exact fp32 sum: the served RS op vs the host reduce of G's own y
            row["rs_op_vs_exact"] = [dict(bitwise=bool(torch.equal(a_, b_)), maxabs=float((a_ - b_).abs().max()),
                                          ndiff=int((a_ != b_).sum())) for a_, b_ in zip(got_rs, self_ref)]  # fmt: skip
            for d_, a_, b_ in zip(row["p15_vs_rs_op"], got["y15"], got_rs):
                d_.update(maxabs=float((a_.reshape(-1, h).float() - b_).abs().max()), ndiff=int((a_.reshape(-1, h).float() != b_).sum()))
            y2 = [t_.reshape(-1, h) for t_ in got["y15"]]
            if args.get("dump"):  # (G v2b TP8 P15 defect, 2026-10-26 gv2c) per chip: G's y, P15's rows, the exact reduce
                os.makedirs(str(args["dump"]), exist_ok=True)
                torch.save(dict(y=[t_.reshape(-1, h).float() for t_ in got["y"]], y15=[t_.float() for t_ in y2], exact=self_ref,
                                rs_op=got_rs, rows=rows_, tag=tag), os.path.join(str(args["dump"]), f"{tag}.pt"))  # fmt: skip
                S.log(f"{tag}: P15 dump -> {args['dump']}/{tag}.pt")
            row["p15_rows"] = dict(bitwise_vs_own_g=[bool(torch.equal(a_, b_)) for a_, b_ in zip(y2, self_ref)],
                                   finite=[bool(torch.isfinite(a_).all()) for a_ in y2],
                                   vs_chain=[_cmp(a_, b_) for a_, b_ in zip(y2, chain_ref)])  # fmt: skip
            ok = ok and (NDEV != 2 or all(x_["bitwise"] for x_ in row["p15_vs_rs_op"])) and all(
                x_["pcc"] >= 0.9999 for x_ in row["p15_vs_rs_op"])
            ok = ok and all(row["p15_rows"]["bitwise_vs_own_g"]) and all(row["p15_rows"]["finite"]) and all(
                abs(x_["scale"] - 1) <= 0.005 and x_["pcc"] >= 0.9999 for x_ in row["p15_rows"]["vs_chain"])
        row["status"] = "pass" if ok else "fail"
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    row.update(status="fail", error=f"{type(e).__name__}: {e}"[:2000])
for k_ in ENV:
    os.environ.pop(k_, None)
S.log(f"{tag}: {row}")
if rec and not C.MOCK:
    C.record("pcc", row)
RESULT = [row]
print("PMK_GV2_UNIT", row, flush=True)
