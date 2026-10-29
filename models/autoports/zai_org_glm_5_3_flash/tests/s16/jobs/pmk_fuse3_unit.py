# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the TP2 pair, a d0 card; <= 15 min): M-A of the per-layer megakernel (operator / program 2026-10-24
03:30:35Z), variant "a15" (default here): program A of layer L with A1 + A2 + A3 + A4 + A5 = {P1 .. P10} in ONE program
(PMK_A15_FUSE: five member windows, the in-program H1 / H6 fan-outs, the mhc round counters cumulative across P1 -> P10)
vs the same five-program chain; the A1 outputs (hn[0], posts[0], combs[0]) join the bitwise set and the spans are
A1..A5 vs the fused program. Variants a2f / a34r = 14's pmk_fuse2_unit (from which this file is copied). Original text:
STAGE 2 of the per-layer kernel (program 2026-10-24 00:13:51Z):
program A of layer L with A2 + A3 + A4 = {P2, P3, P4, P6, P7, P8} in ONE program (PMK_A2_FUSE: three member phase windows
-- the projections | the kda items | o_proj, PfStream's rings re-placed per window --, the in-program H3 barrier, the part
regime (P3 / P7 at HiFi2 under the items' HiFi4 full-sync), C1 then RSR on the lanes with a serial open / close) vs the
five-program chain A1 | A2 | A3 | A4 | A5 of a SECOND PmkProgramA on the same residual / KDA state (the fused object's
items wait the in-program H3, so its A3 cannot run alone). Both TRACED (the traces reset x and the state); one live trace
at a time; every program compiled before the first capture.
Criterion: bitwise per chip on the A2 stores (u, lr, f, gate_out), y, on, S, C, x1, x-hat_f, post_f, comb_f; then the fused
trace replayed `det` times, read at det / 2 and det: bitwise to the first.
Also: the A2 + A3 + A4 spans (the control's, each traced alone) vs the fused program alone; the chain vs the fused chain;
the windows. ARGS: layer (4), rows (2048), replays (20), det (200), tag."""
import os
import sys
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pmk_harness as H  # noqa: E402,F401
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16, FP32, EpMoe, swiglu_prefill  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
L = int(args.get("layer", 4))
N = int(args.get("rows", 2048))
REPLAYS = int(args.get("replays", 20))
DET = int(args.get("det", 200))
# valid_n (GLM53_PMK_ROW_MASK gate, program 2026-10-26 12:10:42Z): after the all-valid replays, the SAME fused trace
# replayed with the chunk's valid_end = valid_n (the padded last chunk); 0 = off
VALID_N = int(args.get("valid_n", 0) or 0)
tag = args.get("tag", f"pmk-fuse3-unit-L{L}")
PHASE_PROBE = bool(
    int(args.get("phase_probe", 0))
)  # GLM53_PMK_PHASE_PROBE (tt/pmk_layer.PHASE_PROBE): per-phase in-program spans
NDEV = int(mesh.get_num_devices())
for k_, v_ in (args.get("env") or {}).items():  # A/B knobs of the engine (e.g. GLM53_PMK_WIN_HOST) before the build
    os.environ[str(k_)] = str(v_)
DUMP_SRC = args.get("dump_src") or None  # a directory: the fused program's generated composites (GLM53_PMK_KEEP_SRC)
if DUMP_SRC:
    os.environ["GLM53_PMK_KEEP_SRC"] = "1"
L1_REPORT = bool(
    args.get("l1_report", False)
)  # host-only: build with PMK_L1_CHECK=0, dump the fused member / root layout
if L1_REPORT:
    os.environ["PMK_L1_CHECK"] = "0"
# W7 host CB model (tt/pmk_cb_model.py, program 2026-10-25 13:40:49Z): host-only, the one-producer / one-consumer RISC
# rule per CB per role per window over the control's programs and the fused one; no device work
CB_MODEL = bool(args.get("cb_model", False))
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ""
S.set_attn(args)
t0 = time.time()
# THE ROW FENCE (program 2026-10-26 20:06:53Z / 20:09:45Z, tt/row_fence.py): fence = "mcast" | "hop1" -> a RowFence
# program before EVERY fused launch (inside its trace); the sync tensor is allocated HERE, before the build and any capture
FENCE = str(args.get("fence", "") or "")
RF_ = None
if FENCE:
    from models.autoports.zai_org_glm_5_3_flash.tt import row_fence as RFm  # noqa: E402

    RF_ = RFm.RowFence(mesh, mode=FENCE)
    S.log(
        f"row fence {FENCE}: axis {RF_.axis}, line {RF_.nc} chips x {RF_.nlines}, lanes {RF_.lanes}, sync L1 {RF_.sync.buffer_address()}"
    )
ent = S.build(mesh, ctx, [L], b=1, buckets=(N,), max_seq=N + 512)
pf = ent["pf"]
pl = pf.layers[0]
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
assert sp and c == N, (sp, c, N)
ci = pf.chunk_inputs(c)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
rows, _ = S.golden_layer(L, N)
rows = rows.reshape(N, 4, 4096).float()
x0 = pf.streams_to_device(rows, sp)  # pristine (staged once, outside every trace)
x_p = pf.streams_to_device(rows, sp)  # the residual both program-A objects are built on (its address is in their args)
kda = pl.kda
state_p = kda.allocate_state(1)
ttnn.copy(
    ttnn.ones(tuple(state_p.slot_mask.shape), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh), state_p.slot_mask
)
S.log(f"{tag}: pf ready in {time.time() - t0:.1f} s (tp{NDEV}, c={c})")
t1 = time.time()
kw = dict(probe=bool(int(args.get("probe", 0))), pfs_park=int(args.get("pfs_park", 4)))  # "probe": 1 = the in-program probe rows (PFS_PROBE / XFFN_PROBE builds)
if NDEV == 8:  # M-C (A15 at TP8): the served construction's PMK_TP8_RESIDENT=0 (X reloaded by P10, not pinned across
    # the windows: A15's no-carry X / XS / IN0 alias group); "tp8_resident": 1 = the pinned form (A/B)
    kw["tp8_resident"] = bool(int(args.get("tp8_resident", 0)))
# variant: "a2f" = stage 2 (A2 + A3 + A4); "a34r" = stage 1 + the o_proj row groups under the scan (PMK_O_RECTS 4)
VARIANT = str(args.get("variant", "a15"))
PP = None
if (
    PHASE_PROBE
):  # (program 20:26:11Z (3): the P11 / P15 spans in the TP8 row) the probe rows + a 128-B L1 scratch per core
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


def phase_spans(prog, mhz: float = 1350.0) -> dict:
    """the probe rows of prog's last run -> per chip, per phase: the max / median duration over the (core, RISC)
    composites that ran it (mark after it - the previous mark: its wait + body) and its END (max over cores of the mark
    after it - the chip's earliest start), us at `mhz` (the AICLK; cycles are the low 32 bits of the RISC wall clock)"""
    rc = prog.engine.place.role_cores()
    out = []
    for t_ in ttnn.get_device_tensors(PP["rows"]) if NDEV > 1 else [PP["rows"]]:
        rows = ttnn.to_torch(t_).to(torch.int64) & 0xFFFFFFFF
        dur, ends, t0 = {}, {}, None
        # the fabric phases' split (non-B levers (2) / (4)): each phase END per core group and RISC -- the RS lanes (TP8
        # stage 1), the tails (stage 2), the stage-1 / stage-2 reducers, the members
        grp = {"lane": set(map(tuple, getattr(prog, "lanes", []) or [])), "tail": set(map(tuple, getattr(prog, "lanes2", []) or [])),
               "red1": set(map(tuple, getattr(prog, "red1", []) or [])), "red2": set(map(tuple, getattr(prog, "red2", []) or [])),
               "member": set(map(tuple, getattr(prog, "members", []) or []))}  # fmt: skip
        gends = {}
        for (role, risc), labels in prog.engine.probe_labels.items():
            ri = 0 if risc == "dm0" else 1
            for xy in rc.get(role, []):
                w = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
                m = rows[((int(w.y) & 31) * 32 + (int(w.x) & 31)) * 2 + ri].tolist()
                n = min(len(labels), 32)
                if m[0] == 0 or any(m[i] == 0 for i in range(n)):
                    continue  # a composite that did not run / flush
                t0 = m[0] if t0 is None else min(t0, m[0])
                for i in range(1, n):
                    dur.setdefault(labels[i], []).append((m[i] - m[i - 1]) & 0xFFFFFFFF)
                    ends.setdefault(labels[i], []).append(m[i])
                    if labels[i] in ("P7", "P8", "P11", "P14", "P15"):
                        for gn, gs in grp.items():
                            if tuple(xy) in gs:
                                gends.setdefault(f"{labels[i]}@{gn}/{risc}", []).append(m[i])
        summ = {}
        for pid, ds in dur.items():
            ds = sorted(ds)
            summ[pid] = dict(max_us=round(ds[-1] / mhz, 1), med_us=round(ds[len(ds) // 2] / mhz, 1), n=len(ds),
                             end_us=round(((max(ends[pid]) - t0) & 0xFFFFFFFF) / mhz, 1) if t0 is not None else None)  # fmt: skip
        if t0 is not None and gends:
            summ["groups"] = {k: round(((max(v) - t0) & 0xFFFFFFFF) / mhz, 1) for k, v in sorted(gends.items())}
        out.append(summ)
    return out


def _mem_view():  # (2026-10-26 w7lv5) the allocator per chip: DRAM / L1 bytes allocated (per bank x banks)
    out_ = {}
    for nm_, bt_ in (("dram", ttnn.BufferType.DRAM), ("l1", ttnn.BufferType.L1)):
        mv_ = ttnn.get_memory_view(mesh, bt_)
        out_[nm_] = int(mv_.total_bytes_allocated_per_bank) * int(mv_.num_banks)
        out_[nm_ + "_free_largest_per_bank"] = int(mv_.largest_contiguous_bytes_free_per_bank)
    return out_


MEM = {"before_programs": _mem_view()}
ctl = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, a34_merge=False, a2_fuse=False, o_rects=0, **kw)
MEM["after_control"] = _mem_view()
if VARIANT in ("p1a2f", "a2fp10"):  # the M-A bisect: A1 (or A5) fused onto stage 2, the other in its own program
    sc_ = (tuple(ASM.P1A2F_SCOPE), ("P10",)) if VARIANT == "p1a2f" else (("P1",), tuple(ASM.A2FP10_SCOPE))
    fus = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, scopes=sc_, **kw)
    FSCOPE = sc_[0] if VARIANT == "p1a2f" else sc_[1]
elif VARIANT in (
    "a15b",
    "a15b11",
    "a15b13",
    "a15b12",
    "a15b15",
    "a15b16",
):  # M-B1 (a15b11 = W7 step 1: + P11, the AG of x-hat_f into XS, in P10's window) (2026-10-24 05:12:04Z): A15 + program B (P14) in ONE program; B's inputs PRE-STAGED by
    # the chain's op fillers (router / gather / inputs / shared expert: the router is not in-program yet)
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe as PM  # noqa: E402
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe2 as M2  # noqa: E402
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PSTK  # noqa: E402

    ep = getattr(pl, "epmoe", None) or EpMoe(pf, pl.moe, xsl=getattr(pl, "xsl", None))
    pl.epmoe = ep
    bcfg = PSTK.b_config({}, tp=NDEV)
    if not int(bcfg.get("out_bfp8", 0)):
        bcfg.update(yw_db=0, yw_coal=1, comb_split=0, yw_brisc=0)
    if VARIANT in (
        "a15b12",
        "a15b15",
        "a15b16",
    ):  # W7 step 3: B reads the router tail's [C, 8] uint16 ids / bf16 weights (both runs)
        bcfg["ids_u16"] = 1
    for k_, v_ in (
        args.get("b_cfg") or {}
    ).items():  # A/B knobs of program B (e.g. mcast_dual; blocks [mg, kc, nc]: the
        # fused program's member CBs sit under the 1,330,000 ruling with B at (1, 4, 4) -- 943,716 B vs (1, 4, 8)'s 1,107,556)
        bcfg[str(k_)] = tuple(int(x) for x in (v_.split(",") if isinstance(v_, str) else v_)) if k_ == "blocks" else v_
    G_ = int(bcfg["g"])
    lv_ = M2.Levers(x_l1=bool(bcfg["x_l1"]), mcast=G_, shared=True, ids_u16=bool(bcfg["ids_u16"]), out_bfp8=bool(bcfg["out_bfp8"]),
                    yw_db=bool(bcfg["yw_db"]), yw_coal=int(bcfg["yw_coal"]), ring_slots=int(bcfg["ring_slots"]),
                    gu_split=bool(bcfg["gu_split"]), comb_split=bool(bcfg["comb_split"]), yw_brisc=bool(bcfg["yw_brisc"]),
                    **({k_: bool(int(bcfg.get(k_, 0))) for k_ in ("valid", "snake", "xs_rows") if k_ in M2.Levers.__dataclass_fields__}))  # fmt: skip
    W_ = int(ep.down[0].shape[-2])
    lv_ = M2.fit_levers(M2.Geometry(c=c, w=W_, p=110, h_split=int(bcfg["h_split"])), lv_)
    strm_ = None
    if (
        NDEV == 8
    ):  # mb8: B's streamers off the lanes / tails (program A's fabric connection words take the ids 10.. there)
        rc_ = ctl.engine.place.role_cores()
        strm_ = M2.adj_pmk_streamers(mesh, members=list(ctl.members), avoid=list(rc_["lane"]) + list(rc_["tail"]))
    b_self_kw = {}
    if int(
        bcfg.get("selfstream", 0)
    ):  # B7g (b_inside_fused_plan.md I-1): 14's self-stream form on the member / MM-only /
        # root cores (the lanes / tails keep their fabric CBs); b_cfg blocks default to the 7g geometry (4, 8, 8)
        rc_ = ctl.engine.place.role_cores()
        # cores ROOTS + MM-ONLY FIRST: PMK_PLAN_P runs the routing plan on the first plan_cores of b.cores (14, 00:30:44Z)
        b_self_kw = dict(selfstream=True, cores=[tuple(c_) for r_ in ("root", "mm_only", "member") for c_ in rc_[r_]])
        if int(bcfg.get("plan_cores", 0) or 0):
            b_self_kw["plan_cores"] = int(bcfg["plan_cores"])
        if bcfg.get("ffn_fid"):  # B2.3 (14, d838042c7e9): the FFN matmuls at their own fidelity under the combine's
            b_self_kw["ffn_fid"] = str(bcfg["ffn_fid"])
        # Stage 14 B2.3 plan levers (w8/progb_plan_design.md): plan_cores = N (the plan on B's first N cores after the
        # hand-over) or "idle" (= the MM-only cores outside A's adj streamer list, i.e. not P13's streamers); plan_early 1
        # = that plan BEFORE the hand-over on those cores (overlapped with P13: PMK_PLAN_EARLY), plan_runs_nb = plan_runs'
        # expert batches (L1 at few plan cores). B's core list puts the plan cores first.
        pc_ = bcfg.get("plan_cores", 0)
        pe_ = int(bcfg.get("plan_early", 0))
        if pe_ or str(pc_) == "idle" or int(pc_ or 0):
            st_ = {tuple(c_) for c_ in ctl.streamers}
            idle_ = [tuple(c_) for c_ in rc_["mm_only"] if tuple(c_) not in st_]
            n_pc = len(idle_) if (pe_ or str(pc_) == "idle") else int(pc_)
            pool_ = idle_ + [tuple(c_) for c_ in rc_["mm_only"] + rc_["root"] if tuple(c_) not in idle_]
            first_ = pool_[:n_pc]
            assert not pe_ or first_ == idle_, (
                "the early plan runs on idle (non-streamer) MM-only cores only",
                n_pc,
                len(idle_),
            )
            b_self_kw["cores"] = first_ + [c_ for c_ in b_self_kw["cores"] if tuple(c_) not in first_]
            b_self_kw["plan_cores"] = n_pc
            b_self_kw["plan_runs_nb"] = int(bcfg.get("plan_runs_nb", 1))
            if pe_:
                # ids 2 / 3: never touched on MM-only by A / P12 / P13 / P15 (id 4 is P15's SEM_P15_TP8 on the TP8 stage-1
                # reducers, MM-only among them): the P12 signal shares id 2 with barrier B (b_wrap_w: base sig_n)
                b_self_kw["plan_early"] = dict(sig=2, sig_n=len(ctl.members), sem_a=3, sem_b=2)
            S.log(
                f"{tag}: B plan on {n_pc} cores ({'EARLY, under P13' if pe_ else 'after the hand-over'}), plan_runs_nb "
                f"{b_self_kw['plan_runs_nb']}; idle MM-only (non-streamer) {len(idle_)} of {len(rc_['mm_only'])}: {idle_}"
            )
        if "blocks" not in (args.get("b_cfg") or {}):
            bcfg["blocks"] = (4, 8, 8)
    b = M2.ProgramB(mesh, g_size=G_, levers=lv_, streamers=strm_, num_experts=ep.e_all, top_k=int(pl.moe.top_k), hidden=int(pl.moe.hidden),
                    limit=float(pl.moe.limit), gate_up=ep.gate_up, down=ep.down, blocks=bcfg["blocks"], fid=str(bcfg["fid"]),
                    probe=False, prefetch=bool(bcfg["prefetch"]), full_sync=bool(bcfg["full_sync"]), rect_mode=str(bcfg["rect"]),
                    gmax=int(bcfg["gmax"]), mcast_linked=bool(bcfg["mcast_linked"]), mcast_unicast=bool(bcfg["mcast_unicast"]),
                    mcast_dual=bool(bcfg["mcast_dual"]), lpt=int(bcfg["lpt"]), gu_split=bool(bcfg["gu_split"]),
                    h_split=int(bcfg["h_split"]), inplace=bool(bcfg["inplace"]), lpt_w=None, comb_split=bool(bcfg["comb_split"]),
                    yw_brisc=bool(bcfg["yw_brisc"]), compute_only=list(ctl.members), **b_self_kw)  # fmt: skip
    assert getattr(b, "inplace", False), "M-B1: program B reads pl.xsl in place (inplace=1)"
    bs_ = b.scratch(c)
    K_, H_ = int(pl.moe.top_k), int(pl.moe.hidden)
    dram = ttnn.DRAM_MEMORY_CONFIG
    BIN = dict(hb=ttnn.empty((1, 1, c, H_), BF16, ttnn.TILE_LAYOUT, mesh, dram),
               x_ext=ttnn.empty((c + 32, H_), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh, dram),
               ids_rm=ttnn.empty((1, K_ * c), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, dram),
               wt=ttnn.empty((1, 1, c, K_), FP32, ttnn.TILE_LAYOUT, mesh, dram))  # fmt: skip
    if lv_.ids_u16:  # the router tail's layouts (M2.tail_layout_inputs = the tail's weight formula to the bit)
        BIN.update(ids_rm=ttnn.empty((1, c * K_), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, mesh, dram),
                   wt=ttnn.empty((1, c * K_), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh, dram))  # fmt: skip
    fus = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, a15_fuse=True,
                          b_p11=VARIANT in ("a15b11", "a15b13", "a15b12", "a15b15", "a15b16"),
                          b_p13=VARIANT in ("a15b13", "a15b12", "a15b15", "a15b16"), b_p12=VARIANT in ("a15b12", "a15b15", "a15b16"),
                          b_p15=VARIANT in ("a15b15", "a15b16"), b_p16=VARIANT == "a15b16",
                          b_window=dict(b=b, x_ext=BIN["x_ext"], ids_rm=BIN["ids_rm"], wt=BIN["wt"], x_tile=BIN["hb"],
                                        y_s=bs_.get("y_s")), **kw)  # fmt: skip
    FSCOPE = tuple(
        {
            "a15b11": ASM.A15B11_SCOPE,
            "a15b13": ASM.A15B13_SCOPE,
            "a15b12": ASM.A15B12_SCOPE,
            "a15b15": ASM.A15B15_SCOPE,
            "a15b16": ASM.A15B16_SCOPE,
        }.get(VARIANT, ASM.A15B_SCOPE)
    )
    S.log(f"{tag}: program B {b.p} receivers / {len(b.streamers)} streamers, rects {b.rect_sizes}, levers {lv_}; "
          f"B window {fus.report.get('b_window')}")
elif VARIANT == "a15":  # M-A: the whole program A of the layer in one launch (P11 off)
    fus = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, a15_fuse=True, **kw)
    FSCOPE = tuple(ASM.A15_SCOPE)
elif VARIANT == "a15p11":  # the W7 step-1 bisect: A15 + P11 (the AG of x-hat_f into XS) in ONE program, no B
    fus = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, a15_fuse=True, a15_p11=True, **kw)
    FSCOPE = tuple(ASM.A15P11_SCOPE)
elif VARIANT == "a34r":
    fus = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, a34_merge=True, a2_fuse=False, o_rects=4, **kw)
    FSCOPE = tuple(ASM.A34_SCOPE)
else:
    # REUSE (fuse8 TP8 hang 02:36Z, A/B): the fused object shares the control's collectives (one C1X / RS8 instance
    # and sync shard on the lanes, as consecutive layers of the served stack do) instead of a second instance
    rk = dict(reuse=ctl) if bool(args.get("reuse", False)) else {}
    fus = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, a2_fuse=True, **kw, **rk)
    FSCOPE = tuple(ASM.A2F_SCOPE)
assert FSCOPE in [tuple(s_) for s_ in fus.scopes], fus.scopes
S.log(
    f"{tag}: program A x2 assembled in {time.time() - t1:.1f} s; control {ctl.scopes}; fused {fus.scopes}; "
    f"knobs {fus.report.get('knobs')}"
)


def l1_layout(prog, scope):
    """the fused program's windowed declaration per role: shared (in place) slots, each window's slots, pad, hosted"""
    e = prog.engine
    e.scope = set(scope)
    out = {}
    try:
        for role in ("member", "root", "mm_only"):
            wl = e._win_layout(role)
            cen = e._census(e.active_phases())[role]
            if wl is None:
                out[role] = dict(windowed=False, cb_bytes=cen["cb_bytes"], cb_end=cen["cb_end"],
                                 slots=[(list(i), b, n[:40]) for i, b, _, n in cen["slots"]][:60])  # fmt: skip
                continue
            sh = []
            for us, b in wl["decl"]:
                sh.append(([u.index for u in us], int(b), us[0].note[:40]))
            out[role] = dict(windowed=True, cb_bytes=wl["bytes"], cb_end=cen["cb_end"], span=wl["span"], pad=wl["pad"],
                             need=wl["need"], hosted=wl["hosted"], windows=wl["windows"], decl=sh,
                             tabs={w: [(t[0], t[2]) for t in tab] for w, tab in wl["tabs"].items()})  # fmt: skip
    finally:
        e.scope = None
    return out


_zeros = {}


def zero(t_):
    key = (tuple(t_.shape), t_.dtype, t_.layout)
    if key not in _zeros:
        _zeros[key] = ttnn.zeros(
            tuple(t_.shape), dtype=t_.dtype, layout=t_.layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    ttnn.copy(_zeros[key], t_)


def use(prog, scopes):
    prog.layer.engine.programs = [set(s_) for s_ in scopes]
    prog.layer.order = [set(s_) for s_ in scopes]


def _copy(src, dst):
    if src.layout != dst.layout or tuple(src.shape) != tuple(dst.shape):
        t_ = ttnn.to_layout(
            ttnn.reshape(src, tuple(int(v) for v in dst.shape)), dst.layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.copy(t_, dst)
        ttnn.deallocate(t_)
    else:
        ttnn.copy(src, dst)


def b_chain(prog):
    """M-B1's reference FFN half on the chain's hn[1]: the op fillers (pmk_stack._ffn_b_steps' router / gather / inputs /
    shared, on the PMK x-hat_f path) into the PERSISTENT inputs the fused program's B reads, then program B standalone
    """
    h1 = prog.hns[1]
    pl._normed["ffn"] = h1
    x32 = pl.rms(h1, "ffn", FP32)
    idx, scores = pl.moe.router(x32, decode=False)
    sel_l = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), FP32)
    ids_l = ttnn.typecast(idx, FP32)
    ids_f, sel = pf.gather(ids_l, sp), pf.gather(sel_l, sp)
    idx16 = ttnn.typecast(ids_f, ttnn.uint16)
    hb = pf.gather(ttnn.typecast(x32, BF16), sp)
    x_ext, ids_rm, wt = PM.inputs_from_op_path(hb, idx16, sel, float(pl.moe.scaling))
    if lv_.ids_u16:  # the router tail's [C, K] uint16 ids / bf16 weights (W7 step 3: what P12's tail writes)
        ttnn.deallocate(ids_rm)
        ttnn.deallocate(wt)
        ids_rm, wt = M2.tail_layout_inputs(idx16, sel, float(pl.moe.scaling))
    y_s = swiglu_prefill(pl.shared, hb)
    for src, dst in (
        (hb, BIN["hb"]),
        (x_ext, BIN["x_ext"]),
        (ids_rm, BIN["ids_rm"]),
        (wt, BIN["wt"]),
        (y_s, bs_["y_s"]),
    ):
        _copy(src, dst)
    for t_ in (x32, idx, scores, sel_l, ids_l, ids_f, sel, idx16, hb, x_ext, ids_rm, wt, y_s):
        ttnn.deallocate(t_)
    b(BIN["x_ext"], BIN["ids_rm"], BIN["wt"], c, x_tile=BIN["hb"])


# POISON (program 01:56:05Z, the rule of record: bitwise + det on reused inputs is BLIND to premature reads): before
# every fused launch, in the trace, B's y / P13's y_s / P15's y15 are overwritten with NaN (device copies from NaN tensors
# allocated once here, outside any capture) -- a phase that reads a row before its producer wrote it this replay reads
# NaN instead of the previous replay's identical value
POISON = bool(int(args.get("poison", 0)))
_PZ = {}


def _poison_targets(prog):
    if not (POISON and prog is fus):
        return []
    t = []
    if VARIANT in ("a15b", "a15b11", "a15b13", "a15b12", "a15b15", "a15b16"):
        t.append(("b_y", bs_["y"]))
    if VARIANT in ("a15b13", "a15b12", "a15b15", "a15b16"):
        t.append(("y_s", prog.sh_dn.y))
    if VARIANT in ("a15b15", "a15b16"):
        t.append(("y15", prog.rsr15.y if prog.rsr15 is not None else prog.rs8_15.y))
    # TP8: RS8 stage 1's fp32 output r1 is written by this chip's own reducers -> poisonable; the landing buffers r / r2
    # are written by the PEERS' fabric sends, which an in-trace copy here would race (no cross-chip order between this
    # chip's copy and a peer already running its program) -- their readers are ordered by the ARR words instead
    for k_, rs_ in (("r1_8", getattr(prog, "rs8", None)), ("r1_15", getattr(prog, "rs8_15", None))):
        if rs_ is not None:
            t.append((k_, rs_.rs1.y))
    return t


def _poison_alloc(prog):
    for k_, t_ in _poison_targets(prog):
        if k_ not in _PZ:
            _PZ[k_] = ttnn.from_torch(torch.full(tuple(t_.shape), float("nan")), dtype=t_.dtype, layout=t_.layout, device=mesh,
                                      memory_config=t_.memory_config(),
                                      mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if NDEV > 1 else None)  # fmt: skip


POISON_PLAN = bool(
    int(args.get("poison_plan", 0))
)  # the poison rule (program 01:56:05Z): B's plan tables overwritten
# with in-range WRONG values (cnt / start / meta / dest 0, disp C) inside the trace before every fused launch -- a plan that
# did not finish before B's readers shows as a y mismatch (vs the chain / the cross-job dumps), never as a hang
PZ = {}
if POISON_PLAN and VARIANT in ("a15b", "a15b11", "a15b13", "a15b12", "a15b15", "a15b16"):
    tb0_ = b.tables(c)
    rp_ = ttnn.ReplicateTensorToMesh(mesh) if NDEV > 1 else None
    for k_, v_ in (("dest", 0), ("cnt", 0), ("start", 0), ("meta", 0), ("disp", c)):
        PZ[k_] = ttnn.from_torch(torch.full(tuple(tb0_[k_].shape), int(v_), dtype=torch.int32), dtype=ttnn.uint32,
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rp_)  # fmt: skip
    S.log(f"{tag}: POISON the plan tables before every fused launch: {sorted(PZ)}")


def fn(prog, scopes=None, reset=True):
    use(prog, scopes or prog.scopes)
    if PZ and prog is fus:  # the poison rule: B's plan tables wrong before the fused program runs
        tb_ = b.tables(c)
        for k_, pz_ in PZ.items():
            ttnn.copy(pz_, tb_[k_])
    if reset:
        ttnn.copy(x0, x_p)
        zero(state_p.S)
        zero(state_p.C)
        for k_, t_ in _poison_targets(prog):
            ttnn.copy(_PZ[k_], t_)
    tbl = (
        VARIANT in ("a15b", "a15b11", "a15b13", "a15b12", "a15b15", "a15b16") or bool(args.get("fus_table", False))
    ) and prog is fus  # M-B1 / the table-mode A/B: the
    # fused program's composites read their args from the DRAM table
    old_ = os.environ.get("GLM53_PMK_ARGS")
    if tbl:  # (2 KB rows: the member BRISC carries A's + B's receiver args); the chain's programs / B stay trace
        os.environ["GLM53_PMK_ARGS"] = "table"
    if RF_ is not None and prog is fus:  # the fence before every fused launch
        RF_()
    try:
        prog(x_p, ci)
    finally:
        if tbl:
            if old_ is None:
                os.environ.pop("GLM53_PMK_ARGS", None)
            else:
                os.environ["GLM53_PMK_ARGS"] = old_
    if (
        VARIANT in ("a15b", "a15b11", "a15b13", "a15b12", "a15b15", "a15b16") and prog is ctl and scopes is None
    ):  # the chain = A1 .. A5 + fillers + program B
        b_chain(prog)


per_chip = lambda t_: [
    ttnn.to_torch(d).float().clone() for d in (ttnn.get_device_tensors(t_) if NDEV > 1 else [t_])
]  # noqa: E731


def outputs(prog):
    out = {k_: per_chip(t_) for k_, t_ in prog.stores().items()}
    if VARIANT in (
        "a15b",
        "a15b11",
        "a15b13",
        "a15b12",
        "a15b15",
        "a15b16",
    ):  # program B's output per chip (before the RS), the same scratch for both runs
        out["b_y"] = per_chip(bs_["y"])
    if VARIANT in (
        "a15b13",
        "a15b12",
        "a15b15",
        "a15b16",
    ):  # W7 step 2: the shared expert's y_s (control: the chain's op filler; fused: P13)
        out["y_s"] = per_chip(prog.sh_dn.y if prog is fus else bs_["y_s"])
    if (
        VARIANT in ("a15b15", "a15b16") and prog is fus
    ):  # W7 step 4: P15's reduced FFN rows (fp32, the chip's owned rows)
        out["y15"] = per_chip(prog.rsr15.y if prog.rsr15 is not None else prog.rs8_15.y)
    if VARIANT in (
        "a15b12",
        "a15b15",
        "a15b16",
    ):  # W7 step 3: the routes (control: the op router + tail_layout_inputs; fused: P12's tail)
        out["ids"] = per_chip(prog.rt_ids if prog is fus else BIN["ids_rm"])
        out["wt"] = per_chip(prog.rt_wt if prog is fus else BIN["wt"])
    if VARIANT in (
        "a15",
        "a15b",
        "a15b11",
        "a15b13",
        "a15b12",
        "a15b15",
        "a15b16",
        "a15p11",
        "p1a2f",
        "a2fp10",
    ):  # A1's outputs: the x-hat rows the P2 holders read and the post coefficients P10 uses
        out.update(xhat=per_chip(prog.hns[0]), post=per_chip(prog.posts[0]), comb=per_chip(prog.combs[0]))
    if "terms" in prog.scr:  # the kda items' DRAM output (H3b), read by the scan: separates items from scan
        out["terms"] = per_chip(prog.scr["terms"])
    out.update(y=per_chip(prog.ys[0]), on=per_chip(prog.scr["on"]), S=per_chip(state_p.S), C=per_chip(state_p.C),
               x1=per_chip(x_p), xhat_f=per_chip(prog.hns[1]), post_f=per_chip(prog.posts[1]), comb_f=per_chip(prog.combs[1]))  # fmt: skip
    return out


def shared_ref():
    """the shared expert in fp32 on the host, per chip: SwiGLU(x @ W_gu) @ W_d with the chip's bf8 weight slice (exact
    values) on the chain's gathered bf16 x-hat_f rows (BIN hb, identical on every chip)"""
    torch.set_num_threads(16)
    sh = pl.shared
    i_ = int(sh.intermediate)
    ie_ = ASM.p13_slice(i_)  # PMK_P13_SLICE: P13 runs the slice's first ie_ columns
    S.log(
        f"{tag}: shared weights gate_up {sh.gate_up.dtype} down {sh.down.dtype}; ds {getattr(getattr(sh, 'ds_gate_up', None), 'weight', None) is not None and sh.ds_gate_up.weight.dtype}; "
        f"ckc {getattr(sh, 'ckc', None)}; P13 w {fus.sh_gu.w_dtype} fidelity {fus.sh_gu.fidelity}"
    )
    xs_ = per_chip(BIN["hb"])
    gus = [ttnn.to_torch(d).float() for d in (ttnn.get_device_tensors(sh.gate_up) if NDEV > 1 else [sh.gate_up])]
    dns = [ttnn.to_torch(d).float() for d in (ttnn.get_device_tensors(sh.down) if NDEV > 1 else [sh.down])]
    out_ = []
    for x_, gu_, dn_ in zip(xs_, gus, dns):
        x2 = x_.reshape(-1, x_.shape[-1])
        gu2 = gu_.reshape(x2.shape[-1], -1)
        h_ = x2 @ gu2
        g_, u_ = h_[:, :ie_], h_[:, i_ : i_ + ie_]
        lim = float(sh.limit)
        act = torch.nn.functional.silu(torch.clamp(g_, max=lim)) * torch.clamp(u_, -lim, lim)
        out_.append(act @ dn_.reshape(i_, -1)[:ie_, : x2.shape[-1]])
    return out_


def compare(a, b):
    out = {}
    for k in a:
        cs = []
        for x_, y_ in zip(a[k], b[k]):
            x2, y2 = x_.reshape(-1, x_.shape[-1]), y_.reshape(-1, y_.shape[-1])
            xd, yd = x2.double(), y2.double()  # scale <x, y> / <y, y>: PCC is scale-blind (program 20:32:15Z rule)
            cs.append(dict(bitwise=bool(torch.equal(x2, y2)), maxabs=float((x2 - y2).abs().max()),
                           pcc=round(C.pcc(x2, y2), 7), finite=bool(torch.isfinite(x2).all()),
                           scale=round(float((xd * yd).sum() / (yd * yd).sum().clamp_min(1e-300)), 6)))  # fmt: skip
        out[k] = cs
    return out


def capture(prog, scopes=None, reset=True):
    assert all(k_ in _PZ for k_, _ in _poison_targets(prog)), "POISON: the NaN sources are allocated before any capture"
    t_ = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn(prog, scopes, reset)
    finally:
        ttnn.end_trace_capture(mesh, t_, cq_id=0)
    return t_


if VARIANT in ("a15b", "a15b11", "a15b13", "a15b12", "a15b15", "a15b16"):
    TEST = (f"PMK per-layer megakernel M-B1: A1 .. A5 + program B (P14: plan / routed experts / combine, the reclaim window) in "
            f"ONE program vs the A1 | .. | A5 chain + op fillers + program B, B's inputs PRE-STAGED by the chain's op fillers "
            f"(router not in-program), program A+B L{L} tp{NDEV} bucket {c}, traced; criterion bitwise per chip incl. B's y + "
            f"{DET} replays")
    MILESTONE = "PMK-M-B1"
    if VARIANT in ("a15b15", "a15b16"):
        TEST = (f"PMK per-layer megakernel M-B2 step 4: A1 .. A5 + P11 + P12 (router) + P13 (shared) + program B (P14) + P15 (the RS "
                f"of B's y to the row owners: a second RSR on the lanes after every member's after-B signal) in ONE program vs "
                f"the chain + op fillers + program B, B's x rows PRE-STAGED, program L{L} tp{NDEV} bucket {c}, traced; criterion "
                f"A bitwise, routes as step 3, y_s as step 2, B's y PCC >= 0.9999, P15's rows BITWISE = own + peer of the "
                f"program's own B y in fp32 and PCC >= 0.9999 vs the chain's + {DET} replays")
        MILESTONE = "PMK-M-B2-S4"
        if VARIANT == "a15b16":  # M1 (program 23:49:25Z): + P16 in the same program
            TEST += (
                " + M1: P16 (the mHC post of the FFN) in-program after P15 (H10: P15's producers +1 every member's "
                "FS_REDUCED; y15 from DRAM), its residual vs the host post of the chain's x1 with P15's y15: PCC >= 0.99999, "
                "|scale - 1| <= 0.005"
            )
            MILESTONE = "PMK-M1"
    elif VARIANT == "a15b12":
        TEST = (
            f"PMK per-layer megakernel M-B2 step 3: A1 .. A5 + P11 (AG of x-hat_f) + P12 (router logits PfStream + router tail "
            f"on the members) + P13 (shared expert) + program B (P14: its ids / weights = P12's, y_s = P13's) in ONE program "
            f"vs the chain + op fillers + program B, B's x rows PRE-STAGED, program L{L} tp{NDEV} bucket {c}, traced; "
            f"criterion A bitwise per chip, ids rows equal >= 99.9 %, weights rel L2 <= 1e-3 on them, y_s no worse than the "
            f"op vs fp32, B's y PCC >= 0.9999 + {DET} replays"
        )
        MILESTONE = "PMK-M-B2-S3"
    elif VARIANT == "a15b13":
        TEST = (
            f"PMK per-layer megakernel M-B2 step 2: A1 .. A5 + P11 (AG of x-hat_f) + P13 (shared expert, two PfStream phases "
            f"on the members) + program B (P14, its y_s = P13's) in ONE program vs the chain + op fillers + program B, B's "
            f"routes / x PRE-STAGED, program L{L} tp{NDEV} bucket {c}, traced; criterion A bitwise per chip, y_s PCC >= "
            f"0.99999, B's y PCC >= 0.9999 + {DET} replays"
        )
        MILESTONE = "PMK-M-B2-S2"
elif VARIANT == "a15":
    TEST = (
        f"PMK per-layer megakernel M-{'C' if NDEV == 8 else 'A'}: A1 + A2 + A3 + A4 + A5 = {{P1 .. P10}} in ONE program "
        f"(five phase windows, in-program H1 / H3 / H4 / H6) vs the A1 | A2 | A3 | A4 | A5 chain, program A L{L} tp{NDEV} "
        f"bucket {c}, traced; criterion bitwise per chip (A1 / A2 stores, y, on, S, C, x1, x-hat_f, post_f, comb_f) + {DET} "
        f"replays"
    )
    MILESTONE = "PMK-M-C" if NDEV == 8 else "PMK-M-A"
else:
    TEST = (
        f"PMK per-layer kernel ({VARIANT}) stage 2 / o_proj row groups: A2 + A3 + A4 = {{P2 .. P8}} in ONE program (three "
        f"phase windows, in-program H3, C1 then RSR on the lanes) vs the A1 | A2 | A3 | A4 | A5 chain, program A L{L} "
        f"tp{NDEV} bucket {c}, traced; criterion bitwise per chip (the A2 stores, y, on, S, C, x1, x-hat_f, post_f, comb_f) "
        f"+ {DET} replays"
    )
    MILESTONE = "PMK-FUSE-2"
MEM[
    "after_fused_build"
] = _mem_view()  # the fused program's objects (+ program B / its inputs, built after the control)


row = dict(variant=VARIANT, test=TEST, milestone=MILESTONE, layer=L, tp=NDEV, bucket=c, control=[list(s_) for s_ in ctl.scopes],
           fused=[list(s_) for s_ in fus.scopes], knobs=fus.report.get("knobs"), mock=C.MOCK, tag=tag)  # fmt: skip
# PMK_RS8_QSLICE (the column-sliced RS8, rs_tail_design.md; args env): Q of P8's RS8 in both programs (1 = today's)
row["rs8_slices"] = dict(fused=(fus.report.get("rs8") or {}).get("slices"), control=(ctl.report.get("rs8") or {}).get("slices"))
if row["rs8_slices"]["fused"] not in (None, 1):
    TEST += f" [PMK_RS8_QSLICE={row['rs8_slices']['fused']}: P8's RS8 per column slice]"
    row["test"] = TEST
# SHA_OUT / SHA_REF (the cross-job bitwise A/B, e.g. PMK_RS8_QSLICE 4 vs 0): per output and chip the sha1 of the first fused
# replay's (and the control's) bytes -> SHA_OUT (json); SHA_REF = another job's SHA_OUT: every common output bitwise equal
SHA_OUT, SHA_REF = args.get("sha_out") or None, args.get("sha_ref") or None
class _L1Done(Exception):
    pass


try:
    if L1_REPORT or CB_MODEL:
        raise _L1Done()
except _L1Done:
    if L1_REPORT:
        rep = l1_layout(fus, FSCOPE)
        for role, v in rep.items():
            S.log(f"{tag}: L1 {role}: " + str({k: v[k] for k in v if k not in ("decl", "slots", "tabs")}))
            for ent in v.get("decl", v.get("slots", [])):
                S.log(f"{tag}:   {role} {ent}")
            for w, tab in (v.get("tabs") or {}).items():
                S.log(f"{tag}:   {role} window {w + 1}: {tab}")
        for role_ in ("member", "mm_only", "root"):  # every declared CB: (index, format, page, pages, live phases)
            for idx_, u_ in sorted((fus.engine.cbs.get(role_) or {}).items()):
                S.log(f"{tag}: CBUSE {role_} {idx_} {u_.fmt} page {u_.page_bytes} pages {u_.pages} live {list(u_.live)} "
                      f"users {list(u_.users)} solo {u_.solo} | {str(u_.note)[:60]}")  # fmt: skip
        sems_ = {}
        for sid_, roles_ in sorted((getattr(fus.engine, "sems", None) or {}).items()):
            for r_ in roles_:
                sems_.setdefault(r_, []).append(sid_)
        S.log(f"{tag}: SEMS {sems_}")
        row.update(status="l1report", l1=rep)
        print("PMK_FUSE3_UNIT l1report done", flush=True)
    if CB_MODEL:
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_cb_model as CBM  # noqa: E402

        cbm = {}
        for nm_, pr_, scs_ in (("chain", ctl, [tuple(s_) for s_ in ctl.scopes]), ("fused", fus, [FSCOPE])):
            for sc_ in scs_:
                m_ = CBM.model(pr_.engine, scope=sc_)
                key_ = f"{nm_}:{'+'.join(sorted(sc_))}"
                cbm[key_] = dict(n_cbs=m_["n_cbs"], n_violations=m_["n_violations"], n_unresolved=m_["n_unresolved"],
                                 violations=m_["violations"][:24], unresolved=m_["unresolved"][:60],
                                 n_reclaim_hazards=m_.get("n_reclaim_hazards", 0), reclaim=m_.get("reclaim", []))  # fmt: skip
                S.log(f"{tag}: CB MODEL {key_}\n" + CBM.report_text(m_))
        row.update(status="cbmodel", cb_model=cbm)
        print(
            "PMK_FUSE3_UNIT cbmodel done "
            + str(
                {k: (v["n_cbs"], v["n_violations"], v["n_unresolved"], v["n_reclaim_hazards"]) for k, v in cbm.items()}
            ),
            flush=True,
        )
try:
    if L1_REPORT or CB_MODEL:
        raise _L1Done()
    # FUSED_FIRST (A/B): the fused chain's compile launch before the control's (no earlier C1X / RS8 user on the lanes)
    order = (
        [("fused", fus), ("control", ctl)]
        if bool(args.get("fused_first", False))
        else [("control", ctl), ("fused", fus)]
    )
    _poison_alloc(fus)  # (POISON: the NaN sources, outside any capture)
    row["poison"] = sorted(_PZ)
    for nm_, pr_ in order:
        S.log(f"{tag}: {nm_} compile launch")
        fn(pr_)
        ttnn.synchronize_device(mesh)
        row["census_" + nm_] = dict(pr_.layer.launch_census)
    row["windows"] = getattr(fus.engine, "window_report", None)
    # the host audits of both programs (each RAISES before a launch on a violation): the runtime-arg audit per program /
    # rank, the fabric packet-header budget and the EDM sender channels (the rs8q mock verdicts)
    row["arg_audit_errors"] = {nm_: {str(k_): int(v_.get("errors", -1)) for k_, v_ in (getattr(pr_.engine, "arg_audits", None) or {}).items()}
                               for nm_, pr_ in (("control", ctl), ("fused", fus))}  # fmt: skip
    row["fabric_audits"] = {nm_: dict(headers=pr_.report.get("fabric_headers"), edm_channels=pr_.report.get("edm_channels"))
                            for nm_, pr_ in (("control", ctl), ("fused", fus))}  # fmt: skip
    S.log(f"{tag}: rs8 slices {row['rs8_slices']}; arg audit errors {row['arg_audit_errors']}; rs8 {fus.report.get('rs8')}")
    if DUMP_SRC:  # (scope tag, role, risc, rank) -> source; the fused program's only
        os.makedirs(DUMP_SRC, exist_ok=True)
        for (tg_, role_, risc_, rank_), src_ in (getattr(fus.engine, "sources", None) or {}).items():
            with open(os.path.join(DUMP_SRC, f"{tg_}_{role_}_{risc_}_r{rank_}.cpp"), "w") as f_:
                f_.write(src_)
    S.log(f"{tag}: census control {row['census_control']}; fused {row['census_fused']}; windows {row['windows']}")
    # the x1 race (w7tp8 / w7tp8c / w7tp8d: the in-place residual corrupted after P10 at TP8 on chips 0-4): the phases whose
    # runtime args hold the residual's (x_p) or the restore source's (x0) DRAM address, per program (host, mock and device)
    xb_ = int(x_p.buffer_address())
    row["addr_hits"] = {nm_: fus.engine.addr_hits({nm_: a_}, spans=sp_) for nm_, a_, sp_ in (
        ("x_p", xb_, None), ("x0", int(x0.buffer_address()), None), ("x_p_range", xb_, {"x_p_range": 2 << 20}))}  # fmt: skip
    row["addr_hits"] = {k_: v_[k_] for k_, v_ in row["addr_hits"].items()}
    S.log(f"{tag}: addr hits (fused) {row['addr_hits']}")
    if C.MOCK:
        row["status"] = "mock"
    else:
        tid = capture(ctl)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ref = outputs(ctl)
        perf_ctl = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} control")
        ttnn.release_trace(mesh, tid)
        tid = capture(fus)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        got = outputs(fus)
        if args.get(
            "dump"
        ):  # the first fused replay's B / shared outputs for a bitwise A/B between two jobs (e.g. B's fid)
            dump_ = {k_: [t_.float() for t_ in got[k_]] for k_ in ("b_y", "y_s", "y15") if k_ in got}
            if VARIANT in (
                "a15b",
                "a15b11",
                "a15b13",
                "a15b12",
                "a15b15",
                "a15b16",
            ):  # + B's plan tables (14's plan levers: bitwise)
                tb_ = b.tables(c)
                dump_.update({f"plan_{k_}": [ttnn.to_torch(d_).clone() for d_ in (ttnn.get_device_tensors(tb_[k_]) if NDEV > 1 else [tb_[k_]])]
                              for k_ in ("dest", "cnt", "start", "meta", "disp")})  # fmt: skip
            torch.save(dump_, str(args["dump"]))
        if SHA_OUT or SHA_REF:  # the cross-job bitwise A/B (PMK_RS8_QSLICE 4 vs 0): sha1 of each output's bits per chip
            import hashlib  # noqa: PLC0415
            import json  # noqa: PLC0415

            sha_ = lambda outs: {k_: [hashlib.sha1(t_.contiguous().view(torch.int32).numpy().tobytes()).hexdigest()[:20]  # noqa: E731
                                      for t_ in v_] for k_, v_ in outs.items()}  # fmt: skip
            row["sha"] = dict(fused=sha_(got), control=sha_(ref))
            if SHA_OUT:
                os.makedirs(os.path.dirname(os.path.abspath(str(SHA_OUT))), exist_ok=True)
                with open(str(SHA_OUT), "w") as f_:
                    json.dump(dict(tag=tag, rs8_slices=row["rs8_slices"], sha=row["sha"]), f_, indent=1)
            if SHA_REF:
                with open(str(SHA_REF)) as f_:
                    sr_ = json.load(f_)
                eq_ = {side: {k_: [a_ == b_ for a_, b_ in zip(v_, sr_["sha"][side][k_])] for k_, v_ in row["sha"][side].items()
                              if k_ in sr_["sha"].get(side, {})} for side in ("fused", "control")}  # fmt: skip
                miss_ = {side: sorted(set(row["sha"][side]) ^ set(sr_["sha"].get(side, {}))) for side in ("fused", "control")}
                row["vs_ref"] = dict(ref=str(SHA_REF), ref_tag=sr_.get("tag"), ref_rs8_slices=sr_.get("rs8_slices"),
                                     ok=all(all(v_) for d_ in eq_.values() for v_ in d_.values()) and not any(miss_.values()),
                                     differ={side: [k_ for k_, v_ in d_.items() if not all(v_)] for side, d_ in eq_.items()},
                                     missing=miss_)  # fmt: skip
                S.log(f"{tag}: vs the reference job {row['vs_ref']}")
        y15 = got.get("y15")  # W7 step 4: checked against the host reduce of B's y below (no chain counterpart; det covers it)
        # M1 (a15b16): P16 updated the residual in place -> the fused x1 is the POST-P16 residual (no chain counterpart: the
        # chain stops at P10's x1); it is checked below against the host mhc post of the chain's x1 with P15's y15
        cmp = compare(
            {k_: v_ for k_, v_ in got.items() if k_ != "y15" and not (k_ == "x1" and VARIANT == "a15b16")}, ref
        )
        if not all(ch["bitwise"] for ch in cmp.get("x1", [])):  # the x1 race: WHERE (residual tile (rb, s, c)) and WHAT
            # (the input x0's tile = a stale / lost store; else another writer's data)
            x0h = per_chip(x0)
            loc = []
            for ch_, (g_, r_, z_) in enumerate(zip(got["x1"], ref["x1"], x0h)):
                g2, r2, z2 = (t_.reshape(-1, t_.shape[-1]) for t_ in (g_, r_, z_))
                bad = (g2 != r2).reshape(g2.shape[0] // 32, 32, g2.shape[1] // 32, 32).any(dim=3).any(dim=1)  # [rb, ct]
                if not bool(bad.any()):
                    loc.append(dict(chip=ch_, n_bad=0))
                    continue
                rb_, ct_ = bad.nonzero(as_tuple=True)
                gt = g2.reshape(g2.shape[0] // 32, 32, g2.shape[1] // 32, 32)
                zt = z2.reshape(gt.shape)
                stale = [bool(torch.equal(gt[a_, :, b_], zt[a_, :, b_])) for a_, b_ in zip(rb_.tolist(), ct_.tolist())]
                zero = [bool((gt[a_, :, b_] == 0).all()) for a_, b_ in zip(rb_.tolist(), ct_.tolist())]
                loc.append(dict(chip=ch_, n_bad=int(bad.sum()), of=int(bad.numel()), rb=sorted(set(rb_.tolist())),
                                s=sorted(set((ct_ // 128).tolist())), c_min=int((ct_ % 128).min()), c_max=int((ct_ % 128).max()),
                                stale_x0=sum(stale), zero=sum(zero), first=[(int(a_), int(b_)) for a_, b_ in zip(rb_[:6], ct_[:6])]))  # fmt: skip
            row["x1_loc"] = loc
            S.log(f"{tag}: x1 mismatch map {loc}")
        if getattr(
            fus, "xext_on", False
        ):  # P11x (PMK_XEXT): the in-program x_ext rows [0, C) vs the harness-staged copy
            xa = [ttnn.to_torch(t_).reshape(-1, H_)[:c] for t_ in ttnn.get_device_tensors(fus.xext)]
            xb = [ttnn.to_torch(t_).reshape(-1, H_)[:c] for t_ in ttnn.get_device_tensors(BIN["x_ext"])]
            row["xext"] = dict(bitwise=[bool(torch.equal(a_, b_)) for a_, b_ in zip(xa, xb)],
                               maxabs=[float((a_.float() - b_.float()).abs().max()) for a_, b_ in zip(xa, xb)])  # fmt: skip
            S.log(f"{tag}: P11x x_ext vs the staged copy {row['xext']}")
        PCC_KEYS = (
            {"y_s": 0.9999, "b_y": 0.9999} if VARIANT in ("a15b13", "a15b12", "a15b15", "a15b16") else {}
        )  # P13 vs the op shared expert
        if VARIANT in (
            "a15b12",
            "a15b15",
            "a15b16",
        ):  # P12: ids judged by the row match rate below; weights by PCC over all rows
            PCC_KEYS.update(ids=-1.0, wt=0.999)
        ok_bw = all(ch["bitwise"] for k_, v in cmp.items() if k_ not in PCC_KEYS for ch in v)
        ok_bw = ok_bw and all(row.get("xext", {}).get("bitwise", [True]))
        pcc_min = {k_: min(ch["pcc"] for ch in cmp[k_]) for k_ in PCC_KEYS if k_ in cmp}
        ok_pcc = all(pcc_min.get(k_, 0.0) >= t_ for k_, t_ in PCC_KEYS.items())
        # every PCC-judged output also gates |scale - 1| <= 0.005 (the program's 20:32:15Z rule; ids are judged by row match)
        scale_dev = {
            k_: max(abs(ch["scale"] - 1.0) for ch in cmp[k_]) for k_, t_ in PCC_KEYS.items() if t_ >= 0 and k_ in cmp
        }
        row["scale_dev_max"] = {k_: round(v_, 6) for k_, v_ in scale_dev.items()}
        # (w7k1 11:11Z) y_s's scale is gated vs the fp32 reference below (the op is the less exact side), not vs the op
        ok_pcc = ok_pcc and all(
            v_ <= 0.005
            for k_, v_ in scale_dev.items()
            if not (k_ == "y_s" and VARIANT in ("a15b13", "a15b12", "a15b15", "a15b16"))
        )
        if y15 is not None:  # P15: own + peer rows of B's y in fp32, per owner chip (chip o = rows [o C / n, (o + 1) C / n))
            rows_ = c // NDEV

            def red(
                by,
            ):  # the device's order: TP2 own + peer (exact); TP8 RS8 = stage 1 the board pair (exact fp32), stage 2
                # ((t0 + t1) + (t2 + t3)) over the row's column positions (chip (r, c) = rank 4 r + c owns rows rank x C / 8)
                b2 = [b_.reshape(-1, b_.shape[-1]).float() for b_ in by]
                out_ = []
                for o in range(NDEV):
                    R = slice(o * rows_, (o + 1) * rows_)
                    if NDEV == 2:
                        out_.append(b2[0][R] + b2[1][R])
                    else:
                        t = [b2[cc][R] + b2[4 + cc][R] for cc in range(4)]
                        out_.append((t[0] + t[1]) + (t[2] + t[3]))
                return out_

            self_ref, chain_ref = red(got["b_y"]), red(ref["b_y"])
            y2 = [t_.reshape(-1, t_.shape[-1]) for t_ in y15]
            p15 = dict(bitwise_vs_own_b=[bool(torch.equal(a_, b_)) for a_, b_ in zip(y2, self_ref)],
                       maxabs_vs_own_b=[float((a_ - b_).abs().max()) for a_, b_ in zip(y2, self_ref)],
                       pcc_vs_chain=[round(C.pcc(a_, b_), 7) for a_, b_ in zip(y2, chain_ref)])  # fmt: skip
            row["p15"] = p15
            ok_pcc = ok_pcc and all(p15["bitwise_vs_own_b"]) and all(v_ >= 0.9999 for v_ in p15["pcc_vs_chain"])
            S.log(f"{tag}: P15 {p15}")
        if (
            VARIANT == "a15b16"
        ):  # M1: P16 = the mHC post (FFN) in-program: x' = post_f y15 + C_f^T x1 (tt/mhc_post.reference_step,
            # float64) on the chain's x1 (bitwise = the fused pre-P16 residual: A is bitwise) and the fused run's y15 / post_f /
            # comb_f (P15's landing; P10's coefficients, bitwise vs the chain's)
            from models.autoports.zai_org_glm_5_3_flash.tt import mhc_post as MPO  # noqa: PLC0415

            p16 = dict(pcc=[], maxabs=[], scale=[], finite=[])
            for x16_, x1_, y_, po_, co_ in zip(got["x1"], ref["x1"], y15, got["post_f"], got["comb_f"]):
                T_ = x1_.reshape(-1, x1_.shape[-1]).shape[0]
                want = MPO.reference_step(
                    y_.reshape(T_, -1), x1_.reshape(T_, 4, -1), po_.reshape(T_, 4), co_.reshape(T_, 4, 4)
                ).reshape(T_, -1)
                g_ = x16_.reshape(T_, -1).double()
                p16["pcc"].append(round(C.pcc(g_.float(), want.float()), 7))
                p16["maxabs"].append(float((g_ - want).abs().max()))
                p16["scale"].append(round(float((g_ * want).sum() / (want * want).sum().clamp_min(1e-300)), 6))
                p16["finite"].append(bool(torch.isfinite(g_).all()))
            row["p16"] = p16
            ok_pcc = (
                ok_pcc
                and all(p16["finite"])
                and all(v_ >= 0.99999 for v_ in p16["pcc"])
                and all(abs(v_ - 1.0) <= 0.005 for v_ in p16["scale"])
            )
            S.log(f"{tag}: P16 (the mHC post in-program) vs the host post of the chain's x1 {p16}")
        if VARIANT in (
            "a15b12",
            "a15b15",
            "a15b16",
        ):  # the router: rows whose 8 ids (in order) equal the op router's; weights on those rows
            match = []
            for a_, r_ in zip(got["ids"], ref["ids"]):
                a2, r2 = a_.reshape(-1, K_), r_.reshape(-1, K_)
                match.append(round(float((a2 == r2).all(dim=-1).float().mean()), 6))
            wrel = []
            for (ai, ri), (aw, rw) in zip(zip(got["ids"], ref["ids"]), zip(got["wt"], ref["wt"])):
                m_ = (ai.reshape(-1, K_) == ri.reshape(-1, K_)).all(dim=-1)
                d_ = aw.reshape(-1, K_)[m_] - rw.reshape(-1, K_)[m_]
                wrel.append(float(d_.norm() / rw.reshape(-1, K_)[m_].norm().clamp(min=1e-30)))
            row["router"] = dict(ids_row_match=match, wt_rel_l2_matched=wrel)
            ok_pcc = ok_pcc and all(m_ >= 0.999 for m_ in match) and all(w_ <= 1e-3 for w_ in wrel)
            S.log(f"{tag}: router ids row match {match}, weights rel L2 on matched rows {wrel}")
        if VARIANT in (
            "a15b13",
            "a15b12",
            "a15b15",
            "a15b16",
        ):  # y_s vs an fp32 HOST reference (the same bf8 weights, the gathered bf16 x-hat_f): P13 must be
            # no worse than the op path (the op rounds gate / up to bf16 before the SwiGLU; P13's epilogue runs on fp32)
            ys_ref = shared_ref()
            vs = {nm_: [round(C.pcc(a_.reshape(-1, a_.shape[-1]), r_), 7) for a_, r_ in zip(src_["y_s"], ys_ref)]
                  for nm_, src_ in (("p13", got), ("op", ref))}  # fmt: skip
            row["y_s_vs_fp32"] = vs
            ok_pcc = ok_pcc and all(p_ >= o_ - 1e-6 for p_, o_ in zip(vs["p13"], vs["op"]))
            # (w7k1 11:11Z) y_s's |scale - 1| gate vs the fp32 reference (the op path is the less exact side: its 0.8 % scale
            # gap to P13 is the op's): re-judge y_s's scale here and drop the vs-op scale verdict for y_s
            sc_ = []
            for a_, r_ in zip(got["y_s"], ys_ref):
                a2_, r2_ = a_.reshape(-1, a_.shape[-1]).double(), r_.double()
                sc_.append(round(float((a2_ * r2_).sum() / (r2_ * r2_).sum().clamp(min=1e-30)), 6))
            row["y_s_scale_vs_fp32"] = sc_
            ok_pcc = ok_pcc and all(abs(v_ - 1.0) <= 0.005 for v_ in sc_)
            S.log(f"{tag}: y_s vs the fp32 reference {vs}")
        row["pcc_min"] = pcc_min
        S.log(f"{tag}: fused vs control {cmp}")
        perf_fus = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} fused")
        det, n_done = {}, 1 + (perf_fus.get("replays", 0) if perf_fus else 0)
        for mark in sorted({max(1, DET // 2), DET}):
            while n_done < mark:
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                n_done += 1
            ttnn.synchronize_device(mesh)
            d_ = compare(outputs(fus), got)
            det[str(n_done)] = all(ch["bitwise"] for v in d_.values() for ch in v)
            if not det[str(n_done)]:
                det[f"{n_done}_diff"] = {k: [ch["maxabs"] for ch in v] for k, v in d_.items()}
        if 0 < VALID_N < c and VARIANT in (
            "a15b12",
            "a15b15",
            "a15b16",
        ):  # the in-program VALID-ROW MASK (P12's router tail writer)
            # valid_end is a device tensor read in-trace: rewrite it, replay the SAME trace; rows < VALID_N of every output
            # bitwise vs the all-valid replay (`got`), rows >= VALID_N = the sentinel expert (288) with weight 0, and B's
            # per-expert counts = VALID_N x K pairs (no padding row in any dispatch)
            assert getattr(fus, "row_mask", False), "valid_n needs GLM53_PMK_ROW_MASK=1 in the job env"
            K_ = int(got["ids"][0].numel() // c)
            cnt_of = lambda: [
                int(ttnn.to_torch(d_).to(torch.int64)[..., :288].sum())  # noqa: E731
                for d_ in (ttnn.get_device_tensors(b.tables(c)["cnt"]) if NDEV > 1 else [b.tables(c)["cnt"]])
            ]
            cnt_all = cnt_of()
            pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=VALID_N)
            rm = dict(valid_n=VALID_N, cnt_all=cnt_all, valid_bitwise={}, skipped=[])
            try:
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                pm = outputs(fus)
                rm["cnt_masked"] = cnt_of()
                for k_, chips_ in pm.items():
                    if k_ in (
                        "S",
                        "C",
                        "terms",
                    ):  # the KDA state after the chunk / its carry rows / the items' per-chunk terms:
                        rm["skipped"].append(
                            (k_, "state")
                        )  # legitimately the state after valid_n rows, not a row output
                        continue
                    res_ = []
                    for ch_, (a_, r_) in enumerate(zip(chips_, got[k_])):
                        a2_, r2_ = a_.reshape(-1, a_.shape[-1]), r_.reshape(-1, r_.shape[-1])
                        if k_ in ("ids", "wt"):
                            a2_, r2_ = a_.reshape(-1, K_), r_.reshape(-1, K_)
                        R_ = int(a2_.shape[0])
                        if R_ == c:
                            g_rows = torch.arange(R_)
                        elif R_ * NDEV == c:
                            g_rows = ch_ * R_ + torch.arange(R_)
                        else:
                            rm["skipped"].append((k_, R_))
                            break
                        v_ = g_rows < VALID_N
                        res_.append(bool(torch.equal(a2_[v_], r2_[v_])))
                    if res_:
                        rm["valid_bitwise"][k_] = res_
                ids_p = [t_.reshape(-1, K_)[VALID_N:] for t_ in pm["ids"]]
                wt_p = [t_.reshape(-1, K_)[VALID_N:] for t_ in pm["wt"]]
                rm["pad_ids_sentinel"] = [bool((t_ == 288).all()) for t_ in ids_p]
                rm["pad_wt_zero"] = [bool((t_ == 0).all()) for t_ in wt_p]
                rm["pad_rows"] = int(ids_p[0].shape[0])
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                d2_ = compare(outputs(fus), pm)
                rm["det_masked"] = all(ch["bitwise"] for v in d2_.values() for ch in v)
            finally:
                pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
            rm["dispatch_pairs_ok"] = all(m_ == VALID_N * K_ for m_ in rm["cnt_masked"]) and all(
                a_ == c * K_ for a_ in cnt_all
            )
            rm["ok"] = (all(all(v_) for v_ in rm["valid_bitwise"].values()) and all(rm["pad_ids_sentinel"]) and all(rm["pad_wt_zero"])
                        and rm["det_masked"] and rm["dispatch_pairs_ok"])  # fmt: skip
            row["row_mask"] = rm
            ok_bw = ok_bw and rm["ok"]
            S.log(f"{tag}: ROW MASK valid_n {VALID_N}: {rm}")
        if PP is not None:  # one more replay of the fused trace: its rows are the probe's
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            row["phase_probe"] = phase_spans(fus)
            S.log(f"{tag}: phase probe (chip 0) {row['phase_probe'][0]}")
            # the RS tail per chip (rs_tail_design.md §5.2): P8 / P10 END after P7 END and P7's own span (the overlapped
            # RS traffic must not slow P7), us from the probe rows
            end_ = lambda ps, k: (ps.get(k) or {}).get("end_us")  # noqa: E731
            row["rs_tail"] = [dict(p6_end=end_(ps, "P6"), p7_end=end_(ps, "P7"), p8_end=end_(ps, "P8"), p10_end=end_(ps, "P10"),
                                   p7_minus_p6=(round(end_(ps, "P7") - end_(ps, "P6"), 1) if end_(ps, "P7") and end_(ps, "P6") else None),
                                   p8_minus_p7=(round(end_(ps, "P8") - end_(ps, "P7"), 1) if end_(ps, "P8") and end_(ps, "P7") else None),
                                   p10_minus_p7=(round(end_(ps, "P10") - end_(ps, "P7"), 1) if end_(ps, "P10") and end_(ps, "P7") else None))
                              for ps in row["phase_probe"]]  # fmt: skip
            S.log(f"{tag}: RS tail per chip (rs8 slices {row['rs8_slices']['fused']}) {row['rs_tail']}")
        ttnn.release_trace(mesh, tid)
        S.log(f"{tag}: determinism {det}")
        # (w7k1 11:11Z) the verdict's inputs land in the row BEFORE the span captures: a span that fails (e.g. a control
        # sub-scope program built inside its capture under the served env's global GLM53_PMK_ARGS=table) loses only itself
        row.update(determinism=det, us_chain=perf_ctl.get("median_us"), us_fused_chain=perf_fus.get("median_us"))
        S.log(f"{tag}: us_chain {row['us_chain']} us_fused_chain {row['us_fused_chain']}")
        spans = {}
        a15x = VARIANT in ("a15", "a15b", "a15b11", "a15b13", "a15b12", "a15b15", "a15p11")
        span_list = ((("A1", ctl, [("P1",)]),) if a15x else ()) + (
            ("A2", ctl, [("P2", "P3")]), ("A3", ctl, [("P4", "P6")]), ("A4", ctl, [("P7", "P8")])) + (
            (("A5", ctl, [tuple(s_) for s_ in ctl.scopes if "P10" in s_]),) if a15x else ()) + (
            (("B", None, None),) if VARIANT in ("a15b", "a15b11", "a15b13", "a15b12", "a15b15", "a15b16") else ()) + (
            ("FUSED", fus, [FSCOPE]),) + ((("SH_op", None, "sh"),) if VARIANT in ("a15b13", "a15b12", "a15b15", "a15b16") else ())  # fmt: skip
        for name, prog, scs in span_list:
            if prog is None and scs == "sh":  # the op path's shared expert alone (the chain's filler P13 replaces; its
                # programs compiled in the chain's compile launch: same shapes, no untraced warm-up); LAST and guarded: a
                # failure here loses only this span
                try:
                    t_ = ttnn.begin_trace_capture(mesh, cq_id=0)
                    try:
                        y_sh_ = swiglu_prefill(pl.shared, BIN["hb"])
                    finally:
                        ttnn.end_trace_capture(mesh, t_, cq_id=0)
                except Exception as e_:  # noqa: BLE001
                    S.log(f"{tag}: SH_op span skipped: {type(e_).__name__}: {e_}"[:400])
                    continue
            elif (
                prog is None and scs == "rt"
            ):  # the op path's router + the tail layouts alone (what P12 replaces; compiled
                # in the chain's compile launch); LAST and guarded
                try:
                    t_ = ttnn.begin_trace_capture(mesh, cq_id=0)
                    try:
                        x32_ = pl.rms(fus.hns[1], "ffn", FP32)
                        idx_, sc_ = pl.moe.router(x32_, decode=False)
                        sel_ = ttnn.typecast(ttnn.gather(sc_, dim=-1, index=idx_), FP32)
                        ids_f_ = pf.gather(ttnn.typecast(idx_, FP32), sp)
                        rt_out_ = M2.tail_layout_inputs(
                            ttnn.typecast(ids_f_, ttnn.uint16), pf.gather(sel_, sp), float(pl.moe.scaling)
                        )
                    finally:
                        ttnn.end_trace_capture(mesh, t_, cq_id=0)
                except Exception as e_:  # noqa: BLE001
                    S.log(f"{tag}: RT_op span skipped: {type(e_).__name__}: {e_}"[:400])
                    continue
            elif prog is None:  # program B alone on the staged inputs (M-B1's B span)
                t_ = ttnn.begin_trace_capture(mesh, cq_id=0)
                try:
                    b(BIN["x_ext"], BIN["ids_rm"], BIN["wt"], c, x_tile=BIN["hb"])
                finally:
                    ttnn.end_trace_capture(mesh, t_, cq_id=0)
            else:
                try:  # (w7m1 12:39Z) its first BUILD outside the capture: under the served env (GLM53_PMK_ARGS=table) a
                    # sub-scope program's first build writes its DRAM arg table -- inside the capture a TT_FATAL host
                    # write (A1..A4 lost in w7k1 / w7m1); prebuild = the descriptors + tables, no launch
                    use(prog, scs)
                    prog.layer.prebuild(x_p, ci)
                except Exception as e_:  # noqa: BLE001
                    S.log(f"{tag}: span {name} prebuild failed: {type(e_).__name__}: {e_}"[:300])
                try:  # (w7k1 11:11Z) a sub-scope program first built here would build its DRAM arg table inside the capture
                    t_ = capture(prog, scs, reset=False)
                except Exception as e_:  # noqa: BLE001
                    S.log(f"{tag}: span {name} skipped: {type(e_).__name__}: {e_}"[:300])
                    continue
            spans[name] = S.replay_stats(mesh, t_, replays=REPLAYS, label=f"{tag} {name}").get("median_us")
            ttnn.release_trace(mesh, t_)
        use(ctl, ctl.scopes)
        use(fus, fus.scopes)
        ks_ = (
            (
                ("A1", "A2", "A3", "A4", "A5")
                + (("B",) if VARIANT in ("a15b", "a15b11", "a15b13", "a15b12", "a15b15", "a15b16") else ())
            )
            if a15x
            else ("A2", "A3", "A4")
        )
        sum3 = sum(spans[k] for k in ks_) if all(spans.get(k) for k in ks_) else None
        row.update(compare=cmp, bitwise=ok_bw, determinism=det, us_chain=perf_ctl.get("median_us"),
                   us_fused_chain=perf_fus.get("median_us"), us_spans=spans, us_a2_a3_a4=round(sum3, 1) if sum3 else None,
                   status="pass" if ok_bw and ok_pcc and all(v for k, v in det.items() if not k.endswith("_diff"))
                   and (row.get("vs_ref") or {}).get("ok", True) else "fail")  # fmt: skip
        S.log(
            f"{tag}: chain {row['us_chain']} us, fused chain {row['us_fused_chain']} us; spans {spans}; status {row['status']}"
        )
        if bool(args.get("record", True)):
            C.record("pcc", dict(test=row["test"], milestone=row["milestone"], layer=L, tp=NDEV, bucket=c, bitwise=ok_bw,
                                 pcc_min=row.get("pcc_min"), determinism=det, status=row["status"], tag=tag,
                                 rs8_slices=row["rs8_slices"]["fused"], vs_ref=row.get("vs_ref")))  # fmt: skip
            C.record("perf", dict(test=f"PMK fused A2 + A3 + A4 L{L} tp{NDEV}: traced spans A2 / A3 / A4 / A2F and the chain vs the fused chain",
                                  milestone=row["milestone"], layer=L, tp=NDEV, bucket=c, us_spans=spans, us_chain=row["us_chain"],
                                  us_fused_chain=row["us_fused_chain"], replays=REPLAYS, status=row["status"], tag=tag,
                                  rs8_slices=row["rs8_slices"]["fused"], rs_tail=row.get("rs_tail")))  # fmt: skip
except _L1Done:
    pass
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    row.update(status="fail", error=f"{type(e).__name__}: {e}"[:1500])
if RF_ is not None:  # the fence's words of the last launch per chip: GEN, TOK, T2 - T0 (cycles) per lane
    try:
        w_ = RF_.words()
        row["fence"] = dict(mode=FENCE, calls=RF_.calls, sync=int(RF_.sync.buffer_address()),
                            gen={str(k_): [l_[RFm.F["GEN"]] for l_ in v_] for k_, v_ in w_.items()},
                            tok={str(k_): [l_[RFm.F["TOK"]] for l_ in v_] for k_, v_ in w_.items()},
                            cyc={str(k_): [(l_[RFm.F["T2"]] - l_[RFm.F["T0"]]) & 0xFFFFFFFF for l_ in v_] for k_, v_ in w_.items()})  # fmt: skip
        S.log(f"{tag}: row fence {row['fence']}")
    except Exception as e_:  # noqa: BLE001
        row["fence_err"] = repr(e_)[:300]
RESULT = row
try:
    MEM["end"] = _mem_view()
    row["mem"] = MEM
    row["mem_fused_build_delta"] = {
        k_: MEM["after_fused_build"][k_] - MEM["after_control"][k_] for k_ in ("dram", "l1")
    }
except Exception as e_:  # noqa: BLE001
    row["mem_err"] = repr(e_)[:200]
print("PMK_FUSE3_UNIT", {k: v for k, v in row.items() if k not in ("compare",)}, flush=True)
