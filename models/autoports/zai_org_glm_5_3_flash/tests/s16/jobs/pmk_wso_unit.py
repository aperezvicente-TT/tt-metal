# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; per chip -- every chip of the mesh runs the whole unit, no fabric): WS-O stage 1, the ONE-CHIP o_proj
unit under a SYNTHETIC SCAN CLOCK (program [session 10] 2026-10-28 05:24:26Z GO; plan_1m/l5_ws_oproj_design.md;
tt/pf_wo.py). P7 at the TP8 per-chip shapes, the served PMK_A_INPLACE form: y [2048, 4096] = on [2048, 1024] bf16
(seeded; DRAM, pre-filled) @ W_o [1024, 4096] (bf8, the resident DSLinear layout, GLM53_PMK_A_FID's HiFi3).
The scan clock (the probe core's BRISC, a timed loop): chunk c (= row tile c of `on`) is released at t0 + c x period
(~12 us: the scan's pace) by +1 on SEM_ON of the members (the served scan leader's H4 semantics) and, in arm B, of the
two WS lanes. The H5 stand-in (the probe core's NCRISC = the RS8 reader's place) stamps when block c is complete (1 raise
per block in A, 64 in B), in block order.
  A  today's PfStream o_proj (slot-4 form, QSLICE 1): PfStreamMM in place (nb 4, nw 16, kb 16, PARK 4), each member
     waiting SEM_ON >= its row + 1 before reading its x row (PFS_WAIT_RT: the O_RECTS mechanism, one rectangle) ->
     the stream starts after the last chunk = today's "P7 waits the whole scan"
  B  WS-O: the members' resident 2-column slices, lanes (2, 0) NoC 0 / (2, 7) NoC 1 outside the 8 x 8 box (one per NoC),
     ring S, mc 1
Reported per arm: the replay us (median of >= 200), the tail = DONE[last block] - REL[last chunk], the per-block lag
DONE[c] - REL[c] (keeping pace), y BITWISE per tile B vs A per chip + determinism over the replays, the lanes' waits, the
L1 census per role (CB base + bytes vs the 1,240,960 floor) and the H5 assert (pf_wo.h5_check).
ARGS: replays (200), period_us (12), slots (2), mc (1), seed (61), tag, record (True)."""
import hashlib
import os
import sys
import time

import torch

import ttnn

tree = os.environ.get("GLM53_ROW_TREE") or os.environ["TT_METAL_HOME"]
sys.path.insert(0, os.path.join(tree, "models", "autoports", "zai_org_glm_5_3_flash", "tests", "s16"))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.pf_stream as PF  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.pf_wo as WO  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.pmk_layer import L1_TOTAL, STAGE16_CB_END_RULING, Placement  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
REPLAYS = int(args.get("replays", 200))
PERIOD_US = float(args.get("period_us", 12.0))
SLOTS = int(args.get("slots", 2))
MC = int(args.get("mc", 1))
tag = args.get("tag", "pmk-wso-unit")
rec = bool(args.get("record", True))
NDEV = int(mesh.get_num_devices())
M, K, N = 2048, 1024, 4096
CPU = 1350.0  # cycles per us (nominal AICLK; d0's clock ramps: the ratios are what carry)
FLOOR = 1_240_960
worker_l1 = int(ctx.get("worker_l1") or os.environ.get("TP16_WORKER_L1") or 1_264_512)
CB_BASE = L1_TOTAL - worker_l1
NCH = M // 32
PERIOD = int(PERIOD_US * CPU)
P_T0, P_REL, P_DONE, P_H5, P_DL, P_WORDS = 0, 1, 65, 129, 193, 208
t0 = time.time()

# ---- the geometry (pmk_layer.Placement's rect form at 11 x 10): members x 3..10 y 0..7 (one 8 x 8 box); the WS lanes on
# roots (2, 0) NoC 0 / (2, 7) NoC 1 -- outside the box, approach routes outside it (NoC 0 X+ then Y+ from (2, 0) reaches
# the box's low corner directly; NoC 1 Y- then X- from (2, 7) runs along row 7's west wrap to the high corner); the
# probe core (clock + H5 stand-in) = the tail (2, 9)
P = Placement(gx=11, gy=10)
members = [tuple(c) for c in P.members]
lanes = [(2, 0), (2, 7)]
probe_core = (2, 9)
assert set(lanes) <= set(P.roots) and probe_core in set(P.tail)
try:
    from models.autoports.zai_org_glm_5_3_flash.tt.pmk_assembly import adj_streamers

    streamers = adj_streamers(mesh, members, P.streamers, list(P.roots) + list(P.mm_only))
except Exception as e:  # noqa: BLE001
    S.log(f"{tag}: adj_streamers unavailable ({e!r}): the placement's streamer list")
    streamers = [tuple(c) for c in P.streamers]
assert probe_core not in streamers

# ---- the seeded inputs (replicated: every chip runs the same unit)
g = torch.Generator().manual_seed(int(args.get("seed", 61)))
onh = torch.randn(1, 1, M, K, generator=g).to(torch.bfloat16)
woh = torch.randn(K, N, generator=g) / K**0.5
rep = ttnn.ReplicateTensorToMesh(mesh)
on = ttnn.from_torch(onh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                     mesh_mapper=rep)  # fmt: skip
w_o = WO.resident_weight(mesh, woh, ttnn.bfloat8_b)
pmc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                        ttnn.ShardSpec(WO._crs([probe_core]), [1, P_WORDS * 4], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
probe = ttnn.from_torch(torch.zeros(1, P_WORDS * 4, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh, memory_config=pmc, mesh_mapper=rep)  # fmt: skip
paddr = probe.buffer_address()
noc = lambda xy: (lambda c: (int(c.x), int(c.y)))(mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy)))  # noqa: E731
pxy = noc(probe_core)
h5 = [(pxy[0], pxy[1], paddr + (P_H5 + c) * 16) for c in range(NCH)]

mm = PF.PfStreamMM(mesh, None, members=members, streamers=streamers, nb=4, nw=N // 32 // 8, kb=16, park=4,
                   fidelity=ttnn.MathFidelity.HiFi3, rt=1, resident=w_o)  # fmt: skip
wo = WO.PfStreamWO(mesh, w_o, members=members, lanes=lanes, m_rows=M, slots=SLOTS, mc=MC, fidelity=ttnn.MathFidelity.HiFi3)


def probe_kernels(targets: list, raises: int) -> list:
    """the scan clock (BRISC) + the H5 stand-in (NCRISC) on the probe core"""
    src = lambda f: open(os.path.join(WO.KDIR, f)).read()  # noqa: E731
    rc, rh = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rc[probe_core[0]][probe_core[1]] = [paddr] + [v for t in targets for v in noc(t)]
    rh[probe_core[0]][probe_core[1]] = [paddr]
    mk = lambda s, ct, rt, cfg: ttnn.KernelDescriptor(kernel_source=s, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                                     core_ranges=WO._crs([probe_core]), compile_time_args=ct,
                                                     runtime_args=rt, config=cfg, defines=[])  # noqa: E731  # fmt: skip
    return [mk(src("unit_clock.cpp"), [NCH, PERIOD, WO.SEM_ON, len(targets)], rc, ttnn.WriterConfigDescriptor()),
            mk(src("unit_h5.cpp"), [NCH, raises], rh, ttnn.ReaderConfigDescriptor())]


def program_a() -> ttnn.ProgramDescriptor:
    """today's PfStream o_proj (PfStreamMM.program()'s assembly) + SEM_ON waits per member + the probe kernels"""
    pt = mm.parts(mm.y, x=on, x_fill=True, wait=(WO.SEM_ON, 0), wait_rt={m_: i + 1 for i, m_ in enumerate(members)},
                  signals={m_: h5[i] for i, m_ in enumerate(members)})  # fmt: skip

    def rt_of(cores):
        r = ttnn.RuntimeArgs()
        for c in cores:
            r[c[0]][c[1]] = pt["rt"][c]
        return r

    dyn = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                            noc=ttnn.NOC.RISCV_0_default, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC)  # fmt: skip
    cc = ttnn.ComputeConfigDescriptor(**pt["compute"], math_approx_mode=False)
    kd = lambda s_, cores, cfg, extra=(): ttnn.KernelDescriptor(
        kernel_source=s_, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=WO._crs(cores),
        compile_time_args=pt["ct"], runtime_args=rt_of(cores), config=cfg, defines=list(pt["defines"]) + list(extra))  # fmt: skip
    s_, m_ = pt["sources"]["streamer"], pt["sources"]["member"]
    kernels = [kd(s_["dm0"], mm.all_streamers, dyn), kd(m_["dm1"], members, ttnn.ReaderConfigDescriptor()),
               kd(m_["c"], members, cc), kd(m_["dm0"], members, ttnn.WriterConfigDescriptor(), pt["wr_defines"])]  # fmt: skip
    kernels += probe_kernels(members, 1)
    cbs = []
    for role, cores in (("streamer", mm.all_streamers), ("member", members)):
        for idx, dt, page, pages in pt["cbs"][role]:
            cbs.append(ttnn.CBDescriptor(total_size=page * pages, core_ranges=WO._crs(cores), format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)]))  # fmt: skip
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=WO._crs(members), initial_value=0) for i in pt["sems"]["member"]]
    sems.append(ttnn.SemaphoreDescriptor(id=WO.SEM_ON, core_ranges=WO._crs(members + [probe_core]), initial_value=0))
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)


def program_b() -> ttnn.ProgramDescriptor:
    kernels, cbs, semc = wo.parts(on, h5)
    kernels = kernels + probe_kernels(members + lanes, len(members))
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=WO._crs(cores + ([probe_core] if i == WO.SEM_ON else [])),
                                     initial_value=0) for i, cores in semc.items()]  # fmt: skip
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)


def launch(prog_fn, io):
    mpd = ttnn.MeshProgramDescriptor()
    for r in range(int(mesh.shape[0])):
        for c in range(int(mesh.shape[1])):
            mc_ = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc_, mc_)] = prog_fn()
    ttnn.generic_op(io, mpd)


arms = {
    "A": dict(fn=lambda: launch(program_a, [on, w_o, mm.y, mm.sync, probe]), y=lambda: mm.y, obj=None,
              form="today's PfStream o_proj, members gated per chunk (the stream starts after the last chunk)"),
    "B": dict(fn=lambda: launch(program_b, [on, w_o, wo.y, wo.sync, probe]), y=lambda: wo.y, obj=wo,
              form=f"WS-O: resident 2-column slices, lanes {lanes} (NoC 0 / 1, outside the box), S {SLOTS}, mc {MC}"),
}

# ---- the L1 census (static CB bytes per role; CB end = the process's CB base + bytes) + the H5 assert
fpa, fpb = mm.footprint(), wo.footprint()
fp = dict(pfs_member=fpa["member"], pfs_streamer=fpa["streamer"], wso_member=fpb["member"], wso_lane=fpb["lane"],
          probe_core=0)  # fmt: skip
census = {k: dict(cb_bytes=v, cb_end=CB_BASE + v, under_floor=CB_BASE + v <= FLOOR,
                  under_ruling=CB_BASE + v <= STAGE16_CB_END_RULING) for k, v in fp.items()}  # fmt: skip
p7_window = dict(member_today_park=4 * 16 * 4 * 1088, member_wso_adds=fpb["ring"] + fpb["w_slice"] + 2 * wo.cols * 2048)
p7_window["member_net_bytes"] = p7_window["member_wso_adds"] - p7_window["member_today_park"]
S.log(f"{tag}: L1 census (worker_l1 {worker_l1:,}, CB base {CB_BASE:,}, floor {FLOOR:,}): {census}; P7 window {p7_window}; "
      f"H5 {wo.h5}; box {wo.box}, lanes {lanes} NoC {[noc(c) for c in lanes]}, probe {probe_core} {pxy}; "
      f"period {PERIOD} cycles ({PERIOD_US} us nominal), chunks {NCH}")  # fmt: skip
assert all(v["under_floor"] for v in census.values()), census


def host(t) -> list:
    return [ttnn.to_torch(ttnn.get_device_tensors(t)[d] if NDEV > 1 else t).reshape(M, -1).contiguous() for d in range(NDEV)]


def probe_read() -> list:
    out = []
    for h in [ttnn.to_torch(d).reshape(P_WORDS, 4)[:, 0].to(torch.int64) & 0xFFFFFFFF
              for d in (ttnn.get_device_tensors(probe) if NDEV > 1 else [probe])]:  # fmt: skip
        t0_ = int(h[P_T0])
        rel = [int(h[P_REL + c]) for c in range(NCH)]
        done = [(int(h[P_DONE + c]) - t0_) & 0xFFFFFFFF for c in range(NCH)]
        lag = [d - r for d, r in zip(done, rel)]
        out.append(dict(tail_us=round((done[-1] - rel[-1]) / CPU, 1), lag_us_max=round(max(lag) / CPU, 1),
                        lag_us_mean=round(sum(lag) / len(lag) / CPU, 1), last_release_us=round(rel[-1] / CPU, 1),
                        last_done_us=round(done[-1] / CPU, 1), first_done_us=round(done[0] / CPU, 1),
                        dl=hex(int(h[P_DL])), period_cycles=PERIOD))  # fmt: skip
    return out


def sha(t) -> str:
    return hashlib.sha1(t.view(torch.int16).numpy().tobytes()).hexdigest()[:16]


rows, results = [], {}
tr = Traced(mesh)
tr.warm(*[a["fn"] for a in arms.values()])  # every program compiled before ANY capture
for name, a in arms.items():
    tr = Traced(mesh)
    tr.run(a["fn"])  # the compile / warm-up launch, the capture, one replay
    if not tr.enabled:
        results[name] = dict(mock=True)
        tr.release()
        continue
    y1 = host(a["y"]())
    pr1 = probe_read()
    st = S.replay_stats(mesh, tr.tid, replays=REPLAYS, label=f"{tag} arm {name}")
    y2 = host(a["y"]())
    pr2 = probe_read()
    det = all(torch.equal(p.view(torch.int16), q.view(torch.int16)) for p, q in zip(y1, y2))
    results[name] = dict(y=y1, stats=st, det=det, probe=pr2, probe_first=pr1,
                         lanes=a["obj"].lane_words() if a["obj"] is not None else None)  # fmt: skip
    tr.release()

ok = True
if C.MOCK:
    row = dict(test="WS-O stage 1: the o_proj weight-stationary unit under a synthetic scan clock (mock build)",
               milestone="P7-WSO", tp=NDEV, census=census, p7_window=p7_window, h5=wo.h5, status="mock", traced=False,
               tag=tag)  # fmt: skip
    S.log(row)
    rows.append(row)
else:
    ya, yb = results["A"]["y"], results["B"]["y"]
    per_chip = []
    for d in range(NDEV):
        a_, b_ = ya[d].view(torch.int16), yb[d].view(torch.int16)
        tiles_bad = int((a_ != b_).reshape(M // 32, 32, N // 32, 32).any(dim=3).any(dim=1).sum())
        per_chip.append(dict(bitwise=torch.equal(a_, b_), tiles_mismatch=tiles_bad, elem_mismatch=int((a_ != b_).sum()),
                             maxabs=float((ya[d].float() - yb[d].float()).abs().max()), sha=[sha(ya[d]), sha(yb[d])]))  # fmt: skip
    dl = [L for chip in (results["B"]["lanes"] or []) for L in chip if L["dl_what"] != "0x0"]
    pdl = [p_["dl"] for n_ in arms for p_ in results[n_]["probe"] if p_["dl"] != "0x0"]
    good = all(c_["bitwise"] for c_ in per_chip) and results["A"]["det"] and results["B"]["det"] and not dl and not pdl
    ok = good
    row = dict(test="WS-O stage 1: P7 o_proj weight-stationary (B) vs today's PfStream o_proj (A) under a synthetic scan "
                    "clock, TP8 per-chip shapes (on 2048 x 1024 bf16 @ W_o 1024 x 4096 bf8 HiFi3 in place), y bitwise per "
                    "tile per chip + determinism", milestone="P7-WSO", tp=NDEV, per_chip=per_chip,
               det={n_: results[n_]["det"] for n_ in arms}, deadlock=dl, probe_deadlock=pdl, census=census,
               p7_window=p7_window, h5=wo.h5, lanes=lanes, status="pass" if good else "fail", traced=True, tag=tag)  # fmt: skip
    S.log(row)
    rows.append(row)
    if rec:
        C.record("pcc", row)
    perf = dict(test="WS-O stage 1: keeping pace with a synthetic scan (one chunk per period): the tail after the last "
                     "chunk's release + the per-block lag, A (today: the stream after the last chunk) vs B (WS-O), traced "
                     "replays", milestone="P7-WSO", tp=NDEV, period_us=PERIOD_US,
                us={n_: results[n_]["stats"].get("median_us") for n_ in arms},
                p90_us={n_: results[n_]["stats"].get("p90_us") for n_ in arms},
                probe={n_: results[n_]["probe"] for n_ in arms}, lanes=results["B"]["lanes"], status="info", traced=True,
                tag=tag)  # fmt: skip
    S.log(perf)
    rows.append(perf)
    if rec:
        C.record("perf", perf)
S.log(f"{tag}: done in {time.time() - t0:.1f} s, ok {ok}")
mm.release()
wo.release()
for t_ in (on, w_o, probe):
    ttnn.deallocate(t_)
RESULT = dict(ok=ok or C.MOCK, rows=rows)
print(f"WSO {'mock' if C.MOCK else ('pass' if ok else 'fail')}", flush=True)
