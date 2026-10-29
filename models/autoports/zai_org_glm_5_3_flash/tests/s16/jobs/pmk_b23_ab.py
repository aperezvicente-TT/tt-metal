# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, d0 pair): Stage 14 B2.3 (program 2026-10-25T23:14:41Z) -- program B's self-stream form (SelfStreamB,
the 7g form of PERF.md sec 19) arms A/B on ONE set of expert slices and ONE routing in TP8's PER-CHIP geometry: chip k
holds layer L's TP8 XSL slice k (W 256), so a pair runs exactly what each TP8 chip runs in program B (program B has no
cross-chip term). Per arm (fid, blocks, Levers valid / snake / xs_rows): a fresh SelfStreamB, its compile launch, a
captured trace, `reps` timed replays (s16common.replay_stats), then `det` replays each read back (the count equal to
the first: determinism), and the y of the last replay per chip compared with the FIRST arm bitwise (PCC when not).
probe=True: the arm's in-program counters (moe_xffn_probe.hpp): per chip the reader spans over the cores that ran
(max / mean = the dealing's imbalance), units per core, weight MB, writer x MB, and the weight-stream rate.
Inputs as moe_xffn_ab: x random bf16 (seed), routing uniform or zipf (zipf_s), weights the served L4 slices.
ARGS: layer (4), xsl_tp (8), experts (288), c (2048), k (8), routing ("zipf"), zipf_s (1.0), seed (0), shared (1: the
P13 partial as the (K + 1)-th combine term = random bf16 rows), arms ([{"name", "fid", "ffn_fid", "blocks", "valid",
"snake", "xs_rows", "expect_bitwise", "stage", "comb_group", "comb_slot", "comb_bufs", "h_split", "out_tm", "plan_cores",
"core_order"}]), tables (0: 1 = compare the plan tables vs arm 1),
reps (200), det (20),
probe (True),
tag, record."""
import gc
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.pmk_moe as PM  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.pmk_moe2 as M2  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402

C = S.C
mesh, args = MESH, ARGS  # noqa: F821
E, CR, K, H = int(args.get("experts", 288)), int(args.get("c", 2048)), int(args.get("k", 8)), 4096
XTP = int(args.get("xsl_tp", 8))
W = 2048 // XTP
REPS, DET = int(args.get("reps", 200)), int(args.get("det", 20))
PROBE = bool(args.get("probe", True))
SHARED = bool(int(args.get("shared", 1)))
tag = args.get("tag", "s14-b23-ab")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
# fid = the kernel's (the combine's) fidelity, ffn_fid = the FFN matmuls' (PMK_FFN_FID; "" = fid). The watched pass of
# 2026-10-25 23:27Z ran fid LoFi on the whole composite: y norm 0.981 x HiFi2's (the combine's bf16 x weight multiplies),
# so the ladder keeps the combine at HiFi2; lofi_all (expect_bitwise 0) records that loss
ARM0 = dict(name="7g", fid="HiFi2", ffn_fid="", blocks="4,8,8", valid=0, snake=0, xs_rows=0, expect_bitwise=1, stage=-1,
            comb_group=1, comb_slot=0, comb_bufs=2, h_split=4, out_tm=0, plan_cores=0, core_order="", comb_dual=0)  # fmt: skip
TABLES = bool(int(args.get("tables", 0)))  # compare the plan's DRAM tables (dest / cnt / start / meta / disp) vs arm 1


def core_list(order: str):
    """the B core list (plan cores = its first plan_cores entries): "" = the whole grid row-major (PmkMoe's default);
    "rootmm_first" = the fused layer's root (x 2) and MM-only (x 0 / 1) columns first (16's placement: members x 3..10,
    y 0..7), so a 32-core plan sits where the idle P13 BRISCs are"""
    if not order:
        return None
    gx, gy = int(mesh.compute_with_storage_grid_size().x), int(mesh.compute_with_storage_grid_size().y)
    allc = [(x, y) for y in range(gy) for x in range(gx)]
    assert order == "rootmm_first", order
    return sorted(allc, key=lambda xy: (0 if xy[0] <= 2 else 1, xy[1], xy[0]))
ARMS = [dict(ARM0, **a) for a in (args.get("arms") or [
    {},
    {"name": "lofi", "ffn_fid": "LoFi"},
    {"name": "lofi_valid", "ffn_fid": "LoFi", "valid": 1},
    {"name": "lofi_valid_snake", "ffn_fid": "LoFi", "valid": 1, "snake": 1},
    {"name": "lofi_valid_snake_mg2", "ffn_fid": "LoFi", "valid": 1, "snake": 1, "blocks": "2,8,8"},
    {"name": "lofi_valid_snake_rows_mg2", "ffn_fid": "LoFi", "valid": 1, "snake": 1, "xs_rows": 1, "blocks": "2,8,8"},
    {"name": "lofi_all", "fid": "LoFi", "expect_bitwise": 0},
])]  # fmt: skip
row = dict(test=f"S14 B2.3 SelfStreamB arms A/B (E {E}, W {W} = TP{XTP} per-chip slices {list(range(TPN))}, c {CR}, k {K}, "
           f"shared {int(SHARED)}) tp{TPN}", milestone="B2.3", tp=TPN, tag=tag, layer=int(args.get("layer", 4)),
           routing=args.get("routing", "zipf"), zipf_s=float(args.get("zipf_s", 1.0)), arms=[])  # fmt: skip


def per_chip(t_):
    return [ttnn.to_torch(p).clone() for p in (ttnn.get_device_tensors(t_) if TPN > 1 else [t_])]


def free_bufs(obj):
    """deallocate every device tensor a PmkMoe holds (bufs / tbl): the arms' scratches must not pile up in DRAM"""
    stack = [obj.bufs, getattr(obj, "tbl", None)]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
        elif isinstance(v, ttnn.Tensor):
            try:
                ttnn.deallocate(v)
            except Exception:  # noqa: BLE001
                pass
    obj.bufs = {}


def probe_summary(chips: list) -> list:
    """per chip: the reader rows of the cores that ran (span 13, units 4, weight KB 5) and the writer's x KB (6)"""
    out = []
    for ch in chips:
        r, w_ = ch["reader"].to(torch.float64), ch["writer"].to(torch.float64)
        live = r[:, 13] > 0
        if int(live.sum()) == 0:
            out.append(dict(mock=True))
            continue
        span, units, kb = r[live, 13], r[live, 4], r[live, 5]
        mhz = 1350.0
        out.append(dict(cores=int(live.sum()), span_max_us=round(float(span.max()) / mhz, 1),
                        span_mean_us=round(float(span.mean()) / mhz, 1), span_min_us=round(float(span.min()) / mhz, 1),
                        imbalance=round(float(span.max() / span.mean()), 3),
                        units_min=int(units.min()), units_max=int(units.max()), units_sum=int(units.sum()),
                        w_mb=round(float(kb.sum()) / 1024, 1), x_mb=round(float(w_[:, 6].sum()) / 1024, 1),
                        y_mb=round(float(w_[:, 7].sum()) / 1024, 1),
                        r_gu_wait_max_us=round(float(r[live, 2].max()) / mhz, 1),
                        w_x_read_max_us=round(float(w_[:, 0].max()) / mhz, 1)))  # fmt: skip
    return out


try:
    from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST
    from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoderConfig, load_xsl_experts
    from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

    gus, dns = load_xsl_experts(FdTileCacheSource(CACHE_DIR, MANIFEST), int(args.get("layer", 4)), E, mesh,
                                FusedDecoderConfig().fused_experts_cache_dir, tp=XTP, slices=list(range(TPN)))  # fmt: skip
    S.log(f"{tag}: L{row['layer']} TP{XTP} slices {list(range(TPN))} loaded: gate_up {tuple(gus[0].shape)} down {tuple(dns[0].shape)}")
    g = torch.Generator().manual_seed(int(args.get("seed", 0)))
    x_h = torch.randn(1, 1, CR, H, generator=g) * 0.5
    if row["routing"] == "zipf":  # skewed expert popularity: the load-imbalance case (moe_xffn_ab's)
        pop = (1.0 / (torch.arange(E, dtype=torch.float32) + 1) ** row["zipf_s"])[torch.randperm(E, generator=g)]
        idx_h = torch.stack([torch.multinomial(pop, K, replacement=False, generator=g) for _ in range(CR)]).reshape(1, 1, CR, K)
    else:
        idx_h = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(CR)]).reshape(1, 1, CR, K)
    sel_h = torch.rand(1, 1, CR, K, generator=g) + 0.05
    ys_h = (torch.randn(CR, H, generator=g) * 0.1).to(torch.bfloat16)
    cnt = torch.bincount(idx_h.reshape(-1), minlength=E)
    row["routing_stats"] = dict(rows_max=int(cnt.max()), rows_mean=round(float(cnt.float().mean()), 1),
                                tile_rows=int(((cnt + 31) // 32).sum()), pad_rows=int(((cnt + 31) // 32 * 32 - cnt).sum()))  # fmt: skip
    mk = lambda t_, dt: ttnn.from_torch(t_, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=PF.DRAM,  # noqa: E731
                                        mesh_mapper=repl)  # fmt: skip
    x, idx, sel = mk(x_h, ttnn.bfloat16), mk(idx_h.to(torch.int32), ttnn.uint16), mk(sel_h, ttnn.float32)
    x_ext, ids_rm, wt = PM.inputs_from_op_path(x, idx, sel, 1.0)
    ys_src = ttnn.from_torch(ys_h, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=PF.DRAM,
                             mesh_mapper=repl)  # fmt: skip

    ref, prev = None, None
    for arm in ARMS:
        blocks = tuple(int(v) for v in str(arm["blocks"]).split(","))
        lv = M2.Levers(x_l1=False, mcast=1, shared=SHARED, valid=bool(int(arm["valid"])), snake=bool(int(arm["snake"])),
                       xs_rows=bool(int(arm["xs_rows"])), comb_group=int(arm["comb_group"]), comb_slot=int(arm["comb_slot"]),
                       comb_bufs=int(arm["comb_bufs"]), out_tm=bool(int(arm["out_tm"])),
                       comb_dual=bool(int(arm.get("comb_dual", 0))), comb_dual_nc=int(arm.get("comb_dual_nc", 0)))  # fmt: skip
        pm = M2.ProgramB(mesh, selfstream=True, levers=lv, num_experts=E, top_k=K, hidden=H, limit=10.0, gate_up=gus,
                         down=dns, blocks=blocks, fid=str(arm["fid"]), ffn_fid=str(arm["ffn_fid"]), probe=PROBE, h_split=int(arm["h_split"]),
                         **({"bw_stage": int(arm["stage"])} if int(arm["stage"]) >= 0 else {}),
                         **({"plan_cores": int(arm["plan_cores"])} if int(arm["plan_cores"]) else {}),
                         **({"cores": core_list(arm["core_order"])} if arm["core_order"] else {}))  # fmt: skip
        assert isinstance(pm, M2.SelfStreamB) and pm.fid == arm["fid"] and pm.ffn_fid == (arm["ffn_fid"] or arm["fid"]) \
            and pm.blocks(CR) == blocks, (type(pm), pm.fid, pm.ffn_fid, pm.blocks(CR))  # fmt: skip
        a_ = dict(arm, levers=str(pm.lv), cb_total=pm.l1_bytes(CR))
        s_ = pm.scratch(CR)
        if SHARED:
            ttnn.copy(ys_src, s_["y_s"])  # the P13 partial, constant over the replays (program B only reads it)

        def run():
            return pm(x_ext, ids_rm, wt, CR, x_tile=x)

        y = run()  # the compile launch
        ttnn.synchronize_device(mesh)
        tid = None
        if not C.MOCK:
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                y = run()
            finally:
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            rs = S.replay_stats(mesh, tid, replays=REPS, label=f"{tag} {arm['name']}")
            a_.update(median_us=rs.get("median_us"), min_us=rs.get("min_us"), p90_us=rs.get("p90_us"), replays=rs.get("replays"))
        # determinism: `det` replays, each read back (the mock runs the readback on its compile launch's buffers)
        out = per_chip(y)
        same = 0
        for _ in range(DET if not C.MOCK else 1):
            if tid is not None:
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            o_ = per_chip(y)
            same += int(all(torch.equal(p_, q_) for p_, q_ in zip(o_, out)))
        a_["det"] = f"{same}/{DET if not C.MOCK else 1}"
        if PROBE:
            try:
                chips = pm.probe_read()
                a_["probe"] = probe_summary(chips)
                a_["stream_rate"] = M2.stream_rate(chips)
            except Exception as ex:  # noqa: BLE001
                a_["probe"] = repr(ex)[:300]
        if tid is not None:
            ttnn.release_trace(mesh, tid)
        if ref is None:
            ref = out
            a_.update(bitwise_vs_first=True)
        else:
            eq = [bool(torch.equal(p_, q_)) for p_, q_ in zip(out, ref)]
            a_.update(bitwise_vs_first=all(eq), bitwise_per_chip=eq,
                      pcc_vs_first=[round(C.pcc(p_.float(), q_.float()), 7) for p_, q_ in zip(out, ref)],
                      maxabs_vs_first=[float((p_.float() - q_.float()).abs().max()) for p_, q_ in zip(out, ref)])  # fmt: skip
        if TABLES and int(arm["stage"]) != 0:  # the plan ran (full program or bisect stage >= 1): its tables
            tb = pm.tables(CR)
            tabs = {k_: per_chip(tb[k_]) for k_ in ("dest", "cnt", "start", "meta", "disp")}
            if "ref_tabs" not in globals():
                globals()["ref_tabs"] = tabs
                a_["tables_vs_first"] = True
            else:
                a_["tables_vs_first"] = {k_: all(torch.equal(p_, q_) for p_, q_ in zip(v_, ref_tabs[k_])) for k_, v_ in tabs.items()}
        if int(arm["stage"]) >= 0:  # a bisect stage writes no y: its time only (no comparison, it does not become prev)
            a_.pop("bitwise_vs_first", None), a_.pop("bitwise_per_chip", None), a_.pop("pcc_vs_first", None), a_.pop("maxabs_vs_first", None)
            row["arms"].append(a_)
            S.log(f"{tag}: {a_}")
            free_bufs(pm)
            del pm, y
            gc.collect()
            continue
        if prev is not None:
            a_["bitwise_vs_prev"] = all(bool(torch.equal(p_, q_)) for p_, q_ in zip(out, prev))
        a_["out_norm"] = [round(float(p_.float().norm()), 3) for p_ in out]
        prev = out
        row["arms"].append(a_)
        S.log(f"{tag}: {a_}")
        free_bufs(pm)
        del pm, y
        gc.collect()
    if C.MOCK:
        row["status"] = "mock"
    else:
        t0 = row["arms"][0].get("median_us")
        for a_ in row["arms"]:
            a_["vs_first"] = round(a_["median_us"] / t0, 4) if t0 and a_.get("median_us") else None
        tab_ok = lambda v_: v_ is None or v_ is True or (isinstance(v_, dict) and all(v_.values()))  # noqa: E731
        ok = all((a_.get("bitwise_vs_first") or not int(a_["expect_bitwise"]) or int(a_["stage"]) >= 0)
                 and a_["det"].split("/")[0] == a_["det"].split("/")[1]
                 and tab_ok(a_.get("tables_vs_first")) for a_ in row["arms"])  # fmt: skip
        row["status"] = "pass" if ok else "fail"
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    row.update(status="fail", error=f"{type(e).__name__}: {e}"[:2000])
S.log(f"{tag}: {row.get('status')} {[(a_['name'], a_.get('median_us'), a_.get('bitwise_vs_first'), a_.get('det')) for a_ in row['arms']]}")
if rec and not C.MOCK:
    C.record("perf", {k_: v_ for k_, v_ in row.items()})
RESULT = row
