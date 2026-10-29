# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, d0 TP2 pair; <= 10 min): the ONE device bench unit of the PMK MoE phase set -- layer L's served XSL
MoE on REAL routed input vs the xffn op path (tt/prefill_tp.py EpMoe, GLM53_PF_MOE_FFN=xffn), both as ONE traced replay
on identical device inputs:
  1. the served prefill is built for layers 0..L (s16common.build, the p2 jobs' construction) and its compile launch
     runs on the refpt prompt (the 2,048-token p2 reference); TPPrefill.ffn is hooked to keep layer L's MoE inputs
     (hb = the gathered FFN input, idx / sel = the router's pairs) -- the op path's own tensors, no mock, no golden;
  2. EpMoe(hb, idx, sel) (xffn) is traced and replayed; PmkMoe(x_ext, ids_rm, wt) on inputs_from_op_path(hb, idx,
     sel) is traced and replayed; the outputs are compared (bitwise, else PCC + max |d|) and both timed.
Criterion: bitwise, or PCC >= 0.9999 with the difference explained (fidelity: the composite runs its combine at the
FFN's fidelity; fid=HiFi4 makes both paths' combine HiFi4). Expected (TP2, L4, 2,048 rows): op path xffn + rm ~ 6-10
ms (the pair's 32-tile slices), pmk <= the op path (the plan / relayout / combine launches are gone; same kernels).
ARGS: tile (1: the captured rows repeated tile x; 4 = the 8,192-row chunk), layer (4), layers ([0..4]), ref (refpt/prefill2048_L0-4.refpt), fid ("" = HiFi2 both | "HiFi4"), reps (5),
blocks (""), probe (False), tag, record; program B (progb 1): g (2: members per streamer rectangle), prefetch (0:
PMK_STREAM_PREFETCH, moe2/PERF.md), full_sync (0: dst_full_sync_en + SUBA 8 -- half the dest round-trips per chunk)."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
from readiness_check.schema import load_reference  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.pmk_moe as PM
import models.autoports.zai_org_glm_5_3_flash.tt.pmk_moe2 as M2  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
L = int(args.get("layer", 4))
LAYERS = [int(v) for v in args.get("layers", list(range(L + 1)))]
REF = args.get("ref", "/opt/glm53/work-s16/refpt/prefill2048_L0-4.refpt")
REPS = int(args.get("reps", 5))
FID = str(args.get("fid", "") or "HiFi2")  # ONE explicit fidelity for BOTH paths (program 12:55Z: the two
# paths never ran at the same fidelity -- the op path's EpMoe kept a cached construction's fidelity); asserted below
tag = args.get("tag", "pmk-moe2-bench")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
os.environ["GLM53_PF_MOE_FFN"] = "xffn"  # the op path's expert-per-core FFN (the pair's 32-tile slices: forced)
os.environ["GLM53_PF_MOE_PLAN"] = "mc2"
os.environ["GLM53_PF_MOE_FID"] = FID
os.environ.setdefault("GLM53_PF_MHC", "comp")
os.environ.setdefault("GLM53_PF_DSA", "0")
args = dict(args)
args.setdefault("mla_form", "absorbed")
args.setdefault("moe_fid", FID if "FID" in globals() else str(args.get("fid", "")))  # set_attn REWRITES
S.set_attn(args)  # GLM53_PF_MOE_FID from args["moe_fid"] (13:4xZ: the env set above was reset to HiFi2 -> the op
# path never ran at the requested fidelity: the 12:55Z norm crossover and the moe2 fid assertion)  # incl. env: the served XSL knobs (GLM53_TP_XSL=1 GLM53_TP_XSL_SLICED=1) come from the job's env arg
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(2048,), max_seq=2560)
pf = ent["pf"]
pf.moe_impl = "ep"
pl = pf.layers[[p_.L for p_ in pf.layers].index(L)] if hasattr(pf.layers[0], "L") else pf.layers[LAYERS.index(L)]
assert getattr(pl, "xsl", None) is not None, "layer L must carry the served XSL slices (GLM53_TP_XSL_SLICED=1 env)"

# 1. capture layer L's MoE inputs from the op path's compile launch (the only eager launch; prefill_tp.compile)
cap = {}
orig_ffn = pf.ffn


def hooked_ffn(pl_, h32_local, sp):
    if pl_ is pl and "hb" not in cap:
        x32 = pl_.rms(h32_local, "ffn", PF.FP32)
        idx, scores = pl_.moe.router(x32, decode=False)
        hb = pf.gather(ttnn.typecast(x32, PF.BF16), sp)
        k = int(idx.shape[-1])
        sel = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), PF.FP32)
        pack = pf.gather(ttnn.concat([ttnn.typecast(idx, PF.FP32), sel], dim=-1), sp)
        rows = int(pack.shape[-2])
        cap["hb"] = ttnn.to_torch(ttnn.get_device_tensors(hb)[0] if TPN > 1 else hb)
        cap["idx"] = ttnn.to_torch(ttnn.get_device_tensors(ttnn.slice(pack, (0, 0, 0, 0), (1, 1, rows, k)))[0] if TPN > 1
                                   else ttnn.slice(pack, (0, 0, 0, 0), (1, 1, rows, k)))  # fmt: skip
        cap["sel"] = ttnn.to_torch(ttnn.get_device_tensors(ttnn.slice(pack, (0, 0, 0, k), (1, 1, rows, 2 * k)))[0] if TPN > 1
                                   else ttnn.slice(pack, (0, 0, 0, k), (1, 1, rows, 2 * k)))  # fmt: skip
        for t_ in (x32, hb, sel, pack):
            ttnn.deallocate(t_)
    return orig_ffn(pl_, h32_local, sp)


pf.ffn = hooked_ffn
prompt = load_reference(REF).entries[0].prompt_tokens.reshape(-1).long()
P_ = int(prompt.numel())
c = pf.bucket_for(P_)
pf.write_inputs(c, prompt, slot=0, chunk_start=0, n_valid=P_)
pf.forward(c, head=False)  # the compile launch with the real prompt (eager, once)
ttnn.synchronize_device(mesh)
pf.ffn = orig_ffn
assert "hb" in cap, "layer L's ffn was not reached"
S.log(f"captured layer {L} MoE inputs: hb {tuple(cap['hb'].shape)} idx {tuple(cap['idx'].shape)}")
# REF_DIR (Stage 14, program 2026-10-24 21:29:57Z: B's expert matmuls vs xffn vs fp32): an hf_ref layer dir -- both
# paths run on the CPU fp32 chain's OWN FFN input (bf16 of sub.pt ffn_in) and routing (router.pt indices / weights), and
# the chips' partials summed on the host are scored against its ffn_out (the fp32 model's MoE + shared output): per-row
# rel err (median / p95 / max), per-row PCC p5, rel L2, scale bias -- program B vs xffn with the routing held fixed
REF_DIR = args.get("ref_dir")
REF_OUT = None
if REF_DIR:
    sub_ = torch.load(os.path.join(REF_DIR, "sub.pt"), map_location="cpu", weights_only=True)
    rr_ = torch.load(os.path.join(REF_DIR, "router.pt"), map_location="cpu", weights_only=True)
    rows_ = int(cap["hb"].shape[-2])
    assert int(sub_["ffn_in"].shape[0]) == rows_ and int(args.get("tile", 1)) == 1, (tuple(sub_["ffn_in"].shape), rows_)
    lead = tuple(cap["hb"].shape[:-2])
    cap["hb"] = sub_["ffn_in"].to(torch.bfloat16).reshape(*lead, rows_, -1)
    cap["idx"] = rr_["indices"].float().reshape(*lead, rows_, -1)
    cap["sel"] = rr_["weights"].float().reshape(*lead, rows_, -1)
    REF_OUT = sub_["ffn_out"].double()
    S.log(f"ref_dir {REF_DIR}: inputs = the fp32 chain's ffn_in (bf16) + its routing; scored vs its ffn_out")
# TTFT lever 2 (operator 2026-10-23 18:47:53Z): the 8,192-row prefill chunk. tile > 1 repeats the captured rows along
# the row axis (tile 4 = 8,192 rows): every expert's routed count is exactly tile x the captured chunk's, so the time per
# 2,048-row equivalent isolates the weight-read amortization (real 8K text spreads the same experts a little more evenly)
TILE = int(args.get("tile", 1))
if TILE > 1:
    for k_ in ("hb", "idx", "sel"):
        cap[k_] = cap[k_].repeat(*([1] * (cap[k_].dim() - 2)), TILE, 1).contiguous()
    S.log(f"tiled x{TILE}: hb {tuple(cap['hb'].shape)} idx {tuple(cap['idx'].shape)}")

# 2. both paths on the captured inputs (replicated on every chip: each chip's slice partial is compared chip 0 vs chip 0)
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
dev = lambda t_, dt: ttnn.from_torch(
    t_, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=PF.DRAM, mesh_mapper=repl
)  # noqa: E731
hb = dev(cap["hb"].to(torch.bfloat16), ttnn.bfloat16)
idx = dev(cap["idx"].to(torch.int32), ttnn.uint16)
sel = dev(cap["sel"].float(), ttnn.float32)
# a FRESH EpMoe for the op path: never the layer's cached one (its programs / ckc were built under an earlier job's
# GLM53_PF_MOE_FID); its xffn program is built at its first call below, after the env above was set
ep = PF.EpMoe(pf, pl.moe, xsl=pl.xsl)
assert ep.ffn == "xffn", ep.ffn
fid_op = str(getattr(ep.ckc, "math_fidelity", "?")).split(".")[-1]
assert fid_op == FID and os.environ["GLM53_PF_MOE_FID"] == FID, (fid_op, FID)
S.log(f"fidelity: op path EpMoe ckc {fid_op}, GLM53_PF_MOE_FID={os.environ['GLM53_PF_MOE_FID']}, pmk {FID}")
blocks = tuple(int(v) for v in str(args.get("blocks", "")).split(",")) if args.get("blocks") else None
lv = M2.Levers(x_l1=bool(int(args.get("x_l1", 1))), mcast=int(args.get("mcast", 2)), shared=False,
               ids_u16=bool(int(args.get("ids_u16", 0))), out_bfp8=bool(int(args.get("out_bfp8", 0))))  # fmt: skip
lv = M2.fit_levers(M2.Geometry(c=int(hb.shape[-2]), w=int(ep.down[0].shape[-2]), p=110, h_split=4), lv)
S.log(f"levers after the budget fit: {lv}")
PROGB = bool(int(args.get("progb", 0)))
if PROGB:  # program B (README sec 14): the 24 adj_pmk streamers + 86 receivers, row-bfp8 partials, the P13 term
    G_B, PREFETCH = int(args.get("g", 2)), bool(int(args.get("prefetch", 0)))
    # b_out_bfp8 0 (Stage 14 C-term A/B, program 19:05:26Z): the expert partials as bf16 rows (packer untilize; the combine
    # tilizes in hardware) instead of row-bfp8 (RISC assembly in the writer + RISC rebuild in the combine); yw_db is bfp8-only
    B_BFP8 = bool(int(args.get("b_out_bfp8", 1)))
    lv = M2.Levers(x_l1=lv.x_l1, mcast=G_B, shared=True, ids_u16=lv.ids_u16, out_bfp8=B_BFP8, yw_db=bool(int(args.get("yw_db", 0))) and B_BFP8, yw_coal=int(args.get("yw_coal", 1)) if B_BFP8 else 1,
                   ring_slots=int(args.get("ring_slots", 2)))  # fmt: skip
    # construction watchdog (PERF.md sec 9: the 'cover' cell sat 13 min in a libtt_metal call): every Python stack is
    # dumped to stderr after `construct_s` seconds (repeating), so the next hang names the call
    import faulthandler

    faulthandler.dump_traceback_later(float(args.get("construct_s", 180)), repeat=True)
    t_c = time.time()
    pm = M2.ProgramB(mesh, g_size=G_B, levers=lv, num_experts=ep.e_all, top_k=int(pl.moe.top_k), hidden=int(pl.moe.hidden),
                     limit=float(pl.moe.limit), gate_up=ep.gate_up, down=ep.down, blocks=blocks, fid=FID,
                     probe=bool(args.get("probe", False)), prefetch=PREFETCH, full_sync=bool(int(args.get("full_sync", 0))),
                     rect_mode=str(args.get("rect_mode", "cover")), gmax=int(args.get("gmax", 4)),
                     gu_rt=bool(int(args.get("gu_rt", 0))), gu_ct=int(args.get("gu_ct", 0)),
                     mcast_linked=bool(int(args.get("mcast_linked", 0))), mcast_unicast=bool(int(args.get("mcast_unicast", 0))), mcast_dual=bool(int(args.get("mcast_dual", 0))),
                     lpt=int(args.get("lpt", 0)), gu_split=bool(int(args.get("gu_split", 0))), h_split=int(args.get("h_split", 4)),
                     inplace=bool(int(args.get("inplace", 1))), lpt_w=M2.lpt_weights_from(args.get("lpt_w")), comb_split=bool(int(args.get("comb_split", 0))), yw_brisc=bool(int(args.get("yw_brisc", 0))),
                     selfstream=bool(int(args.get("selfstream", 0))))  # fmt: skip
    S.log(f"program B constructed in {time.time() - t_c:.1f} s: rects {len(pm.rects)} sizes {pm.rect_sizes} members {sum(pm.rect_sizes)}")
    t_c = time.time()
    # bf16 host copies (the bfp4 slices are exact in bf16): half the 19 GB of float32 of the 18:25Z bench
    if pm.inplace:  # PMK_B_INPLACE: the streamers read pl.xsl in place -- no readback, no stream copy (and per chip correct)
        gus_h = dns_h = []
    else:
        gus_h = [ttnn.to_torch(ttnn.get_device_tensors(t)[0] if TPN > 1 else t).to(torch.bfloat16).reshape(-1, int(t.shape[-1])) for t in ep.gate_up]
        dns_h = [ttnn.to_torch(ttnn.get_device_tensors(t)[0] if TPN > 1 else t).to(torch.bfloat16).reshape(-1, int(t.shape[-1])) for t in ep.down]
    S.log(f"expert slices read back in {time.time() - t_c:.1f} s")
    t_c = time.time()
    pages = pm.load_streams(gus_h, dns_h)
    S.log(f"stream tensor packed + on device in {time.time() - t_c:.1f} s")
    faulthandler.cancel_dump_traceback_later()
    S.log(f"program B: {pages} stream pages, streamers {pm.streamers}, rects {len(pm.rects)} sizes {pm.rect_sizes} "
          f"members {sum(pm.rect_sizes)} of {pm.p} (rect_mode {pm.rect_mode}, gmax {pm.gmax}, gu_rt {pm.gu_rt} ct {pm.gu_ct})")
    y_shared = PF.swiglu_prefill(pl.shared, hb)  # the op path's P13 (bf16 TILE [1, 1, C, H]): the (K + 1)-th term
    # the bf16 combine (SelfStreamB / out_bfp8 0) reads y_s as ROW_MAJOR [C, H]: laid out once here, copied per replay
    y_shared_in = y_shared if pm.lv.out_bfp8 else ttnn.to_layout(ttnn.reshape(y_shared, (int(hb.shape[-2]), int(hb.shape[-1]))), ttnn.ROW_MAJOR_LAYOUT, memory_config=PF.DRAM)
else:
    y_shared = None
pm = pm if PROGB else M2.PmkMoe2(mesh, levers=lv, num_experts=ep.e_all, top_k=int(pl.moe.top_k), hidden=int(pl.moe.hidden), limit=float(pl.moe.limit),
                gate_up=ep.gate_up, down=ep.down, blocks=blocks, fid=FID, probe=bool(args.get("probe", False)))  # fmt: skip
x_ext, ids_rm, wt = PM.inputs_from_op_path(hb, idx, sel, float(pl.moe.scaling))
if lv.ids_u16:  # the router tail's layouts (bf16 weights: PCC-explained vs the op path's fp32 weights)
    ids_rm, wt = M2.tail_layout_inputs(idx, sel, float(pl.moe.scaling))
S.log(f"levers {lv} rects {len(pm.rects)} blocks {pm.blocks(int(hb.shape[-2]))} CB {pm.plan(int(hb.shape[-2]))['total']}")
cr = int(hb.shape[-2])
Y_ref = ttnn.from_torch(torch.zeros(1, 1, cr, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                        memory_config=PF.DRAM, mesh_mapper=repl)  # fmt: skip


def run_ref():
    y = ep(hb, idx, sel)
    if PROGB:  # the op path's MoE output = routed + shared (TPPrefill.ffn)
        y2 = ttnn.add(y, y_shared)
        ttnn.deallocate(y)
        y = y2
    ttnn.copy(y, Y_ref)
    ttnn.deallocate(y)


INP = dict(x_ext=x_ext, ids_rm=ids_rm, wt=wt, hb=hb)  # program B's CALLER tensors (realloc swaps them for new buffers)


def run_pmk():
    if PROGB:
        s_ = pm.scratch(cr)
        ttnn.copy(y_shared_in, s_["y_s"])  # the P13 partial (in the layer: the PfStream pair writes it)
    pm(INP["x_ext"], INP["ids_rm"], INP["wt"], cr, x_tile=INP["hb"])  # x_tile: the holders fill their XS block from hb


def traced(fn):
    fn()  # compile launch
    ttnn.synchronize_device(mesh)
    if C.MOCK:
        return []
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ts.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(mesh, tid)
    return ts


# TRACE-LIGHT A/B (Stage 14, 2026-10-23 16:59:28Z): trace_modes e.g. ["base", "rect", "rect_table"] = program B with
# the core sets as rectangles (GLM53_PMK_RECT_MERGE) and / or its DM args in the DRAM table (GLM53_PMK_ARGS=table) on
# THIS backend and inputs; every mode compiled here BEFORE the first capture (the table is filled at its first build),
# then each captured + replayed below: the trace region bytes per chip it adds, its replay time, its output bitwise vs
# the first mode's
MODES = [str(m) for m in args.get("trace_modes", [])]
mode_of = lambda m: dict(rect="rect" in m, table="table" in m)  # noqa: E731
if MODES:
    assert PROGB, "trace_modes: program B (progb 1) only"
    for m in MODES:
        pm.set_trace_modes(**mode_of(m))
        run_pmk()  # this mode's compile launch (eager, once)
        ttnn.synchronize_device(mesh)
    pm.set_trace_modes(**mode_of(MODES[0]))
ts_ref = traced(run_ref)
a = ttnn.to_torch(ttnn.get_device_tensors(Y_ref)[0] if TPN > 1 else Y_ref).float().clone()
ts_pmk = traced(run_pmk)
y = pm.scratch(cr)["y"]
b = ttnn.to_torch(ttnn.get_device_tensors(y)[0] if TPN > 1 else y).float().clone()
trace_rows = {}
# REALLOC (the dynamic block, 2026-10-23 18:43:37Z): the served chain hands program B NEW x_ext / ids_rm / wt / hb buffers
# per call, so the captures below build with other caller addresses than the table fill (the compile launches above):
# the table's static words must still match and the dynamic block carries the moved addresses
REALLOC = bool(int(args.get("realloc", 0)))
if MODES and REALLOC:
    keep = dict(INP)  # the originals stay alive, so every clone lands at another address
    INP = {k_: ttnn.clone(v_) for k_, v_ in keep.items()}
    moved = sorted(k_ for k_ in INP if int(INP[k_].buffer_address()) != int(keep[k_].buffer_address()))
    S.log(f"realloc: caller tensors at new addresses: {moved} of {sorted(INP)}")
    if C.MOCK:  # the mock captures nothing: one eager build per mode runs the static check + the dynamic block instead
        for m in MODES:
            pm.set_trace_modes(**mode_of(m))
            run_pmk()
        ttnn.synchronize_device(mesh)
        S.log(f"realloc: rebuilt every mode on the new buffers; table {getattr(pm, 'arg_table_report', None)}")
if MODES and not C.MOCK:
    def trace_bytes():
        v_ = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return int(v_.total_bytes_allocated_per_bank) * int(v_.num_banks)

    chips = lambda t_: [ttnn.to_torch(d_).float().clone() for d_ in (ttnn.get_device_tensors(t_) if TPN > 1 else [t_])]  # noqa: E731
    outs = {}
    for m in MODES:
        pm.set_trace_modes(**mode_of(m))
        before = trace_bytes()
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            run_pmk()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        after = trace_bytes()
        ts_m = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ts_m.append((time.perf_counter() - t0) * 1e6)
        outs[m] = chips(pm.scratch(cr)["y"])
        ttnn.release_trace(mesh, tid)
        trace_rows[m] = dict(trace_bytes_per_chip=after - before, trace_mib_per_chip=round((after - before) / 2**20, 4),
                             us=C.stats(ts_m).get("median_us"), table=getattr(pm, "arg_table_report", None) if "table" in m else None)  # fmt: skip
    for m in MODES:
        trace_rows[m]["bitwise_vs_" + MODES[0]] = all(torch.equal(x_, y_) for x_, y_ in zip(outs[m], outs[MODES[0]]))
        trace_rows[m]["bitwise_vs_first_inputs_chip0"] = bool(torch.equal(outs[m][0], b))  # vs the pre-realloc run
    S.log(f"TRACE-LIGHT program B modes: {trace_rows}")
REF_ROW = None
if REF_OUT is not None and not C.MOCK:  # every chip's partial summed on the host = the layer's MoE output (the RS's sum)
    def full(t_):
        return sum(ttnn.to_torch(d_).double().reshape(cr, -1) for d_ in (ttnn.get_device_tensors(t_) if TPN > 1 else [t_]))

    R_ = int(args.get("ref_rows", 2042))  # the prompt's valid rows (goldens/prompts/2048.pt)

    def score(y_):
        y_, r_ = y_[:R_], REF_OUT[:R_]
        d_ = y_ - r_
        rel = d_.norm(dim=-1) / r_.norm(dim=-1).clamp_min(1e-30)
        yc, rc = y_ - y_.mean(-1, keepdim=True), r_ - r_.mean(-1, keepdim=True)
        pr = (yc * rc).sum(-1) / (yc.norm(dim=-1) * rc.norm(dim=-1)).clamp_min(1e-30)
        return dict(rel_l2=float(d_.norm() / r_.norm()), scale_bias=float((y_ * r_).sum() / (r_ * r_).sum() - 1),
                    row_rel_median=float(rel.median()), row_rel_p95=float(rel.quantile(0.95)), row_rel_max=float(rel.max()),
                    row_pcc_p5=float(pr.quantile(0.05)), row_pcc_min=float(pr.min()), worst_rows=rel.topk(5).indices.tolist())  # fmt: skip

    ya, yb = full(Y_ref), full(pm.scratch(cr)["y"])
    REF_ROW = dict(xffn_vs_fp32=score(ya), progb_vs_fp32=score(yb), rows=R_,
                   progb_vs_xffn=dict(rel_l2=float((yb[:R_] - ya[:R_]).norm() / ya[:R_].norm()), maxabs=float((yb - ya).abs().max())))  # fmt: skip
    S.log(f"REF {REF_DIR}: {REF_ROW}")
if args.get("dump"):  # the outputs for the stack's truth script (ffn_ref_L4_r18.py): pmk + the op path, chip 0
    torch.save(dict(pmk=b, op=a, tag=tag, test_fid=FID), str(args["dump"]))
    S.log(f"dumped pmk / op outputs to {args['dump']}")
row = dict(test=f"pmk_moe2{' PROGRAM-B' if PROGB else ''} (levers x_l1={lv.x_l1} G={lv.mcast} u16={lv.ids_u16} bfp8={lv.out_bfp8} shared={lv.shared}{' prefetch=1' if PROGB and pm.prefetch else ''}{' full_sync=1' if pm.full_sync else ''}{' gu_rt=1' if pm.gu_rt else ''}{' yw_db=1' if lv.yw_db else ''}{' linked=1' if PROGB and pm.mcast_linked else ''}{' unicast=1' if PROGB and pm.mcast_unicast else ''}{' dual=1' if PROGB and pm.mcast_dual else ''}{' lpt=' + str(pm.lpt) if PROGB and pm.lpt else ''}{' slots=' + str(lv.ring_slots) if lv.ring_slots != 2 else ''}{' split=1' if pm.lv.gu_split else ''}{' selfstream=1' if PROGB and isinstance(pm, M2.SelfStreamB) else ''}{' inplace=1' if PROGB and pm.inplace else ''}{' lptw=1' if PROGB and pm.lpt_w else ''}{' combsplit=1' if pm.lv.comb_split else ''}{' ywbrisc=1' if pm.lv.yw_brisc else ''}{' hs=' + str(pm.h_split) if pm.h_split != 4 else ''}{' coal=' + str(lv.yw_coal) if lv.yw_db and lv.yw_coal > 1 else ''}{' ' + pm.rect_mode + str(pm.rect_sizes[0]) if PROGB and pm.rect_sizes else ''}) vs xffn op path, layer {L}, {cr} rows, real routing, fid {pm.fid}, tp{TPN}", milestone="PMK",
           status="info", traced=not C.MOCK, blocks=list(pm.blocks(cr)), cb_total=pm.l1_bytes(cr), levers=str(lv), rects=len(pm.rects),
           bitwise=bool(torch.equal(a, b)), pcc=C.pcc(a, b), maxabs=float((a - b).abs().max()),
           members=(sum(pm.rect_sizes) if PROGB else None), rect_sizes=(pm.rect_sizes if PROGB else None), lpt=(pm.lpt if PROGB else None), ring_slots=lv.ring_slots, inplace=(pm.inplace if PROGB else None), gu_split=pm.lv.gu_split,
           ref_norm=float(a.norm()), out_norm=float(b.norm()), op_us=C.stats(ts_ref).get("median_us"),
           pmk_us=C.stats(ts_pmk).get("median_us"), fid_op=fid_op, fid_pmk=pm.fid, trace_modes=trace_rows or None, tag=tag)  # fmt: skip
per_ = cr / 2048.0  # the 2,048-row equivalents in this chunk
row.update(ref_dir=REF_DIR, ref=REF_ROW, b_comb_fid=os.environ.get("GLM53_PMK_B_COMB_FID", ""))
row.update(tile=TILE, rows=cr, realloc=REALLOC, op_us_per_2048=(round(row["op_us"] / per_, 1) if row.get("op_us") else None),
           pmk_us_per_2048=(round(row["pmk_us"] / per_, 1) if row.get("pmk_us") else None))  # fmt: skip
assert row["fid_op"] == row["fid_pmk"] == FID, row
if not C.MOCK:
    row["status"] = "pass" if row["bitwise"] or row["pcc"] >= 0.9999 else "fail"
    if pm.probe:
        chips = pm.probe_read()
        row["probe"] = [{k_: v.tolist()[:4] for k_, v in ch.items()} for ch in chips]
        row["stream_rate"] = M2.stream_rate(chips)  # achieved expert-weight GB/s per chip (vs ~480-512 peak)
        S.log(f"expert-weight streaming rate per chip: {row['stream_rate']}")
        # the program-B scoreboard (moe2/PERF.md): streamer credit / read / multicast waits, member waits -- not on the
        # self-stream form (SelfStreamB has no streamers: progb_per_streamer raised KeyError 'streamer' after the timing,
        # progb8 2026-10-25 22:11Z, and the row was lost); a probe-analysis error never loses the row
        if PROGB and not isinstance(pm, M2.SelfStreamB):
            try:
                row["scoreboard"] = [M2.progb_scoreboard(ch) for ch in chips]
                row["per_streamer"] = M2.progb_per_streamer(pm, chips[0])  # perf5: the 24 streamer rows + their rectangles (chip 0)
                S.log(f"per streamer (chip 0): {row['per_streamer']}")
                S.log(f"program B scoreboard per chip: {row['scoreboard']}")
            except Exception as e_:  # noqa: BLE001
                row["scoreboard_error"] = f"{type(e_).__name__}: {e_}"[:300]
S.log(row)
if rec and not C.MOCK:
    C.record("pcc", row)
RESULT = row
