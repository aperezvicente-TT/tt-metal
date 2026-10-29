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
ARGS: layer (4), layers ([0..4]), ref (refpt/prefill2048_L0-4.refpt), fid ("" = HiFi2 both | "HiFi4"), reps (5),
blocks (""), probe (False), tag, record."""
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

import models.autoports.zai_org_glm_5_3_flash.tt.pmk_moe as PM  # noqa: E402
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
tag = args.get("tag", "pmk-moe-bench")
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
pm = PM.PmkMoe(mesh, num_experts=ep.e_all, top_k=int(pl.moe.top_k), hidden=int(pl.moe.hidden), limit=float(pl.moe.limit),
               gate_up=ep.gate_up, down=ep.down, blocks=blocks, fid=FID, probe=bool(args.get("probe", False)))  # fmt: skip
x_ext, ids_rm, wt = PM.inputs_from_op_path(hb, idx, sel, float(pl.moe.scaling))
cr = int(hb.shape[-2])
Y_ref = ttnn.from_torch(torch.zeros(1, 1, cr, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                        memory_config=PF.DRAM, mesh_mapper=repl)  # fmt: skip


def run_ref():
    y = ep(hb, idx, sel)
    ttnn.copy(y, Y_ref)
    ttnn.deallocate(y)


def run_pmk():
    pm(x_ext, ids_rm, wt, cr)


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


ts_ref = traced(run_ref)
a = ttnn.to_torch(ttnn.get_device_tensors(Y_ref)[0] if TPN > 1 else Y_ref).float().clone()
ts_pmk = traced(run_pmk)
y = pm.scratch(cr)["y"]
b = ttnn.to_torch(ttnn.get_device_tensors(y)[0] if TPN > 1 else y).float().clone()
row = dict(test=f"pmk_moe vs xffn op path, layer {L}, {cr} rows, real routing, fid {pm.fid}, tp{TPN}", milestone="PMK",
           status="info", traced=not C.MOCK, blocks=list(pm.blocks(cr)), cb_total=pm.l1_bytes(cr),
           bitwise=bool(torch.equal(a, b)), pcc=C.pcc(a, b), maxabs=float((a - b).abs().max()),
           ref_norm=float(a.norm()), out_norm=float(b.norm()), op_us=C.stats(ts_ref).get("median_us"),
           pmk_us=C.stats(ts_pmk).get("median_us"), fid_op=fid_op, fid_pmk=pm.fid, tag=tag)  # fmt: skip
assert row["fid_op"] == row["fid_pmk"] == FID, row
if not C.MOCK:
    row["status"] = "pass" if row["bitwise"] or row["pcc"] >= 0.9999 else "fail"
    if pm.probe:
        row["probe"] = [{k_: v.tolist()[:4] for k_, v in ch.items()} for ch in pm.probe_read()]
S.log(row)
if rec and not C.MOCK:
    C.record("pcc", row)
RESULT = row
