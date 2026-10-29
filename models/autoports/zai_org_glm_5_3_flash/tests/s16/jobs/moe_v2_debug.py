# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, debug): EP MoE v2 step by step (eager) on chip 0's view, each step checked on the host: the plan
(meta rows used vs the counts), the bounded dispatch (xd rows vs x rows), the expert output (finite rows, vs v1's
expert output from the same plan), the combine (vs a host emulation of y[t] = sum_j w[j t] out[dest[j t]]).

ARGS: layers ([0, 1, 2, 3, 4]), layer (4), n (2048), versions ([1, 0]: v2 then v1; [1] = v2 alone), steps
("plan,dispatch,experts,combine": stop after a step -- a watched run of one step localises a hang), trace_reps
(0: the full v2 call captured + replayed that many times)."""
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
L = int(args.get("layer", 4))
N = int(args.get("n", 2048))
STEPS = set(str(args.get("steps", "plan,dispatch,experts,combine")).split(","))
os.environ["GLM53_PF_MOE_V2_XDZERO"] = "1" if args.get("xd_zero") else "0"
TPN = int(mesh.get_num_devices())
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "ops"
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,))
pf = ent["pf"]
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
pl = pf.layers[LAYERS.index(L)]


def h0(t_):
    return ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_)


x_in, _ = S.golden_layer(L, N)
pad = torch.zeros(c, 4, 4096)
pad[:N] = x_in.reshape(N, 4, 4096).float()
X = pf.streams_to_device(pad, sp)
hh, post, comb, xs = pl.hc["ffn"].pre_streams(X)
x32 = pl.rms(hh, "ffn", PF.FP32)
idx, scores = pl.moe.router(x32, decode=False)
hb = pf.gather(ttnn.typecast(x32, PF.BF16), sp)
sel = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), PF.FP32)
if sp:
    idx = ttnn.typecast(pf.gather(ttnn.typecast(idx, PF.FP32), sp), ttnn.uint16)
    sel = pf.gather(sel, sp)
k = int(idx.shape[-1])
n = c * k
h = 4096
findings = {}
for v in [str(int(x)) for x in args.get("versions", [1, 0])]:
    os.environ["GLM53_PF_MOE_V2"] = v
    ep = PF.EpMoe(pf, pl.moe)
    ids_t = ttnn.transpose(ttnn.typecast(idx, ttnn.uint32), -2, -1)
    ids_rm = ttnn.to_layout(ttnn.reshape(ids_t, (1, n)), ttnn.ROW_MAJOR_LAYOUT, memory_config=PF.DRAM)
    b = ep.plan(ids_rm, c)
    ttnn.synchronize_device(mesh)
    cnt = h0(b["cnt"]).reshape(-1).to(torch.int64)
    start = h0(b["start"]).reshape(-1).to(torch.int64)
    disp = h0(b["disp"]).reshape(-1).to(torch.int64)
    dest = h0(b["dest"]).reshape(-1).to(torch.int64)
    used_calc = int(((cnt + 31) // 32 * 32).sum())
    f = dict(cnt_sum=int(cnt.sum()), used_calc=used_calc, cap=b["cap"])
    if v == "1":
        f["meta_used"] = int(h0(b["meta"]).reshape(-1)[0])
    S.log(f"v{2 if v == '1' else 1}: plan ok {f}")
    if "dispatch" not in STEPS:
        findings[v] = f
        continue
    x_ext = ttnn.to_layout(ttnn.pad(ttnn.reshape(hb, (c, h)), ((0, PF.TILE), (0, 0)), 0.0), ttnn.ROW_MAJOR_LAYOUT,
                           memory_config=PF.DRAM)  # fmt: skip
    if v == "1":
        mk = ttnn.zeros if args.get("xd_zero") else ttnn.empty
        xd = mk((1, 1, b["cap"], h), dtype=PF.BF16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=PF.DRAM)
        ep._dispatch(x_ext, b, xd, h * 2)
    else:
        xd = ttnn.reshape(ttnn.embedding(b["disp"], x_ext, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=PF.BF16),
                          (1, 1, b["cap"], h))  # fmt: skip
    ttnn.synchronize_device(mesh)
    xe = h0(x_ext).reshape(c + 32, h).float()
    xdh = h0(xd).reshape(b["cap"], h).float()
    u = used_calc
    ref_rows = xe[disp[:u].clamp(max=c + 31)]
    f["xd_rows_mismatch"] = int((xdh[:u] != ref_rows).any(dim=-1).sum())
    S.log(f"v{2 if v == '1' else 1}: dispatch ok, xd_rows_mismatch {f['xd_rows_mismatch']}")
    if "experts" not in STEPS:
        findings[v] = f
        continue
    out = ttnn.empty((1, 1, b["cap"], h), dtype=PF.BF16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=PF.DRAM)
    ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
        xd, ep.gate_up, ep.gate_up, ep.down, b["cnt"], ep.idx_table, dtype=PF.BF16, compute_kernel_config=ep.ckc,
        output=out, core_grid=PF.MFS_GRID, expert_region_offsets=b["start"], read_x_at_offset=True,
        activation=ttnn.RoutedExpertActivation.ClampedSiluGlu, gate_up_interleaved=True,
        l1_budget_bytes=PF.PF_MOE_CB)  # fmt: skip
    ttnn.synchronize_device(mesh)
    oh = h0(out).reshape(b["cap"], h).float()
    local_rows = torch.zeros(b["cap"], dtype=torch.bool)
    for e in range(len(cnt)):
        local_rows[int(start[e]) : int(start[e]) + int(cnt[e])] = True
    f["out_nonfinite_local_rows"] = int((~torch.isfinite(oh[local_rows])).any(dim=-1).sum())
    f["out_local"] = oh[local_rows]
    findings[v] = f
    S.log(f"v{2 if v == '1' else 1}: experts ok, non-finite local rows {f['out_nonfinite_local_rows']}")
    if v == "1" and "combine" in STEPS:
        # combine vs host emulation
        w = ttnn.multiply(sel, ttnn.reciprocal(ttnn.add(ttnn.sum(sel, dim=-1, keepdim=True), 1e-20)))
        w = ttnn.multiply(w, pl.moe.scaling)
        lo, hi, _ = pf.ep_bounds()
        ids_f = ttnn.typecast(idx, PF.FP32)
        wl = ttnn.multiply(w, ttnn.multiply(ttnn.gt(ids_f, lo), ttnn.lt(ids_f, hi)))
        wt = ttnn.to_layout(ttnn.reshape(ttnn.transpose(wl, -2, -1), (1, n)), ttnn.ROW_MAJOR_LAYOUT,
                            memory_config=PF.DRAM)  # fmt: skip
        y = ttnn.empty((1, 1, c, h), dtype=PF.BF16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=PF.DRAM)
        ep._combine(out, b["dest"], wt, y, c, k, h)
        ttnn.synchronize_device(mesh)
        wth = h0(wt).reshape(k, c).float()
        yref = torch.zeros(c, h)
        for j in range(k):
            dj = dest[j * c : (j + 1) * c]
            wj = wth[j]
            m = wj != 0
            yref[m] += wj[m, None] * oh[dj[m]]
        yh = h0(y).reshape(c, h).float()
        f["combine_nonfinite"] = int((~torch.isfinite(yh)).sum())
        fin = torch.isfinite(yh)
        f["combine_pcc_vs_host"] = round(C.pcc(torch.where(fin, yh, 0.0), yref), 6)
        f["combine_max_abs"] = float((torch.where(fin, yh, 0.0) - yref).abs().max())
        f["local_pairs"] = int((wth != 0).sum())
        bad = (~fin).any(dim=-1).nonzero().flatten()
        f["first_bad_rows"] = bad[:8].tolist()
        f["bad_rows_total"] = int(bad.numel())
        f["bad_rows_mod32_hist"] = torch.bincount(bad % 32, minlength=32).tolist()
        loc = (wth != 0).t()  # [C, k]
        f["bad_rows_local_pairs_hist"] = (
            torch.bincount(loc[bad].sum(-1), minlength=k + 1).tolist() if bad.numel() else []
        )
        f["all_rows_local_pairs_hist"] = torch.bincount(loc.sum(-1), minlength=k + 1).tolist()
        if bad.numel():
            r0 = int(bad[0])
            cols = (~fin[r0]).nonzero().flatten()
            f["row0_bad_col_tiles"] = sorted(set((cols // 32).tolist()))[:16]
            f["row0_bad_cols_in_tile"] = sorted(set((cols % 32).tolist()))
            f["row0_local_slots"] = loc[r0].nonzero().flatten().tolist()
            f["row0_dest"] = [int(dest[j * c + r0]) for j in range(k)]
            # the finite part of a good row next to it: exact?
            good = [r for r in range(r0 - 3, r0 + 4) if 0 <= r < c and bool(fin[r].all())]
            f["neighbour_good_rows_max_abs"] = {r: float((yh[r] - yref[r]).abs().max()) for r in good}
f1, f0 = findings.get("1", {}), findings.get("0", {})
if "out_local" in f1 and "out_local" in f0 and f1["out_local"].shape == f0["out_local"].shape:
    a, b_ = f1["out_local"], f0["out_local"]
    fin = torch.isfinite(a) & torch.isfinite(b_)
    f1["out_v2_vs_v1_max_abs"] = float((a[fin] - b_[fin]).abs().max()) if fin.any() else None
for f in (f0, f1):
    f.pop("out_local", None)
S.log({"v2": f1, "v1": f0})
if int(args.get("trace_reps", 0)) and not C.MOCK:  # the full v2 call captured and replayed (the z1 hang was a replay)
    os.environ["GLM53_PF_MOE_V2"] = "1"
    ep = PF.EpMoe(pf, pl.moe)
    y = ep(hb, idx, sel)
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(y)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        y = ep(hb, idx, sel)
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    for i in range(int(args["trace_reps"])):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        S.log(f"v2 trace replay {i} done")
    yh = h0(y).reshape(c, h).float()
    S.log(f"v2 traced output non-finite {int((~torch.isfinite(yh)).sum())}")
    ttnn.release_trace(mesh, tid)
os.environ["GLM53_PF_MOE_V2"] = "0"
RESULT = {"v2": f1, "v1": f0}
