# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, any mesh): Stage 16 fused prefill mHC post-mix (tt/prefill_tp.py mhc_post_prefill, kernels/prefill/
mhc_post_pf_*) vs the composite post_streams + concat it replaces (device vs device, same random inputs) and vs a
float64 host reference; both timed as traced replays (median of reps).

ARGS: ts ([64, 256, 1024]), y_dtype ("bf16"), reps (5), tag, record."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
TS = [int(v) for v in args.get("ts", [64, 256, 1024])]
YDT = ttnn.bfloat16 if args.get("y_dtype", "bf16") == "bf16" else ttnn.float32
REPS = int(args.get("reps", 5))
tag = args.get("tag", "mhc-post-check")
FP32 = ttnn.float32
TPN = int(mesh.get_num_devices())
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
dev = lambda x, dt: ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=repl)  # fmt: skip
host = lambda t_: ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).double()  # noqa: E731


def composite(y, xs, post, comb):
    """OptimizedCompositeHyperConnection.post_streams + the concat (the ops path)"""
    t = int(y.shape[-2])
    col = lambda m, k: ttnn.slice(m, (0, 0, 0, k), (1, 1, t, k + 1))  # noqa: E731
    yf = ttnn.typecast(y, FP32) if y.dtype != FP32 else y
    out = [ttnn.multiply(col(post, j), yf) for j in range(4)]
    for i in range(4):
        for j in range(4):
            out[j] = ttnn.addcmul(out[j], xs[i], col(comb, 4 * i + j))
    flat = ttnn.concat(out, dim=-1)
    for t_ in out:
        ttnn.deallocate(t_)
    return flat


def timed(fn):
    out = fn()
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(out)
    if C.MOCK:
        return fn(), []
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        out = fn()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ts.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(mesh, tid)
    return out, ts


def composite_pre(xs, pre):
    """OptimizedCompositeHyperConnection.pre_streams' h = sum_i pre_i * X_i (the ops path)"""
    t = int(xs[0].shape[-2])
    col = lambda m, k: ttnn.slice(m, (0, 0, 0, k), (1, 1, t, k + 1))  # noqa: E731
    h = ttnn.multiply(xs[0], col(pre, 0))
    for i in range(1, 4):
        h = ttnn.addcmul(h, xs[i], col(pre, i))
    return h


results = []
for t in TS:
    g = torch.Generator().manual_seed(t)
    y_h = torch.randn(t, 4096, generator=g)
    x_h = torch.randn(t, 4, 4096, generator=g) * 3
    post_h = torch.rand(t, 4, generator=g) * 2
    comb_h = torch.rand(t, 4, 4, generator=g)
    pre_h = torch.rand(t, 4, generator=g)
    y = dev(y_h.reshape(1, 1, t, 4096), YDT)
    xs = [dev(x_h[:, i].reshape(1, 1, t, 4096).contiguous(), FP32) for i in range(4)]
    xf = dev(x_h.reshape(1, 1, t, 4 * 4096).contiguous(), FP32)  # the flat residual (stream-major columns)
    post = dev(post_h.reshape(1, 1, t, 4), FP32)
    comb = dev(comb_h.reshape(1, 1, t, 16), FP32)
    pre = dev(pre_h.reshape(1, 1, t, 4), FP32)
    f_out, f_ts = timed(lambda: PF.mhc_post_prefill(y, xf, post, comb))
    c_out, c_ts = timed(lambda: composite(y, xs, post, comb))
    fp_out, fp_ts = timed(lambda: PF.mhc_pre_prefill(xf, pre))
    cp_out, cp_ts = timed(lambda: composite_pre(xs, pre))
    yq = host(y).reshape(t, 4096)  # the device's y (bf16-rounded)
    ref = post_h.double().unsqueeze(-1) * yq.unsqueeze(1) + torch.einsum("tij,tid->tjd", comb_h.double(), x_h.double())
    ref = ref.reshape(t, 4 * 4096)
    ref_pre = torch.einsum("ti,tid->td", pre_h.double(), x_h.double())
    fo, co = host(f_out).reshape(t, -1), host(c_out).reshape(t, -1)
    fpo, cpo = host(fp_out).reshape(t, -1), host(cp_out).reshape(t, -1)
    mx = lambda a, b: float((a - b).abs().max())  # noqa: E731
    row = dict(test=f"fused prefill mHC pre / post mixes vs the composite ops (t={t}, y {YDT}) tp{TPN}",
               milestone="P2", status="info", traced=not C.MOCK, t=t,
               post_fused_vs_composite=mx(fo, co), post_fused_vs_f64=mx(fo, ref), post_composite_vs_f64=mx(co, ref),
               pre_fused_vs_composite=mx(fpo, cpo), pre_fused_vs_f64=mx(fpo, ref_pre),
               pre_composite_vs_f64=mx(cpo, ref_pre), post_fused_us=C.stats(f_ts).get("median_us"),
               post_composite_us=C.stats(c_ts).get("median_us"), pre_fused_us=C.stats(fp_ts).get("median_us"),
               pre_composite_us=C.stats(cp_ts).get("median_us"), tag=tag)  # fmt: skip
    if not C.MOCK:  # absolute: fp32 sums of O(10) values -> |d| ~ 1e-5; a stale / garbage output is O(1) or worse
        ok = row["post_fused_vs_f64"] <= 1e-3 and row["pre_fused_vs_f64"] <= 1e-3
        row["status"] = "pass" if ok else "fail"
    S.log(row)
    if args.get("record", True) and not C.MOCK:
        C.record("pcc", row)
    results.append(row)
    for t_ in [y, xf, post, comb, pre, f_out, c_out, fp_out, cp_out] + xs:
        ttnn.deallocate(t_)
RESULT = results
