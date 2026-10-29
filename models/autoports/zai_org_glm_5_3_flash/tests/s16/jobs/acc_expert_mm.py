# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 ACCURACY, L3 step 7 (program 19:58:04Z): one routed expert of L3 (the busiest;
its fp32 fp8-dequantised weights + TT's own input rows, acc/l3_expert.pt) through PRIMITIVE ttnn ops on the device --
gate|up ttnn.matmul with the bfloat4_b weight (host RNE pack = the cache), clamped SwiGLU from ttnn ops, down matmul
(bfloat4_b) -- each stage vs the host emulation (the same bfp4 values, fp32 math). Splits the fused expert kernels'
+3.7 pct: exact primitive stages here = the gain is inside moe_fused_swiglu / xffn (their SwiGLU or inter-stage pack).
TRACED (one capture of the three stages; the eager call is the compile). ARGS: data, fid (HiFi4), fp32_acc (True),
dump (path), tag, record."""
import gc
import os
import sys

import torch
import torch.nn.functional as F

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
P = args.get("data", "/opt/glm53/data/acc/l3_expert.pt")
if not os.path.exists(P):
    P = P.replace("/opt/glm53/share", "/opt/glm53/home/mpi-shfs")
tag = args.get("tag", "acc-expert-mm")
d = torch.load(P, map_location="cpu")
x, gu, dn, lim = d["x"].float(), d["gu"].float(), d["dn"].float(), float(d["limit"])
n = x.shape[0]
npad = -(-n // 32) * 32
xp = torch.zeros(npad, x.shape[1])
xp[:n] = x
I2, H = gu.shape
I = I2 // 2
rep = ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None
DRAM = ttnn.DRAM_MEMORY_CONFIG
X = ttnn.from_torch(xp.reshape(1, 1, npad, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep,
                    memory_config=DRAM)  # fmt: skip
Wgu = ttnn.from_torch(gu.t().contiguous().reshape(1, 1, H, I2), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                      mesh_mapper=rep, memory_config=DRAM)  # fmt: skip
Wdn = ttnn.from_torch(dn.t().contiguous().reshape(1, 1, I, H), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                      mesh_mapper=rep, memory_config=DRAM)  # fmt: skip
ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=getattr(ttnn.MathFidelity, args.get("fid", "HiFi4")),
                                             math_approx_mode=False, fp32_dest_acc_en=bool(args.get("fp32_acc", True)),
                                             packer_l1_acc=bool(args.get("l1_acc", True)))  # fmt: skip
ODT = ttnn.bfloat16 if args.get("out_bf16", False) else ttnn.float32  # bf16 outputs = moe_fused_swiglu's accumulate path


def fn(T):
    ygu = ttnn.matmul(X, Wgu, compute_kernel_config=ckc, dtype=ODT, memory_config=DRAM)
    g = ttnn.slice(ygu, (0, 0, 0, 0), (1, 1, npad, I))
    u = ttnn.slice(ygu, (0, 0, 0, I), (1, 1, npad, I2))
    a = ttnn.multiply(ttnn.silu(ttnn.clamp(g, max=lim)), ttnn.clamp(u, min=-lim, max=lim))
    ab = ttnn.typecast(a, ttnn.bfloat16)
    y = ttnn.matmul(ab, Wdn, compute_kernel_config=ckc, dtype=ODT, memory_config=DRAM)
    T.update(ygu=ygu, act=a, y=y)
    for t_ in (g, u, ab):
        ttnn.deallocate(t_)


T = {}
fn(T)  # compile
ttnn.synchronize_device(mesh)
row = dict(test="accuracy L3 step 7: one routed expert through primitive ttnn ops (bfp4 matmuls, SwiGLU) vs host",
           milestone="ACC-L3", expert=int(d["e"]), rows=n, fid=args.get("fid", "HiFi4"),
           fp32_acc=bool(args.get("fp32_acc", True)), l1_acc=bool(args.get("l1_acc", True)),
           out_bf16=bool(args.get("out_bf16", False)), tag=tag, status="mock" if C.MOCK else "info")  # fmt: skip
if not C.MOCK:
    for t_ in T.values():
        ttnn.deallocate(t_)
    T = {}
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn(T)
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    h0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0] if mesh.get_num_devices() > 1 else t).float().reshape(npad, -1)[:n]  # noqa: E731

    def bfp_rne(w, mant_bits):
        K, N = w.shape
        wb = w.reshape(K, N // 16, 16)
        mx = wb.abs().amax(-1, keepdim=True).clamp_min(1e-38)
        ex = torch.floor(torch.log2(mx))
        step = torch.pow(2.0, ex - (mant_bits - 1))
        m = torch.round(wb / step).clamp(-(2**mant_bits - 1), 2**mant_bits - 1)
        return (m * step).reshape(K, N)

    qgu = bfp_rne(gu.t().contiguous().to(torch.bfloat16).float(), 3)  # [H, 2I] = the device's values
    qdn = bfp_rne(dn.t().contiguous().to(torch.bfloat16).float(), 3)  # [I, H]
    xb = x.to(torch.bfloat16).float()
    ygu_h = xb @ qgu
    ygu_d, act_d, y_d = h0(T["ygu"]), h0(T["act"]), h0(T["y"])
    g_d, u_d = ygu_d[:, :I], ygu_d[:, I:]
    act_from_dev = F.silu(g_d.clamp(max=lim)) * u_d.clamp(-lim, lim)  # the host SwiGLU of the DEVICE's gate|up
    g_h, u_h = ygu_h[:, :I], ygu_h[:, I:]
    act_h = F.silu(g_h.clamp(max=lim)) * u_h.clamp(-lim, lim)
    y_from_dev_act = act_d.to(torch.bfloat16).float() @ qdn  # the host down matmul of the DEVICE's activation
    y_h = act_h.to(torch.bfloat16).float() @ qdn

    def m(a, b):
        a, b = a.reshape(-1).double(), b.reshape(-1).double()
        return dict(norm_ratio=round(float(a.norm() / b.norm()), 5), proj_scale=round(float((a @ b) / (b @ b)), 5),
                    rel_l2=round(float((a - b).norm() / b.norm()), 5))

    row.update(gate_up=m(ygu_d, ygu_h), swiglu=m(act_d, act_from_dev), down=m(y_d, y_from_dev_act), expert_out=m(y_d, y_h))
    if args.get("dump"):
        torch.save({"y_dev": y_d, "tok": d["tok"], "e": int(d["e"])}, args["dump"])
    ttnn.release_trace(mesh, tid)
S.log(row)
if args.get("record", False):
    C.record("pcc", row)
RESULT = [row]
