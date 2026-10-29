# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14, the PMK accuracy blocker, candidate 2 of numaudit/fidelity_parity.md): the MoE COMBINE's
math fidelity in isolation. The PMK's program B runs the op's combine arithmetic (kernels/prefill/moe_combine_rm_*: per
token, the K gathered bf16 row parts x their fp32 weights -- SCALAR broadcast, the weight is the FPU SrcB operand --
accumulated in fp32 by the packer) at the B composite's ONE fidelity (HiFi2); the op runs it at HiFi4. Here the op's
own combine kernels run on IDENTICAL inputs at each fidelity, TRACED (compile, capture, one replay), and each output is
scored against a host fp32 combine of the same bf16 rows and fp32 weights: relative L2 error and the SCALE BIAS
(<y, y_ref> / <y_ref, y_ref> - 1: a truncated SrcB weight biases every row toward zero).
Inputs: c tokens, k = 8 rows each (the router's top-8), weights = top-8 of sigmoid scores renormalised x 2.5 (the model's
routed scaling), rows N(0, 1) x a per-row scale (the experts' outputs). ARGS: c (256), h (4096), fids (["HiFi2",
"HiFi4"]), seed, tag, record."""
import json
import os
import time

import torch

import ttnn

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
TREE = os.environ["TT_METAL_HOME"]
KDIR = os.path.join(TREE, "models/autoports/zai_org_glm_5_3_flash/tt/kernels")
C_, K_, H_ = int(args.get("c", 256)), int(args.get("k", 8)), int(args.get("h", 4096))  # k 1: no packer-L1 accumulation
FIDS = list(args.get("fids", ["HiFi2", "HiFi4"]))
TILE, BF16, FP32 = 32, ttnn.bfloat16, ttnn.float32
tag = args.get("tag", "combine-fid-unit")
g = torch.Generator().manual_seed(int(args.get("seed", 0)))

# inputs: row j of token t lives at out_rm[j C + t]; dest[j C + t] = that row (identity layout)
rows = (torch.randn(K_ * C_, H_, generator=g) * torch.rand(K_ * C_, 1, generator=g).mul(4).add(0.25)).to(torch.bfloat16)
scores = torch.sigmoid(torch.randn(C_, 256, generator=g) * 2)
top = torch.topk(scores, K_, dim=-1).values  # k 1: the single weight 2.5
wt = (top / top.sum(-1, keepdim=True) * 2.5).float()  # [C, K]
CRM_U2D = bool(args.get("crm_u2d", False))  # UnpackToDestFp32 on the fp32 row CB the tilize reads
W_BF16 = args.get("w_bf16")  # "rne" / "trunc": the weights pre-narrowed on the host (the reference uses the SAME weights):
if W_BF16 == "rne":  # a device bias that vanishes then = the fp32 weight's precision in the SrcB operand
    wt = wt.to(torch.bfloat16).float()
elif W_BF16 == "trunc":
    wt = (wt.contiguous().view(torch.int32) & ~0xFFFF).view(torch.float32)
dest = torch.arange(K_ * C_, dtype=torch.int32)
y_ref = sum(wt[:, j : j + 1].double() * rows[j * C_ : (j + 1) * C_].double() for j in range(K_))  # [C, H] fp64

repl = ttnn.ReplicateTensorToMesh(mesh) if int(mesh.get_num_devices()) > 1 else None
dram = ttnn.DRAM_MEMORY_CONFIG
out_rm = ttnn.from_torch(rows, BF16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=dram, mesh_mapper=repl)
dest_t = ttnn.from_torch(dest.reshape(1, -1), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=dram,
                         mesh_mapper=repl)  # fmt: skip
wt_t = ttnn.from_torch(wt.reshape(1, 1, C_, K_), FP32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram,
                       mesh_mapper=repl)  # fmt: skip
grid = mesh.compute_with_storage_grid_size()


def combine_program(y, fid):
    """prefill_tp.py EpMoe._combine_rm (hs = 1: one part of h * 2 bytes = pt tiles) at math fidelity `fid`"""
    hs, part = 1, H_ * 2
    pt = part // 2048
    assert pt * 2048 == part and 1 <= pt <= 4, (H_, pt)
    units = (C_ // TILE) * hs
    ncores = min(units, int(grid.x) * int(grid.y))
    per = -(-units // ncores)
    cores = [(i % int(grid.x), i // int(grid.x)) for i in range(ncores)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(a, b)) for a, b in cores])
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (a, b_) in enumerate(cores):
        u0 = min(i * per, units)
        nn = min(per, units - u0)
        rr[a][b_] = [out_rm.buffer_address(), dest_t.buffer_address(), wt_t.buffer_address(), u0, nn]
        rc[a][b_] = [nn]
        rw[a][b_] = [y.buffer_address(), u0, nn]
    src = lambda name: os.path.join(KDIR, "prefill", f"moe_combine_rm_{name}.cpp")  # noqa: E731
    dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                      fp32_dest_acc_en=True)  # fmt: skip
    if CRM_U2D:  # the fp32 accumulation rows (CB 17) tilized through DEST in fp32: the only narrowing = the bf16 pack
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[17] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
    kd = lambda name, ct_, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
        kernel_source=src(name), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
        compile_time_args=ct_, runtime_args=rt, config=cfg)  # fmt: skip
    kernels = [kd("reader", [C_, K_, H_ * 2, part, pt, hs], rr, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
               kd("compute", [K_, pt], rc, cc),
               kd("writer", [pt, H_ // TILE, hs], rw, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default))]  # fmt: skip
    cb = lambda i, dt, page, n_: ttnn.CBDescriptor(total_size=page * n_, core_ranges=crs, format_descriptors=[  # noqa: E731
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])  # fmt: skip
    cbs = [cb(0, BF16, 2048, 2 * K_ * pt), cb(1, FP32, 4096, 2 * K_), cb(2, ttnn.uint32, K_ * 32 * 4 + 4096, 1),
           cb(16, BF16, 2048, 32 * pt), cb(17, FP32, 4096, 32 * pt)]  # fmt: skip
    return lambda: ttnn.generic_op([out_rm, dest_t, wt_t, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))


res = {}
ys, runs = {}, {}
for fid in FIDS:  # every program compiled BEFORE any capture (a compile after a capture keeps a live buffer)
    ys[fid] = ttnn.from_torch(torch.zeros(1, 1, C_, H_), BF16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram,
                              mesh_mapper=repl)  # fmt: skip
    runs[fid] = combine_program(ys[fid], fid)
    runs[fid]()  # the one-time compile launch; the scored output is the traced replay's
ttnn.synchronize_device(mesh)
for fid in FIDS:
    y = ys[fid]
    if os.environ.get("GLM53_MOCK"):  # a mock cannot capture a trace (memory: mock-cannot-capture-traces); builds only
        res[fid] = dict(mock=True)
        continue
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    runs[fid]()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    dev = ttnn.get_device_tensors(y)[0] if int(mesh.get_num_devices()) > 1 else y
    yd = ttnn.to_torch(dev).reshape(C_, H_).double()
    ttnn.release_trace(mesh, tid)
    d = yd - y_ref
    res[fid] = dict(rel_l2=float(d.norm() / y_ref.norm()), scale_bias=float((yd * y_ref).sum() / (y_ref * y_ref).sum() - 1),
                    max_abs=float(d.abs().max()), y_ref_absmax=float(y_ref.abs().max()))  # fmt: skip
# the bf16 output's own floor: the host fp32 combine rounded to bf16 (RNE) vs fp64
fl = y_ref.float().to(torch.bfloat16).double() - y_ref
res["bf16_round_floor"] = dict(rel_l2=float(fl.norm() / y_ref.norm()), scale_bias=float(((y_ref + fl) * y_ref).sum() / (y_ref * y_ref).sum() - 1))
row = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           test=f"{os.environ.get('GLM53_ROW_PREFIX', '')}combine_fid_unit", status="mock" if os.environ.get("GLM53_MOCK") else "record",
           c=C_, k=K_, h=H_, w_bf16=W_BF16, crm_u2d=CRM_U2D, traced=True, results=res, tag=tag)  # fmt: skip
print(f"COMBINE-FID {json.dumps(row)}", flush=True)
ev = os.environ.get("GLM53_STAGE_EVIDENCE")
if args.get("record") and ev:
    with open(os.path.join(ev, "pcc.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
RESULT = dict(ok=True, results=res)
