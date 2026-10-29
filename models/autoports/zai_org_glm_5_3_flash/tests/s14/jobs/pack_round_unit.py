# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14, the accuracy blocker: a shared scale bias?): HOW the device narrows fp32 to bf16. The combine
unit (numaudit/combine_fid) found a -0.32 % scale bias even at HiFi4, where an RNE bf16 rounding of the fp32 result
gives ~0 -- the size of a TRUNCATING fp32 -> bf16 narrowing (half an ulp). Here three standard ttnn paths narrow known
fp32 values to bf16, TRACED (compile, capture, one replay), and each result is compared with the host's RNE and the
host's truncation of the SAME fp32 values:
  typecast   ttnn.typecast(x fp32 TILE -> bf16)
  add        ttnn.add(x fp32, 0 fp32, dtype=bf16)                         (FPU eltwise, fp32 DEST, bf16 pack)
  matmul     ttnn.matmul(a bf16, b bf16) -> bf16, HiFi4 + fp32 DEST       (vs the host fp32 product of the same bf16)
Per path: agreement with RNE / with truncation (fraction of elements equal), and the scale bias (<y, ref> / <ref, ref>
- 1) of device / RNE / truncation. ARGS: rows (1,024), cols (4,096), seed, tag, record."""
import json
import os
import time

import torch

import ttnn

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
R_, C_ = int(args.get("rows", 1024)), int(args.get("cols", 4096))
tag = args.get("tag", "pack-round-unit")
MOCK = bool(os.environ.get("GLM53_MOCK"))
g = torch.Generator().manual_seed(int(args.get("seed", 0)))
repl = ttnn.ReplicateTensorToMesh(mesh) if int(mesh.get_num_devices()) > 1 else None
dram = ttnn.DRAM_MEMORY_CONFIG
put = lambda t, dt: ttnn.from_torch(t, dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram, mesh_mapper=repl)  # noqa: E731
chip0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0] if int(mesh.get_num_devices()) > 1 else t)  # noqa: E731


def trunc_bf16(x: torch.Tensor) -> torch.Tensor:
    return (x.float().contiguous().view(torch.int32) & ~0xFFFF).view(torch.float32)


x = (torch.randn(R_, C_, generator=g) * torch.rand(R_, 1, generator=g).mul(8).add(0.1)).float()
a = (torch.randn(R_, 512, generator=g)).to(torch.bfloat16)
b = (torch.randn(512, C_, generator=g) * 0.05).to(torch.bfloat16)
mm_ref = (a.double() @ b.double()).float()  # the fp32 product the matmul should narrow
xt, zt, at, bt = put(x, ttnn.float32), put(torch.zeros_like(x), ttnn.float32), put(a, ttnn.bfloat16), put(b, ttnn.bfloat16)
ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                             fp32_dest_acc_en=True, packer_l1_acc=False)  # fmt: skip
paths = {"typecast": (lambda: ttnn.typecast(xt, ttnn.bfloat16), x),
         "add": (lambda: ttnn.add(xt, zt, dtype=ttnn.bfloat16), x),
         "matmul": (lambda: ttnn.matmul(at, bt, dtype=ttnn.bfloat16, compute_kernel_config=ckc), mm_ref)}  # fmt: skip
res = {}
for name, (fn, ref) in paths.items():
    out = fn()  # the one-time compile launch
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(out)
for name, (fn, ref) in paths.items():
    if MOCK:  # a mock cannot capture a trace
        res[name] = dict(mock=True)
        continue
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    y = chip0(out).reshape(R_, C_).float()
    ttnn.release_trace(mesh, tid)
    rne, trn = ref.to(torch.bfloat16).float(), trunc_bf16(ref)
    refd = ref.double()
    bias = lambda t: float((t.double() * refd).sum() / (refd * refd).sum() - 1)  # noqa: E731
    res[name] = dict(eq_rne=float((y == rne).float().mean()), eq_trunc=float((y == trn).float().mean()),
                     bias_device=bias(y), bias_rne=bias(rne), bias_trunc=bias(trn),
                     maxabs_vs_ref=float((y - ref).abs().max()))  # fmt: skip
row = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test="pack_round_unit",
           status="mock" if MOCK else "record", rows=R_, cols=C_, traced=True, results=res, tag=tag)  # fmt: skip
print(f"PACK-ROUND {json.dumps(row)}", flush=True)
ev = args.get("evidence")
if args.get("record") and ev:
    with open(os.path.join(ev, "pcc.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
RESULT = dict(ok=True, results=res)
