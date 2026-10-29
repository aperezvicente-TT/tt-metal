# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 16, the PMK KDA tf32 RNE cost: program 2026-10-25 17:31:59Z): can the PACKER round DEST fp32 to
tf32 for free? The Blackhole packer gasket has a 10-bit mantissa rounding switch (PCK_DEST_RD_CTRL_Round_10b_mant, HAL:
"rounds to 10b mantissa, regardless of fp_pack format"); llk only sets it for fp16 packs. One program per mode (the
compute tt/kernels/pmk/pack_rnd10_probe_c.cpp: CB 0 read exactly through UnpackToDestFp32, packed fp32 to CB 16):
  plain  PROBE_MODE 0: the control (identity expected);
  gasket PROBE_MODE 1: Round_10b_mant on;
  sfpu   PROBE_MODE 2: the SFPU tf32 RNE of GLM53_PMK_KDA_RNE=1 (rne_tf32.hpp) -- the reference rounding.
Inputs: fp32 values with random sign / exponent / kept mantissa and every class of dropped 13 bits (random, exact ties
0x1000 with kept LSB 0 and 1, 0x0FFF, 0x1001, 0x1FFF, 0, and all-ones kept mantissa = the exponent carry). Each output
is compared with the host's identity / truncation / RNE / round-half-away of the SAME fp32 values, per class.
TRACED: one compile launch per program (never read; binaries cannot load during a capture), then the capture into fresh
zero-filled outputs (ended in a finally), one replay. ARGS: tiles (32), seed,
tag, record, evidence."""
import json
import os
import time

import torch

import ttnn

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
NT = int(args.get("tiles", 32))
tag = args.get("tag", "pack-rnd10-probe")
MOCK = bool(os.environ.get("GLM53_MOCK"))
g = torch.Generator().manual_seed(int(args.get("seed", 0)))
TPN = int(mesh.get_num_devices())
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
DRAM = ttnn.DRAM_MEMORY_CONFIG
KSRC = os.path.join(os.environ["TT_METAL_HOME"], "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/pack_rnd10_probe_c.cpp")

# ---- inputs: [NT * 32, 32] fp32; class per element (dropped low 13 bits)
n = NT * 1024
sign = torch.randint(0, 2, (n,), generator=g, dtype=torch.int64) << 31
expo = torch.randint(100, 150, (n,), generator=g, dtype=torch.int64) << 23  # normal range, |x| ~ 2^-27 .. 2^22
kept = torch.randint(0, 1 << 10, (n,), generator=g, dtype=torch.int64) << 13
CLASSES = ["random", "tie_even", "tie_odd", "below_half", "above_half", "max_low", "zero_low", "carry"]
cls = torch.randint(0, len(CLASSES), (n,), generator=g)
low = torch.randint(0, 1 << 13, (n,), generator=g, dtype=torch.int64)
low = torch.where(cls == 1, torch.full_like(low, 0x1000), low)
kept = torch.where(cls == 1, kept & ~(1 << 13), kept)  # kept LSB 0
low = torch.where(cls == 2, torch.full_like(low, 0x1000), low)
kept = torch.where(cls == 2, kept | (1 << 13), kept)  # kept LSB 1
low = torch.where(cls == 3, torch.full_like(low, 0x0FFF), low)
low = torch.where(cls == 4, torch.full_like(low, 0x1001), low)
low = torch.where(cls == 5, torch.full_like(low, 0x1FFF), low)
low = torch.where(cls == 6, torch.zeros_like(low), low)
kept = torch.where(cls == 7, torch.full_like(kept, 0x3FF << 13), kept)  # all-ones kept mantissa
low = torch.where(cls == 7, torch.full_like(low, 0x1800), low)  # above half: rounds up through the exponent
bits = (sign | expo | kept | low).to(torch.int64)
bits = torch.where(bits >= (1 << 31), bits - (1 << 32), bits).to(torch.int32)
x = bits.view(torch.float32).reshape(NT * 32, 32)


def host_round(b: torch.Tensor, mode: str) -> torch.Tensor:
    """b int32 bit patterns -> int32: identity / trunc / rne / half_away (magnitude rounding on the 13 dropped bits)"""
    u = b.to(torch.int64) & 0xFFFFFFFF
    if mode == "identity":
        r = u
    elif mode == "trunc":
        r = u & ~0x1FFF
    elif mode == "rne":
        r = ((u + 0x0FFF + ((u >> 13) & 1)) >> 13) << 13
    else:  # half_away
        r = ((u + 0x1000) >> 13) << 13
    r = r & 0xFFFFFFFF
    return torch.where(r >= (1 << 31), r - (1 << 32), r).to(torch.int32)


def launch(t_in, t_out, mode: int):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rr[0][0] = [t_in.buffer_address(), NT, 0]
    rc[0][0] = [NT]
    rw[0][0] = [t_out.buffer_address(), NT, 0]
    acc = lambda t_: list(ttnn.TensorAccessorArgs(t_).get_compile_time_args())  # noqa: E731
    un = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow"
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
    st = ttnn.KernelDescriptor.SourceType.FILE_PATH
    kernels = [
        ttnn.KernelDescriptor(kernel_source=f"{un}/reader_unary_interleaved_start_id.cpp", source_type=st, core_ranges=crs,
                              compile_time_args=acc(t_in), runtime_args=rr, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=KSRC, source_type=st, core_ranges=crs, compile_time_args=[],
                              defines=[("PROBE_MODE", str(mode))], runtime_args=rc, config=cc),
        ttnn.KernelDescriptor(kernel_source=f"{un}/writer_unary_interleaved_start_id.cpp", source_type=st, core_ranges=crs,
                              compile_time_args=[16] + acc(t_out), runtime_args=rw, config=ttnn.WriterConfigDescriptor()),
    ]  # fmt: skip
    cb = lambda i_: ttnn.CBDescriptor(total_size=2 * 4096, core_ranges=crs, format_descriptors=[  # noqa: E731
        ttnn.CBFormatDescriptor(buffer_index=i_, data_format=ttnn.float32, page_size=4096)])  # fmt: skip
    ttnn.generic_op([t_in, t_out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb(0), cb(16)]))


put = lambda t: ttnn.from_torch(t, ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,  # noqa: E731
                                mesh_mapper=repl)  # fmt: skip
xt = put(x)
MODES = {"plain": 0, "gasket": 1, "sfpu": 2}
outs = {k: put(torch.zeros_like(x)) for k in MODES}
res = {}
# the compile launch (the one eager launch per program the harness rule allows: binaries cannot load during a capture;
# 2026-10-25 18:03Z a capture without it threw, left the capture open and wedged the server's close) -- never read
for k, m in MODES.items():
    launch(xt, outs[k], m)
ttnn.synchronize_device(mesh)
if MOCK:  # a mock cannot capture a trace
    res = {k: dict(mock=True) for k in MODES}
else:
    # fresh zero-filled outputs (host writes) for the capture: the replay alone writes what is read back (same
    # runtime-arg count as the compile launch: the generic_op cache hit updates the addresses)
    outs = {k: put(torch.zeros_like(x)) for k in MODES}
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        for k, m in MODES.items():
            launch(xt, outs[k], m)
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    xb = x.reshape(-1).contiguous().view(torch.int32)
    refs = {m: host_round(xb, m) for m in ("identity", "trunc", "rne", "half_away")}
    for k in MODES:
        dev = ttnn.get_device_tensors(outs[k])[0] if TPN > 1 else outs[k]
        yb = ttnn.to_torch(dev).float().reshape(-1).contiguous().view(torch.int32)
        row = {}
        for rm, rb in refs.items():
            eq = yb == rb
            row[rm] = dict(all=round(float(eq.float().mean()), 6),
                           **{c: round(float(eq[cls == i].float().mean()), 6) for i, c in enumerate(CLASSES)})  # fmt: skip
        xd = x.reshape(-1).double()
        row["scale_vs_x"] = float((yb.view(torch.float32).double() * xd).sum() / (xd * xd).sum())
        res[k] = row
    ttnn.release_trace(mesh, tid)
row = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test="pack_rnd10_probe",
           status="mock" if MOCK else "record", tiles=NT, traced=True, results=res, tag=tag)  # fmt: skip
print(f"PACK-RND10 {json.dumps(row)}", flush=True)
ev = args.get("evidence")
if args.get("record") and ev:
    with open(os.path.join(ev, "pcc_pack_rnd10.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
RESULT = dict(ok=True, status=row["status"], results=res)
