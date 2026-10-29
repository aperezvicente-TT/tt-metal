# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, any mesh): micro-bisect of the 'device buffer still alive before trace replay' leak traced to
MlaPrefill._mha (01:59Z, allocation traceback: the K / V projection's matmul output survives ttnn.deallocate). A trivial
trace is captured first (the reference trace), then each variant of the _mha op sequence runs EAGERLY and the buffers
allocated since the capture and still alive (get_unsafe_tracked_ids) are listed with their allocation context.

Variants: mm (matmul alone, deallocate), mm_heads (matmul -> nlp_create_qkv_heads -> deallocate both), linear_heads
(ttnn.linear instead of matmul), mm_heads_keep (keep nlp_create_qkv_heads' k / v outputs alive until the end), mha (the
real MlaPrefill._mha incl. SDPA + o_proj).

ARGS: layers ([2, 3]), layer (3), n (2048)."""
import gc
import os
import sys

import torch

import ttnn
from ttnn._ttnn.operations.trace import get_unsafe_tracked_ids

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [2, 3])]
L = int(args.get("layer", 3))
N = int(args.get("n", 2048))
os.environ["GLM53_PF_DSA"] = "0"
S.set_attn(dict(args, mla_form="mha", mha_sdpa=args.get("mha_sdpa", "256,128,HiFi2")))
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,))
pf = ent["pf"]
mla = pf.layers[LAYERS.index(L)].mla
if mla.w_k is None:
    mla.enable_mha()
repl = pf.repl
mk = lambda shape: ttnn.from_torch(torch.randn(*shape) * 0.1, dtype=PF.BF16, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                   memory_config=PF.DRAM, mesh_mapper=repl)  # fmt: skip
src0, q0 = mk((1, 1, N, 512)), mk((1, mla.h, N, 256))
ttnn.synchronize_device(mesh)

# the reference trace: one tiny op, captured (every later allocation is 'after the capture')
tmp = mk((1, 1, 32, 32))
tid = None
if not C.MOCK:
    ttnn.deallocate(ttnn.add(tmp, tmp))  # the op's program into the cache before the capture
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        y = ttnn.add(tmp, tmp)
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)


def alive():
    if C.MOCK:
        return {}
    ttnn.synchronize_device(mesh)
    gc.collect()
    return dict(get_unsafe_tracked_ids(mesh, tid))


def v_mm():
    f = ttnn.matmul(src0, mla.w_k, compute_kernel_config=mla.ckc_head, dtype=PF.BF16, memory_config=PF.DRAM)
    ttnn.deallocate(f)


def v_mm_heads():
    f = ttnn.matmul(src0, mla.w_k, compute_kernel_config=mla.ckc_head, dtype=PF.BF16, memory_config=PF.DRAM)
    t_, k_, v_ = ttnn.experimental.nlp_create_qkv_heads(f, num_heads=mla.h, num_kv_heads=0, transpose_k_heads=False,
                                                        memory_config=PF.DRAM)  # fmt: skip
    ttnn.deallocate(f)
    for x in (t_, k_, v_):
        ttnn.deallocate(x)


def v_linear_heads():
    f = ttnn.linear(src0, mla.w_k, compute_kernel_config=mla.ckc_head, dtype=PF.BF16, memory_config=PF.DRAM)
    t_, _, _ = ttnn.experimental.nlp_create_qkv_heads(f, num_heads=mla.h, num_kv_heads=0, transpose_k_heads=False,
                                                      memory_config=PF.DRAM)  # fmt: skip
    ttnn.deallocate(f)
    ttnn.deallocate(t_)


def v_mha():
    src = ttnn.clone(src0)
    q = ttnn.clone(q0)
    out = mla._mha(q, src)
    ttnn.deallocate(out)


results = {}
for name, fn in (("mm", v_mm), ("mm_heads", v_mm_heads), ("linear_heads", v_linear_heads), ("mha", v_mha)):
    b = alive()
    try:
        fn()
        a = alive()
        new = {k: v for k, v in a.items() if k not in b}
        results[name] = dict(new_alive=len(new), contexts=sorted(set(v[:90] for v in new.values())))
    except Exception as e:  # noqa: BLE001
        results[name] = dict(error=str(e)[:200])
    S.log(f"{name}: {results[name]}")
if tid is not None:
    ttnn.release_trace(mesh, tid)
for t_ in (src0, q0, tmp):
    ttnn.deallocate(t_)
RESULT = results
