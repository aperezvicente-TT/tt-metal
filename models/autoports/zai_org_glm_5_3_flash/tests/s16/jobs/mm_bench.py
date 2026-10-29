# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, any mesh): Stage 16 P3 -- the batched prefill's projection matmuls (pf_mm: 2D multicast program over a
DRAM width-sharded weight) at a given per-chip shape, device vs device: a random weight of shape [k, n] in the role's
DSLinear layout (the decode stack's), an n-row bf16 activation, and every program variant traced (compile launch,
capture of `k_calls` calls, replays); each variant's output vs the first variant's (pcc, max abs) and vs the host fp32
product of the same bf16 / block-float values (pcc).

shapes: [{"name", "k", "n", "wdtype" (bfloat8_b | bfloat4_b | bfloat16), "role", "fid"}] (the per-chip shapes of the
TP to emulate: ds_shapes.py); variants: [{"m_round", "kb", "est_kb", "obt", "nq", "fid"}] (unset keys: pf_mm's defaults).

ARGS: layers ([0]: any built layer, for the decoder config), n (2048), shapes, variants, k_calls (4), reps (8), tag,
record (True)."""
import gc
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import DSLinear, prefill_program  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0])]
N = int(args.get("n", 2048))
SHAPES = list(args["shapes"])
VARIANTS = list(args.get("variants", [{}]))
KC = int(args.get("k_calls", 4))
REPS = int(args.get("reps", 8))
tag = args.get("tag", "p3-mm")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
DRAM = PF.DRAM
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,))
ocfg = next(v for pl in ent["pf"].layers for v in vars(getattr(pl, "kda", None) or pl.mla.attn).values()
            if isinstance(v, DSLinear)).ocfg  # fmt: skip
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None


def chip0(t_):
    return ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()


def ckc_of(fid):
    return ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid),
                                                  math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)  # fmt: skip


def traced(fn):
    t0 = time.time()
    out = fn()
    ttnn.synchronize_device(mesh)
    t_compile = time.time() - t0
    if C.MOCK:
        return out, t_compile, {}
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        for i in range(KC):
            out = fn()
            if i < KC - 1:
                ttnn.deallocate(out)
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    try:  # >= 2,000 steady-state replays (program 03:20Z device duty); per call = per replay / KC
        st = S.replay_stats(mesh, tid, replays=args.get("replays"))
        times = dict(st, median_us=st["median_us"] / KC, min_us=st["min_us"] / KC, p90_us=st["p90_us"] / KC,
                     calls_per_replay=KC)  # fmt: skip
    finally:
        ttnn.release_trace(mesh, tid)
    return out, t_compile, times


torch.manual_seed(0)
results = []
for sh in SHAPES:
    k, n = int(sh["k"]), int(sh["n"])
    wdt = getattr(ttnn, sh.get("wdtype", "bfloat8_b"))
    xh = torch.randn(1, 1, N, k) * 0.5
    wh = torch.randn(k, n) * (1.0 / k**0.5)
    x = ttnn.from_torch(xh, dtype=PF.BF16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=repl)
    w = ttnn.from_torch(wh, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=repl)
    try:  # the role's decode storage-core count is the built TP's; another TP's shape falls back to the default
        ds = DSLinear(mesh, w, role=sh.get("role", "s16_bench"), cfg=ocfg, ckc=ckc_of(sh.get("fid", "HiFi2")),
                      dtype=PF.BF16)  # fmt: skip
    except ValueError:
        ds = DSLinear(mesh, w, role="s16_bench", cfg=ocfg, ckc=ckc_of(sh.get("fid", "HiFi2")), dtype=PF.BF16)
    wq = chip0(ds.weight)[..., :n].reshape(k, n) if not C.MOCK else wh
    ref = (chip0(x).reshape(N, k) @ wq) if not C.MOCK else xh.reshape(N, k) @ wh
    wtb = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}.get(ds.weight.dtype, 2048)
    base = None
    for v in VARIANTS:
        try:
            prog = prefill_program(ds.grid, N // 32, k // 32, ds.n_pad // 32, weight_tile_bytes=wtb,
                                   m_round=int(v.get("m_round", ocfg.prefill_m_block_round)),
                                   max_block_w=int(v.get("kb", ocfg.prefill_k_block)),
                                   l1_budget=int(v.get("est_kb", PF.PF_MM_EST // 1024)) * 1024,
                                   out_block_tiles=int(v.get("obt", 96)),
                                   n_quantum=int(v.get("nq", 4)))  # fmt: skip
            ckc = ckc_of(v.get("fid", sh.get("fid", "HiFi2")))

            def fn():
                return ttnn.linear(x, ds.weight, program_config=prog, memory_config=DRAM, compute_kernel_config=ckc,
                                   dtype=PF.BF16)  # fmt: skip

            out, t_compile, times = traced(fn)
        except Exception as e:  # noqa: BLE001  (host-side program errors: L1 / config limits)
            S.log(f"{sh['name']} {v}: not run: {str(e)[:200]}")
            results.append(dict(shape=sh["name"], variant=v, error=str(e)[:200]))
            continue
        y = chip0(out).reshape(N, -1)[:, :n]
        ttnn.deallocate(out)
        if base is None:
            base = y
        st = dict(times) if times else {}
        row = dict(test=f"p3 prefill matmul {sh['name']} [{N}x{k}] x [{k}x{n}] {sh.get('wdtype', 'bfloat8_b')} "
                        f"variant {v}", milestone="P3", status="info", traced=bool(times), shape=sh["name"], k=k, n=n,
                   tp_emulated=sh.get("tp"), tp=TPN, variant=v,
                   prog=dict(m=prog.per_core_M, n=prog.per_core_N, out_h=prog.out_block_h, out_w=prog.out_block_w,
                             in0_bw=prog.in0_block_w, sub=[prog.out_subblock_h, prog.out_subblock_w]),
                   pcc_ref=round(C.pcc(y, ref), 6), pcc_base=round(C.pcc(y, base), 6),
                   max_abs_base=round(float((y - base).abs().max()), 5), compile_s=round(t_compile, 1), tag=tag,
                   **st)  # fmt: skip
        if times:
            row["tflops"] = round(2 * N * k * n / (st["median_us"] * 1e-6) / 1e12, 1)
        S.log({kk: vv for kk, vv in row.items() if kk != "test"})
        if rec and not C.MOCK:
            C.record("perf", row)
        results.append(row)
    for t_ in (x, w, ds.weight):
        if t_.is_allocated():
            ttnn.deallocate(t_)
    gc.collect()
RESULT = results
