# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 P2 — one prefill LAYER of the TP prefill (tt/prefill_tp.py) on the CPU fp32
prefill goldens (Stage 00c layer_io: the real 2,048-token prompt's streams_in -> streams_out, KDA state after the
prompt), traced: the eager call is the compile, then ONE capture of the layer and a replay whose output is checked
(PCC per layer kind, bar 0.995) and timed (replays, us per layer at C rows).

ARGS: layers (the backend's layer list, default [0, 1, 2, 3, 4]), check (layers to check, default = layers), n (2048),
reps (5), replays / soak_batch (the timing soak), moe, moe_v2, kda_scan, mhc, mhc_post, the P3 knobs
(s16common.set_attn: mla_form, mm_nq, mm_fid, moe_fid ...),
tag, record (True)."""
import gc
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.config import SPARSE_DSA  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
CHECK = [int(v) for v in args.get("check", LAYERS)]
N = int(args.get("n", 2048))
REPS = int(args.get("reps", 5))
tag = args.get("tag", "p2-layer")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())

os.environ["GLM53_PF_MHC"] = args.get("mhc", "comp")
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "ops")
S.set_attn(args)
S.adopt_full(ctx, LAYERS, TPN)  # p2_full's backend in this process, if any (no second full-model build)
ent = S.build(mesh, ctx, LAYERS, b=int(args.get("b", 1)), buckets=(N,), max_seq=int(args.get("max_seq", N + 512)))
pf, kv, backend = ent["pf"], ent["kv"], ent["backend"]
pf.moe_impl = args.get("moe", pf.moe_impl)
S.set_moe_v2(pf, bool(args.get("moe_v2", False)))
pf.kda_scan = args.get("kda_scan", pf.kda_scan)
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
ci = pf.chunk_inputs(c)
results = []
for L in CHECK:
    li = LAYERS.index(L)
    pl = pf.layers[li]
    x_in, x_out = S.golden_layer(L, N)
    x_in = x_in.reshape(N, 4, 4096).float()
    x_out = x_out.reshape(N, 4 * 4096).float()
    pad = torch.zeros(c, 4, 4096)
    pad[:N] = x_in
    X = pf.streams_to_device(pad, sp)
    Y = pf.streams_to_device(torch.zeros(c, 4, 4096), sp)
    pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)

    def zero_state():
        if pl.kind != SPARSE_DSA:
            for t_ in (pl.scratch.S, pl.scratch.C):
                ttnn.copy(ttnn.zeros_like(t_), t_)

    def fn():
        y = pf.layer(li, X, ci, sp, keep_input=True)
        ttnn.copy(y, Y)
        ttnn.deallocate(y)

    zero_state()
    S.log(f"L{L}: compile launch")  # phase markers: a hang is placed by the last marker in the log
    t0 = time.time()
    fn()  # compile launch
    ttnn.synchronize_device(mesh)
    t_compile = time.time() - t0
    tid = None
    if not C.MOCK:
        S.log(f"L{L}: capture")
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            fn()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
    zero_state()
    S.log(f"L{L}: first replay")
    t0 = time.perf_counter()
    if tid is not None:
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    else:
        fn()
        ttnn.synchronize_device(mesh)
    first_us = (time.perf_counter() - t0) * 1e6
    y = pf.streams_from_device(Y, sp).reshape(c, -1)[:N]
    p = C.pcc(y, x_out)
    rel = float((y - x_out).norm() / x_out.norm())
    row = dict(test=f"p2 prefill layer L{L} {pl.kind} vs CPU fp32 golden (n={N})", milestone=S.MILESTONE, layer=L,
               kind=pl.kind, n=N, bucket=c, mode="sp" if sp else "rep", tp=TPN, pcc=round(p, 6), rel_l2=round(rel, 6),
               status="pass" if p >= 0.995 else "fail", traced=tid is not None, mhc=pf.mhc_mode, moe=pf.moe_impl,
               kda_scan=pf.kda_scan, tag=tag)  # fmt: skip
    if pl.kind != SPARSE_DSA:  # the KDA state after the prompt vs the golden (heads of every chip)
        g = S.golden_kda_state(L, N)
        Sg = g["recurrent"].float()  # [1, 64, 128, 128]
        parts = ttnn.get_device_tensors(pl.scratch.S) if TPN > 1 else [pl.scratch.S]
        Sd = torch.cat([ttnn.to_torch(t_).float() for t_ in parts], dim=1)
        hs = Sd.shape[1]
        row["state_pcc"] = round(C.pcc(Sd, Sg[:, :hs]), 6)
        row["state_heads"] = hs
        if row["state_pcc"] < 0.995:
            row["status"] = "fail"
    times = {}
    if tid is not None:  # >= 2,000 steady-state replays (program 03:20Z device duty; a KDA layer's state evolves
        # across them -- timing only, the PCC / state rows above come from the first replay)
        S.log(f"L{L}: soak")
        # soak_batch: replays per synchronize (10 = the device-duty soak; 1 = every replay synchronized, the control for
        # the back-to-back CCL hypothesis of the 03:49Z hang)
        times = S.replay_stats(mesh, tid, replays=args.get("replays"), label=f"L{L}",
                               batch=int(args.get("soak_batch", 10)))  # fmt: skip
        times["soak_batch"] = int(args.get("soak_batch", 10))
        ttnn.release_trace(mesh, tid)
    row.update(compile_s=round(t_compile, 1), first_replay_us=round(first_us, 1), **times)
    S.log(row)
    if rec:
        C.record("pcc", row)
    results.append(row)
    for t_ in (X, Y):
        ttnn.deallocate(t_)
RESULT = results
