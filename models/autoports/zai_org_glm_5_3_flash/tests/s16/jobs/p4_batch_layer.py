# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 P4 (throughput) -- BATCHED prefill of one layer: nb requests' fresh 2,048-token
chunks in ONE traced program of nb x 2,048 rows (TPPrefill.layer_batch: mHC / norms / gathers / FFN-MoE / reduce-
scatters over all rows -- the routed experts' weights read once for every request -- attention per segment on its own
latent pages and KDA scratch) vs the single-request layer (TPPrefill.layer on 2,048 rows), both traced.

Every segment carries the Stage 00c golden prompt's layer input (the real 2,048-token prompt), so each segment's output
and KDA state are checked against the CPU fp32 goldens (PCC bar 0.995). Timing: the steady-state replay of each trace
(s16common.replay_stats); serial = nb x single; the row reports batched vs serial and tokens/s.

ARGS: layers ([0, 3, 4]: the backend's layers), check ([4]), n (2048), nb (4), replays_single (2000), replays_batch
(500), the P2/P3 knobs (moe, moe_v2, kda_scan, mhc_post, s16common.set_attn: mla_form, mm_nq, mm_kb, ...), env, tag,
record."""
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
LAYERS = [int(v) for v in args.get("layers", [0, 3, 4])]
CHECK = [int(v) for v in args.get("check", [4])]
N = int(args.get("n", 2048))
NB = int(args.get("nb", 4))
tag = args.get("tag", "p4-batch-layer")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
S.set_attn(args)
ent = S.build(mesh, ctx, LAYERS, b=NB, buckets=(N,))
pf = ent["pf"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", "grouped")
c = pf.bucket_for(N)
R = NB * c
sp1, spb = pf.mode(c) == "sp", pf.mode(R) == "sp"
assert sp1 == spb, (sp1, spb)
cis = pf.seg_inputs(c, NB)
for b, ci_b in enumerate(cis):  # request b in slot b, a fresh chunk of N tokens
    pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=b, chunk_start=0, n_valid=N, ci=ci_b)

results = []
for L in CHECK:
    li = LAYERS.index(L)
    pl = pf.layers[li]
    x_in, x_out = S.golden_layer(L, N)
    x_in = x_in.reshape(N, 4, 4096).float()
    x_out = x_out.reshape(N, 4 * 4096).float()
    one = torch.zeros(c, 4, 4096)
    one[:N] = x_in
    X1 = pf.streams_to_device(one, sp1)
    Y1 = pf.streams_to_device(torch.zeros(c, 4, 4096), sp1)
    XB = pf.streams_to_device(one.repeat(NB, 1, 1), spb)
    YB = pf.streams_to_device(torch.zeros(R, 4, 4096), spb)
    scratches = [pf.seg_scratch(pl, b) for b in range(NB)] if pl.kind != SPARSE_DSA else []

    def zero_state():
        for sc in scratches:
            for t_ in (sc.S, sc.C):
                ttnn.copy(ttnn.zeros_like(t_), t_)

    def fn_single():
        y = pf.layer(li, X1, cis[0], sp1, keep_input=True)
        ttnn.copy(y, Y1)
        ttnn.deallocate(y)

    def fn_batch():
        y = pf.layer_batch(li, XB, cis, spb, keep_input=True)
        ttnn.copy(y, YB)
        ttnn.deallocate(y)

    zero_state()
    S.log(f"L{L}: compile single + batch (nb {NB})")
    t0 = time.time()
    fn_single()
    zero_state()
    fn_batch()  # every program of both traces compiles before the first capture
    ttnn.synchronize_device(mesh)
    t_compile = time.time() - t0
    tids = {}
    if not C.MOCK:
        for name, fn in (("single", fn_single), ("batch", fn_batch)):
            S.log(f"L{L}: capture {name}")
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                fn()
            finally:
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            tids[name] = tid
    zero_state()
    S.log(f"L{L}: first batched replay")
    if "batch" in tids:
        ttnn.execute_trace(mesh, tids["batch"], cq_id=0, blocking=True)
    else:
        fn_batch()
        ttnn.synchronize_device(mesh)
    yb = pf.streams_from_device(YB, spb).reshape(R, -1)
    seg_pcc = [round(C.pcc(yb[b * c : b * c + N], x_out), 6) for b in range(NB)]
    row = dict(test=f"p4 batched prefill layer L{L} {pl.kind}: {NB} requests x {N} tokens in one traced program vs "
                    f"the single-request layer, tp{TPN}", milestone="P4", layer=L, kind=pl.kind, n=N, nb=NB, rows=R,
               mode="sp" if spb else "rep", tp=TPN, seg_pcc=seg_pcc, pcc=min(seg_pcc),
               status="pass" if min(seg_pcc) >= 0.995 else "fail", traced=bool(tids), moe=pf.moe_impl,
               kda_scan=pf.kda_scan, mla_form=pf.mla_form, tag=tag)  # fmt: skip
    if scratches:  # every segment's KDA state after the prompt vs the golden (heads of every chip)
        Sg = S.golden_kda_state(L, N)["recurrent"].float()
        st = []
        for sc in scratches:
            parts = ttnn.get_device_tensors(sc.S) if TPN > 1 else [sc.S]
            Sd = torch.cat([ttnn.to_torch(t_).float() for t_ in parts], dim=1)
            st.append(round(C.pcc(Sd, Sg[:, : Sd.shape[1]]), 6))
        row["seg_state_pcc"] = st
        if min(st) < 0.995:
            row["status"] = "fail"
    if tids:  # the steady-state replay of each trace (timing only: KDA states evolve across replays)
        ts = S.replay_stats(mesh, tids["single"], replays=int(args.get("replays_single", 2000)), label=f"L{L} single")
        tb = S.replay_stats(mesh, tids["batch"], replays=int(args.get("replays_batch", 500)), label=f"L{L} batch")
        for tid in tids.values():
            ttnn.release_trace(mesh, tid)
        serial = NB * ts["median_us"]
        row.update(single_us=ts["median_us"], single_replays=ts["replays"], batch_us=tb["median_us"],
                   batch_replays=tb["replays"], serial_us=round(serial, 1),
                   batch_vs_serial=round(serial / tb["median_us"], 3),
                   single_tok_s=round(N / (ts["median_us"] * 1e-6), 1),
                   batch_tok_s=round(NB * N / (tb["median_us"] * 1e-6), 1))  # fmt: skip
    row.update(compile_s=round(t_compile, 1))
    S.log(row)
    if rec:
        C.record("perf", row)
    results.append(row)
    for t_ in (X1, Y1, XB, YB):
        ttnn.deallocate(t_)
RESULT = results
