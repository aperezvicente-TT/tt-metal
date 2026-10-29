# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh; pair or 2x4): Stage 16 PMK -- ONE persistent prefill LAYER program vs the op-graph prefill
path (tt/prefill_tp.py) on the SAME input in the SAME process, both traced, ONE replay compared + ONE 200-replay
timing. Nothing else (prefill_megakernel.md §7: one tag, one criterion, <= 10 min, never repeated once passed).

  input    goldens/layer_io/L<L>/2048/streams_in.pt rows [:n], n = 2,048 in BOTH geometries (binding 09:4xZ):
           geometry "pair" = TP2, 1,024 SP rows per chip; "tp8" = 2x4, 256 per chip; bucket c = n, sequence-parallel
  ref      pmk_harness.ref_layer: TPPrefill's fused-mHC layer with the §4b hand-off taps (H1.xhat, H6.y, x1, H10.y_f)
           copied in-program -> compile launch (the one eager launch), capture, ONE replay from zeroed state
  pmk      the builder of the layer's kind (tt/pmk_layer.py; STUB = the op path behind the contract when absent):
           compile launch, capture, EagerGuard snapshot, ONE replay from zeroed state -> output rows, the side effects
           (KDA S / C scratch or the latent pages), the builder's taps and PROBE CB 31 rows (prog.probe_read())
  rows     ONE pcc row: bitwise flag, max |diff|, PCC / rel L2 / norm ratio of the output vs ref, the same per side
           effect and per tap, PCC vs the CPU fp32 golden (info), traced (no eager fallback: the guard RAISES
           otherwise), us per replay (median of `replays` = 200 traced replays, batches of 10 + a synchronize,
           TT_METAL_TRACE_ALLOC_TRACKING from the env) and the op path's us on the same input (the ratio);
           ONE perf row: the same us + the probe-derived MATH utilization (or the arithmetic bound).
  criterion status = pass iff traced AND pcc(out) >= pcc_bar AND every side effect's pcc >= pcc_bar (bitwise is the
           flag the integrator wants; a non-bitwise pass must be PCC-explained in the piece README). impl "stub" ->
           status "info" (the harness proved, the PMK not). Mock -> status "mock" (readback is garbage).

ARGS: layer (4), geometry ("pair" | "tp8", cross-checked against the mesh), kind (optional cross-check of the layer's
kind), rows (2048), replays (200), pcc_bar (0.9999),
probe (True), taps (True), mhc_post ("fused"), moe ("ep"), moe_v2 (True), kda_scan ("op"), dsa (False), the P3 knobs
(s16common.set_attn: mla_form, mm_nq, mm_kb, mm_fid, moe_fid, env ...), max_seq (c + 512), tag, record (True)."""
import gc
import os
import sys
import time

import torch

import ttnn

gc.collect()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pmk_harness as H  # noqa: E402
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import layer_kind  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
TPN = int(mesh.get_num_devices())
L = int(args.get("layer", 4))
# Stage 16 geometry (binding, 09:4xZ): the 2,048-row golden prompt in BOTH geometries -- "pair" = TP2 (1,024 SP rows per
# chip, 32 heads per chip, XSL 1,024 wide), "tp8" = 2x4 (256 per chip); the op-graph reference in the same process
# runs the same bucket / mode, so both paths see the same geometry. The argument is a cross-check against the mesh.
GEOM = args.get("geometry", {2: "pair", 8: "tp8"}.get(TPN, f"tp{TPN}"))
assert {"pair": 2, "tp8": 8}.get(GEOM, TPN) == TPN, f"geometry {GEOM} on a {TPN}-chip mesh"
N = int(args.get("rows", 2048))
REPLAYS = int(args.get("replays", 200))
BAR = float(args.get("pcc_bar", 0.9999))
tag = args.get("tag", f"pmk-unit-L{L}-{GEOM}")
rec = bool(args.get("record", True))
assert N % (32 * TPN) == 0 and N <= 2048, f"rows {N}: a multiple of 32 x {TPN} chips, <= the 2,048-row golden"

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_DSA"] = "1" if args.get("dsa", False) else "0"
os.environ["GLM53_PF_BATCH"] = "0"
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ""
S.set_attn(args)
kind = layer_kind(hf_config(), L)
if args.get("kind") and args["kind"] != kind:
    raise ValueError(f"layer {L} is {kind}, not {args['kind']}")
t_build = time.time()
ent = S.build(mesh, ctx, [L], b=1, buckets=(N,), max_seq=int(args.get("max_seq", N + 512)))
pf, kv, backend = ent["pf"], ent["kv"], ent["backend"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", pf.kda_scan)
li = 0
pl = pf.layers[li]
c = pf.bucket_for(N)
assert c == N, (c, N)
sp = pf.mode(c) == "sp"
ci = pf.chunk_inputs(c)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
H.log(f"L{L} {kind}: backend + TPPrefill ready in {time.time() - t_build:.1f} s; c={c} sp={sp} tp={TPN}")

# ------------------------------------------------------------------------------------------------ inputs / outputs
x_in, x_gold = S.golden_layer(L, 2048)
x_in = x_in.reshape(2048, 4, 4096).float()[:N]
x_gold = x_gold.reshape(2048, 4 * 4096).float()[:N]
X = pf.streams_to_device(x_in, sp)
Y_ref = pf.streams_to_device(torch.zeros(c, 4, 4096), sp)
Y_pmk = pf.streams_to_device(torch.zeros(c, 4, 4096), sp)
taps_ref: dict = {} if args.get("taps", True) else None
factory, impl = H.load_builder(kind)
prog = factory(pf, li, c, sp, probe=bool(args.get("probe", True)))
H.log(f"builder: {impl}")


def fn_ref():
    y = H.ref_layer(pf, li, X, ci, sp, taps_ref)
    ttnn.copy(y, Y_ref)
    ttnn.deallocate(y)


def fn_pmk():
    y = H.run_pmk(pf, li, X, ci, sp, prog)  # through TPPrefill.pmk_layers (what the bucket trace runs)
    ttnn.copy(y, Y_pmk)
    if y is not Y_pmk and y is not X:  # an in-place program returns its input X: never free the harness's input
        ttnn.deallocate(y)


def traced(fn, label: str):
    """compile launch (the one eager launch; never timed) -> capture -> tid (None on the mock)"""
    H.zero_state(pf, li, kv)
    H.log(f"{label}: compile launch")
    t0 = time.time()
    fn()
    ttnn.synchronize_device(mesh)
    t_c = time.time() - t0
    if C.MOCK:
        return None, t_c
    H.log(f"{label}: capture")
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    gc.collect()
    return tid, t_c


def replay_once(tid, fn):
    H.zero_state(pf, li, kv)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    if tid is None:  # mock only
        fn()
    else:
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) * 1e6


# ------------------------------------------------------------------------------------------------ the op-graph reference
# the PMK program's compile launch BEFORE the first capture (a program first compiled after a capture keeps a live
# program-cache buffer: the 10:20Z d0 unit's ref timing replays failed 'buffer alive before trace replay'); the ref's
# compile launch runs inside traced() right before its capture, still before any capture
H.zero_state(pf, li, kv)
fn_pmk()
ttnn.synchronize_device(mesh)
tid_ref, compile_ref = traced(fn_ref, "ref")
first_ref = replay_once(tid_ref, fn_ref)
y_ref = H.rows_host(Y_ref, sp)[:N]
se_ref = H.side_effects(pf, li, kv)
tp_ref = {k: H.rows_host(t_, sp) for k, t_ in (taps_ref or {}).items()}

# ------------------------------------------------------------------------------------------------ the PMK program
tid_pmk, compile_pmk = traced(fn_pmk, "pmk")
guard = H.EagerGuard(mesh, pf, backend, prog)
guard.snapshot()
first_pmk = replay_once(tid_pmk, fn_pmk)
guard.check("the PMK replay")  # raises: an eager fallback is a loud failure
y_pmk = H.rows_host(Y_pmk, sp)[:N]
se_pmk = H.side_effects(pf, li, kv)
tp_pmk = prog.taps() if hasattr(prog, "taps") else {}
probe = H.decode_probe(prog.probe_read()) if hasattr(prog, "probe_read") else None

# ------------------------------------------------------------------------------------------------ compare
out = H.cmp(y_pmk, y_ref)
se = {k: H.cmp(se_pmk[k], se_ref[k]) for k in se_ref}
taps = {k: H.cmp(torch.as_tensor(tp_pmk[k]).float().reshape(-1), tp_ref[k].reshape(-1)) for k in tp_ref if k in tp_pmk}
gold = dict(pmk=round(C.pcc(y_pmk, x_gold), 6), ref=round(C.pcc(y_ref, x_gold), 6))

# ------------------------------------------------------------------------------------------------ timing (traced)
times, times_ref = {}, {}
if tid_pmk is not None:
    H.log(f"pmk: {REPLAYS} replays")
    times = S.replay_stats(mesh, tid_pmk, replays=REPLAYS, min_s=0.0, batch=10, label="pmk", every=100)
    guard.check("the PMK timing replays")
    H.log(f"ref: {REPLAYS} replays")
    times_ref = S.replay_stats(mesh, tid_ref, replays=REPLAYS, min_s=0.0, batch=10, label="ref", every=100)
    for t_ in (tid_pmk, tid_ref):
        ttnn.release_trace(mesh, t_)
us = times.get("median_us")
us_ref = times_ref.get("median_us")
ok = (not C.MOCK) and out["pcc"] >= BAR and all(v["pcc"] >= BAR for v in se.values())
status = "mock" if C.MOCK else ("info" if impl == "stub" else ("pass" if ok else "fail"))
row = dict(test=f"pmk layer unit L{L} {kind}: ONE traced replay of the persistent layer program vs the op-graph "
                f"prefill path (same input = golden streams_in[:{N}], same process, both traced) + {REPLAYS}-replay "
                f"timing, geometry {GEOM} (tp{TPN}, {c // TPN} rows per chip) {'sp' if sp else 'rep'} c={c}", milestone="P4",
           status=status, layer=L, kind=kind, geometry=GEOM, rows_per_chip=c // TPN, impl=impl, n=N, bucket=c, tp=TPN, mode="sp" if sp else "rep", bitwise=out["bitwise"], max_abs=out["max_abs"],
           pcc=out["pcc"], rel_l2=out["rel_l2"], norm_ratio=out["norm_ratio"], side_effects=se, taps=taps,
           pcc_golden=gold, pcc_bar=BAR, traced=tid_pmk is not None, eager_guard=guard.before, us=us, us_ref=us_ref,
           speedup=round(us_ref / us, 3) if us and us_ref else None, replays=times.get("replays", 0),
           first_replay_us=round(first_pmk, 1), compile_s=round(compile_pmk, 1), compile_ref_s=round(compile_ref, 1),
           math=probe, math_bound=H.flop_bound(kind, N, TPN, us), mhc_post=pf.mhc_post, moe=pf.moe_impl,
           kda_scan=pf.kda_scan, tag=tag)  # fmt: skip
H.log({k: v for k, v in row.items() if k not in ("test", "side_effects", "taps", "math")})
H.log(f"  side_effects {se}")
H.log(f"  taps {taps}")
H.log(f"  math {probe}")
if rec:
    C.record("pcc", row)
    C.record("perf", dict(test=f"pmk layer L{L} {kind} replay (ONE execute_trace of the layer program, tp{TPN}, "
                               f"c={c}, {REPLAYS} replays, batches of 10 + synchronize)", milestone="P4",
                          status="info" if status in ("pass", "info") else status, layer=L, kind=kind, geometry=GEOM, impl=impl,
                          n=N, bucket=c, tp=TPN, traced=tid_pmk is not None, us=us, us_ref=us_ref,
                          speedup=row["speedup"], replays=times.get("replays", 0), min_us=times.get("min_us"),
                          p90_us=times.get("p90_us"), math_util=(probe or {}).get("math_util"),
                          math_util_bound=row["math_bound"].get("math_util_bound"), tag=tag))  # fmt: skip
for t_ in (X, Y_ref, Y_pmk, *list((taps_ref or {}).values())):
    ttnn.deallocate(t_)
RESULT = row
