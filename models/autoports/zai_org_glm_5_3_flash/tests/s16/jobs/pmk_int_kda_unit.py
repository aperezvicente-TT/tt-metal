# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, d0 TP2 pair; or the pair mock): the PMK INTEGRATION unit of the KDA piece -- the family's sparse_kda
layer program (tt/pmk_layer.py) with the kda piece's REAL bodies registered (tt/pmk_sparse_kda.kda_bodies: P4 = the item
kernels on the 64 members = P4 conv / gates + P5 prepare; P6 = the scan kernels on the 32 MM-only cores = P6 scan +
gated norm + P9 state), every other phase a stub; CB indices = the engine's free-index map (PK_ITEM_MAP / PK_SCAN_MAP),
H3 PRE-RAISED (P3 is a stub: the reference stages the H3 store, as the standalone unit pmk_kda_unit.py does).
Same inputs and reference as pmk_kda_unit.py (layer L's golden rows, the op path's KDA up to the gated norm), compared
on `on` rows [:valid], the final S and the conv carry C: PCC >= 0.9999 (the standalone phase: on 0.99998, S 0.99997, C
bitwise) -- this unit's job is the COMPOSITION (the bodies inside one program with the family's CB map, semaphores,
argument bases and the runtime-arg audit). One live trace at a time (op path first: it fills the H3 store); every
program compiled before the first capture. ARGS: layer (4), layers ([L]: [0..4] adopts the moe bench's L0-4 build in
the same d0 process), rows (2048), valid (1024), golden_n (2048), replays (200), terms ("fp32"), tag."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_kda as P  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMK  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_sparse_kda as INT  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.kda import CHUNK  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16, DRAM, FP32, KDA_NO_SP, pf_mm  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
L = int(args.get("layer", 4))
LAYERS = [int(v) for v in args.get("layers", [L])]
BUCKET = int(args.get("rows", 2048))
N = int(args.get("valid", 1024))
GN = int(args.get("golden_n", 2048))
REPLAYS = int(args.get("replays", 200))
TERMS = str(args.get("terms", "fp32"))
tag = args.get("tag", f"pmk-int-kda-L{L}-{TERMS}")
TPN = int(mesh.get_num_devices())
PCC_BAR = 0.9999
assert BUCKET % CHUNK == 0 and 0 < N <= BUCKET and N <= GN, (BUCKET, N, GN)

ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(BUCKET,), max_seq=BUCKET + 512)
pf = ent["pf"]
LI = LAYERS.index(L)
pl = pf.layers[LI]
kda = pl.kda
h_, d_, w_ = kda.h, kda.d, kda.width
c = BUCKET
sp = pf.mode(c) == "sp"
ci = pf.chunk_inputs(c)
prog = PMK.PmkSparseKdaLayer(pf, LI, c, sp, probe=False)
geo = prog.engine.kda_geo
assert geo.terms_dtype == TERMS and geo.heads == h_ and geo.rows == c, (geo, TERMS, h_, c)
S.log(f"{tag}: L{L} {pl.kind} heads/chip {h_} bucket {c} valid {N} mode {'sp' if sp else 'rep'} terms {TERMS}; "
      f"plan item {P.plan(geo).item_l1} B scan {P.plan(geo).scan_l1} B spread {geo.spread} cph {geo.cph}")  # fmt: skip

# ---- the layer's attention input on the golden rows (the op graph's own glue: mhc_pre -> RMSNorm -> gather), once
x_in, _ = S.golden_layer(L, GN)
x_in = x_in.reshape(GN, 4, 4096).float()
pad = torch.zeros(c, 4, 4096)
pad[:N] = x_in[:N]
X = pf.streams_to_device(pad, sp)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)  # ci.valid_end = N
h0, _post, _comb = pf._mhc_pre(pl, "attn", X)
hn = pl.rms(h0, "attn", BF16)
xh = pf.gather(hn, sp)  # [1, 1, c, 4096] bf16 on every chip
ttnn.synchronize_device(mesh)

# ---- persistent tensors (addresses are baked into both traces)
phase = INT.kda_phase(prog, kda, nprod=1, h3_preraised=True)  # the family's members / MM-only, its CB maps
scr = phase.scratch()
ref_state, phase_state = kda.allocate_state(1), kda.allocate_state(1)
# FusedKDA.write_slot blends the carry rows into C through state.slot_mask (allocate_state leaves it ZERO: the rows
# would never land -- r2..r5's "constant C" was the reference side); TPPrefill.kda_scratch sets the mask to ones
for st_ in (ref_state, phase_state):
    ttnn.copy(ttnn.ones(tuple(st_.slot_mask.shape), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh), st_.slot_mask)
o_ref = ttnn.empty((1, 1, c, w_), BF16, ttnn.TILE_LAYOUT, mesh, DRAM)
rows = ttnn.slice(pf.row_ids(c), (0, 0, 0, 0), (1, 1, c, 1))
n_chunks = c // CHUNK
# the op path's chunk terms (prepare_chunk_recurrence outputs + the fp32 chunk decay), kept for the term-by-term
# comparison with the phase's TERMS pages (which term goes wrong first, per head / chunk)
TERM_SHAPES = dict(v_beta=(h_, n_chunks, CHUNK, d_), kd=(h_, n_chunks, CHUNK, d_), q_decay=(h_, n_chunks, CHUNK, d_),
                   intra=(h_, n_chunks, CHUNK, CHUNK), k_dec_t=(h_, n_chunks, d_, CHUNK), t_inv=(h_, n_chunks, CHUNK, CHUNK),
                   chunk_decay=(h_, n_chunks, d_, 1))  # fmt: skip
ref_terms = {k: ttnn.empty(v, FP32, ttnn.TILE_LAYOUT, mesh, DRAM) for k, v in TERM_SHAPES.items()}


_zeros = {}


def zero_state(st):
    """fresh request: S and C zeroed from persistent zero tensors (zeros_like on a row-major mesh tensor round-trips
    the host and refuses a distributed tensor); FusedKDAState: S [1, H, K, V], C [1, 3, 1, 3 W] (TILE)"""
    for t_ in (st.S, st.C):
        key = (tuple(t_.shape), t_.dtype, t_.layout)
        if key not in _zeros:
            _zeros[key] = ttnn.zeros(tuple(t_.shape), dtype=t_.dtype, layout=t_.layout, device=mesh, memory_config=DRAM)
        ttnn.copy(_zeros[key], t_)


def ref_fn():
    """kda_prefill_tp (scan="op") up to the gated norm; the H3 store and the reference outputs copied into the
    persistent tensors; the final state / carry via write_slot into ref_state"""
    qkv = pf_mm(kda.ds_qkv, xh)
    lr = pf_mm(kda.ds_lowrank, xh)
    f_a = ttnn.slice(lr, (0, 0, 0, 0), (1, 1, c, d_))
    g_a = ttnn.slice(lr, (0, 0, 0, d_), (1, 1, c, 2 * d_))
    b = ttnn.slice(lr, (0, 0, 0, 2 * d_), (1, 1, c, 2 * d_ + h_))
    gate_out = kda._pf("ds_gb", g_a, kda.w_gb, kda.ckc_fp32, BF16)
    f_raw = ttnn.linear(f_a, kda.w_fb, compute_kernel_config=kda.ckc_fp32, memory_config=DRAM, dtype=FP32)
    g32 = kda._decay(f_a)
    g = ttnn.typecast(g32, BF16)
    beta = ttnn.sigmoid(ttnn.typecast(b, FP32))
    carry, s0 = kda.read_slot(ref_state, 0)
    selections = ChronologicalSelections(
        ttnn.experimental.kda.chronological_selections(kda.zero_u32, KDA_NO_SP, c, h_, d_, d_, actual_end=ci.valid_end)
    )
    qkv_rm = ttnn.to_layout(ttnn.reshape(qkv, (1, c, 3 * w_)), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
    new_carry = selections.select_local_final_history(qkv_rm, carry)
    q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        qkv_rm, carry, *kda.taps_prefill, w_, w_, w_, program_config=kda.conv_program, actual_start=kda.zero_u32,
        sequence_parallel_axis=KDA_NO_SP, predecessor_carry=carry, memory_config=DRAM)  # fmt: skip
    beta_by_chunk = ttnn.reshape(ttnn.permute(ttnn.reshape(beta, (1, c, h_)), (0, 2, 1)), (h_, n_chunks, CHUNK, 1))
    v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv = ttnn.experimental.kda.prepare_chunk_recurrence(
        q, k, v, ttnn.reshape(g, (1, c, w_)), beta_by_chunk, h_, memory_config=DRAM, compute_kernel_config=kda.ckc_kda,
        output_bf16_mask=0, actual_start=kda.zero_u32, actual_end=ci.valid_end, sequence_parallel_axis=KDA_NO_SP)  # fmt: skip
    g_valid = ttnn.multiply(g32, ttnn.lt(rows, ci.valid_end_f))
    g_sum = ttnn.sum(ttnn.reshape(g_valid, (n_chunks, CHUNK, h_ * d_)), dim=1, keepdim=True)
    chunk_decay = ttnn.exp(ttnn.reshape(ttnn.permute(ttnn.reshape(g_sum, (n_chunks, h_, d_)), (1, 0, 2)),
                                        (h_, n_chunks, d_, 1)))  # fmt: skip
    state0 = ttnn.reshape(s0, (h_, d_, d_))
    y, s_fin = ttnn.experimental.kda.recurrent_chunk_scan(
        v_beta, kd, q_decay, intra, k_dec_t, chunk_decay, t_inv, state0, actual_start=kda.zero_u32,
        tail_entry_states=state0, groups_per_head=1, memory_config=DRAM, compute_kernel_config=kda.ckc_fp32,
        sequence_parallel_axis=KDA_NO_SP, actual_end=ci.valid_end)  # fmt: skip
    o = ttnn.experimental.kda.sigmoid_gated_rms_norm(
        ttnn.reshape(y, (h_, c, d_)), ttnn.reshape(gate_out, (1, c, w_)), kda.o_norm, h_, epsilon=kda.eps,
        memory_config=DRAM, compute_kernel_config=kda.ckc_kda, output_dtype=BF16)  # fmt: skip
    for name, t_ in (
        ("v_beta", v_beta),
        ("kd", kd),
        ("q_decay", q_decay),
        ("intra", intra),
        ("k_dec_t", k_dec_t),
        ("t_inv", t_inv),
        ("chunk_decay", chunk_decay),
    ):
        ttnn.copy(t_, ref_terms[name])
    # the H3 store for the phase (what the P3 / P4 stream phases would write) and the reference outputs
    ttnn.copy(qkv, scr["u"])
    ttnn.copy(lr, scr["lr"])
    ttnn.copy(f_raw, scr["f"])
    ttnn.copy(ttnn.typecast(gate_out, scr["gate_out"].dtype), scr["gate_out"])
    ttnn.copy(ttnn.reshape(o, (1, 1, c, w_)), o_ref)
    kda.write_slot(ref_state, 0, new_carry, ttnn.reshape(s_fin, (1, h_, d_, d_)))


INT.register_kda(prog, INT.kda_bodies(phase, scr, phase_state.S, phase_state.C, ci.valid_end))
S.log(f"{tag}: kda P4 (items on {len(prog.engine.place.members)} members) / P6 (scan on {len(prog.engine.place.mm_only)} "
      f"MM-only) registered; CB maps {prog.engine.kda_map}")


def phase_fn():
    prog(X, ci)  # the ONE persistent layer program (stubs elsewhere): its io carries the kda tensors


def capture(fn, st, name, compiled=False):
    """compile launch (the one eager launch; skipped when it already ran), then the trace; -> tid (None on the mock)"""
    zero_state(st)
    if not compiled:
        S.log(f"{tag}: {name} compile launch")
        t0 = time.time()
        fn()
        ttnn.synchronize_device(mesh)
        S.log(f"{tag}: {name} compiled in {time.time() - t0:.1f} s")
    if C.MOCK:
        return None
    S.log(f"{tag}: {name} capture")
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    return tid


def replay(tid, st, name):
    zero_state(st)
    S.log(f"{tag}: {name} replay")
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    return (time.perf_counter() - t0) * 1e6


# the phase's compile launch BEFORE the first capture (a program first compiled after a capture keeps a live
# program-cache buffer -> 'buffers alive before trace replay' at the op path's replays: d0 10:20Z family unit)
zero_state(phase_state)
S.log(f"{tag}: phase compile launch")
phase_fn()
ttnn.synchronize_device(mesh)
launches_before = prog.counters["calls"]
S.log(f"{tag}: runtime-arg audit per rank {prog.engine.arg_audits}; launch census {prog.launch_census}")
result = dict(tag=tag, layer=L, kind=pl.kind, heads=h_, bucket=c, valid=N, tp=TPN, terms=TERMS, mock=C.MOCK,
              spread=geo.spread, cph=geo.cph, arg_audit=prog.engine.arg_audits, census=prog.launch_census,
              kda_map=prog.engine.kda_map)  # fmt: skip
if C.MOCK:
    S.log(f"{tag}: mock -- both paths built and launched once (no data, no traces)")
    result["status"] = "mock"
else:
    # ONE live trace at a time: the op path (fills the H3 store + the reference outputs), timed, released; then the
    # PMK program on that store, timed, released (the compared tensors are persistent: they outlive the traces)
    ref_tid = capture(ref_fn, ref_state, "op path")
    first_ref_us = replay(ref_tid, ref_state, "op path")
    perf_op = S.replay_stats(mesh, ref_tid, replays=REPLAYS, label=f"{tag} op path")
    # the timing replays chain the KDA state (each starts from the previous one's final S / C): ONE more replay from a
    # zeroed state restores the reference outputs to a fresh-request prefill before the trace is released (13:3xZ:
    # without it the reference S was the 21-prefill chain -> S PCC 0.955 for ANY program)
    replay(ref_tid, ref_state, "op path (reference restore)")
    ttnn.release_trace(mesh, ref_tid)
    phase_tid = capture(phase_fn, phase_state, "pmk", compiled=True)
    assert prog.counters["calls"] == launches_before + 1, ("eager guard: the capture is the only launch", prog.counters)
    first_phase_us = replay(phase_tid, phase_state, "pmk")
    ve_host = [
        int(ttnn.to_torch(t_).reshape(-1)[0])
        for t_ in (ttnn.get_device_tensors(ci.valid_end) if TPN > 1 else [ci.valid_end])
    ]
    probes = phase.probe_rows(scr)
    S.log(f"{tag}: host valid_end per chip {ve_host}; probes {probes}")
    result.update(valid_end_host=ve_host, probes=probes)

    # ---- compare on every chip (each chip: its own head shard over the same gathered rows)
    def host_all(t):
        return [ttnn.to_torch(d_t) for d_t in (ttnn.get_device_tensors(t) if TPN > 1 else [t])]

    cmp = {}
    pairs = dict(on=(scr["on"], o_ref), S=(phase_state.S, ref_state.S), C=(phase_state.C, ref_state.C))
    for name, (a_t, b_t) in pairs.items():
        pccs, maxd, bit, stds = [], 0.0, True, []
        for a, b in zip(host_all(a_t), host_all(b_t)):
            if name == "on":
                a, b = a.reshape(c, w_)[:N], b.reshape(c, w_)[:N]
            a, b = a.float(), b.float()
            pccs.append(C.pcc(a, b))
            stds.append((float(a.std()), float(b.std())))
            maxd = max(maxd, float((a - b).abs().max()))
            bit = bit and bool(torch.equal(a, b))
        cmp[name] = dict(
            pcc=round(min(pccs), 7),
            maxabs=maxd,
            bitwise=bit,
            std_phase=round(min(sd[0] for sd in stds), 6),
            std_ref=round(min(sd[1] for sd in stds), 6),
        )  # a NaN PCC: the side with std 0 is the constant one
    ok = all(v["pcc"] >= PCC_BAR for v in cmp.values())
    # term by term (chip 0): phase TERMS item (h, c) = rows [32 m, 32 m + 32) of the [H NCH 32, 22 x 32] scratch, m = h NCH + c;
    # columns: v_beta 0..127 | q_decay 128..255 | kd 256..383 | intra 384..415 | t_inv 416..447 | fd 448..575 (4 column
    # tiles, column 0 of each = final decay rows k 32 .. k 32 + 31) | kdt 576..703 (4 tiles = rows of k_dec_t)
    NP_ = P.out_pages(d_ // 32, d_ // 32)
    terms_host = host_all(scr["terms"])[0].float().reshape(h_ * n_chunks * CHUNK, NP_ * CHUNK)
    ref_host = {k: host_all(t_)[0].float() for k, t_ in ref_terms.items()}
    term_cmp = {}
    for hh, cc in ((0, 0), (0, 1), (1, 0), (0, n_chunks - 1)):
        if cc >= (N + CHUNK - 1) // CHUNK:
            continue
        m = hh * n_chunks + cc
        ph = terms_host[32 * m : 32 * m + 32]
        got = dict(v_beta=ph[:, 0:128], q_decay=ph[:, 128:256], kd=ph[:, 256:384], intra=ph[:, 384:416], t_inv=ph[:, 416:448],
                   chunk_decay=torch.cat([ph[:, 448 + 32 * k : 448 + 32 * k + 1] for k in range(4)], dim=0),
                   k_dec_t=torch.cat([ph[:, 576 + 32 * k : 576 + 32 * k + 32] for k in range(4)], dim=0))  # fmt: skip
        row = {}
        for name, a in got.items():
            b = ref_host[name][hh, cc].reshape(a.shape)
            row[name] = dict(pcc=round(C.pcc(a, b), 6), maxabs=float((a - b).abs().max()), ref_max=float(b.abs().max()),
                             phase_max=float(a.abs().max()))  # fmt: skip
        term_cmp[f"h{hh}c{cc}"] = row
    s_ph = host_all(phase_state.S)[0].float().reshape(h_, -1).abs().max(dim=1).values.tolist()
    s_rf = host_all(ref_state.S)[0].float().reshape(h_, -1).abs().max(dim=1).values.tolist()
    S.log(f"{tag}: terms {term_cmp}")
    S.log(f"{tag}: |S| max per head: phase {[round(v, 3) for v in s_ph]} ref {[round(v, 3) for v in s_rf]}")
    result.update(terms=term_cmp, s_absmax=dict(phase=s_ph, ref=s_rf))
    result.update(compare=cmp, first_replay_us=dict(op=round(first_ref_us, 1), phase=round(first_phase_us, 1)))
    C.record("pcc", dict(test=f"PMK INTEGRATION kda: family layer program with the kda P4 / P6 bodies (stubs elsewhere) L{L} vs op path (bucket {c}, valid {N}, terms {TERMS}), traced",
                         milestone="P2", layer=L, kind=pl.kind, n=N, bucket=c, tp=TPN, terms=TERMS,
                         pcc=min(v["pcc"] for v in cmp.values()), pcc_on=cmp["on"]["pcc"], pcc_s=cmp["S"]["pcc"],
                         pcc_c=cmp["C"]["pcc"], maxabs_on=cmp["on"]["maxabs"], maxabs_s=cmp["S"]["maxabs"],
                         maxabs_c=cmp["C"]["maxabs"], bitwise_on=cmp["on"]["bitwise"], bitwise_s=cmp["S"]["bitwise"],
                         bitwise_c=cmp["C"]["bitwise"], status="pass" if ok else "fail", tag=tag))  # fmt: skip
    # ---- perf: steady-state replays of both traces (state re-zeroed is not needed for timing)
    calls_before_replays = prog.counters["calls"]
    perf_phase = S.replay_stats(mesh, phase_tid, replays=REPLAYS, label=f"{tag} pmk")
    assert prog.counters["calls"] == calls_before_replays, ("eager guard: launched during the replays", prog.counters)
    est = P.plan(geo).estimate()
    C.record("perf", dict(test=f"PMK INTEGRATION kda: layer program (kda bodies only) L{L} replay (bucket {c}, valid {N}, terms {TERMS}) vs op path KDA",
                          milestone="P2", layer=L, kind=pl.kind, n=N, bucket=c, tp=TPN, terms=TERMS,
                          us=perf_phase["median_us"], op_us=perf_op["median_us"], speedup=round(perf_op["median_us"] /
                                                                                              perf_phase["median_us"], 2),
                          replays=perf_phase["replays"], p90_us=perf_phase["p90_us"], min_us=perf_phase["min_us"],
                          est_phase_us=round(est["phase_span_us"], 1), est_busy=round(est["compute_core_busy"], 3),
                          status="pass" if ok else "fail", tag=tag))  # fmt: skip
    result.update(perf_phase=perf_phase, perf_op=perf_op, status="pass" if ok else "fail")
    ttnn.release_trace(mesh, phase_tid)
RESULT = result
print("PMK_INT_KDA_UNIT", RESULT, flush=True)
