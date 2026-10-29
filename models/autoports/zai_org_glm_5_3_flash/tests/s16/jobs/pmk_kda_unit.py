# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, a TP2 pair; also the 2x4 mock): the PMK KDA phase's device bench unit (doc/ttnn/16-prefill-production/
pmk/kda/README.md §5). Layer L's KDA (the chip's head shard) on the golden layer-input rows (layer_io L{L}/{golden_n},
the first `valid` rows of the bucket), in ONE process:
  op path   = prefill_tp.kda_prefill_tp's steps up to the gated norm (scan="op": conv op, prepare_chunk_recurrence,
              recurrent_chunk_scan with the fp32 chunk decay, sigmoid_gated_rms_norm) + write_slot, captured in a trace;
              its projections (qkv, low-rank, the raw f_b / g_b outputs) are copied into the phase's H3 store;
  phase     = tt/pmk_kda.py KdaPhase (P4 + P5 + P6 + P9) on that store, captured in a trace;
both replayed from zero state; compared: `on` rows [:valid] (vs the gated-norm output), the final S, the conv carry C:
PCC >= 0.9999 and max |diff| per tensor, bitwise flags; one pcc ROW; one perf ROW = the phase's steady-state replay
(median of `replays`, default 200) next to the op path's KDA programs' replay. Eager guard: the phase is launched
exactly twice outside the trace (compile + capture) and never during the replays.
ARGS: layer (4), rows (bucket, 2048), valid (1024), golden_n (2048), replays (200), terms ("fp32" | "bf16": the chunk-
term knob), tag; env ({name: value}: knobs set for this job only, restored at its end -- e.g. GLM53_PMK_ITEM_CONV=fpu),
dump (path: the phase's TERMS / on / S / C of every chip saved for a bitwise A/B between two jobs), input_pt (an fp32
layer input instead of the golden layer_io); audit (False) + audit_heads: fp32_audit, the KDA composite PER ITEM vs fp32
(conv / gate / terms / scan, both paths from the same store; Stage 14's audit, tt-metal-s14rne 0ee474a3d36 -- the op
column here is mb2's op path, no RNE: the control for GLM53_PMK_KDA_RNE in env)."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_kda as P  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.kda import CHUNK  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16, DRAM, FP32, KDA_NO_SP, pf_mm  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
_ENV = {str(k): str(v) for k, v in (args.get("env") or {}).items()}
_ENV_OLD = {k: os.environ.get(k) for k in _ENV}
os.environ.update(_ENV)
L = int(args.get("layer", 4))
LAYERS = [int(v) for v in args.get("layers", [L])]  # [0..4]: adopt the d0 session's L0-4 build
MAP = str(args.get("map", ""))  # "family": the family engine's CB index maps (PK_ITEM_MAP / PK_SCAN_MAP)
CORES = str(args.get("cores", "first"))  # "family": items on the family's members, scan on its MM-only cores
BUCKET = int(args.get("rows", 2048))
N = int(args.get("valid", 1024))
GN = int(args.get("golden_n", 2048))
REPLAYS = int(args.get("replays", 200))
TERMS = str(args.get("terms", "fp32"))
tag = args.get("tag", f"pmk-kda-L{L}-{TERMS}")
TPN = int(mesh.get_num_devices())
PCC_BAR = 0.9999
assert BUCKET % CHUNK == 0 and 0 < N <= BUCKET and N <= GN, (BUCKET, N, GN)

ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(BUCKET,), max_seq=BUCKET + 512 if len(LAYERS) > 1 else 4096)
pf = ent["pf"]
pl = pf.layers[LAYERS.index(L)]
kda = pl.kda
h_, d_, w_ = kda.h, kda.d, kda.width
# heads (0 = all): the phase on the chip's heads 0 .. HS - 1 only -- TP8's per-chip geometry on the pair (HS 8: CPH 8 /
# SPREAD 4 / Vt 1 = 64 item + 32 scan cores); the op path runs every head, the comparison takes heads < HS
HS = int(args.get("heads", 0)) or h_
ws_ = HS * d_
assert HS <= h_ and h_ % HS == 0, (HS, h_)
c = BUCKET
sp = pf.mode(c) == "sp"
ci = pf.chunk_inputs(c)
SCAN_BUDGET = int(args.get("scan_budget", 32))  # 8 at TP8 = SPREAD 1 (Vt 4, the pair's validated scan path): the spread A/B
ITEM_CORES = int(args.get("item_cores", 64))  # 32 at TP2 = CPH 1: room for SPREAD 2 (64 scan cores) on a 110-core grid
geo = P.KdaGeometry(rows=c, heads=HS, terms_dtype=TERMS, scan_budget=SCAN_BUDGET, item_cores=ITEM_CORES)
S.log(f"{tag}: L{L} {pl.kind} heads/chip {h_} bucket {c} valid {N} mode {'sp' if sp else 'rep'} terms {TERMS}; "
      f"plan item {P.plan(geo).item_l1} B scan {P.plan(geo).scan_l1} B spread {geo.spread} cph {geo.cph}")  # fmt: skip

# ---- the layer's attention input on the golden rows (the op graph's own glue: mhc_pre -> RMSNorm -> gather), once
if args.get("input_pt"):  # another fp32 layer input [GN, 4, 4096] (e.g. the hf_ref chain's L{L}/streams_in.pt, as 14b's isol)
    x_in = torch.load(str(args["input_pt"]), map_location="cpu", weights_only=True)
else:
    x_in, _ = S.golden_layer(L, GN)
x_in = x_in.reshape(-1, 4, 4096)[:GN].float()
pad = torch.zeros(c, 4, 4096)
pad[:N] = x_in[:N]
X = pf.streams_to_device(pad, sp)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)  # ci.valid_end = N
h0, _post, _comb = pf._mhc_pre(pl, "attn", X)
hn = pl.rms(h0, "attn", BF16)
xh = pf.gather(hn, sp)  # [1, 1, c, 4096] bf16 on every chip
ttnn.synchronize_device(mesh)

# ---- persistent tensors (addresses are baked into both traces)
grid = mesh.compute_with_storage_grid_size()
cores = [ttnn.CoreCoord(x, y) for y in range(int(grid.y)) for x in range(int(grid.x))]
assert len(cores) >= geo.item_cores + geo.scan_cores, (len(cores), geo)
item_cores, scan_cores = cores[: geo.item_cores], cores[geo.item_cores : geo.item_cores + geo.scan_cores]
fam = None
if MAP == "family" or CORES == "family":  # the discriminator of the integration unit (13:05Z): the family's map / cores
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMK  # noqa: PLC0415

    fam = PMK.PmkSparseKdaLayer(pf, LAYERS.index(L), c, sp, probe=False).engine
if CORES == "family":
    item_cores = [ttnn.CoreCoord(int(x), int(y)) for x, y in fam.place.members]
    scan_cores = [ttnn.CoreCoord(int(x), int(y)) for x, y in fam.place.kda_scan][: geo.scan_cores]  # spread A/B: the first H x SPREAD
maps = dict(item_map=fam.kda_map["item"], scan_map=fam.kda_map["scan"]) if MAP == "family" else {}
def _per_chip_cols(t, spans, dtype):
    # every chip's own copy of t [.., W'] -> the column spans [(a, b), ..] concatenated, back on the mesh per chip
    parts = [ttnn.to_torch(d_t).float() for d_t in (ttnn.get_device_tensors(t) if TPN > 1 else [t])]
    sub = [torch.cat([p_[..., a:b] for a, b in spans], dim=-1) for p_ in parts]
    host = torch.cat(sub, dim=0) if TPN > 1 else sub[0]
    return ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                           mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if TPN > 1 else None)  # fmt: skip


if HS < h_:  # the phase's per-head parameters for heads < HS (taps: q | k | v channel blocks; dt_bias / rate: W)
    from types import SimpleNamespace  # noqa: PLC0415

    qkv_spans = [(0, ws_), (w_, w_ + ws_), (2 * w_, 2 * w_ + ws_)]
    kda_ph = SimpleNamespace(device=kda.device, o_norm=kda.o_norm, lower_bound=kda.lower_bound, eps=kda.eps,
                             taps_prefill=[_per_chip_cols(t_, qkv_spans, t_.dtype) for t_ in kda.taps_prefill],
                             dt_bias=_per_chip_cols(kda.dt_bias, [(0, ws_)], kda.dt_bias.dtype),
                             decay_rate=_per_chip_cols(kda.decay_rate, [(0, ws_)], kda.decay_rate.dtype))  # fmt: skip
else:
    kda_ph = kda
phase = P.KdaPhase.from_kda(kda_ph, geo, item_cores=item_cores, scan_cores=scan_cores, nprod=1, h3_preraised=True, **maps)
S.log(f"{tag}: map {MAP or 'contiguous'} {maps}; cores {CORES}: items {[(int(q.x), int(q.y)) for q in item_cores[:3]]}.., "
      f"scan {[(int(q.x), int(q.y)) for q in scan_cores[:3]]}..")
scr = phase.scratch()
ref_state, phase_state = kda.allocate_state(1), kda.allocate_state(1)
if HS < h_:  # the phase's state at the subset's shapes (S [1, HS, K, V], C [1, 3, 1, 3 W_s]); slot_mask unused by it
    from types import SimpleNamespace  # noqa: PLC0415

    _mk = lambda t_, shape: ttnn.zeros(shape, dtype=t_.dtype, layout=t_.layout, device=mesh, memory_config=DRAM)  # noqa: E731
    phase_state = SimpleNamespace(S=_mk(ref_state.S, (1, HS, d_, d_)), C=_mk(ref_state.C, (1, 3, 1, 3 * ws_)),
                                  slot_mask=ref_state.slot_mask)  # fmt: skip
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
    if HS < h_:  # heads < HS of the store: q | k | v blocks, f_a | g_a | b[:HS], f / gate_out columns (row-major slices)
        def _cols(t_, spans):
            rm_ = ttnn.to_layout(t_, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            cat_ = ttnn.concat([ttnn.slice(rm_, (0, 0, 0, a), (1, 1, c, b)) for a, b in spans], dim=-1)
            return ttnn.to_layout(cat_, ttnn.TILE_LAYOUT, memory_config=DRAM)

        ttnn.copy(_cols(qkv, [(0, ws_), (w_, w_ + ws_), (2 * w_, 2 * w_ + ws_)]), scr["u"])
        ttnn.copy(_cols(lr, [(0, 2 * d_), (2 * d_, 2 * d_ + HS)]), scr["lr"])
        ttnn.copy(_cols(f_raw, [(0, ws_)]), scr["f"])
        ttnn.copy(_cols(ttnn.typecast(gate_out, scr["gate_out"].dtype), [(0, ws_)]), scr["gate_out"])
    else:
        ttnn.copy(qkv, scr["u"])
        ttnn.copy(lr, scr["lr"])
        ttnn.copy(f_raw, scr["f"])
        ttnn.copy(ttnn.typecast(gate_out, scr["gate_out"].dtype), scr["gate_out"])
    ttnn.copy(ttnn.reshape(o, (1, 1, c, w_)), o_ref)
    kda.write_slot(ref_state, 0, new_carry, ttnn.reshape(s_fin, (1, h_, d_, d_)))


def phase_fn():
    phase(scr, phase_state.S, phase_state.C, ci.valid_end)


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


AUDIT = bool(args.get("audit", False))
AUDIT_HEADS = int(args.get("audit_heads", 0)) or HS


def _rel(a, b):
    """a vs the reference b: PCC, relative L2 (|a - b| / |b|), max |a - b|, and the magnitude: scale = <a, b> / <b, b>
    (PCC and rel are scale-blind to a uniform shrink -- asmakda 2026-10-25: the op scan's output o is 0.994 of fp32),
    rel after removing that scale, the norm ratio |a| / |b|"""
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    bb = float((b * b).sum().clamp_min(1e-300))
    sc = float((a * b).sum()) / bb
    return dict(pcc=round(C.pcc(a.float(), b.float()), 7), rel=float((a - b).norm() / b.norm().clamp_min(1e-30)),
                maxabs=float((a - b).abs().max()), scale=round(sc, 6),
                rel_after=float((a - sc * b).norm() / b.norm().clamp_min(1e-30)),
                norm_ratio=round(float(a.norm() / b.norm().clamp_min(1e-30)), 6))  # fmt: skip


def _head_scales(a, b):
    """per-head scale <a_h, b_h> / <b_h, b_h> (dim 0 = head) and the worst (the 'worst head >= 0.98' gate, 03:55:13Z)"""
    a, b = a.double().reshape(a.shape[0], -1), b.double().reshape(b.shape[0], -1)
    sc = [round(float((a[h] * b[h]).sum() / (b[h] * b[h]).sum().clamp_min(1e-300)), 6) for h in range(a.shape[0])]
    return dict(heads=sc, worst=min(sc))


def _oracle_terms(q, k, v, g, beta):
    """prepare_chunk_recurrence's seven terms in fp32 / fp64 (the ttnn nightly oracle, test_prepare_chunk_recurrence
    _oracle): q, k, v, g [H, NC, 32, D], beta [H, NC, 32, 1]"""
    dk = q.shape[-1]
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * dk**-0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    cg = torch.cumsum(g, dim=2)
    decay, fg = torch.exp(cg), cg[:, :, -1]
    k64, cg64 = k.double(), cg.double()
    anc = cg64[:, :, -1:].mul(0.5)
    akk = torch.matmul(beta.double() * k64 * torch.exp(cg64 - anc), (k64 * torch.exp(anc - cg64)).transpose(-1, -2))
    intra = torch.matmul(q.double() * torch.exp(cg64 - anc), (k64 * torch.exp(anc - cg64)).transpose(-1, -2)).tril().float()
    eye = torch.eye(CHUNK, dtype=torch.float64).reshape(1, 1, CHUNK, CHUNK)
    return dict(v_beta=beta * v, kd=beta * k * decay, q_decay=q * decay, intra=intra,
                k_dec_t=(k * torch.exp(fg.unsqueeze(2) - cg)).transpose(-1, -2), chunk_decay=torch.exp(fg).unsqueeze(-1),
                t_inv=torch.linalg.inv(eye + torch.tril(akk, diagonal=-1)).float())  # fmt: skip


def _oracle_scan(t):
    """recurrent_chunk_scan from zero state in fp32 (recurrent_chunk_scan_test_utils.recurrent_oracle without its bf16
    output cast) -> y [H, NC, 32, V], S [H, K, V]"""
    t = {k_: v_.double() for k_, v_ in t.items()}
    h, nc, dk, dv = t["kd"].shape[0], t["kd"].shape[1], t["kd"].shape[-1], t["v_beta"].shape[-1]
    st, ys = torch.zeros(h, dk, dv, dtype=torch.float64), []
    for cc in range(nc):
        vn = torch.matmul(t["t_inv"][:, cc], t["v_beta"][:, cc] - torch.matmul(t["kd"][:, cc], st))
        ys.append(torch.matmul(t["q_decay"][:, cc], st) + torch.matmul(t["intra"][:, cc], vn))
        st = st * t["chunk_decay"][:, cc] + torch.matmul(t["k_dec_t"][:, cc], vn)
    return torch.stack(ys, dim=1).float(), st.float()


def fp32_audit(host_all, terms_host, np_):
    """GLM53 program 2026-10-25T09:01:23Z item (3): the KDA composite's error source PER ITEM vs fp32, chip 0, heads
    < AUDIT_HEADS, the valid chunks. Every reference is fed by the SAME H3 store both paths read (u / lr / f /
    gate_out), with the layer's own taps / dt_bias / rate / o_norm:
      conv  = silu(causal conv(u)) (taps row j x input row t - 3 + j, zero history);
      gate  = lower_bound * sigmoid(rate * (f + dt_bias)), beta = sigmoid(b);
      terms = prepare_chunk_recurrence's oracle on those -> each path's terms vs fp32 (chunk_decay = the GATE alone;
              v = v_beta / beta = the CONV alone);
      scan  = each path's OWN device terms through an fp64 scan + the gated norm vs its on / S (the scan + norm alone);
      e2e   = the fp32 chain end to end vs each path's on / S."""
    import torch.nn.functional as F  # noqa: PLC0415

    nv, ha = N // CHUNK, AUDIT_HEADS
    assert N % CHUNK == 0, ("audit: valid rows on chunk bounds", N)
    hd = lambda t_: host_all(t_)[0].float()  # noqa: E731
    u = hd(scr["u"]).reshape(c, -1)[:N]
    lr = hd(scr["lr"]).reshape(c, -1)[:N]
    f = hd(scr["f"]).reshape(c, -1)[:N, :ws_]
    gout = hd(scr["gate_out"]).reshape(c, -1)[:N, :ws_]
    taps = torch.stack([hd(t_).reshape(-1)[: 3 * ws_] for t_ in kda_ph.taps_prefill])  # [4, 3 W_s]
    dtb, rate = hd(kda_ph.dt_bias).reshape(-1)[:ws_], hd(kda_ph.decay_rate).reshape(-1)[:ws_]
    onw = ttnn.to_torch(ttnn.get_device_tensors(kda.o_norm)[0] if TPN > 1 else kda.o_norm).float().reshape(-1)[:d_]
    # conv (zero history: a fresh request), gate, beta
    padu = torch.cat([torch.zeros(3, u.shape[1]), u[:, : 3 * ws_]])
    qkv = F.silu(sum(taps[j] * padu[j : j + N] for j in range(4)))
    g = float(kda.lower_bound) * torch.sigmoid(rate * (f + dtb))
    beta = torch.sigmoid(lr[:, 2 * d_ : 2 * d_ + HS])
    hc = lambda t_, dd: t_.reshape(N, HS, dd).permute(1, 0, 2).reshape(HS, nv, CHUNK, dd)[:ha]  # noqa: E731
    q, k, v = (hc(qkv[:, i_ * ws_ : (i_ + 1) * ws_], d_) for i_ in range(3))
    t32 = _oracle_terms(q, k, v, hc(g, d_), hc(beta, 1))
    # each path's terms [ha, nv, ..]
    pm = {k_: [] for k_ in t32}
    for hh in range(ha):
        for k_ in pm:
            pm[k_].append([])
        for cc in range(nv):
            ph = terms_host[32 * (hh * n_chunks + cc) : 32 * (hh * n_chunks + cc) + 32]
            got = dict(v_beta=ph[:, 0:128], q_decay=ph[:, 128:256], kd=ph[:, 256:384], intra=ph[:, 384:416], t_inv=ph[:, 416:448],
                       chunk_decay=torch.cat([ph[:, 448 + 32 * j : 448 + 32 * j + 1] for j in range(4)], dim=0),
                       k_dec_t=torch.cat([ph[:, 576 + 32 * j : 576 + 32 * j + 32] for j in range(4)], dim=0))  # fmt: skip
            for k_, a_ in got.items():
                pm[k_][-1].append(a_)
    pm = {k_: torch.stack([torch.stack(r_) for r_ in v_]) for k_, v_ in pm.items()}
    op = {k_: hd(t_)[:ha, :nv].reshape(t32[k_].shape) for k_, t_ in ref_terms.items()}
    out = dict(heads=ha, chunks=nv, layer=L)
    out["terms"] = {k_: dict(pmk=_rel(pm[k_], t32[k_]), op=_rel(op[k_], t32[k_])) for k_ in t32}
    # the conv alone: v = v_beta / beta (rows with beta > 0.05), the gate alone: chunk_decay (and its log = sum g)
    bmask = (hc(beta, 1) > 0.05).expand_as(v)
    out["conv_v"] = {p_: _rel((t_["v_beta"] / hc(beta, 1))[bmask], v[bmask]) for p_, t_ in (("pmk", pm), ("op", op))}
    out["gate_logdecay"] = {p_: _rel(torch.log(t_["chunk_decay"].clamp_min(1e-30)), torch.log(t32["chunk_decay"]))
                            for p_, t_ in (("pmk", pm), ("op", op))}  # fmt: skip

    def norm(y):  # [ha, nv, 32, d] -> the gated rms norm rows [N, ha d]
        y = y.reshape(ha, N, d_).permute(1, 0, 2)
        gt = gout[:, : ha * d_].reshape(N, ha, d_)
        o_ = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + float(kda.eps)) * onw * torch.sigmoid(gt)
        return o_.reshape(N, ha * d_)

    on_pm = hd(scr["on"]).reshape(c, -1)[:N, : ha * d_]
    on_op = hd(o_ref).reshape(c, -1)[:N, : ha * d_]
    s_pm = hd(phase_state.S).reshape(-1, d_, d_)[:ha]
    s_op = hd(ref_state.S).reshape(-1, d_, d_)[:ha]
    y32, s32 = _oracle_scan(t32)
    out["e2e"] = dict(pmk=dict(on=_rel(on_pm, norm(y32)), S=_rel(s_pm, s32)), op=dict(on=_rel(on_op, norm(y32)), S=_rel(s_op, s32)))
    out["per_head"] = dict(S=dict(pmk=_head_scales(s_pm, s32), op=_head_scales(s_op, s32)),
                           k_dec_t=dict(pmk=_head_scales(pm["k_dec_t"], t32["k_dec_t"]), op=_head_scales(op["k_dec_t"], t32["k_dec_t"])))
    out["scan_own_terms"] = {}
    for p_, t_, on_, s_ in (("pmk", pm, on_pm, s_pm), ("op", op, on_op, s_op)):
        y_, s2_ = _oracle_scan(t_)
        out["scan_own_terms"][p_] = dict(on=_rel(on_, norm(y_)), S=_rel(s_, s2_))
    return out


CHAIN = [int(v) for v in (args.get("chain") or [])]  # (program 03:19:46Z) state-precision checkpoints, in calls of N rows


CHAIN_XH = str(args.get("chain_xh", ""))  # per-call attention inputs [K, c, 4096] (distinct rows: the slow-decay compounding)
CHAIN_ROLL = int(args.get("chain_roll", 0))  # self-test: call j's input = this unit's own xh rows rolled by j x CHAIN_ROLL


def _chain_terms(host_all, carry: bool, hist_u=None):
    """the fp64 oracle's seven terms of ONE call on the H3 store (heads < AUDIT_HEADS, the valid chunks), as fp32_audit
    builds them; carry: the conv history is the previous call's last 3 rows (the device's C carry; the chained calls
    reuse the same rows), else zero (a fresh request)"""
    import torch.nn.functional as F  # noqa: PLC0415

    nv, ha = N // CHUNK, AUDIT_HEADS
    hd = lambda t_: host_all(t_)[0].float()  # noqa: E731
    u = hd(scr["u"]).reshape(c, -1)[:N]
    lr = hd(scr["lr"]).reshape(c, -1)[:N]
    f = hd(scr["f"]).reshape(c, -1)[:N, :ws_]
    taps = torch.stack([hd(t_).reshape(-1)[: 3 * ws_] for t_ in kda_ph.taps_prefill])
    dtb, rate = hd(kda_ph.dt_bias).reshape(-1)[:ws_], hd(kda_ph.decay_rate).reshape(-1)[:ws_]
    hist = (hist_u if hist_u is not None else u[N - 3 : N, : 3 * ws_]) if carry else torch.zeros(3, 3 * ws_)
    padu = torch.cat([hist, u[:, : 3 * ws_]])
    qkv = F.silu(sum(taps[j] * padu[j : j + N] for j in range(4)))
    g = float(kda.lower_bound) * torch.sigmoid(rate * (f + dtb))
    beta = torch.sigmoid(lr[:, 2 * d_ : 2 * d_ + HS])
    hc = lambda t_, dd: t_.reshape(N, HS, dd).permute(1, 0, 2).reshape(HS, nv, CHUNK, dd)[:ha]  # noqa: E731
    q, k, v = (hc(qkv[:, i_ * ws_ : (i_ + 1) * ws_], d_) for i_ in range(3))
    return {k_: v_.double() for k_, v_ in _oracle_terms(q, k, v, hc(g, d_), hc(beta, 1)).items()}, u[N - 3 : N, : 3 * ws_].clone()


def _oracle_state(t, st):
    """the fp64 recurrence over one call's chunks from state st [ha, K, V] -> the state after it"""
    for cc in range(t["kd"].shape[1]):
        vn = torch.matmul(t["t_inv"][:, cc], t["v_beta"][:, cc] - torch.matmul(t["kd"][:, cc], st))
        st = st * t["chunk_decay"][:, cc] + torch.matmul(t["k_dec_t"][:, cc], vn)
    return st


def chain_run(host_all, ref_tid, phase_tid):
    """S after j calls of the same N rows with the state CARRIED (no re-zeroing), op path and phase, vs the fp64 chain;
    S scale = <a, ref> / <ref, ref>, plus the L2 / L1 norm ratios (14 kprec2's L1 / L2 S scale)"""
    ha = AUDIT_HEADS
    xs = None
    if CHAIN_XH or CHAIN_ROLL:  # distinct rows per call: host-written into the attention input xh between replays (a transfer)
        if CHAIN_XH:
            xs = torch.load(CHAIN_XH, map_location="cpu", weights_only=True).reshape(-1, c, 4096)
        else:  # (the valid rows only: N == c in the chain jobs)
            x0_ = host_all(xh)[0].reshape(c, 4096)
            xs = torch.stack([torch.roll(x0_, shifts=j_ * CHAIN_ROLL, dims=0) for j_ in range(max(CHAIN))])
        assert xs.shape[0] >= max(CHAIN), ("chain_xh: one [c, 4096] input per call", tuple(xs.shape), max(CHAIN))
        rep_ = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
    else:
        (t1, _u1), (tn, _un) = _chain_terms(host_all, False), _chain_terms(host_all, True)
    zero_state(ref_state)
    zero_state(phase_state)
    st = torch.zeros(ha, d_, d_, dtype=torch.float64)
    out, prev_u = [], None
    for j in range(1, max(CHAIN) + 1):
        if xs is not None:
            host_xh = ttnn.from_torch(xs[j - 1].reshape(1, 1, c, 4096).to(torch.bfloat16), dtype=BF16, layout=ttnn.TILE_LAYOUT,
                                      mesh_mapper=rep_)  # fmt: skip
            ttnn.copy_host_to_device_tensor(host_xh, xh)
        ttnn.execute_trace(mesh, ref_tid, cq_id=0, blocking=True)  # the op path (its own carried state; rewrites the store)
        ttnn.execute_trace(mesh, phase_tid, cq_id=0, blocking=True)
        if xs is not None:  # this call's terms from the store just written, the conv history = the previous call's tail
            tj, prev_u = _chain_terms(host_all, j > 1, prev_u)
        else:
            tj = t1 if j == 1 else tn
        st = _oracle_state(tj, st)
        if j in CHAIN:
            ref = st.float()
            row = dict(calls=j, tokens=j * N)
            for p_, t_ in (("pmk", phase_state.S), ("op", ref_state.S)):
                a = host_all(t_)[0].float().reshape(-1, d_, d_)[:ha]
                row[p_] = dict(_rel(a, ref), l1_ratio=round(float(a.abs().sum() / ref.abs().sum().clamp_min(1e-30)), 6),
                               finite=bool(torch.isfinite(a).all()), per_head=_head_scales(a, ref))  # fmt: skip
            S.log(f"{tag}: chain {row}")
            out.append(row)
    return out


# the phase's compile launch BEFORE the first capture (a program first compiled after a capture keeps a live
# program-cache buffer -> 'buffers alive before trace replay' at the op path's replays: d0 10:20Z family unit)
zero_state(phase_state)
S.log(f"{tag}: phase compile launch")
phase_fn()
ttnn.synchronize_device(mesh)
ref_tid = capture(ref_fn, ref_state, "op path")
phase_tid = capture(phase_fn, phase_state, "phase", compiled=True)
launches_after_capture = phase.calls
assert launches_after_capture == (1 if C.MOCK else 2), ("eager guard: phase launches", phase.calls)

result = dict(tag=tag, layer=L, kind=pl.kind, heads=h_, bucket=c, valid=N, tp=TPN, terms=TERMS, mock=C.MOCK,
              spread=geo.spread, cph=geo.cph, heads_phase=HS)  # fmt: skip
if C.MOCK:
    S.log(f"{tag}: mock -- both paths built and launched once (no data, no traces)")
    result["status"] = "mock"
else:
    first_ref_us = replay(ref_tid, ref_state, "op path")  # fills the H3 store + the reference outputs
    first_phase_us = replay(phase_tid, phase_state, "phase")
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
                a, b = a.reshape(c, ws_)[:N], b.reshape(c, w_)[:N, :ws_]
            elif name == "S" and HS < h_:
                a, b = a.reshape(HS, d_, d_), b.reshape(h_, d_, d_)[:HS]
            elif name == "C" and HS < h_:
                b = torch.cat([b[..., 0:ws_], b[..., w_ : w_ + ws_], b[..., 2 * w_ : 2 * w_ + ws_]], dim=-1)
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
    terms_host = host_all(scr["terms"])[0].float().reshape(HS * n_chunks * CHUNK, NP_ * CHUNK)
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
    s_ph = host_all(phase_state.S)[0].float().reshape(HS, -1).abs().max(dim=1).values.tolist()
    s_rf = host_all(ref_state.S)[0].float().reshape(h_, -1).abs().max(dim=1).values.tolist()[:HS]
    S.log(f"{tag}: terms {term_cmp}")
    S.log(f"{tag}: |S| max per head: phase {[round(v, 3) for v in s_ph]} ref {[round(v, 3) for v in s_rf]}")
    result.update(terms=term_cmp, s_absmax=dict(phase=s_ph, ref=s_rf))
    if AUDIT:
        result["audit"] = fp32_audit(host_all, terms_host, NP_)
        S.log(f"{tag}: fp32 audit {result['audit']}")
    result.update(compare=cmp, first_replay_us=dict(op=round(first_ref_us, 1), phase=round(first_phase_us, 1)))
    C.record("pcc", dict(test=f"pmk kda phase L{L} vs op path (bucket {c}, valid {N}, terms {TERMS})",
                         milestone="P2", layer=L, kind=pl.kind, n=N, bucket=c, tp=TPN, terms=TERMS,
                         pcc=min(v["pcc"] for v in cmp.values()), pcc_on=cmp["on"]["pcc"], pcc_s=cmp["S"]["pcc"],
                         pcc_c=cmp["C"]["pcc"], maxabs_on=cmp["on"]["maxabs"], maxabs_s=cmp["S"]["maxabs"],
                         maxabs_c=cmp["C"]["maxabs"], bitwise_on=cmp["on"]["bitwise"], bitwise_s=cmp["S"]["bitwise"],
                         bitwise_c=cmp["C"]["bitwise"], status="pass" if ok else "fail", tag=tag))  # fmt: skip
    if AUDIT:
        result["audit"] = fp32_audit(host_all, terms_host, NP_)
        S.log(f"{tag}: fp32 audit {result['audit']}")
    if args.get("dump"):  # the first replay's outputs (from zero state), before the timing replays advance the state
        torch.save({k: [d_.float() for d_ in host_all(t_)] for k, t_ in
                    dict(terms=scr["terms"], on=scr["on"], S=phase_state.S, C=phase_state.C).items()}, args["dump"])
        result["dump"] = args["dump"]
    if CHAIN:  # (after the dump, before the timing replays) the carried-state precision curve
        result["chain"] = chain_run(host_all, ref_tid, phase_tid)
        result["chain_inputs"] = CHAIN_XH or (f"own rows rolled by {CHAIN_ROLL} per call" if CHAIN_ROLL else "the same rows repeated")
    # ---- perf: steady-state replays of both traces (state re-zeroed is not needed for timing)
    perf_phase = S.replay_stats(mesh, phase_tid, replays=REPLAYS, label=f"{tag} phase")
    perf_op = S.replay_stats(mesh, ref_tid, replays=REPLAYS, label=f"{tag} op path")
    assert phase.calls == launches_after_capture, ("eager guard: the phase launched during the replays", phase.calls)
    est = P.plan(geo).estimate()
    C.record("perf", dict(test=f"pmk kda phase L{L} replay (bucket {c}, valid {N}, terms {TERMS}) vs op path KDA",
                          milestone="P2", layer=L, kind=pl.kind, n=N, bucket=c, tp=TPN, terms=TERMS,
                          us=perf_phase["median_us"], op_us=perf_op["median_us"], speedup=round(perf_op["median_us"] /
                                                                                              perf_phase["median_us"], 2),
                          replays=perf_phase["replays"], p90_us=perf_phase["p90_us"], min_us=perf_phase["min_us"],
                          est_phase_us=round(est["phase_span_us"], 1), est_busy=round(est["compute_core_busy"], 3),
                          status="pass" if ok else "fail", tag=tag))  # fmt: skip
    result.update(perf_phase=perf_phase, perf_op=perf_op, status="pass" if ok else "fail", env=_ENV)
    for t_ in (ref_tid, phase_tid):
        ttnn.release_trace(mesh, t_)
for k_, v_ in _ENV_OLD.items():  # the job's knobs end with it (a standing server runs later jobs in this process)
    if v_ is None:
        os.environ.pop(k_, None)
    else:
        os.environ[k_] = v_
RESULT = result
print("PMK_KDA_UNIT", RESULT, flush=True)
