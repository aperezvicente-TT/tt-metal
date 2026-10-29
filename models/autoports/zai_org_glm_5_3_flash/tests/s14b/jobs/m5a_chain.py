# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd single, chip 3): Stage 14b M5a BASELINE — N resident copies of L0 (dense_kda) run as N
SEQUENTIAL launches of 7a's whole-layer megakernel (MkLayer) in ONE trace, each copy with its own weights (device
clones of L0's DS weights, KDA tensors and mHC parameters: distinct DRAM addresses = per-layer weight bases) and its
own KDA state (S, conv history), each launch fed the previous one's output.

  numerics: one traced chain of NMAX launches with every layer's state restored to the same snapshot inside the trace
      -> the reference for the stacked (one-program) runs: every layer's output and state saved to work-s14b/m5a_ref/
      (bitwise comparisons later read these; device vs device, no CPU oracle);
  timing: chains of N in {ns} launches captured WITHOUT the restores (state drifts; timing only), blocking replays ->
      median us; steady state per layer = (T_N - T_1) / (N - 1) (7a's M1 convention, 879.2 us at T=1 on chip 2).

ARGS: ns ([1, 2, 4, 8, 45]), t (1), reps (20), tag, record (False), env (7a megakernel knobs), probe (False)."""
import gc
import importlib
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.kda_pre as KPRE  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.kda_rec as KREC  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_dense as MKD  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_layer as MKL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_readers as MKR  # noqa: E402

for mod in (KREC, KPRE, MKR, MKD, MKL):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_kda_pre import masks  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_mk_a import mhc_args  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_mk_ffn import ffn_mhc_args  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import cached_decoder  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
KNOBS = ("GLM53_MK_KB_QKV", "GLM53_MK_KB_O", "GLM53_MK_RING_KB_QKV", "GLM53_MK_RING_KB_O", "GLM53_MK_RING_KB",
         "GLM53_MK_PARK", "GLM53_MK_PARK_SLOTS", "GLM53_MK_PARK_PHASES", "GLM53_MK_OPT", "GLM53_MK_READERS")  # fmt: skip
M1_KNOBS = {"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3"}  # 7a's M1 configuration (879.2 us steady state, chip 2)
for k_ in KNOBS:
    os.environ.pop(k_, None)
ENV = {**M1_KNOBS, **{k_: str(v_) for k_, v_ in (args.get("env") or {}).items()}}
assert set(ENV) <= set(KNOBS), ENV
os.environ.update(ENV)
NS = [int(n) for n in args.get("ns", [1, 2, 4, 8, 45])]
NMAX = max(NS)
T = int(args.get("t", 1))
REPS = int(args.get("reps", 20))
tag = args.get("tag", "m5a-base")
W = os.environ.get("GLM53_WORK", "/opt/glm53/work-s14b")
LAYER = 0
cfg = hf_config()


def log(*a):
    print(f"M5A {C.now()}", *a, flush=True)


def dup(t):
    """a device copy of t at a new address (identical bytes: host round trip of the packed storage, no re-quantise)"""
    return ttnn.to_device(ttnn.from_device(t), mesh, memory_config=t.memory_config())


def clone_lin(lin):
    import copy

    c = copy.copy(lin)
    c.weight = dup(lin.weight)
    return c


def clone_kda(kda):
    import copy

    c = copy.copy(kda)
    for name in ("tap_cur", "taps_hist", "w_fb", "w_gb", "dt_bias", "decay_rate", "o_norm"):
        if isinstance(getattr(kda, name, None), ttnn.Tensor):
            setattr(c, name, dup(getattr(kda, name)))
    return c


# ------------------------------------------------------------------ N layer copies (kept in CTX across jobs)
dec = cached_decoder(cfg, LAYER, mesh, ctx)
kda0, mlp0 = dec.layer.attn, dec.layer.mlp
lins0 = {d.role: d for d in dec.layer.ds_linears()}
ROLES = ("kda_qkv", "kda_lowrank", "kda_o", "mlp_gate_up", "mlp_down")
key = ("m5a_layers", T, tuple(sorted(ENV.items())), bool(args.get("probe", False)))
M5.prune_ctx(ctx, [key])
gc.collect()
M5.keep_decoders(ctx, [LAYER])
if key not in ctx:
    ctx[key] = []
layers = ctx[key]
if len(layers) < NMAX:
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    a_args, f_args = mhc_args(source, cfg), ffn_mhc_args(source, cfg)
    t0 = time.time()
    while len(layers) < NMAX:
        first = not layers
        kda_i = kda0 if first else clone_kda(kda0)
        lins_i = {r: (lins0[r] if first else clone_lin(lins0[r])) for r in ROLES}
        mk = MKL.MkLayer(mesh, a_args, f_args, kda_i, lins_i, intermediate=mlp0.intermediate, limit=mlp0.limit,
                         probe=bool(args.get("probe", False)) and first)  # fmt: skip
        if not first:  # one set of the persistent L1 hand-off buffers (on, m) for every copy: L1 is per-core scarce
            mk.persist = layers[0]["mk"].persist
        layers.append(dict(mk=mk, kda=kda_i, st=kda0.allocate_state(T)))
    log(f"{NMAX} layer copies ready in {time.time() - t0:.1f}s")
assert all(layers[i]["mk"].readers.phases[0].ds.weight.buffer_address() !=
           layers[0]["mk"].readers.phases[0].ds.weight.buffer_address() for i in range(1, NMAX))  # fmt: skip

g = golden(LAYER, 2048)["streams_in"].float()
x = ttnn.from_torch(g[0, 300 : 300 + 37 * T : 37].reshape(1, 1, T, 16384), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                    device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
active = torch.tensor([(i % 3) != 2 for i in range(T)]) if T > 1 else torch.tensor([True])
act = masks(mesh, active, kda0.lower_bound)
gen = torch.Generator().manual_seed(1234)
hist0 = ttnn.from_torch((torch.randn(1, 3, T, 24576, generator=gen) * 0.5).to(torch.bfloat16), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
s0 = ttnn.from_torch(torch.randn(T, 64, 128, 128, generator=gen) * 0.1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                     device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip


def chain(n, restore):
    outs = []
    y = x
    for i in range(n):
        L = layers[i]
        if restore:
            ttnn.copy(hist0, L["st"].C)
            ttnn.copy(s0, L["st"].S)
        y = L["mk"](y, L["st"].C, act, t=T, state=L["st"].S)["out"]
        outs.append(y)
    return outs


res = {}
probe = None
# numerics: the NMAX chain with every state restored inside the trace -> references
tr = C.Traced(ttnn, mesh, lambda: chain(NMAX, True))
pc0 = mesh.num_program_cache_entries()
tr.replay()
ref_dir = os.path.join(W, "m5a_ref")
os.makedirs(ref_dir, exist_ok=True)
outs = [ttnn.to_torch(o).float().reshape(-1, 16384)[:T] for o in tr.out]
S = [ttnn.to_torch(L["st"].S).float() for L in layers[:NMAX]]
Cs = [ttnn.to_torch(L["st"].C) for L in layers[:NMAX]]
if layers[0]["mk"].probe is not None:  # layer 0's per-phase marks of the traced numerics call
    probe = MKL.decode_probe(ttnn.to_torch(layers[0]["mk"].probe), layers[0]["mk"].readers.assign,
                             k_cores=len(layers[0]["mk"].place["k"]))  # fmt: skip
    log("PROBE layer 0 of the chain", probe)
tr.replay()  # determinism: the same trace again
outs2 = [ttnn.to_torch(o).float().reshape(-1, 16384)[:T] for o in tr.out]
det = all(torch.equal(a, b) for a, b in zip(outs, outs2))
tr.release()
finite = [bool(torch.isfinite(o).all()) for o in outs]
path = os.path.join(ref_dir, f"chain_T{T}_N{NMAX}_{tag}.pt")
with open(path + ".tmp", "wb") as f:
    torch.save(dict(x=g[0, 300 : 300 + 37 * T : 37].reshape(T, 16384), outs=outs, S=S, C=Cs, T=T, n=NMAX, env=ENV,
                    code=os.environ.get("GLM53_CODE_SHA")), f)  # fmt: skip
    f.flush()
    os.fsync(f.fileno())
os.replace(path + ".tmp", path)
log(f"reference saved {path}: deterministic={det}, finite per layer={sum(finite)}/{NMAX}, "
    f"out rms by layer {[round(float(o.pow(2).mean().sqrt()), 3) for o in outs[:: max(1, NMAX // 9)]]}")  # fmt: skip
gc.collect()
# timing: N-launch chains, no restores
timing = {}
for n in NS:
    trn = C.Traced(ttnn, mesh, lambda n=n: chain(n, False), warm=False)
    st = C.stats(trn.replays(REPS))
    bm = trn.batch(REPS)
    trn.release()
    timing[n] = dict(median_us=st["median_us"], min_us=st["min_us"], batch_mean_us=round(bm, 1) if bm else None)
    log(f"N={n}: {timing[n]}")
SOAK_S = float(args.get("soak_s", 0))
soak = None
if SOAK_S > 0:  # back-to-back non-blocking replays of the NMAX chain for SOAK_S seconds (stability, sustained rate)
    trs = C.Traced(ttnn, mesh, lambda: chain(NMAX, False), warm=False)
    n_rep, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < SOAK_S:
        for _ in range(20):
            ttnn.execute_trace(mesh, trs.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        n_rep += 20
    dt_ = time.perf_counter() - t0
    trs.release()
    soak = dict(seconds=round(dt_, 1), replays=n_rep, us_per_chain=round(dt_ * 1e6 / n_rep, 1),
                us_per_layer=round(dt_ * 1e6 / n_rep / NMAX, 1))  # fmt: skip
    log("SOAK", soak)
pc1 = mesh.num_program_cache_entries()
try:
    from ttnn._ttnn.operations.trace import get_all_unsafe_tracked_ids

    unsafe = len(get_all_unsafe_tracked_ids())
except Exception as e:  # noqa: BLE001
    unsafe = f"n/a: {e}"
t1 = timing.get(1, {}).get("median_us")
steady = {n: round((v["median_us"] - t1) / (n - 1), 1) for n, v in timing.items() if n > 1 and t1}
per_launch_batch = {n: round(v["batch_mean_us"] / n, 1) for n, v in timing.items() if v.get("batch_mean_us")}
row = dict(test=f"m5a baseline: N sequential MkLayer launches (L0 copies) T={T}", status="info", milestone="M5a-base",
           traced=True, us=steady.get(max(NS)), T=T, ns=NS, timing=timing, steady_per_layer_us=steady,
           batch_per_launch_us=per_launch_batch, one_launch_us=t1, tag=tag, knobs=ENV, deterministic=det,
           finite_layers=sum(finite), program_cache_entries=[pc0, pc1], unsafe_allocations_live_at_replay=unsafe,
           ref=path, probe_layer0=probe, soak=soak, ring_slots={str(k): v for k, v in layers[0]["mk"].ring_slots.items()},
           note="one trace = N back-to-back launches of the single-layer megakernel, each copy its own weights / state; "
                "us = steady state per layer (T_N - T_1) / (N - 1) at the largest N")  # fmt: skip
if args.get("record"):
    C.record("perf", row)
else:
    import json

    with open(os.path.join(W, "dev_rows.jsonl"), "a") as f:
        f.write(json.dumps({"time_utc": C.now(), **row}, default=str) + "\n")
log("ROW", row)
RESULT = dict(timing=timing, steady=steady, det=det, finite=sum(finite), ref=path)
