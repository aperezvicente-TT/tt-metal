# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd single, chip 3): Stage 14b M5a — N resident copies of L0 (dense_kda) as ONE program (tt/mk_stack.py
MkStack: the layer loop inside the kernels, per-layer weight / state bases from a DRAM argument table, the residual kept
in the stacked L1 layout between layers, layer l + 1's weights streamed while layer l's tail runs).

  numerics (device vs device): the stack with every layer's state restored to the snapshot of jobs/m5a_chain.py and
      its input written into the residual INSIDE the traced call -> the residual after N layers and every layer's
      state / conv history, BITWISE vs the N sequential MkLayer launches (work-s14b/m5a_ref/chain_T{t}_N45_{ref}.pt;
      layer i's output there = the stack's residual after i + 1 layers);
  timing: the stack launch alone captured (state drifts; timing only), blocking replays -> median us / N per layer;
      batch (non-blocking back to back) mean;
  probe (optional): per-layer marks of every RISC (kernels/mk/probe.hpp, rows 256 l ..) on a COMMON time base ->
      the weight streamer's layer l + 1 start vs the owners' layer l tail (the overlap).

ARGS: ns ([2, 4, 8, 45]), t (1), reps (30), ref (the chain reference tag), tag, record, env (megakernel knobs; 7a's M1
set by default), probe (False), rev (kernel revision define), soak_s (0)."""
import gc
import importlib
import os
import statistics
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
import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402

for mod in (KREC, KPRE, MKR, MKD, MKL, MKS):
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
M1_KNOBS = {"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3"}
for k_ in KNOBS:
    os.environ.pop(k_, None)
ENV = {**M1_KNOBS, **{k_: str(v_) for k_, v_ in (args.get("env") or {}).items()}}
assert set(ENV) <= set(KNOBS), ENV
os.environ.update(ENV)
NS = [int(n) for n in args.get("ns", [2, 4, 8, 45])]
NMAX = max(NS)
T = int(args.get("t", 1))
REPS = int(args.get("reps", 30))
PROBE = bool(args.get("probe", False))
tag = args.get("tag", "m5a-stack")
REF = args.get("ref", f"m5a-base-m1-T{T}")
W = os.environ.get("GLM53_WORK", "/opt/glm53/work-s14b")
LAYER = 0
MHZ = 1350.0
cfg = hf_config()


def log(*a):
    print(f"M5A-STACK {C.now()}", *a, flush=True)


def dup(t):
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


# ------------------------------------------------------------------ N layer copies (CTX; shared with m5a_chain.py)
dec = cached_decoder(cfg, LAYER, mesh, ctx)
kda0, mlp0 = dec.layer.attn, dec.layer.mlp
lins0 = {d.role: d for d in dec.layer.ds_linears()}
ROLES = ("kda_qkv", "kda_lowrank", "kda_o", "mlp_gate_up", "mlp_down")
key = ("m5a_layers", T, tuple(sorted(ENV.items())), PROBE)
M5.prune_ctx(ctx, [key])
gc.collect()
M5.keep_decoders(ctx, [LAYER])
layers = ctx.setdefault(key, [])
if len(layers) < NMAX:
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    a_args, f_args = mhc_args(source, cfg), ffn_mhc_args(source, cfg)
    t0 = time.time()
    while len(layers) < NMAX:
        first = not layers
        kda_i = kda0 if first else clone_kda(kda0)
        lins_i = {r: (lins0[r] if first else clone_lin(lins0[r])) for r in ROLES}
        mk = MKL.MkLayer(mesh, a_args, f_args, kda_i, lins_i, intermediate=mlp0.intermediate, limit=mlp0.limit,
                         probe=PROBE and first)  # fmt: skip
        if not first:
            mk.persist = layers[0]["mk"].persist
        layers.append(dict(mk=mk, kda=kda_i, st=kda0.allocate_state(T)))
    log(f"{NMAX} layer copies ready in {time.time() - t0:.1f}s")

# ------------------------------------------------------------------ inputs and the chain reference
ref_path = os.path.join(W, "m5a_ref", f"chain_T{T}_N45_{REF}.pt")
ref = torch.load(ref_path)
assert ref["T"] == T and ref["n"] >= NMAX, (ref["T"], ref["n"])
x_flat = ref["x"].reshape(T, 16384).float()
x_stk = ttnn.from_torch(x_flat.reshape(T, 1, 4, 4096), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
active = torch.tensor([(i % 3) != 2 for i in range(T)]) if T > 1 else torch.tensor([True])
act = masks(mesh, active, kda0.lower_bound)
gen = torch.Generator().manual_seed(1234)  # the chain job's snapshot (same generator, same order)
hist0 = ttnn.from_torch((torch.randn(1, 3, T, 24576, generator=gen) * 0.5).to(torch.bfloat16), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
s0 = ttnn.from_torch(torch.randn(T, 64, 128, 128, generator=gen) * 0.1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                     device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip


def l1_state():
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    blocks = [b for b in mv.block_table if str(b.get("allocated", "")).lower() in ("yes", "true", "1")]
    return dict(allocated_per_bank=int(mv.total_bytes_allocated_per_bank),
                lowest=min((int(b["address"]) for b in blocks), default=None), n_blocks=len(blocks))  # fmt: skip


log("L1 at start", l1_state())


def stack_for(n):
    skey = ("m5a_stack", key, n, args.get("rev"))
    for k in [k for k in ctx if isinstance(k, tuple) and k[0] == "m5a_stack" and k != skey]:
        ctx.pop(k)  # one stack (its shared intermediates are L1) at a time
    gc.collect()
    if skey not in ctx:
        ctx[skey] = MKS.MkStack(mesh, [L["mk"] for L in layers[:n]], t=T, probe=PROBE, rev=args.get("rev"))
    return ctx[skey]


def run(st, n, restore):
    if restore:
        for L in layers[:n]:
            ttnn.copy(hist0, L["st"].C)
            ttnn.copy(s0, L["st"].S)
        ttnn.copy(x_stk, st.residual())
    return st([L["st"].C for L in layers[:n]], act, [L["st"].S for L in layers[:n]])


def probe_timeline(st, n):
    """per layer: median over the cores of each role of the raw marks, us on the time base of layer 0's first mark"""
    L0 = st.layers[0]
    p = ttnn.to_torch(st.probe).reshape(n, L0.PROBE_ROWS, 16).long() & 0xFFFFFFFF
    nz = p[p > 0]
    if not nz.numel():
        return None
    t0 = int(nz.min())

    def us(v):
        d = (int(v) - t0) & 0xFFFFFFFF
        return round((d - (1 << 32) if d >= (1 << 31) else d) / MHZ, 1)

    def med(lp, rows):
        cols = [[int(v) for v in lp[r] if int(v) > 0] for r in rows]
        m = min((len(c) for c in cols), default=0)
        return [us(statistics.median(c[i] for c in cols)) for i in range(m)]

    kc = len(L0.place["k"])
    assign = L0.readers.assign
    out = []
    for l in range(n):
        lp = p[l]
        d = dict(owner_brisc=med(lp, range(kc)), owner_ncrisc=med(lp, [96 + k for k in range(kc)]))
        idx0 = [i for i, (_, _, ix) in enumerate(assign) if ix == 0]
        d["reader0_w"] = med(lp, [160 + 2 * i for i in idx0])
        d["reader0_a"] = med(lp, [161 + 2 * i for i in idx0])
        out.append(d)
    return out


res = {}
ok_all = True
st = tr = trt = None
for n in sorted(NS, reverse=True):
    st = tr = trt = None  # drop the previous stack (and the traces' references to its residual) before the next build
    gc.collect()
    st = stack_for(n)
    log(f"N={n}: L1 before the build", l1_state())
    tr = C.Traced(ttnn, mesh, lambda st=st, n=n: run(st, n, True))  # compile launch, then the capture
    pc0 = mesh.num_program_cache_entries()
    log(f"N={n}: L1 after the build", l1_state(), st.info)
    tr.replay()
    got = ttnn.to_torch(st.residual()).float().reshape(T, 16384)
    S = [ttnn.to_torch(L["st"].S).float() for L in layers[:n]]
    Cs = [ttnn.to_torch(L["st"].C) for L in layers[:n]]
    timeline = probe_timeline(st, n) if PROBE else None
    tr.replay()
    got2 = ttnn.to_torch(st.residual()).float().reshape(T, 16384)
    tr.release()
    want = ref["outs"][n - 1].reshape(T, 16384)
    out_bitwise = bool(torch.equal(got, want))
    s_bitwise = [bool(torch.equal(S[i], ref["S"][i])) for i in range(n)]
    c_bitwise = [bool(torch.equal(Cs[i], ref["C"][i])) for i in range(n)]
    det = bool(torch.equal(got, got2))
    max_abs = float((got - want).abs().max())
    p_out = C.pcc(got, want)
    # timing: the stack launch alone
    trt = C.Traced(ttnn, mesh, lambda st=st, n=n: run(st, n, False), warm=False)
    walls = trt.replays(REPS)
    bm = trt.batch(REPS)
    soak = None
    if float(args.get("soak_s", 0)) > 0 and n == NMAX:
        n_rep, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < float(args["soak_s"]):
            for _ in range(20):
                ttnn.execute_trace(mesh, trt.tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            n_rep += 20
        dt_ = time.perf_counter() - t0
        soak = dict(seconds=round(dt_, 1), replays=n_rep, us_per_layer=round(dt_ * 1e6 / n_rep / n, 1))
    trt.release()
    pc1 = mesh.num_program_cache_entries()
    try:
        from ttnn._ttnn.operations.trace import get_all_unsafe_tracked_ids

        unsafe = len(get_all_unsafe_tracked_ids())
    except Exception as e:  # noqa: BLE001
        unsafe = f"n/a: {e}"
    stt = C.stats(walls)
    ok = out_bitwise and all(s_bitwise) and all(c_bitwise) and det
    ok_all &= ok
    r = dict(test=f"m5a stack: {n} L0 copies in ONE program vs {n} sequential launches T={T}",
             status="pass" if ok else "fail", milestone="M5a", n=n, T=T, out_bitwise=out_bitwise,
             state_bitwise=all(s_bitwise), conv_bitwise=all(c_bitwise), state_bitwise_layers=sum(s_bitwise),
             conv_bitwise_layers=sum(c_bitwise), deterministic=det, max_abs=max_abs, pcc=p_out, ref=ref_path,
             program_cache_entries=[pc0, pc1], unsafe_allocations_live_at_replay=unsafe, tag=tag, info=st.info)  # fmt: skip
    prow = dict(test=f"m5a stack N={n} T={T} step", status="info", milestone="M5a", traced=True, n=n, T=T,
                us=stt["median_us"], per_layer_us=round(stt["median_us"] / n, 1), replay=stt,
                batch_mean_us=round(bm, 1) if bm else None,
                batch_per_layer_us=round(bm / n, 1) if bm else None, soak=soak, launches_per_step=1, tag=tag,
                knobs=ENV, probe=PROBE, ring_slots={str(k_): v_ for k_, v_ in st.layers[0].ring_slots.items()},
                note="ONE program per step: the layer loop inside the kernels (tt/mk_stack.py); blocking replay of "
                     "one traced launch; per_layer = median / N")  # fmt: skip
    if timeline is not None:
        prow["timeline"] = timeline
    log("ROW", r)
    log("PERF", prow)
    if args.get("record"):
        C.record("pcc", r)
        if not PROBE:
            C.record("perf", prow) if not os.environ.get("TT_METAL_WATCHER") else None
        else:
            C.record("tests", dict(prow, status="info"))
    res[n] = dict(ok=ok, us=stt["median_us"], per_layer=round(stt["median_us"] / n, 1), out_bitwise=out_bitwise,
                  s=sum(s_bitwise), c=sum(c_bitwise), max_abs=max_abs, soak=soak)  # fmt: skip
RESULT = res
