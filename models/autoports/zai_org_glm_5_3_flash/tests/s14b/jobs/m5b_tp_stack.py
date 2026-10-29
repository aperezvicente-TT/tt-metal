# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd pair | 2x4): Stage 14b M5b piece — the dense_kda layers as ONE program per chip on the TP mesh
(tt/tp_stack.py TPStack: the layer loop + Stage 14's in-program fabric all-reduce after every o_proj / down, 2N per
launch) vs N sequential launches of Stage 14's single-layer TP program (tp_layer_launch), device vs device:

  stacks: "real" = the dense_kda layers of the model (ARGS layers, default L0, L1, L2: their own TP shards), "copies" =
      N resident copies of L0's shards (device clones: distinct weight / state addresses), N in ARGS ns;
  numerics: one traced call each side with every layer's state restored to the same random snapshot inside the trace;
      the stack's replicated residual after N layers vs the chain's last output on EVERY chip (bitwise), every layer's
      state / conv history per chip (bitwise), replicas identical across chips;
  timing: the stack launch alone and the N-launch chain alone (no restores; state drifts), blocking replays (median)
      and batch replays -> per layer.
ARGS: layers ([0, 1, 2]), ns ([8, 45]: copy stacks), t (1), reps (20), tag, record, rec_spread (1), env (knobs)."""
import copy
import gc
import importlib
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.fabric_ar as FAR  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.kda_pre as KPRE  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.kda_rec as KREC  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_dense as MKD  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_layer as MKL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_readers as MKR  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8 as TP  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer as TPL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp_stack as TPS  # noqa: E402

for mod in (KREC, KPRE, MKR, MKD, MKL, FAR, TP, TPL, MKS, TPS):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (  # noqa: E402
    FdTileCacheSource,
    OptimizedDecoderConfig,
)

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
# every knob a StackBackend profile sets (tt/stack_model.py KNOB_KEYS: the stack-vs-chain gate of a knob profile)
from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import KNOB_KEYS as KNOBS  # noqa: E402

for k_ in KNOBS:
    os.environ.pop(k_, None)
ENV = {k_: str(v_) for k_, v_ in (args.get("env") or {}).items()}
assert set(ENV) <= set(KNOBS), ENV
os.environ.update(ENV)
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2])]
NS = [int(n) for n in args.get("ns", [8, 45])]
T = int(args.get("t", 1))
REPS = int(args.get("reps", 20))
tag = args.get("tag", "m5b-tp-stack")
TPN = mesh.get_num_devices()
cfg = hf_config()


def log(*a):
    print(f"M5B {C.now()}", *a, flush=True)


def dup(t):
    return ttnn.to_device(ttnn.from_device(t), mesh, memory_config=t.memory_config())


def clone_mk(tp):
    """a device copy of TP layer `tp`'s weights (distinct addresses) as a new MkLayer on the same placement"""
    attn = copy.copy(tp.attn)
    for name in ("tap_cur", "taps_hist", "w_fb", "w_gb", "dt_bias", "decay_rate", "o_norm"):
        if isinstance(getattr(tp.attn, name, None), ttnn.Tensor):
            setattr(attn, name, dup(getattr(tp.attn, name)))
    lins = {}
    for role, ds in dict(kda_qkv=tp.attn.ds_qkv, kda_lowrank=tp.attn.ds_lowrank, kda_o=tp.attn.ds_o,
                         mlp_gate_up=tp.mlp.ds_gate_up, mlp_down=tp.mlp.ds_down).items():  # fmt: skip
        c = copy.copy(ds)
        c.weight = dup(ds.weight)
        lins[role] = c
    mk = MKL.MkLayer(mesh, tp.attn_args, tp.ffn_args, attn, lins, intermediate=tp.mlp.intermediate,
                     limit=tp.mlp.limit)  # fmt: skip
    # the hand-off buffers of L0's program (as MkStack shares them): N copies with their own would pin N sets of L1
    # under the reference chain's CB regions (2026-10-20: 45 copies at T=32 clashed with the single-layer program)
    mk.persist = tp.mk.persist
    return mk


# ------------------------------------------------------------------ TP layers (CTX)
key = ("m5b_tp", T, tuple(LAYERS), int(args.get("rec_spread", 1)), tuple(sorted(ENV.items())))
for k in [k for k in ctx if isinstance(k, tuple) and k[0] in ("m5b_tp", "tp") and k != key]:
    ctx.pop(k)  # one TP layer set at a time (Stage 14's "tp" layers too: L1 / DRAM)
gc.collect()
if key not in ctx:
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    t0 = time.time()
    tps = {L: TPL.TPDenseKDALayer(mesh, cfg, L, source, OptimizedDecoderConfig(), b=T, program="layer",
                                  rec_spread=int(args.get("rec_spread", 1)))
           for L in LAYERS}  # fmt: skip
    ctx[key] = dict(tps=tps, copies=[])
    log(f"TP layers {LAYERS} built in {time.time() - t0:.1f}s (tp {TPN}, lanes {tps[LAYERS[0]].lanes})")
st_ = ctx[key]
tps, copies = st_["tps"], st_["copies"]
fl = tps[LAYERS[0]]
while 1 + len(copies) < max(NS + [1]):
    copies.append(clone_mk(tps[0] if 0 in tps else fl))
sh = fl.shards

# ------------------------------------------------------------------ inputs
g = golden(0, 2048)["streams_in"].float()
x_rows = g[0, 300 : 300 + 37 * T : 37].reshape(T, 16384)
x_flat = fl.replicated(x_rows.reshape(1, 1, T, 16384))
x_stk = fl.replicated(x_rows.reshape((T, 1, 4, 4096) if T <= 4 else (1, 1, T, 16384)))  # t > 4: flat residual
act = fl.masks(torch.ones(T, dtype=torch.bool))
gen = torch.Generator().manual_seed(4321)
S0 = torch.randn(T, 64, 128, 128, generator=gen) * 0.1
C0 = (torch.randn(1, 3, T, 24576, generator=gen) * 0.5).to(torch.bfloat16)
S0m, C0m = sh.shard_state(S0, C0)


def states(n):
    sk = ("m5b_states", key, n)
    if sk not in ctx:
        ctx[sk] = [sh.shard_state(S0, C0) for _ in range(n)]
    return ctx[sk]


def restore(sts):
    for Sm, Cm in sts:
        ttnn.copy(S0m, Sm)
        ttnn.copy(C0m, Cm)


def chain(mks, sts, do_restore):
    if do_restore:
        restore(sts)
    y = x_flat
    for mk, (Sm, Cm) in zip(mks, sts):
        y = TPS.tp_layer_launch(fl, mk, y, Cm, act, Sm, t=T)["out"]
    return y


def per_dev(t, shape):
    return [ttnn.to_torch(d).float().reshape(shape) for d in ttnn.get_device_tensors(t)]


def run_case(name, mks):
    n = len(mks)
    sts = states(n)
    gc.collect()
    # reference: n sequential single-layer TP launches
    trc = C.Traced(ttnn, mesh, lambda: chain(mks, sts, True))
    trc.replay()
    want = [v.reshape(-1, 16384)[:T] for v in per_dev(trc.out, (-1, 16384))]
    want_st = [sh.gather_state(Sm, Cm) for Sm, Cm in sts]
    trc.release()
    tch = C.Traced(ttnn, mesh, lambda: chain(mks, sts, False), warm=False)
    chain_t = C.stats(tch.replays(REPS))
    chain_b = tch.batch(REPS)
    tch.release()
    # the stack: ONE program per chip
    sk = ("m5b_stack", key, name, n)
    for k in [k for k in ctx if isinstance(k, tuple) and k[0] == "m5b_stack" and k != sk]:
        ctx.pop(k)
    gc.collect()
    if sk not in ctx:
        ctx[sk] = TPS.TPStack(mesh, mks, fl, t=T, rev=args.get("rev"))
    stk = ctx[sk]

    def stack_call(do_restore):
        if do_restore:
            restore(sts)
            ttnn.copy(x_stk, stk.residual())
        return stk([Cm for _, Cm in sts], act, [Sm for Sm, _ in sts])

    trs = C.Traced(ttnn, mesh, lambda: stack_call(True))
    pc0 = mesh.num_program_cache_entries()
    trs.replay()
    got = [v.reshape(T, 16384) for v in per_dev(stk.residual(), (T, 16384))]
    got_st = [sh.gather_state(Sm, Cm) for Sm, Cm in sts]
    trs.replay()
    got2 = [v.reshape(T, 16384) for v in per_dev(stk.residual(), (T, 16384))]
    trs.release()
    trt = C.Traced(ttnn, mesh, lambda: stack_call(False), warm=False)
    stack_t = C.stats(trt.replays(REPS))
    stack_b = trt.batch(REPS)
    trt.release()
    pc1 = mesh.num_program_cache_entries()
    replicas = all(torch.equal(got[0], v) for v in got[1:])
    out_bitwise = all(torch.equal(a, b) for a, b in zip(got, want))
    s_bit = sum(bool(torch.equal(a[0], b[0])) for a, b in zip(got_st, want_st))
    c_bit = sum(bool(torch.equal(a[1], b[1])) for a, b in zip(got_st, want_st))
    det = all(torch.equal(a, b) for a, b in zip(got, got2))
    # bitwise equality alone passed NaN == NaN (2026-10-20 wm6: T=32 NaN on both sides) -> finite outputs required
    finite = all(bool(torch.isfinite(v).all()) for v in got + want)
    ok = replicas and out_bitwise and s_bit == n and c_bit == n and det and finite
    r = dict(test=f"m5b tp stack ({name}): {n} dense_kda layers in ONE program per chip vs {n} sequential TP launches "
                  f"T={T} tp{TPN}", status="pass" if ok else "fail", milestone="M5b", n=n, T=T, tp=TPN,
             mgd=CTX["mgd"], layers=LAYERS if name == "real" else f"{n} x L0 copies", out_bitwise=out_bitwise,  # noqa: F821
             replicas_identical=replicas, state_bitwise_layers=s_bit, conv_bitwise_layers=c_bit, deterministic=det,
             finite=finite, max_abs=float(max((a - b).abs().max() for a, b in zip(got, want))),
             pcc=C.pcc(got[0], want[0]),
             program_cache_entries=[pc0, pc1], tag=tag, info=stk.info)  # fmt: skip
    prow = dict(test=f"m5b tp stack ({name}) N={n} T={T} step", status="info", milestone="M5b", traced=True, n=n, T=T,
                tp=TPN, mgd=CTX["mgd"], us=stack_t["median_us"], per_layer_us=round(stack_t["median_us"] / n, 1),  # noqa: F821
                stack=stack_t, stack_batch_us=round(stack_b, 1) if stack_b else None, chain=chain_t,
                chain_batch_us=round(chain_b, 1) if chain_b else None,
                chain_per_layer_us=round(chain_t["median_us"] / n, 1), launches_per_step=1, tag=tag, knobs=ENV,
                note="stack = ONE program per chip per step (layer loop + 2N in-program all-reduces); chain = N "
                     "launches of the single-layer TP program in one trace")  # fmt: skip
    log("ROW", r)
    log("PERF", prow)
    if args.get("record"):
        C.record("pcc", r)
        C.record("perf", prow) if not os.environ.get("TT_METAL_WATCHER") else None
    return dict(ok=ok, per_layer=prow["per_layer_us"], chain_per_layer=prow["chain_per_layer_us"], out=out_bitwise,
                s=s_bit, c=c_bit, replicas=replicas)  # fmt: skip


res = {}
res["real"] = run_case("real", [tps[L].mk for L in LAYERS])
for n in sorted(NS, reverse=True):
    res[f"copies{n}"] = run_case("copies", ([(tps[0] if 0 in tps else fl).mk] + copies)[:n])
RESULT = res
