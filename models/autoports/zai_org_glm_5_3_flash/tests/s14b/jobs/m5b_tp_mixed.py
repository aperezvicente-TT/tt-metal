# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd 2x4): Stage 14b M5b — the REAL layer sequence across KDA kinds on the TP mesh: the dense_kda
layers (default L0-L2, Stage 14's TPDenseKDALayer shards) as ONE TP stack program per chip (tt/tp_stack.py TPStack), then
the sparse_kda layers (default L4-L6, Stage 14's TPSparseKDALayer: KDA heads TP8, routed experts EP8, the router / tail /
shared expert replicated) as ONE TP sparse stack program per chip (TPSparseStack: 2 in-program all-reduces per layer —
o_proj and the routed partial — and the chip's expert range), both on one replicated residual, in ONE trace; vs the same
layers as Stage 14's single-layer TP programs back to back (device vs device: the residual on every chip, every layer's
state / conv history, bitwise; replicas identical). Timing: both traced (no restores). The MoE weights are the
decoder's replicated tensors (one layer ~4.3 GB / chip: a few layers fit; the 45-layer image needs EP-sharded expert
weights, Stage 14). ARGS: dense ([0, 1, 2]), sparse ([4, 5, 6]), t (1), reps (20), rev, tag, record."""
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

import models.autoports.zai_org_glm_5_3_flash.tt.ep_experts as EPX  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer as TPL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8_sparse as TPSP  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp_stack as TPS  # noqa: E402

for mod in (M5, EPX, MKS, TPL, TPSP, TPS):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (  # noqa: E402
    FdTileCacheSource,
    OptimizedDecoderConfig,
)

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
DENSE = [int(v) for v in args.get("dense", [0, 1, 2])]
SPARSE = [int(v) for v in args.get("sparse", [4, 5, 6])]
T = int(args.get("t", 1))
REPS = int(args.get("reps", 20))
REV = args.get("rev")
EPW = bool(args.get("ep", False))  # EP-sharded expert weights (tt/ep_experts.py) instead of the replicated ones
tag = args.get("tag", "m5b-tp-mixed")
TPN = mesh.get_num_devices()
cfg = hf_config()
M5_DENSE_ENV = {"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3"}
M5_SPARSE_ENV = {"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3", "GLM53_MKX_RING": "3", "GLM53_MKX_INFLIGHT": "2"}
M5.set_env(M5_DENSE_ENV)


def log(*a):
    print(f"M5B-MIX {C.now()}", *a, flush=True)


key = ("m5b_tpmix", T, tuple(DENSE), tuple(SPARSE), EPW)
M5.prune_ctx(ctx, [key] + [k for k in ctx if isinstance(k, tuple) and k and k[0] == "m5b_moe"])
for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] in ("tp", "moe", "m5b_tp")]:
    ctx.pop(k)  # Stage 14's / the dense-only job's TP layers (L1 / DRAM)
gc.collect()
if key not in ctx:
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    t0 = time.time()
    dense = [TPL.TPDenseKDALayer(mesh, cfg, L, source, OptimizedDecoderConfig(), b=T, program="layer", rev=REV)
             for L in DENSE]  # fmt: skip
    sparse = []
    for L in SPARSE:
        mk_ = ("m5b_moe", L, EPW)
        if mk_ not in ctx:
            M5.set_env(M5_SPARSE_ENV)
            ctx[mk_] = (EPX.build_moe_ep if EPW else TPSP.build_moe)(mesh, cfg, source, L, OptimizedDecoderConfig())
        M5.set_env(M5_SPARSE_ENV)
        sparse.append(TPSP.TPSparseKDALayer(mesh, cfg, L, source, OptimizedDecoderConfig(), b=T, moe=ctx[mk_],
                                            rev=REV))  # fmt: skip
    M5.set_env(M5_DENSE_ENV)
    ctx[key] = dict(dense=dense, sparse=sparse)
    log(f"TP layers {DENSE} + {SPARSE} built in {time.time() - t0:.1f}s (tp {TPN}; lanes dense {dense[0].lanes}, "
        f"sparse {sparse[0].lanes})")  # fmt: skip
dense, sparse = ctx[key]["dense"], ctx[key]["sparse"]
fd, fs = dense[0], sparse[0]
sh = fd.shards
seq = [("d", lay) for lay in dense] + [("s", lay) for lay in sparse]
n = len(seq)

g = golden(DENSE[0], 2048)["streams_in"].float()
x_rows = g[0, 300 : 300 + 37 * T : 37].reshape(T, 16384)
x_flat = fd.replicated(x_rows.reshape(1, 1, T, 16384))
x_stk = fd.replicated(x_rows.reshape((T, 1, 4, 4096) if T <= 4 else (1, 1, T, 16384)))  # t > 4: flat residual
act = fd.masks(torch.ones(T, dtype=torch.bool))
gen = torch.Generator().manual_seed(4321)
S0 = torch.randn(T, 64, 128, 128, generator=gen) * 0.1
C0 = (torch.randn(1, 3, T, 24576, generator=gen) * 0.5).to(torch.bfloat16)
S0m, C0m = sh.shard_state(S0, C0)
sk = ("m5b_tpmix_states", key)
if sk not in ctx:
    ctx[sk] = [sh.shard_state(S0, C0) for _ in seq]  # every KDA layer shards its heads / state the same way
sts = ctx[sk]


def restore():
    for Sm, Cm in sts:
        ttnn.copy(S0m, Sm)
        ttnn.copy(C0m, Cm)


def chain(do_restore):
    if do_restore:
        restore()
    y = x_flat
    for (kind, lay), (Sm, Cm) in zip(seq, sts):
        if kind == "d":
            y = TPS.tp_layer_launch(fd, lay.mk, y, Cm, act, Sm, t=T)["out"]
        else:
            y = lay.run_layer(y, Cm, act, Sm, t=T)["out"]
    return y


def per_dev(t, shape):
    return [ttnn.to_torch(d).float().reshape(shape) for d in ttnn.get_device_tensors(t)]


trc = C.Traced(ttnn, mesh, lambda: chain(True))
trc.replay()
want = [v.reshape(-1, 16384)[:T] for v in per_dev(trc.out, (-1, 16384))]
want_st = [sh.gather_state(Sm, Cm) for Sm, Cm in sts]
trc.release()
tch = C.Traced(ttnn, mesh, lambda: chain(False), warm=False)
chain_t, chain_b = C.stats(tch.replays(REPS)), tch.batch(REPS)
tch.release()
trc = tch = None
gc.collect()
stk_key = ("m5b_tpmix_stack", key, REV)
for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] in ("m5b_tpmix_stack", "m5b_stack") and k != stk_key]:
    ctx.pop(k)
gc.collect()
if stk_key not in ctx:
    sd = TPS.TPStack(mesh, [lay.mk for lay in dense], fd, t=T, rev=REV)
    ss = TPS.TPSparseStack(mesh, [lay.mk for lay in sparse], fs, t=T, rev=REV, res=sd.residual(), pool=sd.pool)
    ctx[stk_key] = (sd, ss)
sd, ss = ctx[stk_key]
nd = len(dense)


def stack_call(do_restore):
    if do_restore:
        restore()
        ttnn.copy(x_stk, sd.residual())
    sd([Cm for _, Cm in sts[:nd]], act, [Sm for Sm, _ in sts[:nd]])
    ss([Cm for _, Cm in sts[nd:]], act, [Sm for Sm, _ in sts[nd:]])
    return sd.residual()


trs = C.Traced(ttnn, mesh, lambda: stack_call(True))
trs.replay()
got = [v.reshape(T, 16384) for v in per_dev(sd.residual(), (T, 16384))]
got_st = [sh.gather_state(Sm, Cm) for Sm, Cm in sts]
trs.replay()
got2 = [v.reshape(T, 16384) for v in per_dev(sd.residual(), (T, 16384))]
trs.release()
trt = C.Traced(ttnn, mesh, lambda: stack_call(False), warm=False)
stack_t, stack_b = C.stats(trt.replays(REPS)), trt.batch(REPS)
trt.release()
replicas = all(torch.equal(got[0], v) for v in got[1:])
out_bitwise = all(torch.equal(a, b) for a, b in zip(got, want))
s_bit = sum(bool(torch.equal(a[0], b[0])) for a, b in zip(got_st, want_st))
c_bit = sum(bool(torch.equal(a[1], b[1])) for a, b in zip(got_st, want_st))
det = all(torch.equal(a, b) for a, b in zip(got, got2))
ok = replicas and out_bitwise and s_bit == n and c_bit == n and det
names = [f"L{L}d" for L in DENSE] + [f"L{L}s" for L in SPARSE]
r = dict(test=f"m5b tp stack, real mixed-kind sequence {names} as 2 TP stack programs per chip vs {n} sequential TP "
              f"launches T={T} tp{TPN}", status="pass" if ok else "fail", milestone="M5b", n=n, T=T, tp=TPN,
         mgd=CTX["mgd"], sequence=names, out_bitwise=out_bitwise, replicas_identical=replicas,  # noqa: F821
         state_bitwise_layers=s_bit, conv_bitwise_layers=c_bit, deterministic=det,
         max_abs=float(max((a - b).abs().max() for a, b in zip(got, want))), tag=tag,
         info=[sd.info, ss.info])  # fmt: skip
prow = dict(test=f"m5b tp stack mixed {names} T={T} tp{TPN} step", status="info", milestone="M5b", traced=True, n=n,
            T=T, tp=TPN, us=stack_t["median_us"], per_layer_us=round(stack_t["median_us"] / n, 1), stack=stack_t,
            stack_batch_us=round(stack_b, 1) if stack_b else None, chain=chain_t,
            chain_per_layer_us=round(chain_t["median_us"] / n, 1), chain_batch_us=round(chain_b, 1) if chain_b else None,
            launches_per_step=2, tag=tag,
            note="2 TP stack programs per chip (dense run, sparse run; 2 in-program all-reduces per layer) in one trace "
                 "vs one Stage 14 TP program per layer")  # fmt: skip
log("ROW", r)
log("PERF", prow)
if args.get("record"):
    C.record("pcc", r)
    C.record("perf", prow) if not os.environ.get("TT_METAL_WATCHER") else None
RESULT = dict(ok=ok, us=stack_t["median_us"], chain_us=chain_t["median_us"])
