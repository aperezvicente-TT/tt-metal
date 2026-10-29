# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd single, chip 3): Stage 14b — the REAL layer sequence across kinds on one chip: the dense_kda
layers (default L0, L1, L2: 7a's MkLayer) then sparse_kda layers (default L4, L5, L6: 7a's MkSparseLayer, router +
routed experts + shared expert in the program), every layer its own weights / KDA state / mHC parameters / router
bias, as ONE stack program per kind run (tt/mk_stack.py; the runs share one stacked residual: 2 launches per step in
ONE trace) vs the same sequence as single-layer launches in one trace, device vs device: the residual after the last
layer, every layer's state / conv history (bitwise); traced timing of both. ARGS: dense ([0, 1, 2]), sparse ([4, 5, 6]),
t (1), reps (20), tag, record, rev, env, sparse_env (GLM53_MK_* for the sparse layers' construction)."""
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
import models.autoports.zai_org_glm_5_3_flash.tt.mk_experts as MKX  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_layer as MKL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_readers as MKR  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_sparse as MKSP  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402

for mod in (M5, KREC, KPRE, MKR, MKD, MKX, MKL, MKSP, MKS):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_kda_pre import masks  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import cached_decoder  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
KNOBS = ("GLM53_MK_KB_QKV", "GLM53_MK_KB_O", "GLM53_MK_RING_KB_QKV", "GLM53_MK_RING_KB_O", "GLM53_MK_RING_KB",
         "GLM53_MK_OPT", "GLM53_MK_READERS", "GLM53_MK_KB_ROUTER", "GLM53_MK_KB_SGU", "GLM53_MKX_RING",
         "GLM53_MKX_MBUF", "GLM53_MKX_INFLIGHT")  # fmt: skip
DENSE_KNOBS = {"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3", "GLM53_MK_RING_KB_QKV": "750"}
SPARSE_KNOBS = {"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3", "GLM53_MKX_INFLIGHT": "3", "GLM53_MKX_RING": "4"}


def set_env(env):
    for k_ in KNOBS:
        os.environ.pop(k_, None)
    assert set(env) <= set(KNOBS), env
    os.environ.update(env)


DENSE = [int(v) for v in args.get("dense", [0, 1, 2])]
SPARSE = [int(v) for v in args.get("sparse", [4, 5, 6])]
T = int(args.get("t", 1))
REPS = int(args.get("reps", 20))
tag = args.get("tag", "m5a-mixed")
ENV_D = {**DENSE_KNOBS, **{k: str(v) for k, v in (args.get("env") or {}).items()}}
ENV_S = {**SPARSE_KNOBS, **{k: str(v) for k, v in (args.get("sparse_env") or {}).items()}}
cfg = hf_config()


def log(*a):
    print(f"M5A-MIXED {C.now()}", *a, flush=True)


key = ("m5a_mixed", T, tuple(DENSE), tuple(SPARSE), tuple(sorted(ENV_D.items())), tuple(sorted(ENV_S.items())))
M5.prune_ctx(ctx, [key])
gc.collect()
log("dropped cached decoders", M5.keep_decoders(ctx, DENSE + SPARSE))
log("L1 at start", M5.l1_state(mesh))
if key not in ctx:
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    t0 = time.time()
    seq = []
    for L in DENSE + SPARSE:
        dec = cached_decoder(cfg, L, mesh, ctx)
        kda, mlp = dec.layer.attn, dec.layer.mlp
        lins = {d.role: d for d in dec.layer.ds_linears()}
        a_args, f_args = M5.mhc_args(source, cfg, L, "attn"), M5.mhc_args(source, cfg, L, "ffn")
        if L in DENSE:
            set_env(ENV_D)
            mk = MKL.MkLayer(mesh, a_args, f_args, kda, {r: lins[r] for r in ("kda_qkv", "kda_lowrank", "kda_o",
                                                                               "mlp_gate_up", "mlp_down")},
                             intermediate=mlp.intermediate, limit=mlp.limit)  # fmt: skip
        else:
            set_env(ENV_S)
            mk = MKSP.MkSparseLayer(mesh, a_args, f_args, kda, {r: lins[r] for r in ("kda_qkv", "kda_lowrank",
                                                                                     "kda_o")}, mlp)  # fmt: skip
        seq.append(dict(L=L, mk=mk, sparse=L in SPARSE, kda=kda))
    for kind in (False, True):  # one set of persistent hand-off buffers (on, m) per kind, also for the chain
        same = [e["mk"] for e in seq if e["sparse"] == kind]
        for mk_ in same[1:]:
            mk_.persist = same[0].persist
    ctx[key] = seq
    log(f"layers {DENSE} + {SPARSE} ready in {time.time() - t0:.1f}s; L1", M5.l1_state(mesh))
seq = ctx[key]
kda0 = seq[0]["kda"]
g = golden(seq[0]["L"], 2048)["streams_in"].float()
x_rows = g[0, 300 : 300 + 37 * T : 37].reshape(T, 16384)
mk_t = lambda v, sh, dt=ttnn.float32: ttnn.from_torch(v.reshape(sh), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
x_flat, x_stk = mk_t(x_rows, (1, 1, T, 16384)), mk_t(x_rows, (T, 1, 4, 4096))
act = masks(mesh, torch.tensor([(i % 3) != 2 for i in range(T)]) if T > 1 else torch.tensor([True]), kda0.lower_bound)
gen = torch.Generator().manual_seed(4242)
hist0 = mk_t((torch.randn(1, 3, T, 24576, generator=gen) * 0.5).to(torch.bfloat16), (1, 3, T, 24576), ttnn.bfloat16)
s0 = mk_t(torch.randn(T, 64, 128, 128, generator=gen) * 0.1, (T, 64, 128, 128))
sts = [kda0.allocate_state(T) for _ in seq]


def restore():
    for st in sts:
        ttnn.copy(hist0, st.C)
        ttnn.copy(s0, st.S)


def chain(do_restore):
    if do_restore:
        restore()
    y = x_flat
    for e, st in zip(seq, sts):
        set_env(ENV_S if e["sparse"] else ENV_D)
        y = e["mk"](y, st.C, act, t=T, state=st.S)["out"]
    return y


trc = C.Traced(ttnn, mesh, lambda: chain(True))
trc.replay()
want = ttnn.to_torch(trc.out).float().reshape(-1, 16384)[:T]
want_s = [ttnn.to_torch(st.S).float() for st in sts]
want_c = [ttnn.to_torch(st.C) for st in sts]
trc.release()
tch = C.Traced(ttnn, mesh, lambda: chain(False), warm=False)
ch_t, ch_b = C.stats(tch.replays(REPS)), tch.batch(REPS)
tch.release()
trc = tch = None
gc.collect()
# the stacks: one per kind run, sharing one residual
skey = ("m5a_mixed_stack", key, args.get("rev"))
if skey not in ctx:
    runs, cur = [], []
    for i, e in enumerate(seq):
        if cur and seq[cur[-1]]["sparse"] != e["sparse"]:
            runs.append(cur)
            cur = []
        cur.append(i)
    runs.append(cur)
    res = None
    stacks = []
    for r in runs:
        set_env(ENV_S if seq[r[0]]["sparse"] else ENV_D)
        stk = MKS.MkStack(mesh, [seq[i]["mk"] for i in r], t=T, rev=args.get("rev"), res=res,
                          pool=stacks[0][1].pool if stacks else None)  # fmt: skip
        res = stk.residual()
        stacks.append((r, stk))
    ctx[skey] = stacks
stacks = ctx[skey]


def stack_step(do_restore):
    if do_restore:
        restore()
        ttnn.copy(x_stk, stacks[0][1].residual())
    for r, stk in stacks:
        set_env(ENV_S if seq[r[0]]["sparse"] else ENV_D)
        stk([sts[i].C for i in r], act, [sts[i].S for i in r])
    return stacks[0][1].residual()


trs = C.Traced(ttnn, mesh, lambda: stack_step(True))
trs.replay()
got = ttnn.to_torch(stacks[0][1].residual()).float().reshape(T, 16384)
got_s = [ttnn.to_torch(st.S).float() for st in sts]
got_c = [ttnn.to_torch(st.C) for st in sts]
trs.release()
trt = C.Traced(ttnn, mesh, lambda: stack_step(False), warm=False)
st_t, st_b = C.stats(trt.replays(REPS)), trt.batch(REPS)
trt.release()
n = len(seq)
out_bit = bool(torch.equal(got, want))
s_bit = sum(bool(torch.equal(a, b)) for a, b in zip(got_s, want_s))
c_bit = sum(bool(torch.equal(a, b)) for a, b in zip(got_c, want_c))
ok = out_bit and s_bit == n and c_bit == n
names = [f"L{e['L']}{'s' if e['sparse'] else 'd'}" for e in seq]
r = dict(test=f"m5a stack, real mixed-kind sequence {names} as {len(stacks)} stack programs vs {n} single-layer "
              f"launches T={T}", status="pass" if ok else "fail", milestone="M5a", n=n, T=T, sequence=names,
         programs_per_step=len(stacks), out_bitwise=out_bit, state_bitwise_layers=s_bit, conv_bitwise_layers=c_bit,
         max_abs=float((got - want).abs().max()), pcc=C.pcc(got, want), tag=tag,
         info=[stk.info for _, stk in stacks])  # fmt: skip
prow = dict(test=f"m5a stack mixed {names} T={T} step", status="info", milestone="M5a", traced=True, n=n, T=T,
            us=st_t["median_us"], per_layer_us=round(st_t["median_us"] / n, 1), stack=st_t,
            stack_batch_us=round(st_b, 1) if st_b else None, chain=ch_t,
            chain_per_layer_us=round(ch_t["median_us"] / n, 1), chain_batch_us=round(ch_b, 1) if ch_b else None,
            launches_per_step=len(stacks), tag=tag, knobs=dict(dense=ENV_D, sparse=ENV_S),
            note="one stack program per kind run (shared stacked residual), one trace; chain = one launch per layer")  # fmt: skip
log("ROW", r)
log("PERF", prow)
if args.get("record"):
    C.record("pcc", r)
    C.record("perf", prow) if not os.environ.get("TT_METAL_WATCHER") else None
RESULT = dict(ok=ok, per_layer=prow["per_layer_us"], chain_per_layer=prow["chain_per_layer_us"], s=s_bit, c=c_bit)
