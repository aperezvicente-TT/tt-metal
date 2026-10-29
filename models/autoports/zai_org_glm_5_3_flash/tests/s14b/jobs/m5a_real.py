# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd single, chip 3): Stage 14b — the stack program over the REAL dense_kda layer sequence (each layer
its own weights, KDA tensors and mHC parameters from the 00c cache: L0, L1, L2 by default, optionally repeated R times
to a 45-equivalent length) as ONE program (tt/mk_stack.py) vs the same sequence as N sequential MkLayer launches in one
trace, device vs device, bitwise (residual after N layers, every layer's state / conv history); traced timing of both.
ARGS: layers ([0, 1, 2]), repeat ([1, 15]), t (1), reps (20), tag, record, env (megakernel knobs; 7a's M1 by default)."""
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
import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402

for mod in (M5, KREC, KPRE, MKR, MKD, MKL, MKS):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_kda_pre import masks  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import cached_decoder  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
KNOBS = ("GLM53_MK_KB_QKV", "GLM53_MK_KB_O", "GLM53_MK_RING_KB_QKV", "GLM53_MK_RING_KB_O", "GLM53_MK_RING_KB",
         "GLM53_MK_OPT", "GLM53_MK_READERS")  # fmt: skip
M1_KNOBS = {"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3", "GLM53_MK_RING_KB_QKV": "750"}
for k_ in KNOBS:
    os.environ.pop(k_, None)
ENV = {**M1_KNOBS, **{k_: str(v_) for k_, v_ in (args.get("env") or {}).items()}}
assert set(ENV) <= set(KNOBS), ENV
os.environ.update(ENV)
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2])]
REPEATS = [int(v) for v in args.get("repeat", [1, 15])]
T = int(args.get("t", 1))
REPS = int(args.get("reps", 20))
tag = args.get("tag", "m5a-real")
cfg = hf_config()


def log(*a):
    print(f"M5A-REAL {C.now()}", *a, flush=True)


key = ("m5a_real", T, tuple(LAYERS), tuple(sorted(ENV.items())))
M5.prune_ctx(ctx, [key])
gc.collect()
log("dropped cached decoders", M5.keep_decoders(ctx, LAYERS))
log("L1 at start", M5.l1_state(mesh))
if key not in ctx:
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    t0 = time.time()
    mks = {}
    for L in LAYERS:
        dec = cached_decoder(cfg, L, mesh, ctx)
        kda, mlp = dec.layer.attn, dec.layer.mlp
        lins = {d.role: d for d in dec.layer.ds_linears()}
        mks[L] = MKL.MkLayer(mesh, M5.mhc_args(source, cfg, L, "attn"), M5.mhc_args(source, cfg, L, "ffn"), kda,
                             {r: lins[r] for r in ("kda_qkv", "kda_lowrank", "kda_o", "mlp_gate_up", "mlp_down")},
                             intermediate=mlp.intermediate, limit=mlp.limit)  # fmt: skip
    for L in LAYERS[1:]:
        mks[L].persist = mks[LAYERS[0]].persist
    ctx[key] = dict(mks=mks, kda=cached_decoder(cfg, LAYERS[0], mesh, ctx).layer.attn)
    log(f"real layers {LAYERS} ready in {time.time() - t0:.1f}s")
mks, kda0 = ctx[key]["mks"], ctx[key]["kda"]

B = int(args.get("b", T))  # users: B > 4 = B / 4 row chunks of t = 4, each its own stack program (DRAM residual)
CH = min(B, 4) if B > 4 else T
NCH = max(1, B // CH)
assert CH * NCH == B or B == T, (B, CH)
SOAK_S = float(args.get("soak_s", 0))
g = golden(LAYERS[0], 2048)["streams_in"].float()
mk_t = lambda v, sh, dt=ttnn.float32: ttnn.from_torch(v.reshape(sh), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
xs = [g[0, 300 + 7 * c : 300 + 7 * c + 37 * CH : 37].reshape(CH, 16384) for c in range(NCH)]
x_flat = [mk_t(x, (1, 1, CH, 16384)) for x in xs]
x_stk = [mk_t(x, (CH, 1, 4, 4096)) for x in xs]
act = masks(mesh, torch.tensor([(i % 3) != 2 for i in range(CH)]) if CH > 1 else torch.tensor([True]),
            kda0.lower_bound)  # fmt: skip
gen = torch.Generator().manual_seed(777)
hist0 = mk_t((torch.randn(1, 3, CH, 24576, generator=gen) * 0.5).to(torch.bfloat16), (1, 3, CH, 24576), ttnn.bfloat16)
s0 = mk_t(torch.randn(CH, 64, 128, 128, generator=gen) * 0.1, (CH, 64, 128, 128))


def run_case(rep):
    for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] == "m5a_real_stack"]:
        ctx.pop(k)  # the previous case's stacks (their intermediates are L1) before this case's chain allocates its own
    gc.collect()
    seq = [mks[L] for L in LAYERS] * rep
    n = len(seq)
    sts = [[kda0.allocate_state(CH) for _ in range(n)] for _ in range(NCH)]

    def restore():
        for chunk in sts:
            for st in chunk:
                ttnn.copy(hist0, st.C)
                ttnn.copy(s0, st.S)

    def chain(do_restore):
        if do_restore:
            restore()
        outs = []
        for c in range(NCH):
            y = x_flat[c]
            for mk, st in zip(seq, sts[c]):
                y = mk(y, st.C, act, t=CH, state=st.S)["out"]
            outs.append(y)
        return outs

    gc.collect()
    trc = C.Traced(ttnn, mesh, lambda: chain(True))
    trc.replay()
    want = [ttnn.to_torch(o).float().reshape(-1, 16384)[:CH] for o in trc.out]
    want_s = [[ttnn.to_torch(st.S).float() for st in chunk] for chunk in sts]
    want_c = [[ttnn.to_torch(st.C) for st in chunk] for chunk in sts]
    trc.release()
    tch = C.Traced(ttnn, mesh, lambda: chain(False), warm=False)
    ch_t, ch_b = C.stats(tch.replays(REPS)), tch.batch(REPS)
    tch.release()
    trc = tch = None
    gc.collect()
    stacks = []
    for c in range(NCH):  # chunk 0 records the intermediates, the others reuse them; DRAM residual per chunk if B > 4
        res = mk_t(xs[c], (CH, 1, 4, 4096)) if NCH > 1 else None
        stacks.append(MKS.MkStack(mesh, seq, t=CH, rev=args.get("rev"), res=res,
                                  pool=stacks[0].pool if stacks else None))  # fmt: skip
    ctx[("m5a_real_stack", key, rep)] = stacks

    def stack(do_restore):
        if do_restore:
            restore()
            for c in range(NCH):
                ttnn.copy(x_stk[c], stacks[c].residual())
        for c in range(NCH):
            stacks[c]([st.C for st in sts[c]], act, [st.S for st in sts[c]])
        return [stk.residual() for stk in stacks]

    trs = C.Traced(ttnn, mesh, lambda: stack(True))
    trs.replay()
    got = [ttnn.to_torch(stk.residual()).float().reshape(CH, 16384) for stk in stacks]
    got_s = [[ttnn.to_torch(st.S).float() for st in chunk] for chunk in sts]
    got_c = [[ttnn.to_torch(st.C) for st in chunk] for chunk in sts]
    trs.release()
    trt = C.Traced(ttnn, mesh, lambda: stack(False), warm=False)
    st_t, st_b = C.stats(trt.replays(REPS)), trt.batch(REPS)
    soak = None
    if SOAK_S > 0:
        n_rep, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < SOAK_S:
            for _ in range(10):
                ttnn.execute_trace(mesh, trt.tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            n_rep += 10
        dt_ = time.perf_counter() - t0
        soak = dict(seconds=round(dt_, 1), steps=n_rep, us_per_step=round(dt_ * 1e6 / n_rep, 1),
                    us_per_layer=round(dt_ * 1e6 / n_rep / n, 1))  # fmt: skip
    trt.release()
    out_bit = all(bool(torch.equal(a, b)) for a, b in zip(got, want))
    s_bit = sum(bool(torch.equal(a, b)) for ca, cb in zip(got_s, want_s) for a, b in zip(ca, cb))
    c_bit = sum(bool(torch.equal(a, b)) for ca, cb in zip(got_c, want_c) for a, b in zip(ca, cb))
    ok = out_bit and s_bit == n * NCH and c_bit == n * NCH
    seqname = f"{LAYERS} x {rep}"
    r = dict(test=f"m5a stack, real dense_kda sequence {seqname} ({n} layers) in ONE program per {CH}-row chunk vs {n} "
                  f"sequential launches B={B}", status="pass" if ok else "fail", milestone="M5a", n=n, T=CH, B=B,
             chunks=NCH, layers=LAYERS, repeat=rep, out_bitwise=out_bit, state_bitwise_layers=s_bit,
             conv_bitwise_layers=c_bit, max_abs=float(max((a - b).abs().max() for a, b in zip(got, want))),
             pcc=C.pcc(got[0], want[0]), tag=tag, info=stacks[0].info)  # fmt: skip
    prow = dict(test=f"m5a stack real {seqname} N={n} B={B} step", status="info", milestone="M5a", traced=True, n=n,
                T=CH, B=B, chunks=NCH, us=st_t["median_us"], per_layer_us=round(st_t["median_us"] / n, 1),
                stack=st_t, stack_batch_us=round(st_b, 1) if st_b else None, chain=ch_t,
                chain_per_layer_us=round(ch_t["median_us"] / n, 1), chain_batch_us=round(ch_b, 1) if ch_b else None,
                launches_per_step=NCH, soak=soak, tag=tag, knobs=ENV,
                note="real dense_kda layers (own weights / mHC params); stack = ONE program per 4-row chunk (B > 4: "
                     "chunks back to back, DRAM residual each), chain = N MkLayer launches per chunk, one trace")  # fmt: skip
    log("ROW", r)
    log("PERF", prow)
    if args.get("record"):
        C.record("pcc", r)
        C.record("perf", prow) if not os.environ.get("TT_METAL_WATCHER") else None
    return dict(ok=ok, per_layer=prow["per_layer_us"], chain_per_layer=prow["chain_per_layer_us"], soak=soak)


RESULT = {rep: run_case(rep) for rep in sorted(REPEATS, reverse=True)}
