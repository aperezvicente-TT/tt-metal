# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd pair): Stage 14 cross-layer weight ring measurement (doc/ttnn/14-production-tp8-megakernel/
ring_across_layers.md). The model's dense_kda layers ARGS layers (L0, L1, L2) as ONE TPStack program per chip (Stage
14b's tt/tp_stack.py), TP2 (vtp None) or the TP8 per-chip geometry on the pair (vtp 8: timing only), per variant:
  normal | no_dram (GLM53_MK_NO_DRAM=1: the weight streamers skip their reads -> the span with every weight already in
  L1, the upper bound of ANY prefetch scheme; garbage numerics) | env knobs (ring sizes, park, ...);
  timing: blocking replays of the stack launch (median) -> per layer; chain=True: + the same layers as N single-layer TP
  launches in one trace -> per_boundary_us = (chain - stack) / (N - 1); probe=True: per-layer reader-streamer marks
  (layer start + one mark per streamed phase) and owner marks on a COMMON time base (first mark of the launch) ->
  the DRAM-idle windows between phases / layers.
ARGS: layers ([0, 1, 2]), vtp (None), t (1), reps (30), variants ([{"name": .., "env": {..}, "probe": bool}]), tag."""
import gc
import importlib
import os
import sys
import time

import torch

import ttnn

gc.collect()
S14B = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "s14b")
sys.path.insert(0, S14B)
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.mk_layer as MKL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_readers as MKR  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer as TPL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8_sparse as TPSP  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp_stack as TPS  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402

for mod in (MKR, MKL, TPL, TPSP, MKS, TPS, SM):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (  # noqa: E402
    FdTileCacheSource,
    OptimizedDecoderConfig,
)

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
KIND = args.get("kind", "dense")  # dense: TPDenseKDALayer + TPStack; sparse: TPSparseKDALayer + TPSparseStack
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2] if KIND == "dense" else [4, 5, 6])]
VTP = args.get("vtp")
T, REPS = int(args.get("t", 1)), int(args.get("reps", 30))
tag = args.get("tag", "ring-stack")
cfg = hf_config()
OUT = os.environ.get("GLM53_STAGE_EVIDENCE") or "/opt/glm53/kit/doc/ttnn/14-production-tp8-megakernel"


def log(*a):
    print(f"RING {C.now()}", *a, flush=True)


def record(row):
    import json

    print("ROW", json.dumps(row, default=str), flush=True)
    if OUT and not os.environ.get("TT_METAL_WATCHER"):
        with open(os.path.join(OUT, "ring.jsonl"), "a") as f:
            f.write(json.dumps(row, default=str) + "\n")


def mhz():
    try:
        return float(open("/sys/class/tenstorrent/tenstorrent!0/tt_aiclk").read().split()[0])
    except Exception:  # noqa: BLE001
        return 1350.0


def marks(stk, chip: int):
    """per layer: decode_probe of its row block on ONE time base (the launch's first mark)"""
    raw = ttnn.to_torch(ttnn.get_device_tensors(stk.probe)[chip]).reshape(stk.n, -1, 16).long() & 0xFFFFFFFF
    nz = raw[raw > 0]
    if nz.numel() == 0:  # the mock runs no kernels
        return []
    t0 = int(nz.min())
    f = mhz()
    L0 = stk.layers[0]
    out = []
    for i in range(stk.n):
        blk = raw[i]
        bnz = blk[blk > 0]
        off = (int(bnz.min()) - t0) / f if bnz.numel() else 0.0
        d = MKL.decode_probe(blk, L0.readers.assign, k_cores=len(L0.place["k"]), mhz=f)
        out.append({k: [round(v + off, 1) for v in vs] for k, vs in d.items()})
    return out


key = ("ring_tp", KIND, T, tuple(LAYERS), VTP)
for k in [k for k in ctx if isinstance(k, tuple) and k[0] in ("ring_tp", "m5b_tp", "tp") and k != key]:
    ctx.pop(k)
gc.collect()
res = {}
for var in args.get("variants", [{"name": "normal"}]):
    env = {k: str(v) for k, v in (var.get("env") or {}).items()}
    keys = set(env) | set(SM.KNOB_KEYS) | {"GLM53_STACK_KNOBS"}
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update(env)  # first: GLM53_STACK_KNOBS (the profile set_knobs picks: s14 / s7a283 / s7a)
    SM.set_knobs(KIND == "sparse", tp=int(mesh.get_num_devices()), t=T)  # the production stack's knobs
    os.environ.update(env)  # then the variant's own overrides win
    knob_env = {k: os.environ[k] for k in sorted(keys) if k in os.environ}
    try:
        # the layers are rebuilt per variant (ring sizes / NO_DRAM are read at MkLayer construction); weights resident
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        wkey = ("ring_w", KIND, T, VTP)
        wc = ctx.setdefault(wkey, {})
        t0 = time.time()
        cp = bool(var.get("chain_probe"))  # single-layer programs WITH probes: the chain only, marks per layer
        if KIND == "sparse":  # the replicated MoE (vtp: chip r runs experts [36 r, 36 r + 36) of it), per layer
            moes = ctx.setdefault(("ring_moe",), {})
            for L in LAYERS:
                if L not in moes:
                    moes[L] = TPSP.build_moe(mesh, cfg, source, L, OptimizedDecoderConfig())
            tps = {L: TPSP.TPSparseKDALayer(mesh, cfg, L, source, OptimizedDecoderConfig(), b=T, moe=moes[L], vtp=VTP,
                                            weights=wc.get(L), probe=cp) for L in LAYERS}  # fmt: skip
        else:
            tps = {L: TPL.TPDenseKDALayer(mesh, cfg, L, source, OptimizedDecoderConfig(), b=T, program="layer",
                                          vtp=VTP, weights=wc.get(L), probe=cp) for L in LAYERS}  # fmt: skip
        for L in LAYERS:
            wc[L] = tps[L].w
        fl = tps[LAYERS[0]]
        sh = fl.shards
        log(f"variant {var['name']}: layers built in {time.time() - t0:.1f}s")
        g = golden(0, 2048)["streams_in"].float()
        x_rows = g[0, 300 : 300 + 37 * T : 37].reshape(T, 16384)
        x_stk = fl.replicated(x_rows.reshape((T, 1, 4, 4096) if T <= 4 else (1, 1, T, 16384)))
        act = fl.masks(torch.ones(T, dtype=torch.bool))
        gen = torch.Generator().manual_seed(4321)
        S0 = torch.randn(T, 64, 128, 128, generator=gen) * 0.1
        C0 = (torch.randn(1, 3, T, 24576, generator=gen) * 0.5).to(torch.bfloat16)
        sts = [sh.shard_state(S0, C0) for _ in LAYERS]
        RESTORE = bool(var.get("restore"))  # in-trace restore of states + input (data-dependent MoE routing)
        S0m, C0m = sh.shard_state(S0, C0)

        def restore():
            for Sm, Cm in sts:
                ttnn.copy(S0m, Sm)
                ttnn.copy(C0m, Cm)
        if cp:
            x_flat = fl.replicated(x_rows.reshape(1, 1, T, 16384))

            def chain_call():
                if RESTORE:
                    restore()
                y = x_flat
                for L, (Sm, Cm) in zip(LAYERS, sts):
                    y = tps[L](y, Cm, act, Sm, t=T)[1]
                return y

            trc = C.Traced(ttnn, mesh, chain_call)
            trc.replay()
            ch = C.stats(trc.replays(REPS))
            trc.replay()
            ttnn.synchronize_device(mesh)
            f = mhz()
            mk_ = {}
            for d in range(int(mesh.get_num_devices())):
                per = []
                for L in LAYERS:
                    mk = tps[L].mk
                    raw = ttnn.to_torch(ttnn.get_device_tensors(mk.probe)[d])
                    per.append(MKL.decode_probe(raw, mk.readers.assign, k_cores=len(mk.place["k"]), mhz=f)
                               if (raw != 0).any() else {})  # fmt: skip
                mk_[f"chip{d}"] = per
            trc.release()
            record(dict(test=f"ring chain_probe: {len(LAYERS)} {KIND} single-layer TP programs (probes), T={T}",
                        kind=KIND, status="info", variant=var["name"], env=env, knob_env=knob_env, vtp=VTP,
                        tp=int(mesh.get_num_devices()), T=T, layers=LAYERS, chain_us=ch["median_us"],
                        chain_per_layer_us=round(ch["median_us"] / len(LAYERS), 1), stats=ch, tag=tag, restore=RESTORE, aiclk_mhz=f,
                        marks_single=mk_, note="each layer's marks on its OWN program's time base",
                        code=os.environ.get("GLM53_CODE_SHA")))  # fmt: skip
            res[var["name"]] = round(ch["median_us"] / len(LAYERS), 1)
            del tps, trc
            gc.collect()
            continue
        cls = TPS.TPSparseStack if KIND == "sparse" else TPS.TPStack
        stk = cls(mesh, [tps[L].mk for L in LAYERS], fl, t=T, probe=bool(var.get("probe")))
        def call():
            if RESTORE:
                restore()
                ttnn.copy(x_stk, stk.residual())
            return stk([Cm for _, Cm in sts], act, [Sm for Sm, _ in sts])

        ttnn.copy(x_stk, stk.residual())
        tr = C.Traced(ttnn, mesh, call)
        tr.replay()
        st = C.stats(tr.replays(REPS))
        row = dict(test=f"ring stack {var['name']}: {len(LAYERS)} {KIND} layers in one {cls.__name__} program, T={T}",
                   kind=KIND,
                   status="info", variant=var["name"], env=env, knob_env=knob_env, vtp=VTP, tp=int(mesh.get_num_devices()), T=T,
                   layers=LAYERS, us=st["median_us"], per_layer_us=round(st["median_us"] / len(LAYERS), 1), stats=st,
                   tag=tag, restore=RESTORE, aiclk_mhz=mhz(), info=stk.info, code=os.environ.get("GLM53_CODE_SHA"))  # fmt: skip
        if RESTORE:  # restored inputs: the stack output must be BITWISE equal across variants (vs this job's first)
            tr.replay()
            ttnn.synchronize_device(mesh)
            outs = [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(stk.residual())]
            ref = ctx.setdefault(("ring_ref", tag, KIND, T, VTP), outs)
            row["out_bitwise_vs_first_variant"] = all(torch.equal(a, b) for a, b in zip(outs, ref))
            row["out_finite"] = all(bool(torch.isfinite(o).all()) for o in outs)
            row["out_replicas_identical"] = all(torch.equal(outs[0], o) for o in outs[1:])
        if var.get("chain"):  # the same layers as N single-layer TP launches in one trace: the per-boundary cost
            x_flat = fl.replicated(x_rows.reshape(1, 1, T, 16384))

            def chain_call():
                if RESTORE:
                    restore()
                y = x_flat
                for L, (Sm, Cm) in zip(LAYERS, sts):
                    y = tps[L](y, Cm, act, Sm, t=T)[1]  # the single-layer TP program (either kind): its output
                return y

            trc = C.Traced(ttnn, mesh, chain_call)
            trc.replay()
            ch = C.stats(trc.replays(REPS))
            trc.release()
            n = len(LAYERS)
            row.update(chain_us=ch["median_us"], chain_per_layer_us=round(ch["median_us"] / n, 1), chain_stats=ch,
                       per_boundary_us=round((ch["median_us"] - st["median_us"]) / max(n - 1, 1), 1))  # fmt: skip
        if var.get("probe"):
            tr.replay()
            ttnn.synchronize_device(mesh)
            row["marks"] = {f"chip{d}": marks(stk, d) for d in range(int(mesh.get_num_devices()))}
        record(row)
        res[var["name"]] = row["per_layer_us"]
        tr.release()
        del stk, tps, tr
        gc.collect()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
log("RESULT", res)
RESULT = dict(ok=True, **res)
