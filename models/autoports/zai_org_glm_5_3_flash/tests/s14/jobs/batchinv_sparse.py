# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd pair): Stage 14 — the sparse_kda batch-invariance coupling (14b #82 / #85: a sparse_kda layer
FED BY a sparse_dsa layer couples a live row to its idle mates; the idle rows' residuals are ~2.5x the live ones).
ONE TP sparse_kda layer (ARGS layer, default 4) at native t (default 32), the production stack knobs (env), states
restored before every run: live rows = ARGS live (default the even rows), idle rows carry large inputs (golden rows x
ARGS scale, default 2.5), two variants A / B that differ ONLY in the idle rows -> per intermediate of the single-layer
program (MkSparseLayer outputs: s1, hn2, logits, ids, scores, gu, m, y, y2, out, ...): the live rows bitwise A == B?
(the first intermediate whose live rows differ locates the coupling). Also C = A with the idle rows at embedding scale
(x 0.25) and the active mask: idle rows inactive (as the generator runs them). Rows -> GLM53_STAGE_EVIDENCE batchinv.jsonl.
ARGS: layer (4), t (32), vtp (None), scale (2.5), live ("even"), env, tag."""
import gc
import importlib
import json
import os
import sys

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "s14b"))
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8_sparse as TPSP  # noqa: E402

for mod in (TPSP, SM):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (  # noqa: E402
    FdTileCacheSource,
    OptimizedDecoderConfig,
)

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
LAYER, T, VTP = int(args.get("layer", 4)), int(args.get("t", 32)), args.get("vtp")
SCALE = float(args.get("scale", 2.5))
LIVE = list(range(0, T, 2)) if args.get("live", "even") == "even" else [int(v) for v in args["live"]]
IDLE = [r for r in range(T) if r not in LIVE]
tag = args.get("tag", f"batchinv-L{LAYER}-t{T}")
OUT = os.environ.get("GLM53_STAGE_EVIDENCE") or "/opt/glm53/kit/doc/ttnn/14-production-tp8-megakernel"
env = {k: str(v) for k, v in (args.get("env") or {}).items()}
old = {k: os.environ.get(k) for k in set(env) | set(SM.KNOB_KEYS) | {"GLM53_STACK_KNOBS"}}
os.environ.update(env)
SM.set_knobs(True, tp=int(mesh.get_num_devices()), t=T)
os.environ.update(env)
knob_env = {k: os.environ[k] for k in sorted(old) if k in os.environ}
cfg = hf_config()


def log(*a):
    print(f"BATCHINV {C.now()}", *a, flush=True)


try:
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    mkey = ("binv_moe", LAYER)
    if mkey not in ctx:
        ctx[mkey] = TPSP.build_moe(mesh, cfg, source, LAYER, OptimizedDecoderConfig())
    tp = TPSP.TPSparseKDALayer(mesh, cfg, LAYER, source, OptimizedDecoderConfig(), b=T, moe=ctx[mkey], vtp=VTP)
    sh = tp.shards
    g = golden(0, 2048)["streams_in"].float()[0].reshape(2048, -1)  # [2048, 16384] residual rows
    base = g[300 : 300 + 37 * T : 37].reshape(T, 16384).clone()
    xa, xb, xc = base.clone(), base.clone(), base.clone()
    for i, r in enumerate(IDLE):
        xa[r] = g[1000 + 17 * i] * SCALE
        xb[r] = g[1500 + 13 * i] * SCALE
        xc[r] = g[1000 + 17 * i] * 0.25
    gen = torch.Generator().manual_seed(4321)
    S0 = torch.randn(T, 64, 128, 128, generator=gen) * 0.1
    C0 = (torch.randn(1, 3, T, 24576, generator=gen) * 0.5).to(torch.bfloat16)
    S0m, C0m = sh.shard_state(S0, C0)
    Sm, Cm = sh.shard_state(S0, C0)
    act_all = tp.masks(torch.ones(T, dtype=torch.bool))
    am = torch.zeros(T, dtype=torch.bool)
    am[LIVE] = True
    act_live = tp.masks(am)

    def run(x_rows, act):
        ttnn.copy(S0m, Sm)
        ttnn.copy(C0m, Cm)
        x = tp.replicated(x_rows.reshape(1, 1, T, 16384))
        s1, out, o, _ = tp(x, Cm, act, Sm, t=T)
        ttnn.synchronize_device(mesh)
        res = {}
        for k, v in o.items():
            if isinstance(v, ttnn.Tensor) and v.is_allocated():
                try:
                    res[k] = [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(v)]
                except Exception:  # noqa: BLE001
                    pass
        return res

    def rows_of(v):
        """[.., T, cols] -> the T token rows (None if the tensor's rows are not the tokens)"""
        if v.dim() >= 2 and v.shape[-2] == T:
            return v.reshape(-1, T, v.shape[-1])
        if v.dim() >= 2 and v.shape[-2] >= T and v.shape[-2] % 32 == 0 and v.numel() // v.shape[-1] == v.shape[-2]:
            return v.reshape(1, v.shape[-2], v.shape[-1])[:, :T]
        return None

    rep = {}
    for act_name, act in (("all_active", act_all), ("idle_inactive", act_live)):
        A, B, Cc = run(xa, act), run(xb, act), run(xc, act)
        A2 = run(xa, act)  # determinism of A itself
        for k in A:
            if k in ("a", "y"):  # never written under TP (the partials go to the fabric lanes): uninitialised memory
                continue
            for chip in range(len(A[k])):
                ra, rb, rc, ra2 = (rows_of(d[k][chip]) if k in d else None for d in (A, B, Cc, A2))
                if ra is None:
                    rep.setdefault(act_name, {})[f"{k}@{chip}"] = dict(rows="n/a", whole_AB=bool(torch.equal(A[k][chip], B[k][chip])),
                                                                     det=bool(torch.equal(A[k][chip], A2[k][chip])))
                    continue
                live_ab = bool(torch.equal(ra[:, LIVE], rb[:, LIVE]))
                live_ac = bool(torch.equal(ra[:, LIVE], rc[:, LIVE]))
                det = bool(torch.equal(ra, ra2))
                maxd = float((ra[:, LIVE] - rb[:, LIVE]).abs().max())
                rep.setdefault(act_name, {})[f"{k}@{chip}"] = dict(live_AB=live_ab, live_AC=live_ac, det=det, max_live_diff_AB=maxd)
        log(act_name, {k: v for k, v in rep[act_name].items() if not (v.get("live_AB", True) and v.get("live_AC", True))} or "ALL INVARIANT")
    row = dict(test=f"batch invariance: sparse_kda L{LAYER} native t={T}, live {len(LIVE)} rows, idle x{SCALE}", tag=tag,
               layer=LAYER, T=T, vtp=VTP, tp=int(mesh.get_num_devices()), scale=SCALE, knob_env=knob_env, report=rep,
               status="info", code=os.environ.get("GLM53_CODE_SHA"), time_utc=C.now())
    print("ROW", json.dumps(row, default=str), flush=True)
    if not os.environ.get("TT_METAL_WATCHER"):
        with open(os.path.join(OUT, "batchinv.jsonl"), "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    RESULT = dict(ok=True)
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
