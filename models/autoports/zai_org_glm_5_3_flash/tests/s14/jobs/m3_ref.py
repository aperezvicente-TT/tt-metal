# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd single, one chip): the Stage 3 single-chip dense_kda layer references for the M3 layer_check
(tests/s14/jobs/m3_tp.py reads them from work-s14/m3_ref/): per (layer, B) the Stage 3 OptimizedDecoder layer, B users
of golden prompt P (7a's layer_check Users: prefix n_i = P - 1 - i), TRACED prefill of every prefix (each bucket
warmed first), state snapshot, ONE traced decode step -> {x, y_ref, golden, S0, C0, S1, C1}. The decoder is loaded
once per layer and kept in CTX. Never on a submesh (a submesh of a fabric mesh hung chip 0 at 19:46Z).
ARGS: layer (0), bs ([1, 2, 4]), prompt (128)."""
import gc
import os
import sys
import time

import torch

import ttnn

gc.collect()
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config, pcc  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import load_optimized_decoder  # noqa: E402

sys.path.insert(0, "/opt/glm53/kit/doc/ttnn/07a-fused-streaming-layer/scripts")
import layer_check as LC  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYER, PROMPT = int(args.get("layer", 0)), int(args.get("prompt", 128))
W = os.environ.get("GLM53_WORK", "/opt/glm53/work-s14")
cfg = hf_config()
key = ("dec", LAYER)
if key not in ctx:
    for k in [k for k in ctx if isinstance(k, tuple) and k[0] == "dec"]:
        ctx.pop(k)
    gc.collect()
    t0 = time.time()
    ctx[key] = load_optimized_decoder(cfg, LAYER, mesh)
    print(f"M3REF layer {LAYER} loaded in {time.time() - t0:.1f}s", flush=True)
dec = ctx[key]
res = {}
for B in args.get("bs", [1, 2, 4]):
    path = os.path.join(W, "m3_ref", f"L{LAYER}_B{B}_P{PROMPT}.pt")
    if os.path.exists(path):
        res[B] = "exists"
        continue
    t0 = time.time()
    users = LC.Users(LAYER, PROMPT, B, 1)
    cache, pt, snap, traces, mode = LC.prepare(dec, users, B, PROMPT, traced=True, variants=[{}])
    if mode != "traced":
        raise RuntimeError(f"reference prefill was not traced: {mode}")
    S0, C0 = ttnn.to_torch(snap[0]).float(), ttnn.to_torch(snap[1])
    outs, _ = LC.run_variant(dec, cache, pt, snap, users, B, 1, trace=True)
    S1, C1 = ttnn.to_torch(cache.kda.S).float(), ttnn.to_torch(cache.kda.C)
    if traces is not None:
        dec.release_prefill_traces(traces)
    r = dict(x=users.rows(0, "streams_in"), y_ref=outs[0], golden=users.rows(0, "streams_out"), S0=S0, C0=C0, S1=S1,
             C1=C1, positions=users.positions(0), prefill_mode=mode)  # fmt: skip
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:  # atomic + fsync: a host death at 20:02Z left two truncated files
        torch.save(r, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)
    res[B] = dict(s=round(time.time() - t0, 1), ref_vs_golden=pcc(r["y_ref"], r["golden"]))
    print(f"M3REF L{LAYER} B={B} {res[B]}", flush=True)
    del cache, snap
    gc.collect()
RESULT = res
