# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel soak: one trace replayed back to back for GLM53_SOAK_S seconds, the output read back after
EVERY replay and compared bitwise with the first replay's (a race between phases — e.g. the shared-ring phase-start
race fixed in 59d2cfc7563 — shows up as a replay that differs). The trace restores every piece of state it mutates
first, so every replay computes the same thing.

GLM53_SOAK_KIND:
  one     dense layer as ONE program (tt/mk_layer.py MkLayer), 4 chained layer launches per replay (layer GLM53_S7A_LAYER,
          default 0)
  sparse  sparse_kda layer as two programs (MkStepB attention + MkMoE), 4 chained layers per replay (default layer 4)
  moe     the MoE FFN half alone (MkMoE), 4 launches per replay on the same input (default layer 4)
  sparse_one  sparse_kda layer as ONE program (tt/mk_sparse.py MkSparseLayer), 4 chained layers per replay
T = GLM53_MK_TS (first value). Row soak_<kind> -> pcc.jsonl / tests.jsonl / perf.jsonl: replays, mismatching replays,
max |diff| of a mismatch, replay wall-clock median / p99 (blocking replay + 64 KB-per-row readback)."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_kda_pre import masks
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_mk_a import mhc_args
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_mk_ffn import ffn_mhc_args
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_moe_parts import ffn_args
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced, cached_decoder
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

KIND = os.environ.get("GLM53_SOAK_KIND", "one")
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "0" if KIND == "one" else "4"))  # the serve process exports 0: pass it
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
T = int(os.environ.get("GLM53_MK_TS", "1").split(",")[0])
SECONDS = float(os.environ.get("GLM53_SOAK_S", "300"))
CHAIN = 4


def chained(layer, x, n):
    for _ in range(n):
        x = layer(x)
    return x


def build(mesh, ctx):
    """-> (fn returning the list of output tensors, description); every state the trace mutates is restored first"""
    cfg = hf_config()
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    dec = cached_decoder(cfg, LAYER, mesh, ctx)
    kda, mlp = dec.layer.attn, dec.layer.mlp
    lins = {d.role: d for d in dec.layer.ds_linears()}
    g = golden(LAYER, 2048)["streams_in"].float()
    t = T
    if KIND == "moe":
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_moe import MkMoE

        mk = MkMoE(mesh, ffn_args(source, cfg, LAYER), mlp)
        s1 = ttnn.from_torch(g[0, 301 : 301 + 37 * t : 37].reshape(t, 1, 4, 4096), dtype=ttnn.float32,
                             layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
        return (lambda: [mk(s1, t=t)["out"] for _ in range(CHAIN)]), f"MkMoE x{CHAIN} same input"
    a_args, f_args = mhc_args(source, cfg, LAYER), ffn_mhc_args(source, cfg, LAYER)
    x = ttnn.from_torch(g[0, 300 : 300 + 37 * t : 37].reshape(1, 1, t, 16384), dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    active = torch.tensor([(i % 3) != 2 for i in range(t)]) if t > 1 else torch.tensor([True])
    act = masks(mesh, active, kda.lower_bound)
    torch.manual_seed(1000 + t)  # the same random history / state in every process and job (comparable rows)
    hist0 = ttnn.from_torch((torch.randn(1, 3, t, 24576) * 0.5).to(torch.bfloat16), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    s0 = ttnn.from_torch(torch.randn(t, 64, 128, 128) * 0.1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                         device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    st = kda.allocate_state(t)

    def restore():
        ttnn.copy(hist0, st.C)
        ttnn.copy(s0, st.S)

    if KIND == "one":
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_layer import MkLayer

        assert dec.layer.spec.mlp != "moe", "GLM53_SOAK_KIND=one: a dense layer"
        one = MkLayer(mesh, a_args, f_args, kda, lins, intermediate=mlp.intermediate, limit=mlp.limit)

        def fn():
            restore()
            return [chained(lambda x_: one(x_, st.C, act, t=t, state=st.S)["out"], x, CHAIN)]

        return fn, f"MkLayer x{CHAIN} chained"
    if KIND == "sparse":
        from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import MkStepB
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_moe import MkMoE

        assert dec.layer.spec.mlp == "moe", "GLM53_SOAK_KIND=sparse: a sparse layer"
        att = MkStepB(mesh, a_args, kda, lins["kda_qkv"], lins["kda_lowrank"], ds_o=lins["kda_o"], post=True)
        ffn = MkMoE(mesh, f_args, mlp)
        akw = dict(t=t, x_mode=M.X_FLAT, xout_mode=M.XOUT_STACKED, pc_mode=M.PC_STACKED)

        def fn():
            restore()
            layer = lambda x_: ffn(att(x_, st.C, act, state=st.S, **akw)[-1], t=t)["out"]
            return [chained(layer, x, CHAIN)]

        return fn, f"MkStepB + MkMoE x{CHAIN} chained"
    if KIND == "sparse_one":
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_sparse import MkSparseLayer

        assert dec.layer.spec.mlp == "moe", "GLM53_SOAK_KIND=sparse_one: a sparse layer"
        one = MkSparseLayer(mesh, a_args, f_args, kda, lins, mlp)

        def fn():
            restore()
            return [chained(lambda x_: one(x_, st.C, act, t=t, state=st.S)["out"], x, CHAIN)]

        return fn, f"MkSparseLayer x{CHAIN} chained"
    raise ValueError(f"GLM53_SOAK_KIND={KIND}")


def main(mesh=None, ctx=None):
    own = mesh is None
    if ctx is not None and ctx.pop("dev_mk_layer_objs", None) is not None:  # its resident objects pin L1: release
        import gc

        gc.collect()
    if own:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    tr = Traced(mesh)
    r = dict(time_utc=None, run=RUN, layer=LAYER, test=f"soak_{KIND}", t=T, execution="traced",
             watcher=bool(os.environ.get("TT_METAL_WATCHER")))  # fmt: skip
    try:
        fn, desc = build(mesh, ctx)
        outs = tr.run(fn)
        ref = [ttnn.to_torch(o) for o in outs]
        n = bad = 0
        worst = 0.0
        d = []
        t_end = time.time() + SECONDS
        while time.time() < t_end:
            t0 = time.perf_counter()
            tr.replay()
            got = [ttnn.to_torch(o) for o in outs]
            d.append((time.perf_counter() - t0) * 1e6)
            n += 1
            diff = [not torch.equal(a, b) for a, b in zip(got, ref)]
            if any(diff):
                bad += 1
                worst = max(worst, max(float((a.float() - b.float()).abs().max()) for a, b in zip(got, ref)))
                if bad <= 3:
                    print(f"SOAK_MISMATCH replay={n} outputs={diff} maxabs={worst}", flush=True)
        tr.release()
        d.sort()
        r.update(desc=desc, replays=n, mismatching_replays=bad, mismatch_maxabs=worst, seconds=SECONDS,
                 timing_us=dict(replay_readback_median=round(d[len(d) // 2], 1),
                                replay_readback_p99=round(d[int(len(d) * 0.99)], 1)),
                 status="pass" if bad == 0 and n > 0 else "fail")  # fmt: skip
    except Exception:
        traceback.print_exc()
        tr.release()
        r.update(status="fail", error=traceback.format_exc()[-400:])
    r["time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print("ROW", json.dumps(r), flush=True)
    for name, keys in (("pcc.jsonl", None), ("tests.jsonl", ("time_utc", "run", "test", "status", "t")),
                       ("perf.jsonl", ("time_utc", "run", "test", "t", "execution", "timing_us", "watcher", "replays",
                                       "mismatching_replays", "desc"))):  # fmt: skip
        if name == "perf.jsonl" and not r.get("timing_us"):
            continue
        with open(os.path.join(OUT, name), "a") as f:
            f.write(json.dumps(r if keys is None else {k: r[k] for k in keys if k in r}) + "\n")
    if own:
        ttnn.close_mesh_device(mesh)
    print(f"DEV_MK_SOAK_DONE kind={KIND} status={r['status']}", flush=True)
    return 0 if r["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
