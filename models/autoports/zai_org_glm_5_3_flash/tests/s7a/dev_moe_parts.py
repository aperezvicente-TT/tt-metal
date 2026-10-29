# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a M4 sizing: the sparse_kda layer's MoE FFN sublayer, piece by piece, traced on one chip (the inputs of the
M4 megakernel's FFN phases). Layer GLM53_S7A_LAYER (default 4), T = GLM53_MK_TS (default 1,4,8,32): FFN mhc_pre (s1 ->
hn2, fp32 for the router when router_fp32_input), router (DS matmul + the fused tail), routed experts (fused_experts
with the router's ids / scores), shared expert (DS gate_up -> clamped SwiGLU -> DS down), mhc_post (y + y2), and the
whole FFN sublayer chain. Timing only (traced, median of REPLAYS); rows -> perf.jsonl / tests.jsonl."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced, cached_decoder
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import OUT_FLAT, X_FLAT, X_STACKED, MhcPost
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

LAYER = int(os.environ.get("GLM53_S7A_LAYER", "4"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
TS = [int(v) for v in os.environ.get("GLM53_MK_TS", "1,4,8,32").split(",")]
REPLAYS = 20
ROWS = []


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, layer=LAYER, test=name,
             watcher=bool(os.environ.get("TT_METAL_WATCHER")), **kw)  # fmt: skip
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)


def flush():
    with open(os.path.join(OUT, "perf.jsonl"), "a") as f:
        for r in ROWS:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT, "tests.jsonl"), "a") as f:
        for r in ROWS:
            f.write(json.dumps({k: r[k] for k in ("time_utc", "run", "test", "status", "t") if k in r}) + "\n")


def ffn_args(source, cfg, layer):
    """MhcPre arguments of `layer`'s FFN sublayer (explicit layer: a long-lived process keeps modules imported)"""
    n = cfg.hc_mult
    scale = source.host(layer, "hc_ffn_scale").float().reshape(-1)
    return dict(
        fn_t=source.host(layer, "hc_ffn_fn").reshape(n * cfg.hidden_size, (2 + n) * n),
        scale=torch.cat([scale[0].repeat(n), scale[1].repeat(n), scale[2].repeat(n * n)]),
        bias=source.host(layer, "hc_ffn_base").float().reshape(-1),
        norm_w=source.host(layer, "post_attention_layernorm.weight").float().reshape(-1),
        hc_eps=cfg.hc_eps, x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps, iters=cfg.hc_sinkhorn_iters,
        hidden=cfg.hidden_size, n=n,
    )  # fmt: skip


def timed(tr):
    d = []
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        tr.replay()
        d.append((time.perf_counter() - t0) * 1e6)
    return round(sorted(d)[len(d) // 2], 1)


def main(mesh=None, ctx=None):
    """mesh / ctx: run inside the long-lived device process (serve.py)."""
    own = mesh is None
    cfg = hf_config()
    if own:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    tr = Traced(mesh)
    try:
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        dec = cached_decoder(cfg, LAYER, mesh, ctx)
        layer, mlp = dec.layer, dec.layer.mlp
        if layer.spec.mlp != "moe":
            raise ValueError(f"layer {LAYER} is not an MoE layer")
        f_args = ffn_args(source, cfg, LAYER)
        fp32_router = layer.fcfg.router_fp32_input
        ref_f = M.MhcPre(mesh, **f_args)
        mpost = MhcPost(mesh)
        g = golden(LAYER, 2048)["streams_in"].float()
        cases = {}
        for t in TS:
            # the decoder's layouts: stacked residual for t <= 4, flat (stream-major coefficients) above
            stk = t <= 4
            rows = g[0, 301 : 301 + 37 * t : 37] if 301 + 37 * t <= g.shape[1] else g[0, 300 : 300 + t]
            s1 = ttnn.from_torch(rows.reshape((t, 1, 4, 4096) if stk else (1, 1, t, 16384)), dtype=ttnn.float32,
                                 layout=ttnn.TILE_LAYOUT, device=mesh,
                                 memory_config=ttnn.L1_MEMORY_CONFIG if stk else ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            xm, pm = (M.X_STACKED, M.PC_STACKED) if stk else (M.X_FLAT, M.PC_STREAM_MAJOR)
            pxm = X_STACKED if stk else X_FLAT
            pre = lambda s1=s1, t=t, xm=xm, pm=pm: ref_f(s1, hn_dtype=ttnn.float32 if fp32_router else ttnn.bfloat16,
                                                         t=t, x_mode=xm, xout_mode=M.XOUT_NONE, pc_mode=pm)  # fmt: skip
            o0 = pre()  # fixed inputs for the piece timings (allocated before any capture)
            x32 = o0.hn
            xb = ttnn.typecast(x32, ttnn.bfloat16) if fp32_router else x32
            idx0, sc0 = mlp.router(x32, decode=True)
            y0 = mlp.routed(xb, idx0, sc0, mlp.cfg.moe_decode_micro_batch, mlp.cfg.experts_block_size)
            y20 = mlp.shared(xb, decode=True)

            def whole(s1=s1, t=t, pre=pre, pxm=pxm):
                o = pre()
                xr = o.hn
                xb_ = ttnn.typecast(xr, ttnn.bfloat16) if fp32_router else xr
                y, y2 = mlp.parts(xb_, decode=True, x_router=xr if fp32_router else None)
                return mpost(y, s1, o.coef, t=t, x_mode=pxm, out_mode=OUT_FLAT, y2=y2,
                             out_shape=(1, 1, t, 16384), out_memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip

            pieces = dict(
                mhc_pre=pre,
                router=lambda x32=x32: mlp.router(x32, decode=True),
                routed=lambda xb=xb, idx0=idx0, sc0=sc0: mlp.routed(xb, idx0, sc0, mlp.cfg.moe_decode_micro_batch,
                                                                    mlp.cfg.experts_block_size),
                shared=lambda xb=xb: mlp.shared(xb, decode=True),
                mhc_post=lambda y0=y0, y20=y20, s1=s1, o0=o0, t=t, pxm=pxm: mpost(
                    y0, s1, o0.coef, t=t, x_mode=pxm, out_mode=OUT_FLAT, y2=y20, out_shape=(1, 1, t, 16384),
                    out_memory_config=ttnn.DRAM_MEMORY_CONFIG),
                ffn_chain=whole,
            )  # fmt: skip
            cases[t] = pieces
        tr.warm(*[f for p in cases.values() for f in p.values()])
        for t, pieces in cases.items():
            us = {}
            try:
                for name, fn in pieces.items():
                    tr.run(fn)
                    us[name] = timed(tr)
                    tr.release()
                row("moe_ffn_parts", t=t, execution=tr.mode, timing_us=us, status="record",
                    sum_of_pieces=round(sum(v for k, v in us.items() if k != "ffn_chain"), 1),
                    top_k=mlp.top_k, experts=mlp.num_experts, router_fp32_input=fp32_router)  # fmt: skip
            except Exception:
                traceback.print_exc()
                row("moe_ffn_parts", t=t, status="fail", error=traceback.format_exc()[-400:])
                tr.release()
                if "TIMEOUT" in traceback.format_exc():
                    flush()
                    os._exit(3)
    finally:
        flush()
        if own:
            ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MOE_PARTS_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
