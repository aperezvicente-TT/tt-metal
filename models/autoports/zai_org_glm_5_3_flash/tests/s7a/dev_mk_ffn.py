# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel, FFN half check: P7 mhc_pre (FFN) -> P8 gate_up (16 readers) -> P8b swiglu (64 owners) -> P9 down
(readers) -> P10 mhc_post (owners) in ONE program (tt/mk_ffn.py MkFFN) vs the chain it replaces (MhcPre, DSLinear
gate_up, slices + ttnn.clamped_silu_glu, DSLinear down, MhcPost flat DRAM), dense_kda L0, T = 1 / 2 / 4, device vs
device; traced (every case warmed before any capture). hn2 / gu bitwise (same programs' arithmetic: mhc_pre, gate_up
K blocking = DSLinear's), m (SFPU product in fp32 dest vs the chain's bf16-rounded activations), y and the layer output
PCC; traced time of each. Rows -> pcc.jsonl / tests.jsonl."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced, cached_decoder
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import OUT_FLAT, X_STACKED, MhcPost
from models.autoports.zai_org_glm_5_3_flash.tt.mk_ffn import MkFFN
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "0"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
REPLAYS = 20
ROWS = []
_FLUSHED = 0


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, layer=LAYER, test=name,
             watcher=bool(os.environ.get("TT_METAL_WATCHER")), **kw)  # fmt: skip
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)


def flush():
    global _FLUSHED
    new, _FLUSHED = ROWS[_FLUSHED:], len(ROWS)
    with open(os.path.join(OUT, "pcc.jsonl"), "a") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT, "tests.jsonl"), "a") as f:
        for r in new:
            f.write(json.dumps({k: r[k] for k in ("time_utc", "run", "test", "status", "t") if k in r}) + "\n")


def host(t):
    return ttnn.to_torch(t).float()


def timed(tr):
    if not tr.enabled:
        return None
    d = []
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        tr.replay()
        d.append((time.perf_counter() - t0) * 1e6)
    return sorted(d)[len(d) // 2]


def ffn_mhc_args(source, cfg, layer=None):
    """layer: explicit in a long-lived process (this module's LAYER is fixed at its first import)"""
    layer = LAYER if layer is None else layer
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


def main(mesh=None, ctx=None):
    """mesh / ctx: run inside the long-lived device process (serve.py): no open / close, the decoder cached."""
    own = mesh is None
    cfg = hf_config()
    wl1 = int(os.environ.get("GLM53_WORKER_L1", "0")) if own else (ctx or {}).get("worker_l1", 0)
    extra = dict(worker_l1_size=wl1) if wl1 else {}
    if own:
        mesh = (
            ttnn.open_device(device_id=0, **extra)
            if MOCK
            else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000, **extra)
        )
    tr = Traced(mesh)
    try:
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        dec = cached_decoder(cfg, LAYER, mesh, ctx)
        mlp = dec.layer.mlp
        lins = {d.role: d for d in dec.layer.ds_linears()}
        args = ffn_mhc_args(source, cfg, LAYER)
        mk = MkFFN(mesh, args, lins["mlp_gate_up"], lins["mlp_down"], intermediate=mlp.intermediate, limit=mlp.limit)
        ref_mhc = M.MhcPre(mesh, **args)
        mpost = MhcPost(mesh)
        i_ = mlp.intermediate
        g = golden(LAYER, 2048)["streams_in"].float()
        cases = {}
        for t in (1, 2, 4):
            # s1 = the attention half's stacked residual (here: golden streams, stacked)
            s1 = ttnn.from_torch(g[0, 301 : 301 + 37 * t : 37].reshape(t, 1, 4, 4096), dtype=ttnn.float32,
                                 layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip

            def run_mk(s1=s1, t=t):
                return mk(s1, t=t)

            def run_chain(s1=s1, t=t):
                r = ref_mhc(s1, hn_dtype=ttnn.bfloat16, t=t, x_mode=M.X_STACKED, xout_mode=M.XOUT_NONE,
                            pc_mode=M.PC_STACKED)  # fmt: skip
                gu_ = lins["mlp_gate_up"](r.hn, ttnn.L1_MEMORY_CONFIG)
                shape = list(gu_.shape)
                gate = ttnn.slice(gu_, [0, 0, 0, 0], shape[:-1] + [i_])
                up = ttnn.slice(gu_, [0, 0, 0, i_], shape[:-1] + [2 * i_])
                m_ = ttnn.clamped_silu_glu(gate, up, mlp.limit)
                y_ = lins["mlp_down"](m_, ttnn.L1_MEMORY_CONFIG)
                out_ = mpost(y_, s1, r.coef, t=t, x_mode=X_STACKED, out_mode=OUT_FLAT, out_shape=(1, 1, t, 16384),
                             out_memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
                return r, gu_, m_, y_, out_

            cases[t] = (run_mk, run_chain)
        tr.warm(*[f for c in cases.values() for f in c])  # every program compiled before any capture
        for t, (run_mk, run_chain) in cases.items():
            try:
                o, gu, m, y, out = tr.run(run_mk)
                got = dict(hn2=host(o.hn), gu=host(gu)[..., : 2 * i_], m=host(m), y=host(y)[..., :4096],
                           out=host(out))  # fmt: skip
                us_mk = timed(tr)
                tr.release()
                o2, gu2, m2, y2, out2 = tr.run(run_chain)
                want = dict(hn2=host(o2.hn), gu=host(gu2), m=host(m2), y=host(y2), out=host(out2))
                us_chain = timed(tr)
                tr.release()
                r = dict(t=t, step="ffn", execution=tr.mode, mk_us=us_mk, chain_us=us_chain,
                         mk_opt=os.environ.get("GLM53_MK_OPT", "O3"), worker_l1=wl1 or None,
                         alloc_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1")  # fmt: skip
                for k_ in got:
                    a, b = got[k_].reshape(-1), want[k_].reshape(-1)[: got[k_].numel()]
                    r[f"{k_}_bitwise"] = bool(torch.equal(a, b))
                    r[f"{k_}_pcc"] = pcc(a, b)
                    r[f"{k_}_maxabs"] = float((a - b).abs().max())
                ok = all(r[f"{k_}_bitwise"] for k_ in ("hn2", "gu")) and min(r[f"{k_}_pcc"] for k_ in got) >= 0.9999
                r["bitwise_required"] = ["hn2", "gu"]
                r["status"] = "record" if MOCK else ("pass" if ok else "fail")
                row("mk_ffn_vs_chain", **r)
            except Exception:
                traceback.print_exc()
                err = traceback.format_exc()
                row("mk_ffn_vs_chain", t=t, status="fail", error=err[-400:])
                if "TIMEOUT" in err or "hang" in err:
                    flush()
                    print("DEV_MK_FFN_ABORT device timeout", flush=True)
                    os._exit(3)
                tr.release()
    finally:
        flush()
        if own:
            ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MK_FFN_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
