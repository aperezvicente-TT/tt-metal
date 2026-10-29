# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a M4: the MoE FFN half of a sparse layer as ONE program (tt/mk_moe.py MkMoE: mhc_pre -> router -> router tail
-> routed experts + shared expert -> mhc_post) vs the op chain it replaces (MhcPre, router DSLinear + RouterTail,
fused_experts, the shared SwiGLU's DSLinears + clamped_silu_glu, MhcPost with y + y2), layer GLM53_S7A_LAYER (default
4), T = GLM53_MK_TS (default 1,2,4), device vs device, traced (every case warmed before any capture): hn2 / ids /
routed output bitwise expected, the shared expert / layer output PCC; traced time of each.
Rows -> pcc.jsonl / tests.jsonl / perf.jsonl."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_moe_parts import ffn_args
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced, cached_decoder
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import OUT_FLAT
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import X_FLAT as PX_FLAT
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import X_STACKED, MhcPost
from models.autoports.zai_org_glm_5_3_flash.tt.mk_moe import MkMoE
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "4"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
TS = [int(v) for v in os.environ.get("GLM53_MK_TS", "1,2,4").split(",")]
REPLAYS = 20
ROWS = []


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, layer=LAYER, test=name,
             watcher=bool(os.environ.get("TT_METAL_WATCHER")), **kw)  # fmt: skip
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)


def flush():
    for name, keys in (
        ("pcc.jsonl", None),
        ("tests.jsonl", ("time_utc", "run", "test", "status", "t")),
        ("perf.jsonl", ("time_utc", "run", "test", "t", "execution", "timing_us", "watcher")),
    ):
        with open(os.path.join(OUT, name), "a") as f:
            for r in ROWS:
                if name == "perf.jsonl" and not r.get("timing_us"):
                    continue
                f.write(json.dumps(r if keys is None else {k: r[k] for k in keys if k in r}) + "\n")


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
    return round(sorted(d)[len(d) // 2], 1)


def main(mesh=None, ctx=None):
    """mesh / ctx: run inside the long-lived device process (serve.py)."""
    own = mesh is None
    if ctx is not None and ctx.pop("dev_mk_layer_objs", None) is not None:  # its resident objects pin L1: release
        import gc

        gc.collect()
    cfg = hf_config()
    if own:
        mesh = (ttnn.open_device(device_id=0) if MOCK
                else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000))  # fmt: skip
    tr = Traced(mesh)
    try:
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        dec = cached_decoder(cfg, LAYER, mesh, ctx)
        mlp = dec.layer.mlp
        assert not dec.layer.fcfg.router_fp32_input, "MkMoE feeds the router the bf16 hn2"
        f_args = ffn_args(source, cfg, LAYER)
        mk = MkMoE(mesh, f_args, mlp)
        ref_f = M.MhcPre(mesh, **f_args)
        mpost = MhcPost(mesh)
        g = golden(LAYER, 2048)["streams_in"].float()
        cases = {}
        for t in TS:
            big = t > 4  # t > 4: the flat residual (mk_layer.residual_modes)
            s1 = ttnn.from_torch(g[0, 301 : 301 + 37 * t : 37].reshape((1, 1, t, 16384) if big else (t, 1, 4, 4096)),
                                 dtype=ttnn.float32,
                                 layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip

            def chain(s1=s1, t=t, big=big):
                o = ref_f(s1, hn_dtype=ttnn.bfloat16, t=t, x_mode=M.X_FLAT if big else M.X_STACKED,
                          xout_mode=M.XOUT_NONE, pc_mode=M.PC_STREAM_MAJOR if big else M.PC_STACKED)  # fmt: skip
                ids, scores = mlp.router(o.hn, decode=True)
                y = ttnn.reshape(mlp.routed(o.hn, ids, scores, mlp.cfg.moe_decode_micro_batch,
                                            mlp.cfg.experts_block_size), (1, 1, t, mlp.hidden))  # fmt: skip
                y2 = mlp.shared(o.hn, decode=True)
                out = mpost(y, s1, o.coef, t=t, x_mode=PX_FLAT if big else X_STACKED, out_mode=OUT_FLAT, y2=y2,
                            out_shape=(1, 1, t, 16384),
                            out_memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
                return dict(out=out, hn2=o.hn, ids=ids, y=y, y2=y2)

            cases[t] = (lambda s1=s1, t=t: mk(s1, t=t), chain)
        tr.warm(*[f for c in cases.values() for f in c])
        for t, (run_mk, run_chain) in cases.items():
            try:
                r_mk = tr.run(run_mk)
                got = dict(hn2=host(r_mk["hn2"]), ids=host(r_mk["ids"])[..., :t, :], y=host(r_mk["y"]),
                           y2=host(r_mk["y2"])[..., :mlp.hidden], out=host(r_mk["out"]))  # fmt: skip
                us_mk = timed(tr)
                tr.release()
                r_ch = tr.run(run_chain)
                want = dict(hn2=host(r_ch["hn2"]), ids=host(r_ch["ids"])[..., :t, :], y=host(r_ch["y"]),
                            y2=host(r_ch["y2"])[..., :mlp.hidden], out=host(r_ch["out"]))  # fmt: skip
                us_chain = timed(tr)
                tr.release()
                r = dict(t=t, execution=tr.mode, timing_us=dict(mk_moe=us_mk, chain=us_chain))
                for k_ in got:
                    a, b = got[k_].reshape(-1), want[k_].reshape(-1)[: got[k_].numel()]
                    r[f"{k_}_bitwise"] = bool(torch.equal(a, b))
                    r[f"{k_}_pcc"] = pcc(a, b)
                    r[f"{k_}_maxabs"] = float((a - b).abs().max())
                ok = r["hn2_bitwise"] and r["ids_bitwise"] and min(r[f"{k_}_pcc"] for k_ in got) >= 0.9999
                r["status"] = "record" if MOCK else ("pass" if ok else "fail")
                row("mk_moe_vs_chain", **r)
            except Exception:
                traceback.print_exc()
                err = traceback.format_exc()
                row("mk_moe_vs_chain", t=t, status="fail", error=err[-400:])
                tr.release()
                if "TIMEOUT" in err or "hang" in err:
                    flush()
                    print("DEV_MK_MOE_ABORT device timeout", flush=True)
                    os._exit(3)
    finally:
        flush()
        if own:
            ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MK_MOE_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
