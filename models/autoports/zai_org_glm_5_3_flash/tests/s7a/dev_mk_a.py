# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel step A check: P1 (mhc_pre, attention sublayer) + P2 (qkv matmul) in ONE program (tt/mk_dense.py)
vs the two programs it replaces run back to back (MhcPre at its default placement, then the layer's DSLinear(qkv)) on
the same golden residual rows of a dense_kda layer (L0), T = 1 / 4 (stacked path) and 8 / 32 (stream-major path);
traced (eager only as the compile step): hn bitwise, qkv bitwise, post / comb bitwise, then the traced step time of
each. Rows -> $GLM53_STAGE_EVIDENCE/{pcc.jsonl,tests.jsonl}. GLM53_MOCK=1: dry run."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import load_optimized_decoder
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import MkStepA, decode_probe
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "0"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
REPLAYS = 20
ROWS = []


def row(name, **kw):
    r = dict(
        time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        run=RUN,
        layer=LAYER,
        test=name,
        watcher=bool(os.environ.get("TT_METAL_WATCHER")),
        **kw,
    )
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)


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


def mhc_args(source, cfg, layer=None):
    """layer: explicit in a long-lived process (this module's LAYER is fixed at its first import)"""
    layer = LAYER if layer is None else layer
    n = cfg.hc_mult
    scale = source.host(layer, "hc_attn_scale").float().reshape(-1)
    return dict(
        fn_t=source.host(layer, "hc_attn_fn").reshape(n * cfg.hidden_size, (2 + n) * n),
        scale=torch.cat([scale[0].repeat(n), scale[1].repeat(n), scale[2].repeat(n * n)]),
        bias=source.host(layer, "hc_attn_base").float().reshape(-1),
        norm_w=source.host(layer, "input_layernorm.weight").float().reshape(-1),
        hc_eps=cfg.hc_eps, x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps, iters=cfg.hc_sinkhorn_iters,
        hidden=cfg.hidden_size, n=n,
    )  # fmt: skip


def main():
    cfg = hf_config()
    mesh = (
        ttnn.open_device(device_id=0)
        if MOCK
        else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    )
    tr = Traced(mesh)
    try:
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        dec = load_optimized_decoder(cfg, LAYER, mesh)
        ds_qkv = {d.role: d for d in dec.layer.ds_linears()}["kda_qkv"]
        args = mhc_args(source, cfg, LAYER)
        mk = MkStepA(mesh, args, ds_qkv, probe=os.environ.get("GLM53_MK_PROBE") == "1",
                     rect=os.environ.get("GLM53_MK_RECT", "ds_rect"))  # fmt: skip
        ref = M.MhcPre(mesh, **args)
        row("mk_a_placement", owners=[(int(c.x), int(c.y)) for c in mk.place["k"]], root=(int(mk.place["root"].x),
            int(mk.place["root"].y)), readers=[(int(c.x), int(c.y)) for c, _, _ in mk.mm.assign], status="info")  # fmt: skip
        g = golden(LAYER, 2048)["streams_in"].float()
        for t in (1, 4, 8, 32):
            try:
                stacked = t <= 4
                kw = dict(t=t, x_mode=M.X_FLAT, xout_mode=M.XOUT_STACKED if stacked else M.XOUT_FLAT,
                          pc_mode=M.PC_STACKED if stacked else M.PC_STREAM_MAJOR)  # fmt: skip
                x = ttnn.from_torch(g[0, 300 : 300 + 37 * t : 37].reshape(1, 1, t, 16384), dtype=ttnn.float32,
                                    layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
                o, qkv = tr.run(lambda: mk(x, **kw))
                got = dict(hn=host(o.hn), post=host(o.post), comb=host(o.comb), qkv=host(qkv)[..., : ds_qkv.n])
                us_mk = timed(tr)
                if mk.probe is not None and not MOCK:  # the last replay's phase marks
                    row("mk_a_probe", t=t, status="info", **decode_probe(ttnn.to_torch(mk.probe)))
                tr.release()

                def chain():
                    r = ref(x, hn_dtype=ttnn.bfloat16, **kw)
                    return r, ds_qkv(r.hn)

                o2, qkv2 = tr.run(chain)
                want = dict(hn=host(o2.hn), post=host(o2.post), comb=host(o2.comb), qkv=host(qkv2))
                us_ref = timed(tr)
                tr.release()
                r = dict(t=t, execution=tr.mode, mk_us=us_mk, chain_us=us_ref)
                for k in got:
                    a, b = got[k].reshape(-1), want[k].reshape(-1)[: got[k].numel()]
                    r[f"{k}_bitwise"] = bool(torch.equal(a, b))
                    r[f"{k}_pcc"] = pcc(a, b)
                ok = all(r[f"{k}_bitwise"] for k in got)
                r["status"] = "record" if MOCK else ("pass" if ok else "fail")
                row("mk_a_vs_chain", **r)
            except Exception:
                traceback.print_exc()
                err = traceback.format_exc()
                row("mk_a_vs_chain", t=t, status="fail", error=err[-400:])
                if "TIMEOUT" in err or "hang" in err:
                    print("DEV_MK_A_ABORT device timeout", flush=True)
                    flush()
                    os._exit(3)
                tr.release()
    finally:
        flush()
        ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MK_A_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


_FLUSHED = 0


def flush():
    global _FLUSHED
    new, _FLUSHED = ROWS[_FLUSHED:], len(ROWS)
    with open(os.path.join(OUT, "pcc.jsonl"), "a") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT, "tests.jsonl"), "a") as f:
        for r in new:
            f.write(json.dumps({k: r[k] for k in ("time_utc", "run", "test", "status", "t") if k in r}) + "\n")


if __name__ == "__main__":
    sys.exit(main())
