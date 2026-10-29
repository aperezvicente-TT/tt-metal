# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a device numerics of the fused mhc_post (moe_post) program on one chip with a real layer's attention mHC:
the fused mhc_pre program produces the residual copy, post / comb and the compact coefficient tile from golden residual
rows; then on the SAME post / comb / residual
  * the fused post-mix program vs `mhc_post.reference_step` (float64 on the device's own inputs) and vs today's op chain
    (mix_streams on the stacked path, OptimizedStreamMajorHyperConnection.post on the stream-major path; with a second
    summand the chain first adds the two bf16 tensors, as OptimizedMoE.__call__ does),
  * the coefficient tile vs the post / comb outputs (bitwise: pure data movement),
  * determinism (two calls bitwise equal),
at T = 1, 2, 4 (stacked residual; stacked output = the attention sublayer's use, flat DRAM output + y2 = the FFN's) and
T = 8, 32 (stream-major residual, flat output, with and without y2). Dump-then-compare; rows ->
$GLM53_STAGE_EVIDENCE/{pcc.jsonl,tests.jsonl}. GLM53_MOCK=1: logic dry run on the mock cluster."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_mhc_pre import Chain, make_program, post_comb_host
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_post as P
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "4"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
DUMP = os.environ.get("GLM53_DUMP_DIR", "/opt/glm53/work-s7a/dumps")
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
PCC_GATE = 0.9999
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
ROWS = []
TR = None  # Traced(mesh), set in main


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, layer=LAYER, test=name, **kw)
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)


def host(t):
    return ttnn.to_torch(t).float()


def stats(name, got, ref):
    d, r = got.double().reshape(-1), ref.double().reshape(-1)
    err = (d - r).abs()
    return {
        f"{name}_pcc": pcc(d, r),
        f"{name}_max_abs": float(err.max()),
        f"{name}_max_rel": float(err.max() / r.abs().max()),
    }


def check_case(dumps, mesh, prog, chain, mp, x_rows, t, add):
    stacked = t <= 4
    pc = M.PC_STACKED if stacked else M.PC_STREAM_MAJOR
    x_host = x_rows.reshape(t, 16384).float()
    x_dram = ttnn.from_torch(x_host.reshape(1, 1, t, 16384), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                             memory_config=DRAM)  # fmt: skip
    g = torch.Generator().manual_seed(t * 2 + int(add))
    y = (torch.randn(t, 4096, generator=g) * 0.8).to(torch.bfloat16)
    y2 = (torch.randn(t, 4096, generator=g) * 0.3).to(torch.bfloat16) if add else None
    mk = lambda v: ttnn.from_torch(v.reshape(1, 1, t, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                   memory_config=L1)  # fmt: skip
    y_d, y2_d = mk(y), (mk(y2) if add else None)
    out_mode = (P.OUT_FLAT if add else P.OUT_STACKED) if stacked else P.OUT_FLAT

    def step():
        """mhc_pre (the post-mix's inputs) -> the fused post-mix -> today's chain on the same post / comb / residual."""
        o = prog(x_dram, t=t, x_mode=M.X_FLAT, xout_mode=M.XOUT_STACKED if stacked else M.XOUT_FLAT, pc_mode=pc)
        if stacked:
            # attention use: stacked in -> stacked out (L1); FFN use: stacked in -> the flat DRAM layer output (+ y2)
            kw = dict(t=t, x_mode=P.X_STACKED, out_mode=out_mode, y2=y2_d)
            if add:
                kw.update(out_shape=(1, 1, t, 16384), out_memory_config=DRAM)
        else:
            kw = dict(t=t, x_mode=P.X_FLAT, out_mode=out_mode, y2=y2_d, out_shape=tuple(o.x_out.shape))
        out = mp(y_d, o.x_out, o.coef, **kw)
        ysum = ttnn.add(y_d, y2_d) if add else y_d  # the chain: bf16 routed + shared
        if stacked:
            ch = chain.hc.post(ysum, o.x_out, o.post, o.comb)  # mix_streams, [T,1,4,4096]
        else:
            ch = chain.sm.post(ysum, o.x_out, o.post, o.comb)  # [1,4,T,4096]
        return o, out, ch

    o, out, ch = TR.run(step)  # traced (eager only as the compile step)
    shape = (
        lambda v: v.reshape(t, 4, 4096)
        if (out_mode == P.OUT_STACKED or stacked)
        else v.reshape(4, t, 4096).permute(1, 0, 2)
    )
    got = shape(host(out))
    ch_h = host(ch).reshape(t, 4, 4096) if stacked else host(ch).reshape(4, t, 4096).permute(1, 0, 2)
    TR.replay()  # determinism: a second replay of the same trace
    got2 = shape(host(out))
    post, comb = post_comb_host(o.post, o.comb, t, pc)
    coef = host(o.coef).reshape(32, 32)
    x_res = host(o.x_out)
    x_res = x_res.reshape(t, 4, 4096) if stacked else x_res.reshape(4, t, 4096).permute(1, 0, 2)
    key = f"T{t}_{'add' if add else 'single'}"
    dumps[key] = dict(y=y, y2=y2, x=x_res, post=post, comb=comb, coef=coef, out=got, out2=got2, chain=ch_h)
    torch.save(dumps, os.path.join(DUMP, f"{RUN}_mhc_post.pt"))  # dump before comparing
    ref = P.reference_step(y, x_res, post, comb, y2)
    r = dict(t=t, add=add, stacked_residual=stacked, out_mode=out_mode, execution=TR.mode)
    r.update(stats("out_vs_oracle", got, ref))
    r.update(stats("out_vs_chain", got, ch_h))
    r.update(stats("chain_vs_oracle", ch_h, ref))
    r["coef_bitwise"] = bool(torch.equal(coef[:t, :4], post) and torch.equal(coef[:t, 4:20], comb.reshape(t, 16)))
    r["residual_bitwise"] = bool(torch.equal(x_res, x_host.reshape(t, 4, 4096)))
    r["deterministic"] = bool(torch.equal(got, got2))
    ok = min(r["out_vs_oracle_pcc"], r["out_vs_chain_pcc"]) >= PCC_GATE and r["coef_bitwise"] and r["deterministic"]
    r["status"] = "record" if MOCK else ("pass" if ok else "fail")
    row("mhc_post_numerics", **r)
    TR.release()
    for tns in (x_dram, y_d) + ((y2_d,) if add else ()):
        ttnn.deallocate(tns)


def main():
    os.makedirs(DUMP, exist_ok=True)
    cfg = hf_config()
    global TR
    mesh = (
        ttnn.open_device(device_id=0)
        if MOCK
        else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    )
    TR = Traced(mesh)
    try:
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        prog = make_program(mesh, source, cfg, "attn", "input_layernorm.weight")
        chain = Chain(mesh, source, cfg, "attn", "input_layernorm.weight")
        mp = P.MhcPost(mesh)
        g = golden(LAYER, 2048)["streams_in"].float()
        dumps = {}
        for t in (1, 2, 4, 8, 32):
            for add in (False, True):
                try:
                    check_case(dumps, mesh, prog, chain, mp, g[0, 900 : 900 + 29 * t : 29], t, add)
                except Exception:
                    traceback.print_exc()
                    row("mhc_post_numerics", t=t, add=add, status="fail", error=traceback.format_exc()[-400:])
    finally:
        with open(os.path.join(OUT, "pcc.jsonl"), "a") as f:
            for r in ROWS:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(OUT, "tests.jsonl"), "a") as f:
            for r in ROWS:
                f.write(
                    json.dumps({k: r[k] for k in ("time_utc", "run", "test", "status", "t", "add") if k in r}) + "\n"
                )
        ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MHC_POST_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
