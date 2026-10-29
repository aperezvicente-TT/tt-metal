# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7b baseline batch: the sparse_dsa attention half of layer GLM53_S7B_LAYER (default 3) as the op chain the
megakernel replaces, traced, T = GLM53_MK_TS (default 1,2,4), every case warmed before any capture:
  * att_half (MhcPre -> OptimizedMLA.decode -> MhcPost) and its segments (mhc_pre, the 30 MLA ops, mhc_post);
  * the flash-MLA op's configurations (heads per call x K chunk) on a fixed q_lat (device vs the default config);
  * the whole layer: chain attention + 7a's MkMoE, chain attention + the MoE op chain, both alone and 2 layers chained
    in one trace (steady state per layer);
  * determinism: a second replay of the attention half bitwise equal to the first.
Rows -> pcc.jsonl / tests.jsonl / perf.jsonl of the stage evidence dir."""

from __future__ import annotations

import os
import sys
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.s7b import reload_deps

reload_deps.refresh()  # helper modules fresh per job (serve.py reloads only the job module)
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tests.s7b.mla_common import MlaLayer, Rows, compare, host, timed
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import OUT_FLAT, X_STACKED

LAYER = int(os.environ.get("GLM53_S7B_LAYER", "3"))
TS = [int(v) for v in os.environ.get("GLM53_MK_TS", "1,2,4").split(",")]
# (heads per call, K chunk, grid): the flash-MLA op configurations that may fit next to the megakernel servers'
# kernel-config ring
FLASH = [
    (h, k, g)
    for g in ("8x4", "8x8", "4x4", "11x4", "8x2")
    for h, k in ((32, 32), (64, 32), (32, 64), (64, 64), (64, 128))
]


def main(mesh=None, ctx=None):
    own = mesh is None
    if own:  # standalone / mock (GLM53_MOCK=1): the serve processes' worker_l1_size (GLM53_WORKER_L1, default 1264512)
        wl1 = int(os.environ.get("GLM53_WORKER_L1", "1264512"))
        mesh = (ttnn.open_device(device_id=0, worker_l1_size=wl1) if os.environ.get("GLM53_MOCK") == "1" else
                ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000, worker_l1_size=wl1))  # fmt: skip
    rows = Rows(LAYER)
    tr = Traced(mesh)
    try:
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_moe import MkMoE

        L = MlaLayer(mesh, ctx, LAYER)
        mlp = L.mlp
        mkmoe = MkMoE(mesh, L.f_args, mlp)
        ref_f = M.MhcPre(mesh, **L.f_args)

        def moe_chain(s1, t):
            o = ref_f(s1, hn_dtype=ttnn.bfloat16, t=t, x_mode=M.X_STACKED, xout_mode=M.XOUT_NONE,
                      pc_mode=M.PC_STACKED)  # fmt: skip
            ids, scores = mlp.router(o.hn, decode=True)
            y = ttnn.reshape(mlp.routed(o.hn, ids, scores, mlp.cfg.moe_decode_micro_batch, mlp.cfg.experts_block_size),
                             (1, 1, t, mlp.hidden))  # fmt: skip
            return L.mpost(y, s1, o.coef, t=t, x_mode=X_STACKED, out_mode=OUT_FLAT, y2=mlp.shared(o.hn, decode=True),
                           out_shape=(1, 1, t, 16384), out_memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip

        cases = {}
        for t in TS:
            c = L.case(t)
            # fixed inputs of the segment timings (eager, before any capture)
            r0 = L.mhc_pre(c)
            hn_fix = ttnn.clone(r0.hn)
            q_lat, _ = L.attn._project_decode(hn_fix)
            q_fix = ttnn.permute(q_lat, (0, 2, 1, 3))  # [1,t,64,512]
            a_fix = ttnn.from_torch(torch.randn(1, 1, t, 4096).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
            fns = dict(
                att_half=lambda c=c: L.chain_att(c),
                mhc_pre=lambda c=c: L.mhc_pre(c).hn,
                mla=lambda c=c, h=hn_fix: L.mla(c, h),
                mhc_post=lambda c=c, a=a_fix, r=r0: L.mhc_post(c, a, r),
                layer_mkmoe=lambda c=c, t=t: mkmoe(L.chain_att(c), t=t)["out"],
                layer_chain=lambda c=c, t=t: moe_chain(L.chain_att(c), t),
            )
            for heads, kc, gs in FLASH:
                gx, gy = (int(v) for v in gs.split("x"))
                fns[f"flash_h{heads}_k{kc}_g{gs}"] = lambda c=c, q=q_fix, h=heads, k=kc, g=ttnn.CoreCoord(
                    gx, gy
                ): L.flash(c, q, heads=h, kc=k, grid=g)
            cases[t] = (c, fns)
        # warm every case (compile) before any capture; a configuration the op rejects (L1) is recorded and dropped
        for t, (c, fns) in cases.items():
            for name in list(fns):
                try:
                    tr.warm(fns[name])
                except Exception as e:  # noqa: BLE001
                    rows(f"warm_{name}", t=t, status="info", rejected=str(e)[:300])
                    del fns[name]
        for t, (c, fns) in cases.items():
            try:
                us, outs = {}, {}
                for name, fn in fns.items():
                    out = tr.run(fn)
                    outs[name] = host(out)
                    us[name] = timed(tr, 50)
                    if name == "att_half":  # determinism: one more replay, same bits
                        tr.replay()
                        rows(
                            "att_half_determinism", t=t, status="pass" if torch.equal(host(out), outs[name]) else "fail"
                        )
                    tr.release()
                rows("chain_baseline", t=t, execution=tr.mode, timing_us=us, variant=f"op_chain_k{L.kchunk}",
                     positions=c["pos"], status="info")  # fmt: skip
                ref = (
                    f"flash_h{L.attn.h if t >= L.attn.fcfg.mla_single_call_min_batch else L.attn.decode_heads}"
                    f"_k{L.kchunk}_g{L.flash_grid.x}x{L.flash_grid.y}"
                )
                for heads, kc, gs in FLASH:
                    k = f"flash_h{heads}_k{kc}_g{gs}"
                    if k not in outs or ref not in outs:
                        continue
                    r = dict(t=t, ref=ref)
                    compare(r, {"o_lat": outs[k]}, {"o_lat": outs[ref]})
                    r["status"] = "pass" if r["o_lat_pcc"] >= 0.9999 else "fail"
                    rows(f"flash_cfg_{k}", **r)
            except Exception:
                traceback.print_exc()
                err = traceback.format_exc()
                rows("chain_baseline", t=t, status="fail", error=err[-600:])
                tr.release()
                if "TIMEOUT" in err or "hang" in err:
                    rows.flush()
                    os._exit(3)
        # steady state: two layers chained in one trace (layer i+1 fed layer i's output)
        for t, (c, fns) in cases.items():
            try:
                us = {}
                x0 = c["x"]

                def two(layer_fn, x0=x0, c=c):
                    y = layer_fn(dict(c, x=x0))
                    return layer_fn(dict(c, x=y))

                for name, lf in (("layer_mkmoe_x2", lambda cc, t=t: mkmoe(L.chain_att(cc), t=t)["out"]),
                                 ("layer_chain_x2", lambda cc, t=t: moe_chain(L.chain_att(cc), t))):  # fmt: skip
                    tr.warm(lambda lf=lf: two(lf))
                    tr.run(lambda lf=lf: two(lf))
                    us[name] = timed(tr, 30)
                    tr.release()
                rows("chain_layer_steady", t=t, execution=tr.mode, timing_us=us, variant="op_chain",
                     steady_per_layer_us={k: round(v / 2, 1) for k, v in us.items() if v}, status="info")  # fmt: skip
            except Exception:
                traceback.print_exc()
                rows("chain_layer_steady", t=t, status="fail", error=traceback.format_exc()[-600:])
                tr.release()
    finally:
        rows.flush()
        if own:
            ttnn.close_device(mesh) if os.environ.get("GLM53_MOCK") == "1" else ttnn.close_mesh_device(mesh)
    bad = rows.failed()
    print(f"DEV_MLA_CHAIN_DONE rows={len(rows.rows)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
