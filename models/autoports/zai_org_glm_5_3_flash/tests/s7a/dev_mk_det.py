# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a determinism probe: ONE traced call of the sparse_kda layer program (tt/mk_sparse.py MkSparseLayer, the conv
history and the recurrent state restored inside the trace) replayed GLM53_DET_N times; after every replay EVERY
intermediate the program writes (hn, hn2, s1, logits, ids, scores, gu, m, y, y2, out) is read back and compared bitwise
with the first replay's -> per tensor: how many replays differ and the largest |diff|. The first tensor in program
order that differs localises a race. Layer GLM53_S7A_LAYER (default 4), T = GLM53_MK_TS (first value).
Row -> pcc.jsonl / tests.jsonl (status info: a measurement, not a gate)."""

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
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced, cached_decoder
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

LAYER = int(os.environ.get("GLM53_S7A_LAYER", "4"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
T = int(os.environ.get("GLM53_MK_TS", "32").split(",")[0])
N = int(os.environ.get("GLM53_DET_N", "20"))
KIND = os.environ.get("GLM53_DET_KIND", "mk")  # mk = MkSparseLayer; chain = the op chain it is compared against
ORDER = ("hn", "s1", "hn2", "logits", "ids", "scores", "gu", "m", "y", "y2", "out")


def main(mesh=None, ctx=None):
    own = mesh is None
    if own:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    if ctx is not None and ctx.pop("dev_mk_layer_objs", None) is not None:  # release the layer harness's resident set
        import gc

        gc.collect()
    tr = Traced(mesh)
    r = dict(time_utc=None, run=RUN, layer=LAYER, test="mk_sparse_determinism", t=T,
             watcher=bool(os.environ.get("TT_METAL_WATCHER")), knobs={k: v for k, v in os.environ.items()
                                                                        if k.startswith(("GLM53_MK", "GLM53_MHC", "GLM53_REC"))})  # fmt: skip
    try:
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_sparse import MkSparseLayer

        cfg = hf_config()
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        dec = cached_decoder(cfg, LAYER, mesh, ctx)
        kda, mlp = dec.layer.attn, dec.layer.mlp
        lins = {d.role: d for d in dec.layer.ds_linears()}
        a_args, f_args = mhc_args(source, cfg, LAYER), ffn_mhc_args(source, cfg, LAYER)
        one = MkSparseLayer(mesh, a_args, f_args, kda, lins, mlp) if KIND == "mk" else None
        g = golden(LAYER, 2048)["streams_in"].float()
        t = T
        x = ttnn.from_torch(g[0, 300 : 300 + 37 * t : 37].reshape(1, 1, t, 16384), dtype=ttnn.float32,
                            layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        active = torch.tensor([(i % 3) != 2 for i in range(t)]) if t > 1 else torch.tensor([True])
        act = masks(mesh, active, kda.lower_bound)
        torch.manual_seed(7)
        hist0 = ttnn.from_torch((torch.randn(1, 3, t, 24576) * 0.5).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        s0 = ttnn.from_torch(torch.randn(t, 64, 128, 128) * 0.1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                             device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        st = kda.allocate_state(t)

        def fn():
            ttnn.copy(hist0, st.C)
            ttnn.copy(s0, st.S)
            o = one(x, st.C, act, t=t, state=st.S)
            return {k: o[k] for k in ORDER}

        if KIND == "chain":  # dev_mk_layer's reference chain (flat modes above 4 rows), its intermediates
            from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
            from models.autoports.zai_org_glm_5_3_flash.tt.kda_pre import KdaPre
            from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import OUT_FLAT, OUT_STACKED
            from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import X_FLAT as PX_FLAT
            from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import X_STACKED, MhcPost

            ref_a, ref_f, mpost = M.MhcPre(mesh, **a_args), M.MhcPre(mesh, **f_args), MhcPost(mesh)
            ref_kda = KdaPre(kda, rows_max=32)
            big = t > 4
            pcm = M.PC_STREAM_MAJOR if big else M.PC_STACKED

            def fn():
                ttnn.copy(hist0, st.C)
                ttnn.copy(s0, st.S)
                r = ref_a(x, hn_dtype=ttnn.bfloat16, t=t, x_mode=M.X_FLAT,
                          xout_mode=M.XOUT_NONE if big else M.XOUT_STACKED, pc_mode=pcm)  # fmt: skip
                ko_ = ref_kda(lins["kda_qkv"](r.hn), lins["kda_lowrank"](r.hn), st.C, act, rows=t)
                a_ = kda._recurrence_and_output(*ko_[:6], st, t)
                s1 = mpost(a_, x if big else r.x_out, r.coef, t=t, x_mode=PX_FLAT if big else X_STACKED,
                           out_mode=OUT_FLAT if big else OUT_STACKED)  # fmt: skip
                f = ref_f(s1, hn_dtype=ttnn.bfloat16, t=t, x_mode=M.X_FLAT if big else M.X_STACKED,
                          xout_mode=M.XOUT_NONE, pc_mode=pcm)  # fmt: skip
                ids_, sc_ = mlp.router(f.hn, decode=True)
                y_ = ttnn.reshape(mlp.routed(f.hn, ids_, sc_, mlp.cfg.moe_decode_micro_batch, mlp.cfg.experts_block_size),
                                  (1, 1, t, mlp.hidden))  # fmt: skip
                y2_ = mlp.shared(f.hn, decode=True)
                out = mpost(y_, s1, f.coef, t=t, x_mode=PX_FLAT if big else X_STACKED, out_mode=OUT_FLAT, y2=y2_,
                            out_shape=(1, 1, t, 16384), out_memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
                return dict(hn=r.hn, s1=s1, hn2=f.hn, ids=ids_, scores=sc_, y=y_, y2=y2_, out=out)

        outs = tr.run(fn)
        ref = {k: ttnn.to_torch(v) for k, v in outs.items()}
        stats = {k: [0, 0.0] for k in ORDER if k in outs}
        for _ in range(N):
            tr.replay()
            for k, v in outs.items():
                a = ttnn.to_torch(v)
                if not torch.equal(a, ref[k]):
                    stats[k][0] += 1
                    stats[k][1] = max(stats[k][1], float((a.float() - ref[k].float()).abs().max()))
        tr.release()
        r.update(replays=N, differing={k: dict(replays=v[0], maxabs=v[1]) for k, v in stats.items()},
                 first_differing=next((k for k in ORDER if k in stats and stats[k][0]), None), kind=KIND,
                 status="info")  # fmt: skip
    except Exception:
        traceback.print_exc()
        tr.release()
        r.update(status="fail", error=traceback.format_exc()[-400:])
    r["time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print("ROW", json.dumps(r), flush=True)
    for name, keys in (("pcc.jsonl", None), ("tests.jsonl", ("time_utc", "run", "test", "status", "t"))):
        with open(os.path.join(OUT, name), "a") as f:
            f.write(json.dumps(r if keys is None else {k: r[k] for k in keys if k in r}) + "\n")
    if own:
        ttnn.close_mesh_device(mesh)
    return 0 if r["status"] != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())
