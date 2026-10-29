# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a batch-invariance bisect of the sparse_kda layer program (tt/mk_sparse.py MkSparseLayer, ONE program, one
chip): 14b #85 found the live rows of a B=32 step moving with the MAGNITUDE of idle chunk-mates' inputs. Here: T rows
(GLM53_MK_TS, default 32), live rows = golden residual rows, idle rows = other golden rows; two launches from the same
restored KDA state / conv history that differ ONLY in the idle rows (run B: idle rows x GLM53_INV_SCALE, default 2.5),
then the live rows of every hand-off tensor the program returns (hn, qkv, lr, kda, on, a, s1, hn2, logits, ids, scores,
gu, m, y2, y, out) and of the new state / history are compared bitwise: the first tensor, in phase order, whose live rows
differ is where idle rows leak into live rows. Live-row patterns (GLM53_INV_PATTERN): 'first' (rows 0..T/2-1 live) and
'alt' (even rows live), both by default. The active mask marks the live rows (as the decoder does for idle users).
Rows -> $GLM53_STAGE_EVIDENCE/{tests.jsonl,pcc.jsonl}."""

from __future__ import annotations

import gc
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
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import cached_decoder
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "4"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
TS = [int(v) for v in os.environ.get("GLM53_MK_TS", "32").split(",")]
SCALE = float(os.environ.get("GLM53_INV_SCALE", "2.5"))
PATTERNS = os.environ.get("GLM53_INV_PATTERN", "first,alt").split(",")
# phase order of the program's hand-off tensors (MkSparseLayer.__call__'s dict)
ORDER = ["hn", "qkv", "lr", "on", "a", "s1", "hn2", "logits", "scores", "ids", "gu", "m", "y2", "y", "out"]
ROWS = []


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, layer=LAYER, test=name, **kw)
    ROWS.append(r)
    print("ROW", json.dumps(r, default=str), flush=True)


def flush():
    for fn in ("tests.jsonl", "pcc.jsonl"):
        with open(os.path.join(OUT, fn), "a") as f:
            for r in ROWS:
                f.write(json.dumps(r, default=str) + "\n")
    ROWS.clear()


def host(t):
    return ttnn.to_torch(t).float()


def live_rows(t: torch.Tensor, live: list, T: int):
    """The live token rows of a hand-off tensor (token axis = the one of size T among the last two; ids / scores [T, k])."""
    if t.dim() >= 2 and t.shape[-2] >= T:
        return t[..., live, :]
    if t.shape[0] >= T:
        return t[live]
    return t


def main(mesh=None, ctx=None):
    own = mesh is None
    if ctx is not None and ctx.pop("dev_mk_layer_objs", None) is not None:  # their resident objects pin L1: release
        gc.collect()
    cfg = hf_config()
    if own:
        mesh = ttnn.open_device(device_id=0) if MOCK else ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    rc = 0
    try:
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_sparse import MkSparseLayer

        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        dec = cached_decoder(cfg, LAYER, mesh, ctx)
        kda, mlp = dec.layer.attn, dec.layer.mlp
        assert dec.layer.spec.mlp == "moe", f"layer {LAYER} is not sparse"
        lins = {d.role: d for d in dec.layer.ds_linears()}
        # GLM53_INV_XSL=1: the batched expert-TP routed phase (slice 0 of 8 of every selected expert, MKX_XSL)
        xsl = dict(xsl=0, xsl_tp=8) if os.environ.get("GLM53_INV_XSL") == "1" else {}
        one = MkSparseLayer(mesh, mhc_args(source, cfg, LAYER), ffn_mhc_args(source, cfg, LAYER), kda, lins, mlp, **xsl)
        g = golden(LAYER, 2048)["streams_in"].float()
        for T in TS:
            for pat in PATTERNS:
                live = list(range(T // 2)) if pat == "first" else list(range(0, T, 2))
                idle = [i for i in range(T) if i not in live]
                xa = torch.empty(T, 16384)
                xa[live] = g[0, 300 : 300 + 37 * len(live) : 37].reshape(len(live), 16384)
                xa[idle] = g[0, 1300 : 1300 + 37 * len(idle) : 37].reshape(len(idle), 16384)
                xb = xa.clone()
                xb[idle] = xa[idle] * SCALE
                act = masks(mesh, torch.tensor([i in live for i in range(T)]), kda.lower_bound)
                torch.manual_seed(2000 + T)
                hist0 = ttnn.from_torch((torch.randn(1, 3, T, 24576) * 0.5).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
                s0 = ttnn.from_torch(torch.randn(T, 64, 128, 128) * 0.1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                     device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
                st = kda.allocate_state(T)
                got = {}
                for tag, xh in (("A", xa), ("B", xb)):
                    ttnn.copy(hist0, st.C)
                    ttnn.copy(s0, st.S)
                    x = ttnn.from_torch(xh.reshape(1, 1, T, 16384), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
                    r_ = one(x, st.C, act, t=T, state=st.S)
                    ttnn.synchronize_device(mesh)
                    d = {k: host(r_[k]) for k in ORDER if k in r_ and isinstance(r_[k], ttnn.Tensor)}
                    d["state"] = host(st.S)
                    d["hist"] = host(st.C)
                    got[tag] = d
                    del r_, x  # this call's L1 hand-off tensors go before the next call (else they lower its L1 floor)
                    gc.collect()
                first_bad, cmp = None, {}
                for k in ORDER + ["state", "hist"]:
                    if k not in got["A"]:
                        continue
                    a_, b_ = got["A"][k], got["B"][k]
                    if k == "state":
                        la, lb = a_[live], b_[live]
                    elif k == "hist":
                        la, lb = a_[..., live, :], b_[..., live, :]
                    else:
                        la, lb = live_rows(a_, live, T), live_rows(b_, live, T)
                    same = bool(torch.equal(la, lb))
                    diff = float((la - lb).abs().max()) if la.numel() else 0.0
                    cmp[k] = dict(bitwise=same, max_abs=diff, shape=list(a_.shape))
                    if not same and first_bad is None and k in ORDER:
                        first_bad = k
                ok = first_bad is None and all(v["bitwise"] for v in cmp.values())
                row("mk_sparse_batch_invariance", t=T, pattern=pat, idle_scale=SCALE, live=len(live),
                    first_differing_phase=first_bad, compare=cmp, status="record" if MOCK else ("pass" if ok else "fail"))  # fmt: skip
                if not ok:
                    rc = 1
    except Exception:
        traceback.print_exc()
        row("mk_sparse_batch_invariance", status="fail", error=traceback.format_exc()[-600:])
        rc = 1
    finally:
        flush()
        if own:
            ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    print(f"DEV_MK_INV_DONE rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
