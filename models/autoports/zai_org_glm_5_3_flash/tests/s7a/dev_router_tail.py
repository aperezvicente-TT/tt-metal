# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a device check of the fused router_tail program on one chip with a real MoE layer's router (L4 by default):
the router's DS logits matmul on real normalised inputs, then (a) the op chain (OptimizedRouter decode path) and (b) the
fused program on the SAME logits tensor: ids identical (order included) and scores bitwise, plus both vs the host
oracle (`router_tail.reference_step` on the device logits), at B = 1, 2, 8, 32. Dump-then-compare; rows ->
$GLM53_STAGE_EVIDENCE/{pcc.jsonl,tests.jsonl}. GLM53_MOCK=1: dry run on the mock cluster."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tt import router_tail as R
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedRouter
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (
    FdTileCacheSource,
    OptimizedDecoderConfig,
    OptimizedRouter,
    _upgrade,
)

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "4"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
DUMP = os.environ.get("GLM53_DUMP_DIR", "/opt/glm53/work-s7a/dumps")
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
ROWS = []


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, layer=LAYER, test=name, **kw)
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)


def main():
    os.makedirs(DUMP, exist_ok=True)
    cfg = hf_config()
    mesh = (
        ttnn.open_device(device_id=0)
        if MOCK
        else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=100_000_000)
    )
    tr = Traced(mesh)
    try:
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        ocfg = OptimizedDecoderConfig(fused_router_tail=True)
        router = FusedRouter(mesh, source.device(LAYER, "mlp.gate.weight", mesh),
                             source.host(LAYER, "mlp.gate.e_score_correction_bias"),
                             num_experts=cfg.n_routed_experts, top_k=cfg.num_experts_per_tok, fcfg=ocfg)  # fmt: skip
        _upgrade(router, OptimizedRouter, ocfg)
        g = golden(LAYER, 2048)["streams_in"].float()
        dumps = {}
        for b in (1, 2, 8, 32):
            try:
                x = g[0, 600 : 600 + 17 * b : 17].mean(1)
                x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
                xd = ttnn.from_torch(x.reshape(1, 1, b, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
                tail = router.tail

                def step():
                    """router logits -> the fused tail; the op chain on a fresh logits tensor of the same input."""
                    logits = router.ds(xd, None)
                    ids_f, sc_f = tail(logits, rows=b)
                    lg_i = ttnn.sharded_to_interleaved(logits, ttnn.L1_MEMORY_CONFIG)
                    router.tail = None
                    try:
                        ids_c, sc_c = router(xd, decode=True)
                    finally:
                        router.tail = tail
                    return lg_i, ids_f, sc_f, ids_c, sc_c

                lg_d, ids_f, sc_f, ids_c, sc_c = tr.run(step)  # traced (eager only as the compile step)
                lg = ttnn.to_torch(lg_d).float().reshape(b, -1)
                d = dict(logits=lg, ids_f=ttnn.to_torch(ids_f).reshape(b, -1), sc_f=ttnn.to_torch(sc_f).reshape(b, -1),
                         ids_c=ttnn.to_torch(ids_c).reshape(b, -1), sc_c=ttnn.to_torch(sc_c).reshape(b, -1))  # fmt: skip
                dumps[b] = d
                torch.save(dumps, os.path.join(DUMP, f"{RUN}_router_tail.pt"))
                ref_ids, ref_sc = R.reference_step(lg, tail.host_bias)
                same_ids = bool(torch.equal(d["ids_f"].long(), d["ids_c"].long()))
                same_sets = all(
                    set(a.tolist()) == set(c.tolist()) for a, c in zip(d["ids_f"].long(), d["ids_c"].long())
                )
                tr.replay()  # determinism: a second replay of the same trace
                r_ids, r_sc = ttnn.to_torch(ids_f).reshape(b, -1), ttnn.to_torch(sc_f).reshape(b, -1)
                deterministic = bool(torch.equal(r_ids, d["ids_f"]) and torch.equal(r_sc, d["sc_f"]))
                tr.release()
                r = dict(b=b, execution=tr.mode, deterministic=deterministic, ids_equal_chain=same_ids, id_sets_equal_chain=same_sets,
                         scores_bitwise_chain=bool(torch.equal(d["sc_f"], d["sc_c"])),
                         ids_equal_oracle=bool(torch.equal(d["ids_f"].long(), ref_ids)),
                         scores_bitwise_oracle_bf16=bool(torch.equal(d["sc_f"].to(torch.bfloat16), ref_sc)))  # fmt: skip
                ok = r["id_sets_equal_chain"] and r["scores_bitwise_chain"] and deterministic
                r["status"] = "record" if MOCK else ("pass" if ok else "fail")
                row("router_tail_vs_chain", **r)
            except Exception:
                traceback.print_exc()
                row("router_tail_vs_chain", b=b, status="fail", error=traceback.format_exc()[-400:])
    finally:
        with open(os.path.join(OUT, "pcc.jsonl"), "a") as f:
            for r in ROWS:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(OUT, "tests.jsonl"), "a") as f:
            for r in ROWS:
                f.write(json.dumps({k: r[k] for k in ("time_utc", "run", "test", "status", "b") if k in r}) + "\n")
        ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_ROUTER_TAIL_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
