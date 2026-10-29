# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel: compute headroom of the 16 DS readers per weight phase (input to the forwarding-ring design).
Per dense_kda weight (L0: kda_qkv, kda_lowrank, kda_o, mlp_gate_up, mlp_down), one reader program with that one phase,
traced, M = 1 / 32: (a) the normal stream (DRAM-bound), (b) MK_NO_DRAM (the weight streamer skips its reads: the ring
slots are pushed as they are) = the phase's compute-bound time. If (b) << (a), weights prefetched into L1 during the
glue phases are consumed faster than DRAM delivers them; if (b) ~ (a), the ring needs more compute cores as well.
Timing only (MK_NO_DRAM outputs are garbage). Rows -> perf.jsonl / tests.jsonl."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced, cached_decoder
from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import TILE_BYTES, MMPhase, ReaderPhases

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "0"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
KB = dict(kda_qkv=2, kda_lowrank=8, kda_o=8, mlp_gate_up=2, mlp_down=8)
REPLAYS = 30
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
            f.write(json.dumps({k: r[k] for k in ("time_utc", "run", "test", "status", "m") if k in r}) + "\n")


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
    """mesh / ctx: run inside the long-lived device process (serve.py): no open / close, the decoder cached."""
    own = mesh is None
    cfg = hf_config()
    if own:
        mesh = (
            ttnn.open_device(device_id=0)
            if MOCK
            else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
        )
    tr = Traced(mesh)
    try:
        dec = cached_decoder(cfg, LAYER, mesh, ctx)
        lins = {d.role: d for d in dec.layer.ds_linears()}
        cases = []
        for role, kb in KB.items():
            ds = lins[role]
            progs = {nd: ReaderPhases(mesh, [MMPhase(ds=ds, kb=kb)], no_dram=nd) for nd in (False, True)}
            for m in (1, 32):
                x = ttnn.from_torch((torch.randn(1, 1, m, ds.k) * 0.5).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
                for nd, rp in progs.items():
                    cases.append((role, m, nd, ds, lambda rp=rp, x=x: rp([x])))
        tr.warm(*[c[-1] for c in cases])  # every program compiled before any capture
        res = {}
        for role, m, nd, ds, fn in cases:
            try:
                tr.run(fn)
                res[(role, m, nd)] = timed(tr)
                tr.release()
            except Exception:
                traceback.print_exc()
                row("mk_rate", role=role, m=m, no_dram=nd, status="fail", error=traceback.format_exc()[-400:])
                tr.release()
        for role, kb in KB.items():
            ds = lins[role]
            wbytes = (ds.k // 32) * (ds.n_pad // 32) * TILE_BYTES[ds.weight.dtype]
            for m in (1, 32):
                a, b = res.get((role, m, False)), res.get((role, m, True))
                if a is None or b is None:
                    continue
                r = dict(role=role, m=m, kb=kb, k=ds.k, n=ds.n, execution=tr.mode,
                         timing_us=dict(dram=a, no_dram=b), compute_headroom=round(a / b, 3) if b else None,
                         weight_bytes=wbytes, status="record")  # fmt: skip
                if wbytes and a:
                    r["dram_gbps"] = round(wbytes / a / 1e3, 1)
                row("mk_rate", **r)
    finally:
        flush()
        if own:
            ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MK_RATE_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
