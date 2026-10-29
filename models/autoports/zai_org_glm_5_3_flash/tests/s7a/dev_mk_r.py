# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel step R check: the reader cores' multi-phase program (tt/mk_readers.py: qkv, low-rank, o_proj,
gate_up, down of a dense_kda layer back to back in ONE program on the 16 DS readers, weights streamed across phases)
vs the five DSLinear calls in sequence, on random activations per phase (M = 1 / 32); traced (eager only as the compile
step, every case warmed before any capture): per-phase PCC / bitwise vs DSLinear (every phase at its DSLinear's fidelity: low-rank HiFi4), determinism, and the
traced time of the one program vs the five calls. Rows -> pcc.jsonl / tests.jsonl (watcher-tagged)."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import load_optimized_decoder
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import MMPhase, ReaderPhases

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "0"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
REPLAYS = int(os.environ.get("GLM53_MK_REPLAYS", "20"))
KB = dict(kda_qkv=2, kda_lowrank=32, kda_o=8, mlp_gate_up=2, mlp_down=8)
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
            f.write(json.dumps({k: r[k] for k in ("time_utc", "run", "test", "status", "m") if k in r}) + "\n")


def timed(tr):
    if not tr.enabled:
        return None
    d = []
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        tr.replay()
        d.append((time.perf_counter() - t0) * 1e6)
    return sorted(d)[len(d) // 2]


def main():
    cfg = hf_config()
    mesh = (
        ttnn.open_device(device_id=0)
        if MOCK
        else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    )
    tr = Traced(mesh)
    try:
        dec = load_optimized_decoder(cfg, LAYER, mesh)
        lins = {d.role: d for d in dec.layer.ds_linears()}
        phases = [MMPhase(ds=lins[r], kb=KB[r]) for r in KB if r in lins]
        rp = ReaderPhases(mesh, phases)
        row("mk_r_setup", roles=[p.ds.role for p in phases], rings={str(k): [str(v[0]), v[1]] for k, v in rp.rings.items()},
            l1_bytes=rp.l1_bytes(), status="info")  # fmt: skip
        cases = {}
        for m in (1, 32):
            xs = [ttnn.from_torch((torch.randn(1, 1, m, p.ds.k) * 0.5).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG)
                  for p in phases]  # fmt: skip
            cases[m] = (xs, lambda xs=xs: rp(xs), lambda xs=xs: [p.ds(x) for p, x in zip(phases, xs)])
        tr.warm(*[f for c in cases.values() for f in c[1:]])  # every program compiled before any capture
        for m, (xs, run_mk, run_chain) in cases.items():
            try:
                ys = tr.run(run_mk)
                got = [ttnn.to_torch(y).float().reshape(m, -1)[:, : p.ds.n] for y, p in zip(ys, phases)]
                us_mk = timed(tr)
                tr.replay()
                got2 = [ttnn.to_torch(y).float().reshape(m, -1)[:, : p.ds.n] for y, p in zip(ys, phases)]
                tr.release()
                refs = tr.run(run_chain)
                want = [ttnn.to_torch(y).float().reshape(m, -1)[:, : p.ds.n] for y, p in zip(refs, phases)]
                us_chain = timed(tr)
                tr.release()
                r = dict(m=m, execution=tr.mode, mk_us=us_mk, chain_us=us_chain,
                         deterministic=all(torch.equal(a, b) for a, b in zip(got, got2)),
                         alloc_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1",
                         fidelity={p.ds.role: str(p.fidelity()) for p in phases})  # fmt: skip
                for p, a, b in zip(phases, got, want):
                    r[f"{p.ds.role}_pcc"] = pcc(a, b)
                    r[f"{p.ds.role}_bitwise"] = bool(torch.equal(a, b))
                ok = r["deterministic"] and all(r[f"{p.ds.role}_pcc"] >= 0.9999 for p in phases)
                r["status"] = "record" if MOCK else ("pass" if ok else "fail")
                row("mk_r_vs_dslinear", **r)
            except Exception:
                traceback.print_exc()
                err = traceback.format_exc()
                row("mk_r_vs_dslinear", m=m, status="fail", error=err[-400:])
                if "TIMEOUT" in err or "hang" in err:
                    flush()
                    print("DEV_MK_R_ABORT device timeout", flush=True)
                    os._exit(3)
                tr.release()
    finally:
        flush()
        ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MK_R_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
