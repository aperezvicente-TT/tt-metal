# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""8-chip validation of per-chip parallel weight loading (MAINTENANCE WINDOW #2 — do not run outside an announced slot).

One process, the 2x4 mesh, plan pipeline8_ring_wrap (45 layers):
  1. parallel load (MultichipDecoder(parallel_load=True); page cache COLD for most files) -> per-chip and total wall,
     per-layer sampled device-tensor hashes; free everything;
  2. serial load (the accepted path; page cache now WARM: a conservative baseline) -> the same hashes;
  3. compare: every sampled tensor bit-identical between the two loads (same files, same loader calls; a race in the
     threaded path would show here), and the timing.
Sample per layer: 5 evenly spaced tensors of the decoder's device tensors (<= 256 MiB each) + gate|up and down of
routed experts 0 / 143 / 287 (MoE layers). Hash = sha256 of the host bytes (ttnn.to_torch).
The GIL-release patch (load_tensor_flatbuffer / to_device / copy_host_to_device_tensor) must be in the _ttnn.so that
runs (worktree glm53/glm53-flash-s4-parload, rebuilt); without it the threads serialize on the GIL (still
correct, no speedup).

run (window #2, all 8 chips):
  GLM53_LOCK=all GLM53_EXCLUSIVE=1 scripts/dev.sh timeout -s INT --preserve-status 1800 python \
    $TT_METAL_HOME/models/autoports/zai_org_glm_5_3_flash/tests/test_multichip_parallel_load.py
rows -> $GLM53_OUT_DIR (default $GLM53_WS/doc/ttnn/04-multichip-decoder/parload)/parload.jsonl
"""
import gc
import hashlib
import json
import os
import sys
import time

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config
from models.autoports.zai_org_glm_5_3_flash.tests.test_multichip_decoder import MESH_DIR, E, Loader, free_decoder
from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import (
    MultichipDecoder,
    _device_tensors,
    chip_coords,
    close_pipeline_mesh,
    open_pipeline_mesh,
    plan_from_partition,
)

WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
OUT = os.environ.get("GLM53_OUT_DIR", os.path.join(WS, "doc/ttnn/04-multichip-decoder/parload"))
EXPERTS = (0, 143, 287)
MAX_BYTES = 256 << 20


def record(row):
    os.makedirs(OUT, exist_ok=True)
    row = dict(
        time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), code=os.environ.get("GLM53_CODE_SHA"), **row
    )
    with open(os.path.join(OUT, "parload.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print("ROW", json.dumps(row)[:300], flush=True)


def tensor_hash(t):
    h = ttnn.to_torch(t).contiguous().view(-1)
    return hashlib.sha256(h.view(dtype=__import__("torch").uint8).numpy().tobytes()).hexdigest()[:16]


def small(t):
    try:
        return t.volume() * max(1, t.element_size()) <= MAX_BYTES
    except Exception:  # noqa: BLE001  (block-float dtypes may not report an element size)
        return t.volume() <= MAX_BYTES


def sample_hashes(dec):
    ts = [t for t in _device_tensors(dec) if t.is_allocated() and small(t)]
    pick = sorted({round(i * (len(ts) - 1) / 4) for i in range(5)}) if ts else []
    out = {f"t{i}": tensor_hash(ts[i]) for i in pick}
    moe = getattr(dec.layer, "mlp", None)
    for name in ("gate_up", "down"):
        lst = getattr(moe, name, None) or getattr(moe, f"w_{name}", None)
        if isinstance(lst, (list, tuple)) and len(lst) == 288:
            for e in EXPERTS:
                out[f"{name}{e}"] = tensor_hash(lst[e])
    return out


def load(mesh, stages, parallel, cfg):
    loader = Loader(cfg, "optimized", prefetch=True)
    t0 = time.perf_counter()
    m = MultichipDecoder(mesh, stages, loader=loader, handoff="host", parallel_load=parallel)
    for st in m.stages:
        ttnn.synchronize_device(st.mesh)
    wall = time.perf_counter() - t0
    per_chip = {}
    for st in m.stages:
        per_chip[st.plan.chip] = round(
            per_chip.get(st.plan.chip, 0.0) + sum(loader.load_s[l] for l in st.plan.layers), 1
        )
    hashes = {f"S{si}/L{l}": sample_hashes(st.decoders[l]) for si, st in enumerate(m.stages) for l in st.plan.layers}
    if loader.prefetch is not None:
        loader.prefetch.close()
    return m, dict(mode="parallel" if parallel else "serial", wall_s=round(wall, 1), sum_layer_load_s=per_chip), hashes


def main():
    cfg = hf_config()
    mesh = open_pipeline_mesh(
        (2, 4), trace_region_size=600_000_000, mgd=os.path.join(MESH_DIR, "glm8_2x4_mesh_graph_descriptor.textproto")
    )
    try:
        stages = plan_from_partition(os.path.join(E, "partition.json"), chip_coords(mesh), key="pipeline8_ring_wrap")
        results = {}
        for parallel in (True, False):  # parallel first: cold page cache = conservative for the parallel path
            m, row, hashes = load(mesh, stages, parallel, cfg)
            results[row["mode"]] = hashes
            record(dict(test="multichip_parallel_load/time", **row))
            for st in m.stages:
                for dec in st.decoders.values():
                    free_decoder(dec)
            m.close_links()
            del m
            gc.collect()
        a, b = results["parallel"], results["serial"]
        diff = sorted(k for k in a if a[k] != b.get(k))
        record(
            dict(
                test="multichip_parallel_load/bitexact",
                layers=len(a),
                tensors=sum(len(v) for v in a.values()),
                mismatched_layers=diff,
                status="pass" if not diff and a.keys() == b.keys() else "fail",
            )
        )
        return 0 if not diff else 1
    finally:
        close_pipeline_mesh(mesh, None)


if __name__ == "__main__":
    sys.exit(main())
