# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the GLM-5.3-Flash functional-decoder tests (test-harness boundary: torch/host I/O lives here).

References come from the stage-0c fp32 CPU goldens (`$GLM53_GOLDENS`, `$glm53-goldens`): per captured layer and prompt
length T, `streams_in/out [1, T, 4, 4096]` (prefill) and `decode0_streams_in/out` (first greedy decode step at position
T). The decoder layers are causal, so for any prefix length n <= T the golden output of the prefix is
`streams_out[:, :n]`, and a decode step at position p after a prefix of p tokens must reproduce row p of the golden
prefill (`streams_in[:, p]` -> `streams_out[:, p]`). MLA/DSA layers are exact-dense only while kv_len <= 2051, which
holds for every row p <= 2050 of every golden prompt (the 4096-token golden attends sparsely only from p >= 2051).
"""

from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache

import torch

import ttnn

GOLDENS = os.environ.get("GLM53_GOLDENS", "/opt/glm53/kit/goldens")
EVIDENCE_DIR = os.environ.get(
    "GLM53_STAGE_EVIDENCE",
    os.path.join(
        os.environ.get("GLM53_WS", "/opt/glm53/kit"),
        "doc",
        "ttnn",
        "01-functional-decoder",
    ),
)
PCC_BAR = 0.995
HC_WIDTH = 4 * 4096

if GOLDENS not in sys.path:
    sys.path.insert(0, GOLDENS)
try:
    import compare  # noqa: E402  (goldens/compare.py: golden loaders for the real-weight tests)
except ImportError:  # synthetic / host-only tests run without the goldens checkout
    compare = None

# captured golden layers per layer kind
LAYERS = {"dense_kda": (0, 1, 2), "sparse_kda": (4, 44), "sparse_dsa": (3, 7, 43)}


@lru_cache(maxsize=4)
def golden(layer: int, prompt: int) -> dict:
    return compare.load_layer_io(layer, prompt)


def to_dev(t: torch.Tensor, mesh, dtype=ttnn.float32) -> ttnn.Tensor:
    """[1, T, 4, 4096] (or [1, T, 16384]) host streams -> [1, 1, T, 16384] device tensor."""
    rows = t.shape[1]
    return ttnn.from_torch(
        t.reshape(1, 1, rows, HC_WIDTH).float().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh
    )


def to_host(t: ttnn.Tensor) -> torch.Tensor:
    """[1, 1, T, 16384] device -> [1, T, 4, 4096] fp32 host."""
    x = ttnn.to_torch(t).float()
    return x.reshape(1, x.shape[-2], 4, 4096)


def pcc(a, b) -> float:
    """Pearson correlation of the flattened tensors in float64 (same definition as goldens/compare.py)."""
    a = torch.as_tensor(a).reshape(-1).to(torch.float64)
    b = torch.as_tensor(b).reshape(-1).to(torch.float64)
    if a.numel() != b.numel():
        raise ValueError(f"size mismatch {a.numel()} vs {b.numel()}")
    a, b = a - a.mean(), b - b.mean()
    den = (a.norm() * b.norm()).item()
    if den == 0.0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float((a @ b).item() / den)


def delta_pcc(out, ref_out, ref_in) -> float:
    """PCC of the layer's update (out - in): stricter than the stream PCC, which the residual dominates."""
    return pcc(out - ref_in, ref_out - ref_in)


def int_tensor(values, mesh, dtype=ttnn.int32) -> ttnn.Tensor:
    return ttnn.from_torch(torch.tensor(values, dtype=torch.int32), dtype=dtype, device=mesh)


def record(row: dict, name: str = "pcc_real_weights.jsonl"):
    """Append one evidence row (JSON line) under the stage evidence dir."""
    from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_TRACKING

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    row = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **row,
        "trace_alloc_tracking": bool(TRACE_ALLOC_TRACKING),
        "watcher": os.environ.get("TT_METAL_WATCHER", ""),
    }
    with open(os.path.join(EVIDENCE_DIR, name), "a") as f:
        f.write(json.dumps(row) + "\n")


def open_mesh(trace_region_size: int = 0):
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=trace_region_size)


# Real-weight inputs of this host (test boundary): HF snapshot (config.json) and the pre-packed tile cache.
SNAPSHOT = os.environ.get("GLM53_HF_SNAPSHOT", "")
CACHE_DIR = os.environ.get("GLM53_CACHE_DIR", "")
MANIFEST = os.path.join(os.environ.get("GLM53_WS", ""), "weights", "manifest.json")


def hf_config():
    from models.autoports.zai_org_glm_5_3_flash.tt.config import load_hf_text_config

    return load_hf_text_config(SNAPSHOT)


def load_decoder(cfg, layer: int, mesh):
    from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import FunctionalDecoder

    return FunctionalDecoder.from_weight_cache(
        hf_config=cfg, layer_idx=layer, mesh_device=mesh, cache_dir=CACHE_DIR, manifest_path=MANIFEST
    )


def streams4(y: torch.Tensor) -> torch.Tensor:
    """[1, T, 16384] (prefill_forward output) -> [1, T, 4, 4096]."""
    return y.reshape(1, -1, 4, 4096)


class RecordingRouter:
    """Test-side wrapper around MoE.router: keeps the dense routing weights (device tensors) of the calls made while
    `recording` is True (prefill); no device reads, so it is safe while a decode trace is captured."""

    def __init__(self, router):
        self.router = router
        self.num_experts = router.num_experts
        self.recording = False
        self.log = []

    def __call__(self, x):
        out = self.router(x)
        if self.recording:
            self.log.append(out)
        return out


def row_stats(out: torch.Tensor, ref: torch.Tensor, top: int = 4) -> dict:
    """Per-token PCC statistics of [1, T, 4, 4096] tensors and the PCC without the `top` highest-variance golden rows
    (a few massive-activation tokens can dominate the aggregate PCC)."""
    o = out.reshape(out.shape[1], -1).double()
    r = ref.reshape(ref.shape[1], -1).double()
    per = torch.tensor([pcc(o[t], r[t]) for t in range(o.shape[0])])
    var = (r - r.mean()).pow(2).sum(-1)
    keep = torch.ones(o.shape[0], dtype=torch.bool)
    if o.shape[0] > top + 1:
        keep[var.topk(top).indices] = False
    q = per.sort().values
    return dict(
        row_pcc_min=float(q[0]),
        row_pcc_p1=float(q[int(0.01 * (len(q) - 1))]),
        row_pcc_median=float(q[len(q) // 2]),
        rows_below_bar=int((per < PCC_BAR).sum()),
        top_var_row_share=float(var.max() / var.sum()),
        pcc_excl_top4_var_rows=pcc(o[keep], r[keep]),
    )


# GLM53_RECORD_ONLY=1: metrics are recorded but not asserted, so an exploratory device run that measures a sub-bar
# value still exits 0 (a non-zero device-process exit triggers a device reset). Gated runs leave it unset.
RECORD_ONLY = os.environ.get("GLM53_RECORD_ONLY") == "1"


def check(failures, what: str = "rows below the acceptance bar"):
    if failures and RECORD_ONLY:
        print(f"[record-only] {what}: {failures}")
        return
    assert not failures, f"{what}: {failures}"
