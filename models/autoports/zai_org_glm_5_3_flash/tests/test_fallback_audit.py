# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runtime host-fallback audit of the measured prefill chunk passes and one decode pass per layer kind.

Inside the audited window every host<->device conversion entry point of ttnn (`from_torch`, `to_torch`, `as_tensor`,
`to_device`, `from_device`, `copy_host_to_device_tensor`, `copy_device_to_host_tensor`, `to_layout` on host tensors is
not used) is replaced by a tripwire, and `sys.setprofile` records every Python call that lands in a `torch` source file.
Each pass must finish with zero tripwire hits and zero torch calls, and its output must be a device tensor. The prefill
covers every chunk pass of a 2,100-token KDA prompt (a full 2048-token chunk and a 52-token tail padded to 64) and of a
2,000-token MLA prompt (one chunk padded to the 2048 bucket); the host chunk boundary (`begin_prefill_chunk`, input transfer) runs outside the audited window. Decode is the exact
device-only function captured by the trace.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import (
    check,
    golden,
    hf_config,
    int_tensor,
    load_decoder,
    open_mesh,
    record,
    to_dev,
)
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV
from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import HC_WIDTH, plan_prefill_chunks

TORCH_DIR = os.path.dirname(torch.__file__)
GUARDED = [
    "from_torch",
    "to_torch",
    "as_tensor",
    "to_device",
    "from_device",
    "copy_host_to_device_tensor",
    "copy_device_to_host_tensor",
]


class Audit:
    def __init__(self):
        self.torch_calls = {}
        self.tripwires = []
        self._saved = {}

    def _profile(self, frame, event, arg):
        key = None
        if event == "call" and frame.f_code.co_filename.startswith(TORCH_DIR):
            key = f"{frame.f_code.co_filename[len(TORCH_DIR):]}:{frame.f_code.co_name}"
        elif event == "c_call" and (getattr(arg, "__module__", None) or "").startswith("torch"):
            key = f"builtin:{arg.__module__}.{getattr(arg, '__name__', '?')}"
        if key is not None:
            self.torch_calls[key] = self.torch_calls.get(key, 0) + 1

    def __enter__(self):
        for name in GUARDED:
            self._saved[name] = getattr(ttnn, name)

            def wire(*a, _n=name, **k):
                self.tripwires.append(_n)
                return self._saved[_n](*a, **k)

            setattr(ttnn, name, wire)
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)
        for name, fn in self._saved.items():
            setattr(ttnn, name, fn)


@pytest.mark.parametrize("kind,layer", [("dense_kda", 0), ("sparse_kda", 4), ("sparse_dsa", 3)])
def test_fallback_audit(kind, layer):
    cfg = hf_config()
    mesh = open_mesh()
    try:
        dec = load_decoder(cfg, layer, mesh)
        n = 2100 if dec.is_kda else 2000  # two chunks + non-aligned tail (MLA: kv_len <= 2051)
        g = golden(layer, 4096)
        cache = dec.allocate_cache(max_batch=1, max_seq_len=n + 64 if dec.is_kda else DSA_DENSE_EXACT_MAX_KV)
        pt = None if dec.is_kda else dec.make_page_table(cache)
        rows = g["streams_in"].reshape(-1, HC_WIDTH)
        chunks = []
        for off, nv, phys in plan_prefill_chunks(n, dec.prefill_align):
            x = torch.zeros(phys, HC_WIDTH)
            x[:nv] = rows[off : off + nv]
            chunks.append((off, nv, to_dev(x.reshape(1, phys, 4, 4096), mesh)))
        streams = to_dev(g["streams_in"][:, n : n + 1], mesh)
        cur_pos = int_tensor([n], mesh)
        dec.prefill_forward(g["streams_in"][:, :n], cache=cache, user_slot=0)  # warm (program cache)
        # positive control: the audit must see a torch builtin and a host->device conversion
        with Audit() as ctl:
            ttnn.from_torch(torch.zeros(1, 32), layout=ttnn.TILE_LAYOUT, device=mesh)
        assert ctl.torch_calls and ctl.tripwires == ["from_torch"], (ctl.torch_calls, ctl.tripwires)
        results = {}
        for i, (off, nv, x) in enumerate(chunks):
            dec.begin_prefill_chunk(cache, user_slot=0, chunk_start=off, n_valid=nv)  # host boundary, not audited
            with Audit() as a:
                y = dec.prefill_chunk(x, cache=cache, user_slot=0)
            results[f"prefill_chunk{i}_{nv}of{x.shape[-2]}"] = (a, y)
        with Audit() as a:
            yd = dec.decode_forward(streams, cache=cache, page_table=pt, cur_pos=cur_pos)
        results["decode"] = (a, yd)
        ttnn.synchronize_device(mesh)
        for mode, (a, out) in results.items():
            row = dict(
                layer_kind=kind,
                layer=layer,
                mode=mode,
                tokens=int(out.shape[-2]),
                torch_calls=sum(a.torch_calls.values()),
                torch_call_sites=sorted(a.torch_calls)[:20],
                host_transfer_calls=a.tripwires,
                output_on_device=ttnn.is_tensor_storage_on_device(out),
                positive_control_detected=sorted(ctl.torch_calls)[:3] + ctl.tripwires,
            )
            record(row, name="fallback_audit.jsonl")
            bad = row["torch_calls"] or row["host_transfer_calls"] or not row["output_on_device"]
            check([row] if bad else [], "host fallback inside a measured pass")
    finally:
        ttnn.close_mesh_device(mesh)
