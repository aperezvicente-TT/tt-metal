# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill and traced warmed decode of one decoder layer per kind, delimited by Tracy signposts.

Run under Tracy (one invocation per layer kind; outputs post-processed into ops_perf_results_*.csv):
    python -m tracy -r -p -v -m pytest models/autoports/zai_org_glm_5_3_flash/tests/test_perf.py -k <kind>
Signposts: PERF_PREFILL .. PERF_PREFILL_END (one warmed 2048-token `prefill_chunk` pass, eager ops),
           PERF_DECODE[_B32] .. _END (DECODE_REPLAYS replays of the captured decode trace, B users with distinct
           golden inputs/positions, so the MoE expert union is representative).
Also prints wall-clock (host) timings for the same windows.
"""

from __future__ import annotations

import os
import time

import pytest

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import (
    golden,
    hf_config,
    int_tensor,
    load_decoder,
    open_mesh,
    record,
    to_dev,
)
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV

try:
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(*_args, **_kwargs):
        pass


DECODE_REPLAYS = int(os.environ.get("GLM53_PERF_DECODE_REPLAYS", "4"))
PROFILING = bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))


def _flush(mesh):
    """Drain the on-device profiler buffers between phases (trace replays otherwise overflow them)."""
    if PROFILING:
        ttnn.ReadDeviceProfiler(mesh)


PREFILL_LEN = int(os.environ.get("GLM53_PERF_PREFILL_LEN", "2048"))


@pytest.mark.parametrize("kind,layer", [("dense_kda", 0), ("sparse_kda", 4), ("sparse_dsa", 3)])
@pytest.mark.parametrize("batch", [1, 32])
def test_perf(kind, layer, batch):
    tag = "" if batch == 1 else f"_B{batch}"
    cfg = hf_config()
    mesh = open_mesh()
    try:
        dec = load_decoder(cfg, layer, mesh)
        g = golden(layer, 2048)
        cache = dec.allocate_cache(
            max_batch=batch, max_seq_len=PREFILL_LEN + 64 if dec.is_kda else DSA_DENSE_EXACT_MAX_KV
        )
        pt = None if dec.is_kda else dec.make_page_table(cache)
        x = to_dev(g["streams_in"][:, :PREFILL_LEN], mesh)
        # warm-up chunk (compiles every program of this physical bucket), then the measured warmed chunk pass
        dec.begin_prefill_chunk(cache, user_slot=0, chunk_start=0, n_valid=PREFILL_LEN)
        ttnn.deallocate(dec.prefill_chunk(x, cache=cache, user_slot=0))
        dec.begin_prefill_chunk(cache, user_slot=0, chunk_start=0, n_valid=PREFILL_LEN)
        ttnn.synchronize_device(mesh)
        _flush(mesh)
        signpost(f"PERF_PREFILL{tag}")
        t0 = time.perf_counter()
        y = dec.prefill_chunk(x, cache=cache, user_slot=0)
        ttnn.synchronize_device(mesh)
        prefill_s = time.perf_counter() - t0
        signpost(f"PERF_PREFILL{tag}_END")
        ttnn.deallocate(y)
        _flush(mesh)

        # decode: persistent inputs, warm, restore, capture, replay
        # B users with distinct golden rows and positions (MLA: kv_len <= 2051)
        last = PREFILL_LEN - 1 if dec.is_kda else min(PREFILL_LEN, 2050) - 1
        positions = list(range(last - batch + 1, last + 1))
        streams = to_dev(g["streams_in"][:, positions[0] : last + 1], mesh)
        cur_pos = int_tensor(positions, mesh)
        snap = dec.snapshot_state(cache)
        dec.decode_forward(streams, cache=cache, page_table=pt, cur_pos=cur_pos)
        dec.restore_state(cache, snap)
        ttnn.synchronize_device(mesh)
        _flush(mesh)
        trace = dec.capture_decode_trace(streams, cache=cache, page_table=pt, cur_pos=cur_pos)
        for _ in range(2):  # warm replays
            dec.restore_state(cache, snap)
            trace.replay()
        ttnn.synchronize_device(mesh)
        _flush(mesh)
        signpost(f"PERF_DECODE{tag}")
        t0 = time.perf_counter()
        for _ in range(DECODE_REPLAYS):
            trace.replay(blocking=False)
            if PROFILING:  # one replay per profiler drain; host time is not the metric in profiled runs
                ttnn.synchronize_device(mesh)
                _flush(mesh)
        ttnn.synchronize_device(mesh)
        decode_s = (time.perf_counter() - t0) / DECODE_REPLAYS
        _flush(mesh)
        signpost(f"PERF_DECODE{tag}_END")
        trace.release()
        record(
            dict(
                layer_kind=kind,
                layer=layer,
                batch=batch,
                prefill_tokens=PREFILL_LEN,
                prefill_wall_ms=prefill_s * 1e3,
                decode_wall_ms_per_step=decode_s * 1e3,
                decode_replays=DECODE_REPLAYS,
                profiler=bool(os.environ.get("TT_METAL_DEVICE_PROFILER")),
            ),
            name="perf_wall.jsonl",
        )
        print(
            f"PERF {kind} L{layer}: prefill {PREFILL_LEN} tok {prefill_s*1e3:.1f} ms, traced decode B={batch} "
            f"{decode_s*1e3:.3f} ms/step"
        )
    finally:
        ttnn.close_mesh_device(mesh)
