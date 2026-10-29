# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill and traced warmed decode of one decoder layer per kind, fused (default) or functional baseline
(GLM53_DECODER_IMPL=functional), delimited by Tracy signposts; same protocol as Stage 1's tests/test_perf.py.

    python -m tracy -r -p -v -m pytest models/autoports/zai_org_glm_5_3_flash/tests/test_fused_perf.py -k <kind>
Signposts: PERF_PREFILL .. _END (one warmed 2048-token prefill_chunk pass, eager ops); PERF_DECODE[_B32] .. _END
(DECODE_REPLAYS replays of the captured decode trace, B users with distinct golden rows/positions).
Wall-clock rows (non-profiled runs) go to perf_wall.jsonl with the implementation name.
"""

from __future__ import annotations

import os
import time

import pytest

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import golden, hf_config, int_tensor, record, to_dev
from models.autoports.zai_org_glm_5_3_flash.tests.common_fused import IMPL, load_impl, open_mesh
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV

try:
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(*_args, **_kwargs):
        pass


DECODE_REPLAYS = int(os.environ.get("GLM53_PERF_DECODE_REPLAYS", "4"))
WALL_REPLAYS = int(os.environ.get("GLM53_PERF_WALL_REPLAYS", "20"))
PROFILING = bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))
PREFILL_LEN = int(os.environ.get("GLM53_PERF_PREFILL_LEN", "2048"))
KINDS = [("dense_kda", 0), ("sparse_kda", 4), ("sparse_dsa", 3)]
ONLY = os.environ.get("GLM53_PERF_KINDS")  # e.g. "sparse_kda" (tracy re-runs pytest through /bin/sh; select by env)
BATCHES = [int(b) for b in os.environ.get("GLM53_PERF_BATCHES", "1,32").split(",")]


def _flush(mesh):
    if PROFILING:
        ttnn.ReadDeviceProfiler(mesh)


@pytest.mark.parametrize("kind,layer", [k for k in KINDS if not ONLY or k[0] in ONLY.split(",")])
def test_perf(kind, layer):
    cfg = hf_config()
    mesh = open_mesh(trace_region_size=int(os.environ.get("GLM53_TRACE_REGION", "300000000")))
    try:
        dec = load_impl(cfg, layer, mesh)
        g = golden(layer, 2048)
        for batch in BATCHES:
            tag = "" if batch == 1 else f"_B{batch}"
            cache = dec.allocate_cache(
                max_batch=batch, max_seq_len=PREFILL_LEN + 64 if dec.is_kda else DSA_DENSE_EXACT_MAX_KV
            )
            pt = None if dec.is_kda else dec.make_page_table(cache)
            prefill_ms = None
            if batch == BATCHES[0]:
                x = to_dev(g["streams_in"][:, :PREFILL_LEN], mesh)
                dec.begin_prefill_chunk(cache, user_slot=0, chunk_start=0, n_valid=PREFILL_LEN)
                ttnn.deallocate(dec.prefill_chunk(x, cache=cache, user_slot=0))  # warm-up (compile)
                dec.begin_prefill_chunk(cache, user_slot=0, chunk_start=0, n_valid=PREFILL_LEN)
                ttnn.synchronize_device(mesh)
                _flush(mesh)
                signpost("PERF_PREFILL")
                t0 = time.perf_counter()
                y = dec.prefill_chunk(x, cache=cache, user_slot=0)
                ttnn.synchronize_device(mesh)
                prefill_ms = (time.perf_counter() - t0) * 1e3
                signpost("PERF_PREFILL_END")
                ttnn.deallocate(y)
                ttnn.deallocate(x)
                _flush(mesh)

            last = PREFILL_LEN - 1 if dec.is_kda else min(PREFILL_LEN, 2050) - 1
            positions = list(range(last - batch + 1, last + 1))
            streams = to_dev(g["streams_in"][:, positions[0] : last + 1], mesh)
            cur_pos = int_tensor(positions, mesh)
            snap = dec.snapshot_state(cache)
            ttnn.deallocate(dec.decode_forward(streams, cache=cache, page_table=pt, cur_pos=cur_pos))
            dec.restore_state(cache, snap)
            ttnn.synchronize_device(mesh)
            _flush(mesh)
            trace = dec.capture_decode_trace(streams, cache=cache, page_table=pt, cur_pos=cur_pos)
            for _ in range(2):
                dec.restore_state(cache, snap)
                trace.replay()
            ttnn.synchronize_device(mesh)
            _flush(mesh)
            reps = DECODE_REPLAYS if PROFILING else WALL_REPLAYS
            signpost(f"PERF_DECODE{tag}")
            t0 = time.perf_counter()
            for _ in range(reps):
                trace.replay(blocking=False)
                if PROFILING:
                    ttnn.synchronize_device(mesh)
                    _flush(mesh)
            ttnn.synchronize_device(mesh)
            decode_ms = (time.perf_counter() - t0) * 1e3 / reps
            _flush(mesh)
            signpost(f"PERF_DECODE{tag}_END")
            trace.release()
            record(
                dict(
                    impl=IMPL,
                    fcfg_overrides=os.environ.get("GLM53_FCFG", ""),
                    layer_kind=kind,
                    layer=layer,
                    batch=batch,
                    prefill_tokens=PREFILL_LEN if prefill_ms is not None else None,
                    prefill_wall_ms=prefill_ms,
                    decode_wall_ms_per_step=decode_ms,
                    decode_replays=reps,
                    profiler=PROFILING,
                ),
                name="perf_wall.jsonl",
            )
            print(
                f"PERF {IMPL} {kind} L{layer} B={batch}: prefill {prefill_ms} ms, traced decode {decode_ms:.3f} ms/step"
            )
            del cache
    finally:
        ttnn.close_mesh_device(mesh)
