# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Positive control for the trace-allocation-tracking runs: with TT_METAL_TRACE_ALLOC_TRACKING=1 a device buffer
allocated after capture and alive at replay must make `ttnn.execute_trace` raise. Skipped when tracking is off."""

import pytest
import torch
from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_TRACKING

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import open_mesh, record


@pytest.mark.skipif(not TRACE_ALLOC_TRACKING, reason="run with TT_METAL_TRACE_ALLOC_TRACKING=1")
def test_tracker_detects_unsafe_allocation():
    mesh = open_mesh()
    try:
        x = ttnn.from_torch(torch.randn(32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        ttnn.add(x, x)  # warm
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        ttnn.add(x, x)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        unsafe = ttnn.from_torch(torch.randn(32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        raised = False
        try:
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        except RuntimeError:
            raised = True
        ttnn.deallocate(unsafe)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)  # clean once the unsafe buffer is gone
        ttnn.release_trace(mesh, tid)
        record(
            dict(test="trace_tracker_positive_control", raised_on_unsafe_allocation=raised),
            name="trace_tracker_control.jsonl",
        )
        assert raised
    finally:
        ttnn.close_mesh_device(mesh)
