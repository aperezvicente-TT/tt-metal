# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced execution for the Stage 7a device checks (operator order 2026-10-19T05:33Z: every device run that measures or
validates runs traced; eager only as the compile / warm-up step).

    tr = Traced(mesh)
    tr.warm(f1, f2, ...)                 # every case of the run, eagerly, before ANY capture (program rule 07:25Z)
    out = tr.run(lambda: program(...))   # eager call (compile / warm-up, outputs dropped) -> capture -> replay
    a = host(out)                        # the traced call's outputs (trace-owned buffers)
    tr.replay(); b = host(out)           # determinism: another replay of the same trace
    tr.release()                         # before the next capture

No program may compile after a capture (a program compiled while a trace exists can take memory the trace uses): warm
every case first; run under TT_METAL_TRACE_ALLOC_TRACKING=1 the first time a new program set is captured.
A trace is replayed only right after its capture (no device allocation in between): intermediates freed after the
capture may be reused by later allocations. Functions that mutate persistent state must reset that state inside the
traced function (a device copy from a saved tensor), so the warm-up call and every replay start from the same state.
GLM53_MOCK=1 (mock cluster, no traces): plain eager calls."""

from __future__ import annotations

import os

import ttnn


class Traced:
    def __init__(self, mesh):
        self.mesh = mesh
        self.enabled = os.environ.get("GLM53_MOCK") != "1" and os.environ.get("GLM53_TRACED", "1") == "1"
        self.tid = None

    def warm(self, *fns):
        """Eager call of every case (compiles every program) before any capture; outputs dropped."""
        if not self.enabled:
            return
        self.release()
        for fn in fns:
            fn()
        ttnn.synchronize_device(self.mesh)

    def run(self, fn, replays: int = 1):
        if not self.enabled:
            return fn()
        self.release()
        fn()  # the compile / warm-up step (eager); its outputs are dropped
        ttnn.synchronize_device(self.mesh)
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            out = fn()
        except BaseException:
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)  # never leave the queue in capture mode
            ttnn.release_trace(self.mesh, tid)
            raise
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.tid = tid
        for _ in range(replays):
            self.replay()
        return out

    def replay(self):
        if self.enabled and self.tid is not None:
            ttnn.execute_trace(self.mesh, self.tid, cq_id=0, blocking=True)

    def release(self):
        if self.tid is not None:
            ttnn.release_trace(self.mesh, self.tid)
            self.tid = None

    @property
    def mode(self) -> str:
        return "traced" if self.enabled else "eager"


def cached_decoder(cfg, layer: int, mesh, ctx):
    """The optimized decoder of `layer` on `mesh`: loaded once per long-lived process (serve.py: ctx shared across
    jobs), or per call without a ctx."""
    from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import load_optimized_decoder

    if ctx is None:
        return load_optimized_decoder(cfg, layer, mesh)
    key = ("dec", layer)
    if key not in ctx:
        ctx[key] = load_optimized_decoder(cfg, layer, mesh)
    return ctx[key]
