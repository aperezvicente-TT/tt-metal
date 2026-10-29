# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, a long-lived server): Stage 14b — release every trace of the process that no live CTX backend owns
(the traces of failed jobs whose backends were dropped without close() stay in the trace region: 2026-10-20 16:08Z
'Creating trace buffers of size 583729152B ... only 536870912B'). Closes the CTX backends first (M5.prune_ctx), then
tries ttnn.release_trace on trace ids 0..ARGS max_id (unknown ids raise, ignored). ARGS: max_id (512)."""
import gc
import os
import sys

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
dropped = M5.prune_ctx(ctx, [])
gc.collect()
freed = []
for tid in range(int(args.get("max_id", 512))):
    try:
        ttnn.release_trace(mesh, ttnn.MeshTraceId(tid))
        freed.append(tid)
    except Exception:  # noqa: BLE001 -- an id that is not (or no longer) a trace
        pass
ttnn.synchronize_device(mesh)
print(f"RELEASE_TRACES dropped={len(dropped)} freed={freed}", flush=True)
RESULT = dict(ok=True, freed=freed)
