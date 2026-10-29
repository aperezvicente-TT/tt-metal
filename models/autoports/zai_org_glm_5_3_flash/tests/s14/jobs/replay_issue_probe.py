# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14 W2b, the launch skew): the HOST cost of issuing a trace replay to every device of the mesh
(FDMeshCommandQueue::submit_replay_buffer loops issue_trace_commands over the devices). A trivial trace (one small eltwise
op per device) is captured once; then execute_trace(blocking=False) is timed on the host, alone (after a synchronize:
the queues empty) and back-to-back; device time is negligible, so the host return time = the per-replay issue cost.
Also the mesh's device order (get_devices / device ids), to compare with a profile's per-chip entry order.
ARGS: reps (200), rows (32), tag, record."""
import json
import os
import statistics
import time

import torch

import ttnn

mesh, args = MESH, ARGS  # noqa: F821
REPS = int(args.get("reps", 200))
n = int(mesh.get_num_devices())
repl = ttnn.ReplicateTensorToMesh(mesh) if n > 1 else None
a = ttnn.from_torch(torch.ones(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=repl)  # fmt: skip
b = ttnn.add(a, a)  # the compile launch
ttnn.synchronize_device(mesh)
ttnn.deallocate(b)
tid = ttnn.begin_trace_capture(mesh, cq_id=0)
b = ttnn.add(a, a)
ttnn.end_trace_capture(mesh, tid, cq_id=0)
ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
alone, b2b = [], []
for _ in range(REPS):
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    alone.append((time.perf_counter() - t0) * 1e6)
ttnn.synchronize_device(mesh)
t0 = time.perf_counter()
for _ in range(REPS):
    ta = time.perf_counter()
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    b2b.append((time.perf_counter() - ta) * 1e6)
ttnn.synchronize_device(mesh)
total_b2b = (time.perf_counter() - t0) * 1e6
ttnn.release_trace(mesh, tid)
try:
    ids = [int(d.id()) for d in mesh.get_devices()]
except Exception as e:  # noqa: BLE001
    ids = repr(e)[:80]
med = lambda v: round(statistics.median(v), 1)  # noqa: E731
row = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test="replay_issue_probe", status="info",
           devices=n, device_order=ids, issue_us_alone_median=med(alone), issue_us_alone_p90=round(sorted(alone)[int(0.9 * len(alone))], 1),
           issue_us_b2b_median=med(b2b), b2b_total_us_per_replay=round(total_b2b / REPS, 1),
           issue_us_per_device=round(med(alone) / n, 1), reps=REPS, traced=True, tag=args.get("tag", "replay-issue-probe"))  # fmt: skip
print(f"REPLAY-ISSUE {json.dumps(row)}", flush=True)
ev = os.environ.get("GLM53_STAGE_EVIDENCE")
if args.get("record") and ev:
    with open(os.path.join(ev, "perf.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
RESULT = row
