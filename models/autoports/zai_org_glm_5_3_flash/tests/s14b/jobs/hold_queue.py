# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, the long-lived 2x4 server): Stage 14b — HOLD the mesh with the resident model (program 2026-10-20
15:12Z: the full-stack server holds the mesh continuously, weights resident, a job queue inside it). The job is a nested
serve loop over ARGS queue (default work-s14b/q_hold): the same job contract as serve.py (*.job JSON {script, args} ->
.running -> .done / .failed + .out; MESH / ARGS / CTX globals: CTX is the server's, so the resident weights stay), the
outer server's deadline does not apply (it is checked between ITS jobs only). Ends on queue/STOP (removed on exit),
after which the outer server closes as usual. After a failed job: one synchronize; a dead device ends the hold.
ARGS: queue, poll_s (0.2)."""
import gc
import glob
import json
import os
import runpy
import sys
import time
import traceback

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s14common as C  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
Q = args.get("queue", "/opt/glm53/work-s14b/q_hold")
POLL = float(args.get("poll_s", 0.2))
os.makedirs(Q, exist_ok=True)
print(f"HOLD_START {C.now()} queue={Q}", flush=True)
n_jobs, n_fail = 0, 0
while True:
    stop = os.path.join(Q, "STOP")
    if os.path.exists(stop):
        os.remove(stop)
        print(f"HOLD_STOP {C.now()} after {n_jobs} jobs ({n_fail} failed)", flush=True)
        break
    jobs = sorted(glob.glob(os.path.join(Q, "*.job")), key=os.path.getmtime)
    if not jobs:
        time.sleep(POLL)
        continue
    job = jobs[0]
    run = job[:-4] + ".running"
    os.rename(job, run)
    try:
        spec = json.load(open(run))
    except ValueError as e:  # a half-written job file: back to .job, retried next poll
        os.rename(run, job)
        time.sleep(POLL)
        continue
    name = os.path.basename(job)[:-4]
    print(f"JOB_START {name} {C.now()} {spec}", flush=True)
    t0 = time.time()
    res, err = None, None
    try:
        ns = runpy.run_path(spec["script"], init_globals={"MESH": mesh, "ARGS": spec.get("args", {}), "CTX": ctx},
                            run_name="__job__")  # fmt: skip
        res = ns.get("RESULT")
    except BaseException as e:  # noqa: BLE001 -- recorded, then the device health decides
        err = traceback.format_exc()
        print(f"JOB_ERROR {name}\n{err}", flush=True)
        if isinstance(e, KeyboardInterrupt):
            raise
    ns = None
    gc.collect()
    out = dict(name=name, start=t0, wall_s=round(time.time() - t0, 2), ok=err is None, result=res,
               error=err[-4000:] if err else None)  # fmt: skip
    with open(job[:-4] + ".out", "w") as f:
        json.dump(out, f, default=str, indent=1)
    os.rename(run, job[:-4] + (".done" if err is None else ".failed"))
    print(f"JOB_END {name} ok={err is None} wall_s={out['wall_s']}", flush=True)
    n_jobs += 1
    if err is not None:
        n_fail += 1
        ttnn.synchronize_device(mesh)  # raises on a dead device -> the outer server's error path
        print("HOLD_HEALTH ok after job error", flush=True)
RESULT = dict(ok=True, jobs=n_jobs, failed=n_fail)
