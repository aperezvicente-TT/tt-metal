# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14 long-lived device process (PRODUCTION_PATH.md §6: one long-lived process per chip / mesh; every piece
runs the moment it compiles). Opens ONE mesh with fabric (pair = card0 chips 0,1 | 2x4 = all 8 chips), then runs job
scripts dropped into a queue directory one after another INSIDE this process, so the device, the fabric, the program
cache and the JIT cache stay warm across jobs.

  queue/<name>.job    JSON {"script": <path>, "args": {...}}; renamed .running -> .done / .failed (+ .out JSON)
  queue/STOP          stop after the current job; --deadline <ISO-8601 UTC> = hard stop (no job starts after it,
                      a job is given at most until the deadline to finish via its own timeout)
A job script runs with globals MESH, ARGS, CTX (CTX = dict shared across jobs: persistent objects), sets RESULT.
A job that raises is recorded; then a device health check (synchronize) decides: healthy -> next job; unhealthy (the CQ
is not trusted after a dispatch / timeout error) -> the process closes the mesh and exits rc 6. Before every job the
COORDINATOR_NOTE is checked: a note newer than NOTE_SEEN pauses the queue until it is acknowledged."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import glob
import json
import os
import runpy
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s14common as C  # noqa: E402

import ttnn  # noqa: E402

W = os.environ.get("GLM53_WORK", "/opt/glm53/work-s14")


CONTROL_WORDS = ("OPENS", "CLOSED", "CAP", "STOP", "RESET", "GO", "HOLD")


def note_pending():
    """True if the program note has an unacknowledged CONTROL line: a line newer than NOTE_SEEN whose message
    (after 'PROGRAM[-SCHEDULER] — ') starts, within its first 40 characters, with one of CONTROL_WORDS (throughput
    rule, program 2026-10-20 14:27Z: informational notes never pause a server)."""
    note, seen = os.path.join(W, "COORDINATOR_NOTE.txt"), os.path.join(W, "NOTE_SEEN")
    if not os.path.exists(note):
        return False
    t_seen = os.path.getmtime(seen) if os.path.exists(seen) else 0.0
    if os.path.getmtime(note) <= t_seen:
        return False
    for line in open(note, errors="replace").read().splitlines()[-50:]:
        ts, _, rest = line.partition(" ")
        try:
            t_line = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        if t_line <= t_seen:
            continue
        body = rest.split(" — ", 1)[1] if " — " in rest else rest
        if any(w in body[:40] for w in CONTROL_WORDS):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mgd", default="pair", choices=list(C.MGDS))
    ap.add_argument("--queue", required=True)
    ap.add_argument("--deadline", default="", help="hard stop, ISO-8601 UTC (e.g. 2026-10-19T21:00:00Z)")
    ap.add_argument("--idle-exit-min", type=float, default=0, help="exit after this many idle minutes (0 = never)")
    ap.add_argument("--worker-l1", type=int, default=int(os.environ.get("GLM53_WORKER_L1", "0")))
    ap.add_argument("--trace-region-mb", type=int, default=256)
    ap.add_argument("--payload", type=int, default=0)
    a = ap.parse_args()
    deadline = dt.datetime.fromisoformat(a.deadline.replace("Z", "+00:00")).timestamp() if a.deadline else None
    os.makedirs(a.queue, exist_ok=True)
    mesh, open_s = C.open_mesh(ttnn, a.mgd, trace_region=a.trace_region_mb << 20, worker_l1_size=a.worker_l1,
                               payload=a.payload)  # fmt: skip
    cmap = C.chip_map(mesh)
    print(f"SERVE_OPEN mgd={a.mgd} open_s={open_s:.2f} chips={ {str(k): v for k, v in cmap.items()} } "
          f"worker_l1={a.worker_l1} deadline={a.deadline}", flush=True)  # fmt: skip
    ctx = {"cmap": cmap, "mgd": a.mgd, "open_s": open_s, "worker_l1": a.worker_l1, "payload": a.payload}
    rc, idle_since, paused = 0, time.time(), False
    try:
        while True:
            if os.path.exists(os.path.join(a.queue, "STOP")):
                print("SERVE_STOP file", flush=True)
                break
            if deadline is not None and time.time() >= deadline:
                print("SERVE_DEADLINE reached", flush=True)
                break
            if note_pending():
                if not paused:
                    print(f"SERVE_PAUSED {C.now()}: COORDINATOR_NOTE newer than NOTE_SEEN", flush=True)
                paused = True
                time.sleep(1)  # job pipeline rule (program 23:52:53Z): 1 s when paused
                continue
            paused = False
            jobs = sorted(glob.glob(os.path.join(a.queue, "*.job")), key=os.path.getmtime)
            if not jobs:
                if a.idle_exit_min and time.time() - idle_since > 60 * a.idle_exit_min:
                    print("SERVE_IDLE_EXIT", flush=True)
                    break
                time.sleep(0.2)  # job pipeline rule: 0.2 s between jobs
                continue
            job = jobs[0]
            run = job[:-4] + ".running"
            os.rename(job, run)
            spec = json.load(open(run))
            name = os.path.basename(job)[:-4]
            print(f"JOB_START {name} {C.now()} {spec}", flush=True)
            t0 = time.time()
            res, err = None, None
            try:
                ns = runpy.run_path(spec["script"], init_globals={"MESH": mesh, "ARGS": spec.get("args", {}),
                                                                  "CTX": ctx}, run_name="__job__")  # fmt: skip
                res = ns.get("RESULT")
            except BaseException as e:  # noqa: BLE001 -- recorded, then the device health decides
                err = traceback.format_exc()
                print(f"JOB_ERROR {name}\n{err}", flush=True)
                if isinstance(e, KeyboardInterrupt):
                    rc = 130
            ns = None
            gc.collect()  # a failed job's frames (traceback cycles) keep device tensors alive otherwise
            out = dict(name=name, start=t0, wall_s=round(time.time() - t0, 2), ok=err is None, result=res,
                       error=err[-4000:] if err else None)  # fmt: skip
            with open(job[:-4] + ".out", "w") as f:
                json.dump(out, f, default=str, indent=1)
            os.rename(run, job[:-4] + (".done" if err is None else ".failed"))
            print(f"JOB_END {name} ok={err is None} wall_s={out['wall_s']}", flush=True)
            idle_since = time.time()
            if rc == 130:
                break
            if err is not None and ("device timeout" in err or "unrecoverable" in err):
                # Stage 14 (2026-10-20 03:42Z): after a device timeout synchronize_device "passes" while nothing
                # executes any more (the next job's fabric probe read zeros, its PCC was 0): close instead
                print("SERVE_HEALTH device timeout in the job -> exit rc 7", flush=True)
                rc = 7
                break
            if err is not None:
                try:
                    ttnn.synchronize_device(mesh)
                    print("SERVE_HEALTH ok after job error", flush=True)
                except Exception as e2:  # noqa: BLE001
                    print(f"SERVE_HEALTH FAILED after job error: {e2!r} -> exit rc 6", flush=True)
                    rc = 6
                    break
    finally:
        for k, v in list(ctx.items()):
            rel = getattr(v, "release", None)
            if callable(rel):
                try:
                    rel()
                except Exception as e:  # noqa: BLE001
                    print(f"release {k}: {e!r}", flush=True)
        C.close_mesh(ttnn, mesh, fabric=a.mgd != "single")
        print(f"SERVE_CLOSED rc={rc} {C.now()}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
