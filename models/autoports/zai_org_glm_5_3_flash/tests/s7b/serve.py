# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7b long-lived device process (copy of tests/s7a/serve.py) (program HARD RULE 2026-10-19T19:10Z, PRODUCTION_PATH.md §6): ONE process
per chip for the whole session — the chip is opened once (unit mesh; worker_l1_size fixed at open), then the device
checks run as jobs INSIDE this process; never close and reopen between jobs. Pattern of Stage 14's tests/s14/serve.py.

  <queue>/<name>.job   JSON {"module": "dev_mk_layer", "env": {"GLM53_MK_ONE": "1", ...}} -> the tests/s7a module is
                       (re)loaded with that environment and its main(mesh=MESH, ctx=CTX) runs; or {"script": <path>,
                       "args": {...}} (runpy with globals MESH, ARGS, CTX; sets RESULT). Renamed .running -> .done /
                       .failed, result in <name>.out (JSON).
  <queue>/STOP         stop after the current job; --deadline <ISO-8601 UTC>: no job starts after it.
A job that raises is recorded; then a device health check (synchronize) decides: healthy -> next job; unhealthy (the
command queue is not trusted after a dispatch / timeout error) -> close and exit rc 6 (no reset by this stage). A job
whose main returns 3 (device timeout) ends the process the same way. Before every job the COORDINATOR_NOTE is checked:
a note newer than NOTE_SEEN pauses the queue until it is acknowledged. CTX keeps the loaded decoder layers (one load
per session) and is shared across jobs."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import glob
import importlib
import json
import os
import runpy
import sys
import time
import traceback

import ttnn

W = os.environ.get("GLM53_WORK", "/opt/glm53/work-s7b")
PKG = "models.autoports.zai_org_glm_5_3_flash.tests.s7b"


MESH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "mesh")
MGDS = {  # --mgd: name -> (mesh graph descriptor, shape); "single" = one chip (TT_VISIBLE_DEVICES), no fabric
    "single": (None, (1, 1)),
    "pair": ("glm_pair_1x2_mesh_graph_descriptor.textproto", (1, 2)),
    "2x4": ("glm8_2x4_mesh_graph_descriptor.textproto", (2, 4)),
}


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def note_pending():
    note, seen = os.path.join(W, "COORDINATOR_NOTE.txt"), os.path.join(W, "NOTE_SEEN")
    return os.path.exists(note) and (not os.path.exists(seen) or os.path.getmtime(note) > os.path.getmtime(seen))


def run_module(mesh, ctx, spec):
    env = {k: str(v) for k, v in spec.get("env", {}).items()}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        name = f"{PKG}.{spec['module']}"
        mod = importlib.reload(sys.modules[name]) if name in sys.modules else importlib.import_module(name)
        return mod.main(mesh=mesh, ctx=ctx)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--deadline", default="", help="hard stop, ISO-8601 UTC")
    ap.add_argument("--worker-l1", type=int, default=int(os.environ.get("GLM53_WORKER_L1", "0")))
    ap.add_argument("--trace-region-mb", type=int, default=200)
    ap.add_argument("--idle-exit-min", type=float, default=0.0,
                    help="> 0: close once the queue has been empty this long (a granted mesh slot ends with its jobs)")
    ap.add_argument("--mgd", default="single", choices=list(MGDS),
                    help="mesh: single chip, or a fabric mesh (the TP programs, tt/tp_mla.py; Stage 14's descriptors)")
    a = ap.parse_args()
    deadline = dt.datetime.fromisoformat(a.deadline.replace("Z", "+00:00")).timestamp() if a.deadline else None
    os.makedirs(a.queue, exist_ok=True)
    extra = dict(worker_l1_size=a.worker_l1) if a.worker_l1 else {}
    desc, shape = MGDS[a.mgd]
    if desc is not None:  # the fabric mesh (Stage 14's s14common.open_mesh)
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = os.path.join(MESH_DIR, desc)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    t0 = time.time()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape), trace_region_size=a.trace_region_mb * 1_000_000, **extra)
    open_s = time.time() - t0
    print(f"SERVE_OPEN {now()} open_s={open_s:.2f} mgd={a.mgd} chip={os.environ.get('GLM53_CHIP')} worker_l1={a.worker_l1} "
          f"watcher={bool(os.environ.get('TT_METAL_WATCHER'))} deadline={a.deadline}", flush=True)  # fmt: skip
    ctx = {"open_s": open_s, "worker_l1": a.worker_l1, "mgd": a.mgd}
    rc, paused = 0, False
    idle_since = time.time()
    try:
        while True:
            if os.path.exists(os.path.join(a.queue, "STOP")):
                print(f"SERVE_STOP file {now()}", flush=True)
                break
            if deadline is not None and time.time() >= deadline:
                print(f"SERVE_DEADLINE reached {now()}", flush=True)
                break
            if note_pending():
                if not paused:
                    print(f"SERVE_PAUSED {now()}: COORDINATOR_NOTE newer than NOTE_SEEN", flush=True)
                paused = True
                time.sleep(1)  # paused (JOB PIPELINE RULE 2026-10-19T23:52:53Z: 1 s when paused)
                continue
            paused = False
            jobs = sorted(glob.glob(os.path.join(a.queue, "*.job")), key=os.path.getmtime)
            if not jobs:
                if a.idle_exit_min > 0 and time.time() - idle_since >= a.idle_exit_min * 60:
                    print(f"SERVE_IDLE_EXIT {now()} (queue empty {a.idle_exit_min} min)", flush=True)
                    break
                time.sleep(0.2)  # JOB PIPELINE RULE: poll 0.2 s between jobs
                continue
            job = jobs[0]
            run = job[:-4] + ".running"
            os.rename(job, run)
            name = os.path.basename(job)[:-4]
            res, err, t1 = None, None, time.time()
            try:
                spec = json.load(open(run))
                print(f"JOB_START {name} {now()} {spec}", flush=True)
                if "module" in spec:
                    res = run_module(mesh, ctx, spec)
                else:
                    ns = runpy.run_path(spec["script"], init_globals={"MESH": mesh, "ARGS": spec.get("args", {}),
                                                                      "CTX": ctx}, run_name="__job__")  # fmt: skip
                    res = ns.get("RESULT")
                    ns = None
            except BaseException as e:  # noqa: BLE001 -- recorded, then the device health decides
                err = traceback.format_exc()
                print(f"JOB_ERROR {name}\n{err}", flush=True)
                if isinstance(e, KeyboardInterrupt):
                    rc = 130
            gc.collect()  # a failed job's frames keep device tensors alive otherwise
            out = dict(name=name, start=t1, wall_s=round(time.time() - t1, 2), ok=err is None and res in (None, 0),
                       result=res, error=err[-4000:] if err else None)  # fmt: skip
            with open(job[:-4] + ".out", "w") as f:
                json.dump(out, f, default=str, indent=1)
            os.rename(run, job[:-4] + (".done" if out["ok"] else ".failed"))
            print(f"JOB_END {name} {now()} ok={out['ok']} result={res} wall_s={out['wall_s']}", flush=True)
            idle_since = time.time()
            if rc == 130:
                break
            if res == 3:
                print("SERVE_DEVICE_TIMEOUT in the job -> close, exit rc 3 (no reset by this stage)", flush=True)
                rc = 3
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
        ctx.clear()
        gc.collect()
        ttnn.close_mesh_device(mesh)
        if desc is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        print(f"SERVE_CLOSED rc={rc} {now()}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
