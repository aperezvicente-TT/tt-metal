# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a long-lived device process (program HARD RULE 2026-10-19T19:10Z, PRODUCTION_PATH.md §6): ONE process
per chip for the whole session — the chip is opened once (unit mesh; worker_l1_size fixed at open), then the device
checks run as jobs INSIDE this process; never close and reopen between jobs. Pattern of Stage 14's tests/s14/serve.py.

  <queue>/<name>.job   JSON {"module": "dev_mk_layer", "env": {"GLM53_MK_ONE": "1", ...}} -> the tests/s7a module is
                       (re)loaded with that environment and its main(mesh=MESH, ctx=CTX) runs; or {"script": <path>,
                       "args": {...}} (runpy with globals MESH, ARGS, CTX; sets RESULT). Renamed .running -> .done /
                       .failed, result in <name>.out (JSON).
  <queue>/STOP         stop after the current job; --deadline <ISO-8601 UTC>: no job starts after it.
  <queue>/PAUSE        no job starts while it exists (edit protocol: PAUSE, wait until no .running, edit + bump the
                       kernel's *_REV, remove PAUSE).
Code revision (Stage 14's m3_tp pattern): before every job the tree's revision (HEAD + a hash of the uncommitted model
tt/ diff: kernels + model Python) is compared with the loaded one; on a change every loaded models...tt module is
reloaded (two passes) and CTX's cached objects are dropped, so a job never runs new kernels with old imported Python.
--mgd pair (iteration rules v2, 2026-10-20: 7a's standing card = card1, chips 2,3, a TP2 fabric pair): the pair is opened
once with FABRIC_1D; a job runs on the unit submesh of the pair's chip spec["chip"] (0 | 1, default 0) or, with
spec["mesh"] == "pair", on the whole 1x2 mesh (TP2 jobs: script jobs see MESH = the pair, CTX["subs"] = the submeshes).
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

W = os.environ.get("GLM53_WORK", "/opt/glm53/work-s7a")
PKG = "models.autoports.zai_org_glm_5_3_flash.tests.s7a"
TT_PKG = "models.autoports.zai_org_glm_5_3_flash.tt"
KEEP_CTX = ("open_s", "worker_l1", "mgd", "subs")
MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mesh")


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


CONTROL = ("OPENS", "CLOSED", "CAP", "STOP", "RESET", "GO", "HOLD")


def is_control(line: str) -> bool:
    """Program throughput rule (2026-10-20 14:27Z): only control lines pause the queue — every scheduler line, and a
    program line whose message (after the first ' — ') has OPENS / CLOSED / CAP / STOP / RESET / GO / HOLD as a word
    in its first 40 characters."""
    import re

    if "COORDINATOR-SCHEDULER" in line:
        return True
    head = (line.split(" \u2014 ", 1)[1] if " \u2014 " in line else line)[:40]
    return any(re.search(rf"\b{t}\b", head) for t in CONTROL)


def note_pending():
    """A note newer than NOTE_SEEN pauses only if a line after the session's handled baseline (.note_handled) is a
    control line (no baseline: any newer note pauses)."""
    note, seen = os.path.join(W, "COORDINATOR_NOTE.txt"), os.path.join(W, "NOTE_SEEN")
    if not os.path.exists(note) or (os.path.exists(seen) and os.path.getmtime(note) <= os.path.getmtime(seen)):
        return False
    try:
        handled = int(open(os.path.join(W, ".note_handled")).read().split()[0])
    except (OSError, ValueError, IndexError):
        return True
    return any(is_control(line) for line in open(note).read().splitlines()[handled:])


def tree_rev() -> str:
    import hashlib
    import subprocess

    home, md = os.environ.get("TT_METAL_HOME", "."), "models/autoports/zai_org_glm_5_3_flash/tt"
    head = subprocess.run(["git", "-C", home, "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout
    diff = subprocess.run(["git", "-C", home, "diff", "HEAD", "--", md], capture_output=True, text=True).stdout
    return head.strip() + ("_" + hashlib.sha1(diff.encode()).hexdigest()[:8] if diff else "")


def reload_tt(ctx) -> int:
    """Reload every loaded models...tt module, two passes (sys.modules order is not leaves-first: the second pass
    rebinds every `from x import y` to the reloaded code); drop CTX's objects (built from the old classes)."""
    names = [n for n in list(sys.modules) if n.startswith(TT_PKG + ".") and sys.modules[n] is not None]
    for _ in range(2):
        for n in names:
            importlib.reload(sys.modules[n])
    for k in [k for k in ctx if k not in KEEP_CTX]:
        ctx.pop(k)
    gc.collect()
    return len(names)


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
    ap.add_argument("--mgd", default="unit", choices=["unit", "pair"])
    ap.add_argument(
        "--idle-exit-min", type=float, default=0, help="exit after this many idle minutes (0 = never; mocks)"
    )
    a = ap.parse_args()
    deadline = dt.datetime.fromisoformat(a.deadline.replace("Z", "+00:00")).timestamp() if a.deadline else None
    os.makedirs(a.queue, exist_ok=True)
    extra = dict(worker_l1_size=a.worker_l1) if a.worker_l1 else {}
    t0 = time.time()
    if a.mgd == "pair":
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = os.path.join(MESH_DIR, "glm_pair_1x2_mesh_graph_descriptor.textproto")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=a.trace_region_mb * 1_000_000, **extra)
        subs = [mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, c)) for c in range(2)]
    else:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=a.trace_region_mb * 1_000_000, **extra)
        subs = [mesh]
    open_s = time.time() - t0
    print(f"SERVE_OPEN {now()} open_s={open_s:.2f} mgd={a.mgd} chip={os.environ.get('GLM53_CHIP')} "
          f"visible={os.environ.get('TT_VISIBLE_DEVICES')} device_ids={list(mesh.get_device_ids())} "
          f"worker_l1={a.worker_l1} watcher={bool(os.environ.get('TT_METAL_WATCHER'))} deadline={a.deadline}",
          flush=True)  # fmt: skip
    ctx = {"open_s": open_s, "worker_l1": a.worker_l1, "mgd": a.mgd, "subs": subs}
    rc, paused, rev, jm, idle_since, last_kind = 0, False, tree_rev(), None, time.time(), None
    print(f"SERVE_REV {rev}", flush=True)
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
                time.sleep(1)  # job-pipeline rule: 1 s polling while paused
                continue
            paused = False
            if os.path.exists(os.path.join(a.queue, "PAUSE")):
                time.sleep(1)
                continue
            jobs = sorted(glob.glob(os.path.join(a.queue, "*.job")), key=os.path.getmtime)
            if not jobs:
                if a.idle_exit_min and time.time() - idle_since > 60 * a.idle_exit_min:
                    print(f"SERVE_IDLE_EXIT {now()}", flush=True)
                    break
                time.sleep(0.2)  # job-pipeline rule: 0.2 s between jobs
                continue
            new_rev = tree_rev()
            if new_rev != rev:
                n_mod = reload_tt(ctx)
                print(
                    f"SERVE_RELOAD {now()} rev {rev} -> {new_rev}: {n_mod} tt modules reloaded, CTX cleared", flush=True
                )
                rev = new_rev
            job = jobs[0]
            run = job[:-4] + ".running"
            os.rename(job, run)
            name = os.path.basename(job)[:-4]
            res, err, t1 = None, None, time.time()
            try:
                spec = json.load(open(run))
                print(f"JOB_START {name} {now()} {spec}", flush=True)
                kind = "pair" if spec.get("mesh") == "pair" else "unit"
                if kind == "pair" and a.mgd != "pair":
                    raise RuntimeError("job wants the pair mesh; this server is --mgd unit")
                if last_kind is not None and kind != last_kind and not spec.get("allow_mix"):
                    # one kind per process: pair (fabric) programs and unit programs in one process corrupted card1
                    # three times (2026-10-20 08:06Z sparse NaN after pair jobs; 07:31Z / 14:14Z the pair fabric probe
                    # refused after unit jobs even with CTX dropped; 14:14Z the watcher's NoC state corrupted in the
                    # unit job after it) -> queue them in separate processes
                    raise RuntimeError(f"job kind {kind} after {last_kind} jobs in this process (one kind per process)")
                last_kind = kind
                jm = mesh if kind == "pair" else subs[int(spec.get("chip", 0))]
                if a.mgd == "pair":  # the physical chip(s) of this job, for the rows / logs
                    vis = (os.environ.get("TT_VISIBLE_DEVICES") or "0,1").split(",")
                    os.environ["GLM53_CHIP"] = (
                        ",".join(vis) if spec.get("mesh") == "pair" else vis[int(spec.get("chip", 0))]
                    )
                if "module" in spec:
                    res = run_module(jm, ctx, spec)
                else:
                    ns = runpy.run_path(spec["script"], init_globals={"MESH": jm, "ARGS": spec.get("args", {}),
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
                rc = 7  # close without synchronizing the timed-out device (dev.sh: 7 is not a clean code -> escalate)
                break
            if err is not None and ("device timeout" in err or "unrecoverable" in err):
                # Stage 14's lesson (2026-10-20 03:42Z) and card1 06:53Z: after a device timeout the synchronize
                # "health check" blocks on (or falsely passes) the hung device -> close without it, exit rc 7
                print("SERVE_HEALTH device timeout in the job -> exit rc 7 (no reset by this stage)", flush=True)
                rc = 7
                break
            if err is not None:
                try:
                    ttnn.synchronize_device(mesh)
                    for m in subs[1:] if a.mgd == "pair" else []:
                        ttnn.synchronize_device(m)
                    print("SERVE_HEALTH ok after job error", flush=True)
                except Exception as e2:  # noqa: BLE001
                    print(f"SERVE_HEALTH FAILED after job error: {e2!r} -> exit rc 6", flush=True)
                    rc = 6
                    break
    finally:
        ctx.clear()
        # no submesh reference may outlive the parent's close (card1 06:58Z: a submesh closed at interpreter exit threw
        # inside a destructor -> terminate -> SIGABRT)
        subs = jm = None
        gc.collect()
        try:
            if rc != 7:  # not on a hung device
                ttnn.synchronize_device(mesh)
        except Exception as e:  # noqa: BLE001
            print(f"SERVE_CLOSE synchronize failed: {e!r}", flush=True)
            rc = rc or 6
        if a.mgd == "pair" and rc != 7 and os.environ.get("GLM53_MOCK") != "1":  # (the mock cluster segfaults in it)
            # a parent whose CQ is "in use" by its submeshes refuses to close (and the submesh then throws in its
            # destructor -> SIGABRT, card1 06:58Z / 07:39Z): quiesce_devices() quiesces the submeshes and clears the
            # flags first (tests/mesh8_bench.py's recipe, 2026-10-17)
            try:
                mesh.quiesce_devices()
            except Exception as e:  # noqa: BLE001
                print(f"SERVE_CLOSE quiesce_devices failed: {e!r}", flush=True)
        try:
            ttnn.close_mesh_device(mesh)
        except Exception as e:  # noqa: BLE001
            print(f"SERVE_CLOSE close_mesh_device failed: {e!r}", flush=True)
            rc = rc or 6
        mesh = None
        gc.collect()
        if a.mgd == "pair":
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        print(f"SERVE_CLOSED rc={rc} {now()}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
