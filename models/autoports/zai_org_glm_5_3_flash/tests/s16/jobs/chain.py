# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py): several jobs in order inside ONE job (one server, one CTX), for runners that start a fresh server per
unit -- e.g. p2_full then p3_long_on_full, which must share the process. ARGS: jobs ([{"script", "args"}, ...]),
stop_on_error (True). RESULT: per sub-job {script, ok, error tail}; the first error is re-raised at the end (the
unit fails) unless every later job still ran with stop_on_error False."""
import os
import runpy
import traceback

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
STOP = bool(args.get("stop_on_error", True))
RESULT, first_err = [], None
for spec in args.get("jobs", []):
    name = os.path.basename(spec["script"])
    print(f"CHAIN_START {name}", flush=True)
    try:
        runpy.run_path(spec["script"], init_globals={"MESH": mesh, "ARGS": spec.get("args", {}), "CTX": ctx},
                       run_name="__job__")  # fmt: skip
        RESULT.append(dict(script=name, ok=True))
        print(f"CHAIN_END {name} ok", flush=True)
    except Exception as e:  # noqa: BLE001
        err = traceback.format_exc()
        RESULT.append(dict(script=name, ok=False, error=err[-2000:]))
        print(f"CHAIN_END {name} FAILED\n{err}", flush=True)
        first_err = first_err or e
        if STOP:
            break
if first_err is not None:
    raise first_err
