# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job wrapper (Stage 14): set ARGS["env"] (knobs read at import / construction), then run either another
serve job script (ARGS["script"] with ARGS["args"] as its ARGS) or a test module's main(MESH, CTX)
(ARGS["module"], reloaded); the env is restored after. RESULT = the inner job's RESULT / {"rc": main's return}."""

import importlib
import os
import runpy

ENV = {k: str(v) for k, v in (ARGS.get("env") or {}).items()}  # noqa: F821
old = {k: os.environ.get(k) for k in ENV}
os.environ.update(ENV)
try:
    if ARGS.get("module"):  # noqa: F821
        mod = importlib.reload(importlib.import_module(ARGS["module"]))  # noqa: F821
        rc = mod.main(MESH, CTX)  # noqa: F821
        RESULT = dict(rc=rc, ok=not rc)
        if rc:
            raise RuntimeError(f"{ARGS['module']}: rc {rc}")  # noqa: F821
    else:
        ns = runpy.run_path(ARGS["script"], init_globals={"MESH": MESH, "ARGS": ARGS.get("args", {}), "CTX": CTX},  # noqa: F821
                            run_name="__job__")  # fmt: skip
        RESULT = ns.get("RESULT")
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
