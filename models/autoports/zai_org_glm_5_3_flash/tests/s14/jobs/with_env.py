# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14): run another job script (ARGS["script"], e.g. tests/s14b/jobs/m5c_tf_lag.py) with ARGS["env"]
set for its whole run (restored after) and ARGS["args"] as its ARGS - for job scripts that take no env of their own
(the knobs must hold while the backend is built: GLM53_STACK_DSA, ...)."""

import os
import runpy

ENV = {k: str(v) for k, v in (ARGS.get("env") or {}).items()}  # noqa: F821
old = {k: os.environ.get(k) for k in ENV}
os.environ.update(ENV)
try:
    ns = runpy.run_path(ARGS["script"], init_globals={"MESH": MESH, "ARGS": ARGS.get("args", {}), "CTX": CTX},  # noqa: F821
                        run_name="__job__")  # fmt: skip
    RESULT = ns.get("RESULT")
    ns = None
    if isinstance(RESULT, dict) and RESULT.get("ok") is False:
        raise RuntimeError(f"{os.path.basename(ARGS['script'])}: {RESULT}")  # noqa: F821
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
