# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; outside the tree: runs on the server's tree): run another job script under extra GLM53_* knobs for
ONE job only (an A/B inside a standing queue). Sets `env`, closes every cached Stage 14b backend first (the next build
captures with the knobs), runs `script` with `inner` as its ARGS, then closes the backends it built and restores the
environment (later jobs never see the knobs: env-leak-resident-build). ARGS: script (path; relative = the server
tree's tests/s14b/jobs/), env (dict), inner (dict)."""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.environ["TT_METAL_HOME"], "models/autoports/zai_org_glm_5_3_flash/tests/s14b"))
import m5common as M5  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
script = args["script"]
if not os.path.isabs(script):
    script = os.path.join(os.environ["TT_METAL_HOME"], "models/autoports/zai_org_glm_5_3_flash/tests/s14b/jobs", script)
ENV = {k: str(v) for k, v in (args.get("env") or {}).items()}
old = {k: os.environ.get(k) for k in ENV}
print("ENV-WRAP", script, ENV, "closed before:", M5.prune_ctx(ctx, []), flush=True)
os.environ.update(ENV)
try:
    ns = runpy.run_path(script, init_globals={"MESH": mesh, "ARGS": dict(args.get("inner") or {}), "CTX": ctx},
                        run_name="__job__")  # fmt: skip
    RESULT = dict(ns.get("RESULT") or {}, env=ENV)
finally:
    print("ENV-WRAP closed after:", M5.prune_ctx(ctx, []), flush=True)
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
