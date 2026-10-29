# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14, MkMla native t <= 32): Stage 7b's TPMla harness (tests/s7b/dev_mk_tp_mesh.py: TPMla on the
open mesh vs the replicated one-chip MkMla, s1 / cache rows bitwise across chips, o_lat vs float64) with ARGS["env"]
(GLM53_MK_TS = the row counts, ...) set BEFORE the module is (re)loaded (its knobs are module constants), restored
after. ARGS: env (dict), dsa (bool: tests/s7b/dev_mk_dsa.py instead)."""

import importlib
import os

ENV = {k: str(v) for k, v in (ARGS.get("env") or {}).items()}  # noqa: F821
old = {k: os.environ.get(k) for k in ENV}
os.environ.update(ENV)
try:
    name = "dev_mk_dsa" if ARGS.get("dsa") else "dev_mk_tp_mesh"  # noqa: F821
    mod = importlib.import_module(f"models.autoports.zai_org_glm_5_3_flash.tests.s7b.{name}")
    mod = importlib.reload(mod)
    rc = mod.main(MESH, CTX)  # noqa: F821
    RESULT = dict(rc=rc, ok=rc == 0)
    if rc:
        raise RuntimeError(f"{name}: {rc} failing row(s)")
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
