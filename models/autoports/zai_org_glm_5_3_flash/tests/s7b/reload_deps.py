# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""refresh() is called first by every Stage 7b job module: the long-lived device process (serve.py) reloads only the
job module, so the helper modules a job imports would otherwise stay at their first-imported version for the whole
session. A helper is reloaded only when its source changed since its last load: CTX-resident objects (the JOB PIPELINE
RULE's resident state) keep instances of the helpers' classes, and a reload of unchanged code would make them fail
every isinstance check against the new class objects (a resident GroupedWeight taken for a DSLinear)."""

import hashlib
import importlib
import sys

MODULES = (  # dependencies first: mk_mla binds mhc_pre's module object (an in-place reload updates it)
    "models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre",
    "models.autoports.zai_org_glm_5_3_flash.tt.mk_mla",
    "models.autoports.zai_org_glm_5_3_flash.tt.tp_mla",
    "models.autoports.zai_org_glm_5_3_flash.tests.s7b.mla_common",
)


def _digest(mod) -> str:
    with open(mod.__file__, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def refresh():
    for name in MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        d = _digest(mod)
        if getattr(mod, "_s7b_digest", None) == d:
            continue
        mod = importlib.reload(mod)
        mod._s7b_digest = d
