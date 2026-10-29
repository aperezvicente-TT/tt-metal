# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Script job (tests/s7a/serve.py): reload every loaded models...tt module (two passes, with the server's current
environment) and drop CTX's objects — module-level constants read from the environment at import come back clean
after a job that imported / reloaded them under other knobs (Stage 14's m3_tp reloads its modules with its knobs set)."""
import gc
import importlib
import sys

PKG = "models.autoports.zai_org_glm_5_3_flash.tt."
names = [n for n in list(sys.modules) if n.startswith(PKG) and sys.modules[n] is not None]
for _ in range(2):
    for n in names:
        importlib.reload(sys.modules[n])
keep = ("open_s", "worker_l1", "mgd", "subs")
for k in [k for k in CTX if k not in keep]:  # noqa: F821
    CTX.pop(k)  # noqa: F821
gc.collect()
print(f"RELOAD_TT {len(names)} modules, CTX cleared", flush=True)
RESULT = 0
