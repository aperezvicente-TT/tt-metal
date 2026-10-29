# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; any tree): close every backend the long-lived process caches (Stage 14b "m5..." keys and Stage 16
"s16-..." entries: their traces released, buffers freed) so the next job's construction compiles on a clean device
(no program compiled after another backend's trace capture). ARGS: none."""
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.environ["TT_METAL_HOME"], "models/autoports/zai_org_glm_5_3_flash/tests/s14b"))
import m5common as M5  # noqa: E402

ctx = CTX  # noqa: F821
dropped = [str(k[:2]) for k in M5.prune_ctx(ctx, [])]
for k in [k for k in ctx if isinstance(k, tuple) and k and str(k[0]).startswith("s16")]:
    ent = ctx.pop(k)
    try:
        if isinstance(ent, dict) and ent.get("pf") is not None:
            ent["pf"].release()
        if isinstance(ent, dict) and ent.get("backend") is not None:
            ent["backend"].close()
    except Exception as e:  # noqa: BLE001
        print(f"ctx_prune: {k}: {e!r}", flush=True)
    dropped.append(str(k[:2]))
gc.collect()
print("CTX-PRUNE dropped", dropped, flush=True)
RESULT = dict(ok=True, dropped=dropped)
