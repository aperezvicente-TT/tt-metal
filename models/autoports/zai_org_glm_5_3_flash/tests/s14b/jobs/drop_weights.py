# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py): Stage 14b — drop every cached backend and the shared weight cache (CTX["s14b_weights"]) before a
build in another expert layout (sliced XSL <-> EP slots): the two layouts' expert weights do not fit in DRAM together."""
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402

ctx = CTX  # noqa: F821
dropped = M5.prune_ctx(ctx, [])
n = len(ctx.pop("s14b_weights", {}) or {})
gc.collect()
print(f"DROP_WEIGHTS backends={len(dropped)} weight_entries={n}", flush=True)
RESULT = dict(ok=True, backends=len(dropped), weight_entries=n)
