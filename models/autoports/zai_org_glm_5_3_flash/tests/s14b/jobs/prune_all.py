# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py): Stage 14b — close every cached backend / layer set / stack of the long-lived process (their L1
buffers and fabric sync buffers reserve address ranges on every core), keeping the shared weight cache."""
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402

ctx = CTX  # noqa: F821
dropped = M5.prune_ctx(ctx, [])
gc.collect()
print(f"PRUNE_ALL dropped={len(dropped)} kept={[k for k in ctx]}", flush=True)
RESULT = dict(ok=True, dropped=len(dropped))
