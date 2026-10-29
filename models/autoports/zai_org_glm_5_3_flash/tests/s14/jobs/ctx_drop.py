# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14): close and drop the CTX-resident backends / objects whose key starts with one of ARGS
prefixes (default: every tuple key) before a job builds another backend in the same long-lived process (StackBackend
objects keep traces, L1 intermediates and DRAM state; m5common.prune_ctx drops only "m5..." keys)."""

import gc

import ttnn

prefixes = tuple(ARGS.get("prefixes", ()))  # noqa: F821
drop = [k for k in CTX if (not prefixes and isinstance(k, tuple))  # noqa: F821
        or (isinstance(k, tuple) and k and str(k[0]).startswith(prefixes))]  # fmt: skip
closed = []
for k in drop:
    v = CTX.get(k)  # noqa: F821
    be = v.get("backend") if isinstance(v, dict) else None
    if be is not None and hasattr(be, "close"):
        try:
            be.close()
            closed.append(str(k[0]))
        except Exception as e:  # noqa: BLE001 -- best effort
            print(f"ctx_drop: {k}: close {e!r}", flush=True)
for k in drop:
    CTX.pop(k, None)  # noqa: F821
v = be = None
gc.collect()
ttnn.synchronize_device(MESH)  # noqa: F821
print(f"CTX_DROP dropped={[str(k[0]) for k in drop]} closed={closed}", flush=True)
RESULT = dict(ok=True, dropped=len(drop), closed=closed)
