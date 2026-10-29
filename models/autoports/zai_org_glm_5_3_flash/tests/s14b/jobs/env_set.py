# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; any tree): set the process environment for the following jobs -- ARGS env: {key: value}; a null value
REMOVES the key (probe_hooks' noop env can only set, so a knob one unit set would leak into later units' builds)."""
import os

env = ARGS.get("env") or {}  # noqa: F821
for k, v in env.items():
    if v is None:
        os.environ.pop(k, None)
    else:
        os.environ[k] = str(v)
print("ENV-SET", {k: os.environ.get(k) for k in env}, flush=True)
RESULT = dict(ok=True, env={k: os.environ.get(k) for k in env})
