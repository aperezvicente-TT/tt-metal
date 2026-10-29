# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Evidence rows for the host-only tests (stage 06a): `record(test, status, **metrics)` appends one JSON line to
$GLM53_06A_RECORDS when set (no-op otherwise), so a test only leaves a row after its assertions ran."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from typing import Any


def _default(o: Any):
    try:
        import torch

        if isinstance(o, torch.Tensor):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def expect_error(error, message=None):
    """Local stand-in for the root conftest `expect_error` fixture (host-only tests run with --noconftest)."""
    import pytest

    return pytest.raises(error, match=message)  # allow-pytest.raises: the one wrapper behind expect_error


_GIT: dict[str, Any] = {}


def _git_state() -> dict[str, Any]:
    """tt-metal commit of the code under test and whether the model directory differs from it."""
    if not _GIT:
        import subprocess

        here = os.path.dirname(os.path.abspath(__file__))
        git = ["git", "-c", "safe.directory=*", "-C", here]
        try:
            sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
            dirty = subprocess.run(
                git + ["status", "--porcelain", "--", os.path.dirname(here)], capture_output=True, text=True, check=True
            ).stdout.strip()
            _GIT.update(tt_metal_sha=sha, tt_metal_dirty=bool(dirty))
        except (OSError, subprocess.CalledProcessError):
            _GIT.update(tt_metal_sha=None, tt_metal_dirty=None)
    return dict(_GIT)


def record(test: str, status: str = "pass", **fields: Any) -> dict[str, Any]:
    tag = os.environ.get("GLM53_06A_RUN_TAG")
    row = {
        "test": f"{test}[{tag}]" if tag else test,
        "status": status,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.executable,
        "host": platform.node(),
        **_git_state(),
        **fields,
    }
    path = os.environ.get("GLM53_06A_RECORDS")
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=_default, ensure_ascii=False) + "\n")
    return row
