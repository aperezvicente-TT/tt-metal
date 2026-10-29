# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Advertised-context run (1,048,576 tokens) of the KDA layer kinds on the OPTIMIZED decoder: Stage 1's
`test_long_context.test_advertised_context_kda` (1,048,575 streamed prefill tokens through `prefill_forward`, then a
traced decode at position 1,048,575; first call vs the fp32 golden, last 2,048-token chunk and the decode step vs the
fp32 CPU reference layer from the device's carried state; with GLM53_KDA_STATE_DIR the device states at 32K / 64K /
256K / 1M are saved for `compare_kda_drift.py`) run unmodified against `OptimizedDecoder`."""

from __future__ import annotations

import os

import pytest

from models.autoports.zai_org_glm_5_3_flash.tests import test_long_context as s1lc
from models.autoports.zai_org_glm_5_3_flash.tests.common import record
from models.autoports.zai_org_glm_5_3_flash.tests.test_optimized_decoder import _load_optimized_checked


@pytest.fixture(autouse=True)
def _optimized_path(monkeypatch):
    assert not os.environ.get("GLM53_LONG_RESUME_DIR"), "resume is Stage 1-only"
    monkeypatch.setattr(s1lc, "load_decoder", _load_optimized_checked)
    monkeypatch.setattr(s1lc, "record", lambda row, name: record({"impl": "optimized", **row}, name=name))


test_advertised_context_kda = s1lc.test_advertised_context_kda
