# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runtime host-fallback audit of the FUSED decoder: Stage 1's `test_fallback_audit` (every prefill chunk pass of a
2,100 / 2,000-token prompt incl. the padded non-aligned tail, and one decode pass, under ttnn host-transfer tripwires
and a torch call profiler, with a positive control) run unmodified against `FusedDecoder`."""

from __future__ import annotations

import pytest

from models.autoports.zai_org_glm_5_3_flash.tests import test_fallback_audit as s1
from models.autoports.zai_org_glm_5_3_flash.tests.common import record
from models.autoports.zai_org_glm_5_3_flash.tests.common_fused import load_fused_decoder
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoder


def _load_fused_checked(cfg, layer, mesh):
    dec = load_fused_decoder(cfg, layer, mesh)
    assert isinstance(dec, FusedDecoder), type(dec)
    return dec


@pytest.fixture(autouse=True)
def _fused_path(monkeypatch):
    monkeypatch.setattr(s1, "load_decoder", _load_fused_checked)
    monkeypatch.setattr(s1, "record", lambda row, name="fallback_audit.jsonl": record({"impl": "fused", **row}, name))


test_fallback_audit = s1.test_fallback_audit
