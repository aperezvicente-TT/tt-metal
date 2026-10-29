# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Advertised-context run (1,048,576 tokens) of the KDA layer kinds on the FUSED decoder.

Stage 1's `test_long_context.test_advertised_context_kda` is imported unmodified and run against `FusedDecoder` (the
module-level loader is swapped by an autouse fixture, rows are tagged `impl: fused`): 1,048,575 prefill tokens
streamed through `prefill_forward` (32 calls, non-aligned), then a traced decode at position 1,048,575; first call vs
the fp32 golden, last 2,048-token chunk and the decode step vs the fp32 CPU reference layer started from the device's
own carried state. With GLM53_KDA_STATE_DIR set, the device states after 32K / 64K / 256K / 1M tokens are saved and
compared with Stage 1's independent fp32 CPU reference states (`compare_kda_drift.py`, kda_drift.jsonl).
(GLM53_LONG_RESUME_DIR is not supported here: it writes into Stage 1's per-slot state list.)
"""

from __future__ import annotations

import os

import pytest

from models.autoports.zai_org_glm_5_3_flash.tests import test_long_context as s1lc
from models.autoports.zai_org_glm_5_3_flash.tests.common import record
from models.autoports.zai_org_glm_5_3_flash.tests.common_fused import load_fused_decoder
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoder


def _load_fused_checked(cfg_, layer, mesh_):
    dec = load_fused_decoder(cfg_, layer, mesh_)
    assert isinstance(dec, FusedDecoder), type(dec)
    return dec


@pytest.fixture(autouse=True)
def _fused_path(monkeypatch):
    assert not os.environ.get("GLM53_LONG_RESUME_DIR"), "resume is Stage 1-only"
    monkeypatch.setattr(s1lc, "load_decoder", _load_fused_checked)
    monkeypatch.setattr(s1lc, "record", lambda row, name: record({"impl": "fused", **row}, name=name))


test_advertised_context_kda = s1lc.test_advertised_context_kda
