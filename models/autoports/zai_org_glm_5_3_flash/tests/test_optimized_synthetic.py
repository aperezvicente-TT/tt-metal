# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CI-style synthetic-weight test of the OPTIMIZED decoder (no checkpoint weights, no weight cache, no goldens): Stage 2's
`test_fused_synthetic.test_fused_synthetic_weights` protocol (deterministic per-tensor normal weights with the real
names / shapes / dtypes, fp32 reference layer loaded with the exact device values incl. the per-expert fused_experts
weights, 200-token paged prefill + one traced decode step, PCC >= 0.995 per layer kind, from_state_dict == WeightSource
bitwise for dense_kda) run unmodified against `OptimizedDecoder` (the module's FusedDecoder symbol is swapped by an
autouse fixture; rows tagged `impl: optimized`; the loaded layer is checked to be the optimized class with DRAM-sharded
decode projections)."""

from __future__ import annotations

import pytest

from models.autoports.zai_org_glm_5_3_flash.tests import test_fused_synthetic as s2
from models.autoports.zai_org_glm_5_3_flash.tests.common import record
from models.autoports.zai_org_glm_5_3_flash.tests.test_optimized_decoder import assert_optimized_path
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedDecoder


class _Checked:
    """FusedDecoder stand-in: the same constructors, returning the optimized decoder (asserted on the optimized path)."""

    @staticmethod
    def from_weight_cache(**kw):
        dec = OptimizedDecoder.from_weight_cache(**kw)
        assert_optimized_path(dec)
        return dec

    @staticmethod
    def from_state_dict(sd, **kw):
        dec = OptimizedDecoder.from_state_dict(sd, **kw)
        assert_optimized_path(dec)
        return dec


@pytest.fixture(autouse=True)
def _optimized_path(monkeypatch):
    monkeypatch.setattr(s2, "FusedDecoder", _Checked)
    monkeypatch.setattr(
        s2, "record", lambda row, name="pcc_synthetic.jsonl": record({**row, "impl": "optimized"}, name)
    )


test_optimized_synthetic_weights = s2.test_fused_synthetic_weights
