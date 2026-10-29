# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the prefill chunk planner and the MLA/DSA context guard (no device).

    pytest models/autoports/zai_org_glm_5_3_flash/tests/test_contract.py
"""

from __future__ import annotations

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV
from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import (
    PREFILL_ALIGN,
    PREFILL_CHUNK,
    bucket_for,
    plan_prefill_chunks,
)
from models.autoports.zai_org_glm_5_3_flash.tt.indexer import check_dense_exact, dense_selection_is_exact

LENGTHS = [1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 333, 1000, 2047, 2048, 2049, 2050, 2051, 2052, 3000, 4095]
LENGTHS += [4096, 32768, 1048575]


@pytest.mark.parametrize("attention", ["kda", "mla"])
def test_prefill_plan(attention, expect_error):
    align = PREFILL_ALIGN[attention]
    buckets = set(range(align, PREFILL_CHUNK + 1, align))  # what FunctionalDecoder.prepare_prefill compiles
    for n in LENGTHS:
        plan = plan_prefill_chunks(n, align)
        assert sum(v for _, v, _ in plan) == n
        assert [off for off, _, _ in plan] == list(range(0, n, PREFILL_CHUNK))
        for i, (_, valid, phys) in enumerate(plan):
            assert valid == PREFILL_CHUNK or i == len(plan) - 1  # only the last chunk is partial
            assert phys in buckets and valid <= phys < valid + align
    with expect_error(ValueError, "at least one token"):
        plan_prefill_chunks(0, align)
    with expect_error(ValueError, "outside"):
        bucket_for(PREFILL_CHUNK + 1, align)


def test_kda_padding_never_fills_a_recurrence_chunk():
    # prepare_chunk_recurrence leaves fully padded 32-token recurrence chunks unspecified
    assert all(bucket_for(n, PREFILL_ALIGN["kda"]) - n < 32 for n in range(1, PREFILL_CHUNK + 1))


def test_dsa_dense_exact_guard(expect_error):
    assert DSA_DENSE_EXACT_MAX_KV == 2051  # index_topk 2048 + kpool 4 - 1
    assert dense_selection_is_exact(2051) and not dense_selection_is_exact(2052)
    check_dense_exact(3, 2051)
    with expect_error(NotImplementedError, "kv_len 2052 > 2051"):
        check_dense_exact(3, 2052)
