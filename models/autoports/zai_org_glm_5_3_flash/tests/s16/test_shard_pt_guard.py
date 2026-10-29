# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host test (CPU only): TPPrefill.page_row's null-row fallback under GLM53_DSA_SHARD raises when the unit harness
requires a published owner-aware page table (s16common.build sets require_pt under GLM53_DSA_SHARD), and is unchanged
otherwise (no shard: identity rows; published table: its row; require_pt off: the owners' null row)."""
from types import SimpleNamespace

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import TPPrefill

BPS, NB = 4, 16


def _pf(*, shard: bool, pt=None, require: bool = False):
    sm = SimpleNamespace(null_row=lambda bps: torch.zeros(bps, dtype=torch.int64))
    pf = SimpleNamespace(b=SimpleNamespace(_pt_host=pt), shard=(object() if shard else None), sm=sm,
                         kv=SimpleNamespace(blocks_per_seq=BPS, num_blocks=NB), require_pt=require)  # fmt: skip
    return pf


def test_shard_unpublished_required_raises():
    with pytest.raises(RuntimeError, match="no owner-aware page table"):
        TPPrefill.page_row(_pf(shard=True, require=True), 0)


def test_shard_unpublished_not_required_null_row():
    assert TPPrefill.page_row(_pf(shard=True), 0).tolist() == [0] * BPS


def test_shard_published_required_ok():
    pt = torch.arange(2 * BPS).reshape(2, BPS) + 1
    assert TPPrefill.page_row(_pf(shard=True, pt=pt, require=True), 1).tolist() == pt[1].tolist()


def test_no_shard_identity_unchanged():
    assert TPPrefill.page_row(_pf(shard=False, require=True), 1).tolist() == list(range(BPS, 2 * BPS))


def test_compile_and_warm_passes_are_lenient():
    # 14's SHOULD (2026-10-27T10:09:52Z): inside compile() / warm() (eager passes whose page rows are never read) the
    # null-row fallback stays; outside, an unpublished table still raises
    pf = _pf(shard=True, require=True)
    pf._pt_eager = lambda: TPPrefill._pt_eager(pf)
    with pf._pt_eager():
        assert TPPrefill.page_row(pf, 0).tolist() == [0] * BPS
    with pytest.raises(RuntimeError, match="no owner-aware page table"):
        TPPrefill.page_row(pf, 0)
    seen = []
    pf._compile = lambda buckets=None, head=True: seen.append(TPPrefill.page_row(pf, 0).tolist())
    pf._warm = lambda buckets=None, head=True: seen.append(TPPrefill.page_row(pf, 0).tolist())
    TPPrefill.compile(pf)
    TPPrefill.warm(pf)
    assert seen == [[0] * BPS, [0] * BPS] and not pf._pt_lenient
    with pytest.raises(RuntimeError, match="no owner-aware page table"):
        TPPrefill.page_row(pf, 0)
