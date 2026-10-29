# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the sharded page-table publish's host half (cf3h-d lever A; tt/dsa_shard.py). The one-gather
localization equals the per-chip ShardMap.page_table_local bitwise (values, dtype, shape, the off-owner refusal), and
LocalTableCache's row-incremental tables equal a full localization after every publish.

    no-tt-devices python -m pytest --noconftest -q models/autoports/zai_org_glm_5_3_flash/tests/test_page_table_publish.py
"""

from __future__ import annotations

import random

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DSH


def _owner_table(rows, bps, C, bpu, rng):
    own = torch.arange(bps) // bpu % C
    return own[None].repeat(rows, 1) + C * torch.randint(0, 5000, (rows, bps), generator=rng)


@pytest.mark.parametrize("C,bpu,bps", [(8, 2, 16392), (8, 2, 1024), (4, 2, 333), (2, 1, 77), (1, 2, 64), (8, 4, 130)])
def test_one_gather_localization_equals_the_per_chip_tables(C, bpu, bps):
    sm = DSH.ShardMap(C, DSH.UNIT // bpu)
    rng = torch.Generator().manual_seed(C * 1000 + bps)
    for extra in (0, 3):  # bps_local past the table: zero columns
        bpl = sm.local_cap(bps * sm.block) // sm.block + extra
        for rows in (1, 5, 16):
            pt = _owner_table(rows, bps, C, bpu, rng)
            got = sm.page_tables_local_rows(pt, bpl)
            want = [sm.page_table_local(pt, r, bpl) for r in range(C)]
            for a, b in zip(got, want):
                assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
            if C > 1:
                bad = pt.clone()
                bad[rows - 1, bps // 2] += 1
                with pytest.raises(ValueError) as e1:
                    sm.page_tables_local_rows(bad, bpl)
                with pytest.raises(ValueError) as e2:
                    [sm.page_table_local(bad, r, bpl) for r in range(C)]
                assert str(e1.value) == str(e2.value)


@pytest.mark.parametrize("incremental", ["1", "0"])
def test_local_table_cache_equals_a_full_localization_after_every_publish(monkeypatch, incremental):
    monkeypatch.setenv("GLM53_PT_INCREMENTAL", incremental)
    C, bpu, bps, B, rows = 8, 2, 1030, 16, 32  # rows > B: padding rows past the table are every column's null block
    sm = DSH.ShardMap(C, DSH.UNIT // bpu)
    bpl = sm.local_cap(bps * sm.block) // sm.block
    loc = lambda v, n: DSH.local_rows(v, n, C, bpu, bpl)  # noqa: E731
    rng, gen = random.Random(5), torch.Generator().manual_seed(5)
    cache, key = DSH.LocalTableCache(), (object(), object())
    table = (torch.arange(bps) // bpu % C)[None].repeat(B, 1).to(torch.int32)
    for step in range(60):
        if step == 30:
            key = (object(), object())  # a new device buffer: the full path
        for s in rng.sample(range(B), rng.choice([0, 1, 1, 3])):
            table[s] = _owner_table(1, bps, C, bpu, gen)[0].to(torch.int32)
        local, changed = cache.update(key, table, rows, loc)
        want = loc(table, rows)
        assert all(torch.equal(a, b) for a, b in zip(local, want))
        if incremental == "0" or step in (0, 30):
            assert changed is None
        else:
            assert changed is not None
    assert cache.full == (60 if incremental == "0" else 2)
    bad = table.clone()
    bad[2, 7] += 1
    before = [t.clone() for t in cache.local]
    with pytest.raises(ValueError):
        cache.update(key, bad, rows, loc)
    if incremental == "1":  # a refused publish leaves the cache as it was
        assert all(torch.equal(a, b) for a, b in zip(cache.local, before))
