# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/owner_blocks.py OwnerBlockMap (GLM53_DSA_SHARD owner-aware physical blocks), host-only."""

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt.owner_blocks import OwnerBlockMap


def _owners_ok(m, phys_row, covered):
    for j, g in enumerate(phys_row):
        assert g % m.C == m.owner(j), (j, g)
        if j >= covered:
            assert g == m.null(j)


def _state(m):
    return (dict(m.phys), dict(m.refs), [list(f) for f in m.free], [list(od) for od in m.idle],
            {s: list(r) for s, r in m.rows.items()}, m.evictions)  # fmt: skip


def _total(m):
    return [f + e for f, e in zip(m.free_counts(), m.evictable_counts())]


def test_columns_land_on_their_unit_owner_and_tail_is_null():
    m = OwnerBlockMap(C=4, nb_local=16, bpu=2)
    row = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, -1, -1]
    out = m.set_row(0, row, covered=10)
    _owners_ok(m, out, 10)
    assert len(set(out[:10])) == 10  # distinct physical blocks
    assert all(g // m.C >= 1 for g in out[:10])  # never a null block
    assert out[10:] == [m.null(10), m.null(11)]


def test_prefix_hit_reuses_the_physical_block_and_refcounts():
    m = OwnerBlockMap(C=2, nb_local=8, bpu=2)
    a = m.set_row(0, [5, 6, 7, 8], covered=4)
    b = m.set_row(1, [5, 6, 9, 10], covered=4, keep=2)  # the shared prefix: blocks 5, 6 at the same columns
    assert a[:2] == b[:2] and a[2:] != b[2:]
    assert m.refs[5] == 2 and m.refs[7] == 1


def test_a_released_block_stays_mapped_for_a_later_prefix_hit():
    """vLLM keeps a finished request's blocks cached: a later request's hit (keep columns) finds the same physical
    block, content intact"""
    m = OwnerBlockMap(C=2, nb_local=8, bpu=2)
    a = m.set_row(0, [5, 6, 7], covered=3)
    m.release_slot(0)
    assert m.physical(5) == a[0] and m.evictable_counts() == [2, 1] and _total(m) == [7, 7]
    b = m.set_row(3, [5, 6, 11], covered=3, keep=2)
    assert b[:2] == a[:2] and m.evictable_counts() == [0, 1]  # 5, 6 live again; 7 still cached


def test_rewrite_releases_blocks_no_longer_listed_to_the_cache():
    m = OwnerBlockMap(C=2, nb_local=6, bpu=2)
    m.set_row(0, [1, 2, 3, 4], covered=4)
    m.set_row(0, [1, 2, 7, 8], covered=4)  # columns 2, 3 now list new blocks
    assert m.physical(3) is not None and not m.refs.get(3) and m.evictable_counts() == [0, 2]


def test_a_dry_owner_evicts_its_least_recently_released_block():
    m = OwnerBlockMap(C=2, nb_local=3, bpu=2)  # two usable blocks per chip
    m.set_row(0, [1, 2], covered=2)  # chip 0 full
    m.release_slot(0)  # 1 then 2 cached (1 least recent)
    out = m.set_row(1, [3], covered=1)  # chip 0 dry: evicts 1
    assert m.physical(1) is None and m.physical(2) is not None and m.evictions == 1
    assert out[0] % 2 == 0
    with pytest.raises(ValueError, match="no device content"):
        m.set_row(2, [1, 2], covered=2, keep=2)  # a hit that claims evicted block 1: refused
    assert m.set_row(2, [2], covered=1, keep=1)[0] == m.physical(2)  # the surviving cached block still hits


def test_all_live_dry_owner_raises():
    m = OwnerBlockMap(C=2, nb_local=2, bpu=2)  # one usable block per chip
    m.set_row(0, [1], covered=1)
    with pytest.raises(RuntimeError, match="no free physical block"):
        m.set_row(1, [2], covered=1)


def test_a_live_block_moved_to_another_owner_raises_and_rolls_back():
    m = OwnerBlockMap(C=2, nb_local=6, bpu=2)
    m.set_row(0, [1, 2, 3, 4], covered=4)  # block 1 at column 0 -> chip 0
    st = _state(m)
    with pytest.raises(ValueError, match="live on chip"):
        m.set_row(1, [9, 10, 1, 11], covered=4)  # block 1 at column 2 -> chip 1 while still live
    assert _state(m) == st  # nothing mapped, nothing taken


def test_a_cached_block_reused_at_another_owner_moves():
    """vLLM evicted block b from its cache and hands it to a new request at another column: new content -> b moves"""
    m = OwnerBlockMap(C=2, nb_local=6, bpu=2)
    a = m.set_row(0, [1, 2, 3], covered=3)
    m.release_slot(0)
    out = m.set_row(1, [9, 10, 1], covered=3)  # 1 (cached on chip 0) at column 2 = chip 1, fresh content
    _owners_ok(m, out, 3)
    assert out[2] != a[0] and m.physical(1) == out[2] and _total(m) == [5 - 2, 5 - 1]  # live: 9, 10 | 1


def test_local_blocks_sizing_has_null_and_per_slot_headroom():
    assert OwnerBlockMap.local_blocks_for(8, 1000, max_batch=32) == 125 + 1 + 64


def test_remap_moves_rows_without_touching_blocks():
    m = OwnerBlockMap(C=2, nb_local=8, bpu=2)
    a = m.set_row(0, [5, 6, 7], covered=3)
    b = m.set_row(1, [8, 9], covered=2)
    free = m.free_counts()
    m.remap({0: 3, 1: 0})  # a chain
    assert m.free_counts() == free and m.physical(5) == a[0] and m.physical(8) == b[0]
    m.release_slot(3)  # slot 0's old row now lives in slot 3
    assert not m.refs.get(5) and m.refs[8] == 1
    m.release_slot(0)
    assert _total(m) == [7, 7] and not m.refs
    m.set_row(2, [1], covered=1)
    with pytest.raises(ValueError, match="holds a live row"):
        m.remap({0: 2})
    assert m.physical(1) is not None and 2 in m.rows


def test_own_row_reshuffled_moves_blocks_to_the_new_owner():
    """a new request in the same slot lists the slot's previous blocks at other columns (test_generator's reversed
    identity row): blocks only this slot held move to the new column's owner; a block another live row also lists keeps
    its owner (the row is refused and nothing changes)"""
    m = OwnerBlockMap(C=2, nb_local=8, bpu=2)
    m.set_row(0, [1, 2, 3, 4], covered=4)  # columns 0, 1 -> chip 0; 2, 3 -> chip 1
    out = m.set_row(0, [4, 3, 2, 1], covered=4)  # every block changes owner
    _owners_ok(m, out, 4)
    assert len(set(out)) == 4 and _total(m) == [5, 5]  # each move frees one block and takes one
    m.set_row(1, [9, 4], covered=2)  # slot 1 shares block 4 at column 1: chip 0, as slot 0's column 0
    assert m.physical(4) == out[0]
    st = _state(m)
    with pytest.raises(ValueError, match="live on chip"):
        m.set_row(0, [3, 2, 4, 1], covered=4)  # 2 would move (only slot 0 holds it), 4 may not (slot 1 holds it)
    assert _state(m) == st  # rolled back exactly


def test_eviction_rolls_back_with_a_refused_row():
    m = OwnerBlockMap(C=2, nb_local=3, bpu=2)
    m.set_row(0, [1, 2, 3, 4], covered=4)
    m.release_slot(0)  # all four cached
    m.set_row(1, [5, 6], covered=2)  # chip 0: evicts 1 and 2
    st = _state(m)
    with pytest.raises(RuntimeError, match="no free physical block"):
        m.set_row(2, [-1, -1, 9, 10, 7, 8], covered=6)  # chip 1: evicts 3, 4; then chip 0 (5, 6 live) is dry
    assert _state(m) == st and m.physical(3) is not None and m.physical(4) is not None


def test_a_duplicate_block_in_one_row_raises():
    m = OwnerBlockMap(C=2, nb_local=8, bpu=2)
    with pytest.raises(ValueError, match="listed twice"):
        m.set_row(0, [1, 2, 1], covered=3)
    assert not m.phys


# ---- Stage 12 x 14: a KV-connector LOAD admits a hit's blocks at their columns before any row lists them
def test_admit_maps_a_hit_before_its_row_and_set_row_keeps_it():
    m = OwnerBlockMap(C=8, nb_local=6, bpu=2)
    pairs = [(b, j) for j, b in enumerate([20, 21, 22, 23, 24, 25])]  # columns 0-5: units 0, 1, 2 = chips 0, 1, 2
    phys = m.admit(pairs)
    assert [g % 8 for g in phys] == [0, 0, 1, 1, 2, 2] and all(g // 8 >= 1 for g in phys)
    assert not m.refs and m.evictable_counts() == [2, 2, 2, 0, 0, 0, 0, 0]  # mapped, cached, evictable
    row = m.set_row(0, [20, 21, 22, 23, 24, 25, 30], covered=7, keep=6)  # the hit's continuation lists them
    assert row[:6] == phys and m.refs[20] == 1 and m.evictable_counts() == [0] * 8


def test_admit_reuses_moves_or_refuses_like_set_row():
    m = OwnerBlockMap(C=2, nb_local=6, bpu=2)
    a = m.set_row(0, [1, 2, 3, 4], covered=4)  # 1, 2 on chip 0; 3, 4 on chip 1
    st = _state(m)
    with pytest.raises(ValueError, match="live on chip"):
        m.admit([(9, 0), (1, 2)])  # block 1 live on chip 0, loaded at column 2 (chip 1)
    assert _state(m) == st  # atomic: 9 was not mapped either
    assert m.admit([(1, 0)]) == [a[0]]  # a live block at its own owner: reused in place
    m.release_slot(0)
    out = m.admit([(3, 0)])  # cached on chip 1, loaded at column 0: moves to chip 0
    assert out[0] % 2 == 0 and m.physical(3) == out[0] and _total(m) == [5, 5]
    with pytest.raises(ValueError, match="admitted twice"):
        m.admit([(7, 0), (7, 1)])


def test_admit_never_evicts_its_own_blocks():
    m = OwnerBlockMap(C=2, nb_local=3, bpu=2)  # two usable blocks per chip
    m.set_row(0, [1, 2], covered=2)
    m.release_slot(0)  # chip 0: 1, 2 cached
    out = m.admit([(5, 0), (6, 1)])  # chip 0 dry twice: evicts 1 then 2, never 5
    assert len(set(out)) == 2 and m.physical(1) is None and m.physical(2) is None and m.evictions == 2
    assert m.physical(5) == out[0] and m.physical(6) == out[1]
    with pytest.raises(RuntimeError, match="no free physical block"):
        m.admit([(7, 0), (8, 1), (9, 2 * 2)])  # 7, 8 take chip 0 by evicting 5, 6; 9 at column 4 (chip 0) finds it dry
    assert m.physical(5) == out[0] and m.physical(7) is None  # rolled back: the evictions undone


@pytest.mark.parametrize("seed", range(6))
def test_extending_rows_walk_only_new_columns_with_the_full_walks_result(seed):
    """cf3h-d lever A: set_row's fast path (the row extends the slot's live row) returns the same physical row, raises
    the same refusals and leaves the same map (phys, refs, free lists, idle LRU order, rows, evictions) as the full
    walk, through a random history of growth, fresh and prefix-shared rows, -1 entries, keep bounds, releases, admits
    and forgets on a pool small enough to evict and to run dry"""
    import random

    rng = random.Random(seed)
    fast, full = OwnerBlockMap(C=4, nb_local=9, bpu=2), OwnerBlockMap(C=4, nb_local=9, bpu=2)
    full.EXTEND = False
    rows, nid, extended, refused = {}, 1, 0, []

    def both(op, *args):
        res = []
        for m in (fast, full):
            try:
                res.append(("ok", getattr(m, op)(*args)))
            except (RuntimeError, ValueError) as e:
                res.append((type(e).__name__, str(e)))
        assert res[0] == res[1], (op, args, res)
        assert _state(fast) == _state(full), (op, args)
        if res[0][0] != "ok":
            refused.append(res[0][1].split(" ")[0] + " " + res[0][1].split(" ")[-1])
        return res[0][0] == "ok"

    for _ in range(600):
        s, x = rng.randrange(4), rng.random()
        if x < 0.12 and s in rows:
            both("release_slot", s)
            rows.pop(s)
            continue
        if x < 0.17:
            pairs = [(rng.randrange(1, nid + 3), rng.randrange(12)) for _ in range(rng.randrange(1, 3))]
            both("admit", pairs)
            continue
        if x < 0.2:
            both("forget", rng.randrange(1, nid + 1))
            continue
        old = rows.get(s)
        if old is not None and x < 0.75:  # grow the live row (the fast path), sometimes with a -1 hole or a repeat
            row = list(old) + [nid + k for k in range(rng.choice([1, 1, 2, 5]))]
            nid += len(row) - len(old)
            if rng.random() < 0.08:
                row.append(-1)
            if rng.random() < 0.05:
                row.append(rng.choice(row[: len(old)] or [1]))
            keep = len(old) * rng.choice([0, 1])
            extended += 1
        else:  # a fresh row, maybe sharing another slot's prefix (shared pages)
            donor = rows.get(rng.randrange(4)) or []
            row = list(donor[: rng.randrange(len(donor) + 1)]) + [nid + k for k in range(rng.randrange(1, 6))]
            nid += 6
            keep = 0
        covered = len(row) if rng.random() < 0.9 else max(len(row) - 1, 0)
        padded = row + [-1] * rng.randrange(0, 4)
        if both("set_row", s, padded, covered, keep):
            rows[s] = [b for b in row[:covered] if b >= 0]
    assert extended > 100 and len(refused) > 10 and fast.evictions > 0, (extended, len(refused), fast.evictions)

