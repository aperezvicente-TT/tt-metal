# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The paged pooled-key store (GLM53_DSA_PAGED_POOLS, MkMla rev 63; doc/ttnn/12-fast-start/paged_pooled_keys.md rev 3):
a host model of the decode kernels' addressing, checked against the dense per-slot store.

Dense (today): chip r's local pool q of slot s is row q % 32 of pooled page s x pps + q / 32 (4 tiles of [32, 32]).
Paged: one [32, 128] page per LOCAL latent block lb (64 tokens = 16 pools at rows 0..15, rows 16..31 unused):
  ix_dm1.cpp (the row write, MLA_DSA_IX_ROWS): pool q -> row q % 16 of page ptab_local[s][q / 16];
  att_dsa_shard_dm0.cpp (the I4 scoring read): unit tile u = rows 0..15 of page ptab_local[s][2u] then rows 0..15 of
  page ptab_local[s][2u + 1] (two 1 KiB top-half reads per tile: faces 0-1 = rows 0..15).
Checked:
- every unit tile an owner scores equals the dense store's, for every row of the slot's closed local pools;
- a write lands only in its own block's page (rows 16..31 and every other page untouched);
- blocks shared between slots (prefix caching) are read back identically by both;
- a slot move (copy_slot = a page-table row move) carries the pooled rows without a copy.
The negative control (the dense row index q % 32 on the paged page) must fail."""

import random

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt.dsa_shard import KPOOL, UNIT, ShardMap

DIM = 128
BLOCK = 64
PPB = BLOCK // KPOOL  # pools per latent block (16)


def ix_write(paged, row_tab, q, key, *, dense_row=False):
    """ix_dm1 MLA_DSA_PAGED_POOLS: pool q of the slot whose local table row is row_tab"""
    lb = row_tab[q // PPB]
    r = q % 32 if dense_row else q % PPB
    paged[lb, r] = key


def owner_unit_tile(paged, row_tab, u):
    """att_dsa_shard_dm0 MLA_DSA_PAGED_POOLS: unit tile u = top halves of pages row[2u], row[2u + 1]"""
    return torch.cat([paged[row_tab[2 * u], :PPB], paged[row_tab[2 * u + 1], :PPB]], dim=0)


def build(C, slots, n_pos, rng, share=0):
    """per chip: local tables for `slots` slots (random distinct local blocks; `share` leading blocks of slot 1 = slot
    0's, as a prefix-cache hit), the dense store and the paged store after decoding positions 0 .. n_pos - 1 of every
    slot"""
    sm = ShardMap(C)
    cap = n_pos + UNIT
    lcap = sm.local_cap(cap)
    bps_local = lcap // BLOCK
    pps = lcap // KPOOL // 32
    nb_local = 1 + slots * bps_local + 4
    chips = []
    for r in range(C):
        free = list(range(1, nb_local))
        rng.shuffle(free)
        tabs = []
        for s in range(slots):
            row = [free.pop() for _ in range(bps_local)]
            if s == 1 and share:
                row[:share] = tabs[0][:share]
            tabs.append(row)
        dense = torch.zeros(slots * pps, 32, DIM)
        paged = torch.full((nb_local, 32, DIM), float("nan"))  # unwritten rows read as NaN (a stray read fails)
        paged[:, PPB:] = -7.0  # rows 16..31: must stay untouched
        chips.append(dict(tabs=tabs, dense=dense, paged=paged))
    keys = {}
    for s in range(slots):
        for p in range(KPOOL - 1, n_pos, KPOOL):  # the pooled key of pool p / 4 is written at its last token
            g = p // KPOOL
            r = sm.owner(p)
            q = sm.local_pool(g)
            if s == 1 and share and q // PPB < share:
                key = keys[(0, r, q)]  # a shared block holds slot 0's pooled rows (the same prefix)
            else:
                key = torch.randn(DIM)
            keys[(s, r, q)] = key
            ch = chips[r]
            ch["dense"][s * pps + q // 32, q % 32] = key
            ix_write(ch["paged"], ch["tabs"][s], q, key)
    return sm, chips, pps


@pytest.mark.parametrize("C", [2, 8])
@pytest.mark.parametrize("n_pos", [2048 + 3, 4096, 6 * UNIT * 8 + 77])
def test_paged_reads_match_dense(C, n_pos):
    rng = random.Random(C * 1000 + n_pos)
    torch.manual_seed(C + n_pos)
    sm, chips, pps = build(C, 3, n_pos, rng)
    for r, ch in enumerate(chips):
        for s in range(3):
            n_closed = sm.n_closed_local(n_pos - 1, r)
            for u in range(-(-n_closed // 32)):
                got = owner_unit_tile(ch["paged"], ch["tabs"][s], u)
                ref = ch["dense"][s * pps + u]
                k = min(32, n_closed - 32 * u)  # the closed rows of this tile (the rest is masked by n_closed)
                assert torch.equal(got[:k], ref[:k]), (C, n_pos, r, s, u)
        assert torch.all(ch["paged"][:, PPB:] == -7.0), "a write left its block's 16 rows"


@pytest.mark.parametrize("C", [2, 8])
def test_shared_prefix_blocks(C):
    rng = random.Random(11 + C)
    torch.manual_seed(5)
    n_pos = 8 * UNIT * C
    sm, chips, pps = build(C, 2, n_pos, rng, share=6)
    for r, ch in enumerate(chips):
        n_closed = sm.n_closed_local(n_pos - 1, r)
        for u in range(3):  # the shared units (blocks 0..5)
            if 32 * u < n_closed:
                assert torch.equal(owner_unit_tile(ch["paged"], ch["tabs"][0], u),
                                   owner_unit_tile(ch["paged"], ch["tabs"][1], u))  # fmt: skip


def test_slot_move_is_a_table_move():
    rng = random.Random(3)
    torch.manual_seed(3)
    C, n_pos = 2, 4096
    sm, chips, pps = build(C, 2, n_pos, rng)
    for r, ch in enumerate(chips):
        moved = list(ch["tabs"][0])  # copy_slot(dst=1, src=0) + remap: the destination's row lists the source's blocks
        n_closed = sm.n_closed_local(n_pos - 1, r)
        for u in range(-(-n_closed // 32)):
            k = min(32, n_closed - 32 * u)
            assert torch.equal(owner_unit_tile(ch["paged"], moved, u)[:k], ch["dense"][u][:k])


def test_dense_row_index_breaks():
    """negative control: writing pool q at the dense row q % 32 of the block's page loses rows 16..31's pools"""
    rng = random.Random(9)
    torch.manual_seed(9)
    sm = ShardMap(2)
    tab = list(range(1, 64))
    rng.shuffle(tab)
    paged = torch.zeros(64, 32, DIM)
    keys = {}
    for q in range(64):
        keys[q] = torch.randn(DIM)
        ix_write(paged, tab, q, keys[q], dense_row=True)
    with pytest.raises(AssertionError):
        for u in range(2):
            got = owner_unit_tile(paged, tab, u)
            assert torch.equal(got, torch.stack([keys[32 * u + i] for i in range(32)]))
