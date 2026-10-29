# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host model of the TP prefill's GLM53_DSA_PAGED_POOLS path (n29h7s v1; Stage 12's paged pooled keys rev 3, the padded
layout: page lb of a chip's ppool = its local latent block lb, the block's 16 pooled rows in rows 0-15), emulated on the
real ShardMap helpers (tt/dsa_shard.py):
- WRITE (DsaIndexer.keys under the shard): paged_pool_src = one padded [32, 128] tile row per 64-token block, its tiles
  in block-major order; tiles_to by Stage 12's prefill_pool_idx_paged(blocks, n_valid) -> tile 4 k + j of the chunk to
  tile 4 lb + j of chip g % C's store (g = the chunk's physical block k), only for blocks holding a valid row;
- READ (SparseCtx.paged_pools): the slot's local blocks of every chip via the latent LOCAL table (page_table_local), the
  rank-order all-gather, the global order through lat_g2l = gather_table(T / 64, T_local / 64, 2), rows 0-15 per block.
Checks: the read returns the written pooled keys in global pool order (owner-aware physical rows, several chunks of a
prompt, a partial last chunk), pool_idx_tiles matches 12's index width, and pad blocks are never written."""
from __future__ import annotations

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DSH
from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as P

BLK = 64
ROWS = BLK // 4  # pooled rows per block


def owner_row(C: int, bps: int, gen: torch.Generator) -> torch.Tensor:
    """an owner-aware PHYSICAL page row: logical block i -> a global block g on chip (i // 2) % C (g % C = the chip),
    distinct random local ids per chip (as OwnerBlockMap hands them out, not as aligned pairs)"""
    per_chip = {r: torch.randperm(4 * bps, generator=gen).tolist() for r in range(C)}
    row = []
    for i in range(bps):
        r = (i // 2) % C
        row.append(per_chip[r].pop() * C + r)
    return torch.tensor(row, dtype=torch.int64)


def write_chunk(sm, store, pooled_chunk, blocks, n_valid):
    """paged_pool_src + tiles_to(prefill_pool_idx_paged): store[r] = [nb_local, 32, 128] (rows emulate tile rows)"""
    nblk = int(blocks.numel())
    idx = sm.prefill_pool_idx_paged(blocks, n_valid)  # [C, nblk x 4]
    assert idx.shape[1] == P.pool_idx_tiles(nblk * BLK, True), (idx.shape, nblk)
    for k in range(nblk):
        src_rows = torch.zeros(32, 128)
        src_rows[:ROWS] = pooled_chunk[ROWS * k : ROWS * (k + 1)]
        for r in range(sm.C):
            for j in range(4):  # tile j = columns 32 j .. 32 j + 31 of the block's tile row
                d = int(idx[r, 4 * k + j])
                if d == DSH.SKIP:
                    continue
                lb, jj = divmod(d, 4)
                assert jj == j
                store[r][lb, :, 32 * j : 32 * (j + 1)] = src_rows[:, 32 * j : 32 * (j + 1)]


def read_slot(sm, store, row, t):
    """SparseCtx.paged_pools: local blocks via the latent local table -> rank-order gather -> lat_g2l -> rows 0-15"""
    t_local = sm.local_cap(t)
    nb, nb_l = t // BLK, t_local // BLK
    gathered = []
    for r in range(sm.C):
        tab = sm.page_table_local(row.reshape(1, -1), r, nb_l).reshape(-1)  # chip r's local blocks of the slot
        gathered += [store[r][int(tab[p])] for p in range(nb_l)]
    g2l = sm.gather_table(nb, nb_l, 2).reshape(-1)  # ppu = 2: two 64-token pages per 128-token unit (as lat_g2l)
    pages = [gathered[int(g2l[i])] for i in range(nb)]
    return torch.cat([pg[:ROWS] for pg in pages])  # [T / 4, 128]


@pytest.mark.parametrize("C", [2, 8])
def test_paged_write_then_read_is_the_identity(C):
    gen = torch.Generator().manual_seed(C)
    sm = DSH.ShardMap(C)
    t, chunk = 8192, 2048
    bps = t // BLK
    row = owner_row(C, bps, gen)
    n_tok = 7000  # a partial last chunk
    pooled = torch.randn(t // 4, 128, generator=gen).bfloat16().float()
    nb_local = 4 * bps
    store = [torch.zeros(nb_local, 32, 128) for _ in range(C)]
    for c0 in range(0, n_tok, chunk):
        nv = min(chunk, n_tok - c0)
        blocks = row[c0 // BLK : (c0 + chunk) // BLK]
        write_chunk(sm, store, pooled[c0 // 4 : (c0 + chunk) // 4], blocks, nv)
    got = read_slot(sm, store, row, t)
    n_pools_written = -(-n_tok // BLK) * ROWS  # whole blocks holding a valid row
    assert torch.equal(got[:n_pools_written], pooled[:n_pools_written])
    # pad blocks (no valid row) were never written: their pages stay zero
    assert bool((got[n_pools_written:] == 0).all())


def test_pool_idx_width():
    for c in (512, 2048, 8192):
        assert P.pool_idx_tiles(c, True) == c // 64 * 4
        assert P.pool_idx_tiles(c, False) == c // 128 * 4
