# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (torch, no ttnn) checks of the TP prefill's OWNER-ONLY writes and OWNER-AWARE reads on the sequence-sharded
DSA pool (tt/dsa_shard.py ShardMap.prefill_lat_idx / prefill_pool_idx / packed_* / gather_table / page_tables_local;
tt/prefill_tp.py under GLM53_DSA_SHARD):
  writes: the chunk's latent tiles (tiles_to by the per-chip index) land on each unit's owner chip only, at its local
      block, and only the tile rows that hold valid rows; a per-chip replay of kernels/prefill/rows_to.cpp's contract
      (dst tile idx[i] <- src tile i, SKIP = 0xFFFFFFFF) over a multi-chunk prompt rebuilds exactly ShardMap.shard_rows
      of the replicated store; no chip's null block (local 0) is ever written; the pooled keys likewise (unit pages);
      the PACKED prefill's tables agree with the chunk path's;
  reads: each chip's page_gather over its LOCAL table (the slot's physical row through page_tables_local), the blocks
      all-gathered in rank order and the static un-permute (gather_table) give the replicated path's key store and
      pool buffer BITWISE (the selection and the attention then see the same bytes as the replicated prefill);
  the physical null row, the B = 32 x 131K local tables (vectorised page_table_local == the per-column reference).
Run: no-tt-devices python -m pytest --noconftest -q tests/s14/test_shard_prefill.py (PYTHONPATH = the tt-metal tree)."""

from __future__ import annotations

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DS

BLOCK, TILE, W = 64, 32, 512 // 32  # latent page rows, tile rows, latent tiles per row
WP = 128 // 32  # pooled-key tiles per unit page
SKIP = DS.SKIP


def _phys_row(sm: DS.ShardMap, bps: int, cov: int, seed: int = 0) -> torch.Tensor:
    """a physical page-table row of `cov` covered blocks obeying g % C == owner(j) (distinct local blocks per chip,
    never local 0), the tail = the owners' null blocks (tt/owner_blocks.py)"""
    g = torch.Generator().manual_seed(seed)
    row = sm.null_row(bps).clone()
    per = {r: (torch.randperm(4096, generator=g)[:bps] + 1).tolist() for r in range(sm.C)}
    for j in range(cov):
        r = sm.owner(j * BLOCK)
        row[j] = per[r].pop() * sm.C + r
    return row


def _tiles_to(store: torch.Tensor, src: torch.Tensor, idx: torch.Tensor):
    """kernels/prefill/rows_to.cpp over tile pages: store[idx[i]] <- src[i] unless SKIP"""
    for i, d in enumerate(idx.tolist()):
        if d != SKIP:
            store[d] = src[i]


def _latent_tiles(rows: torch.Tensor) -> torch.Tensor:
    """[L, 512] rows -> tile pages [L / 32 x W, 32, 32] (tile row k, column w = rows 32 k .., cols 32 w ..)"""
    L = rows.shape[0]
    return rows.reshape(L // TILE, TILE, W, TILE).permute(0, 2, 1, 3).reshape(-1, TILE, TILE)


@pytest.mark.parametrize("C", [2, 8])
@pytest.mark.parametrize("prompt,bucket", [(4096, 2048), (4000, 2048), (3000, 512), (131072, 2048)])
def test_owner_only_latent_writes_rebuild_the_sharded_store(C, prompt, bucket):
    sm = DS.ShardMap(C, BLOCK)
    bps = -(-prompt // BLOCK) + 8
    cap = bps * BLOCK
    bpl = sm.local_cap(cap) // BLOCK
    row = _phys_row(sm, bps, -(-prompt // BLOCK), seed=C)
    rows = torch.randn(prompt, 512).to(torch.bfloat16)
    nbl = int(row.max()) // C + 2  # local blocks per chip
    stores = [torch.zeros(nbl * (BLOCK // TILE) * W, TILE, TILE, dtype=torch.bfloat16) for _ in range(C)]
    writes = [0] * C
    for cs in range(0, prompt, bucket):
        nv = min(bucket, prompt - cs)
        if prompt - cs < bucket and prompt > 4096:  # (the last chunk of a long prompt: the 2048 bucket, padded)
            pass
        chunk = torch.zeros(bucket, 512, dtype=torch.bfloat16)
        chunk[:nv] = rows[cs : cs + nv]
        pad = torch.randn(bucket - nv, 512).to(torch.bfloat16)  # padding rows carry garbage: never written
        chunk[nv:] = pad
        blocks = row[cs // BLOCK : (cs + bucket) // BLOCK]
        idx = sm.prefill_lat_idx(blocks, nv)
        assert idx.shape == (C, bucket // TILE * W)
        tiles = _latent_tiles(chunk)
        for r in range(C):
            _tiles_to(stores[r], tiles, idx[r])
            writes[r] += int((idx[r] != SKIP).sum())
        # every written tile sits on its page's owner (the invariant) and never in local block 0
        for r in range(C):
            d = idx[r][idx[r] != SKIP]
            assert ((d // (W * (BLOCK // TILE))) > 0).all(), "a chip's null block was written"
        # exactly the tile rows holding valid rows are written, once, on one chip
        written = (idx != SKIP).sum(0)
        k = torch.arange(bucket // TILE).repeat_interleave(W)
        assert torch.equal(written, (k * TILE < nv).to(written.dtype))
    # the local stores, read through the local tables, equal ShardMap.shard_rows of the replicated store
    local_tables = sm.page_tables_local(row.reshape(1, -1), bpl)  # [C, 1, bpl]
    for r in range(C):
        want = sm.shard_rows(rows, r, cap)  # [local_cap, 512], zero past the prompt
        lt = local_tables[r, 0].tolist()
        n_loc = sm.n_local_rows(prompt - 1, r)
        got = torch.zeros(sm.local_cap(cap), 512, dtype=torch.bfloat16)
        for lb in range(-(-n_loc // BLOCK)):
            page = lt[lb]
            assert page > 0, (r, lb, page)  # a covered local block, never the null block
            t0 = page * (BLOCK // TILE) * W
            tiles = stores[r][t0 : t0 + (BLOCK // TILE) * W]
            got[lb * BLOCK : (lb + 1) * BLOCK] = (
                tiles.reshape(BLOCK // TILE, W, TILE, TILE).permute(0, 2, 1, 3).reshape(BLOCK, 512)
            )
        assert torch.equal(got[:n_loc], want[:n_loc]), (C, prompt, bucket, r)
        assert writes[r] == -(-n_loc // TILE) * W


@pytest.mark.parametrize("C", [2, 8])
def test_owner_only_pool_writes_and_the_packed_tables(C):
    sm = DS.ShardMap(C, BLOCK)
    max_seq = 8192
    pps_local = sm.local_cap(max_seq) // DS.UNIT
    prompt, bucket, slot = 4000, 2048, 3
    pools = torch.randn(-(-prompt // 4), 128).to(torch.bfloat16)  # pooled key of every pool
    stores = [torch.zeros(8 * pps_local * WP, TILE, TILE, dtype=torch.bfloat16) for _ in range(C)]
    for cs in range(0, prompt, bucket):
        nv = min(bucket, prompt - cs)
        idx = sm.prefill_pool_idx(bucket, cs, nv, slot, pps_local)
        n_units = -(-nv // DS.UNIT)
        assert idx.shape == (C, bucket // DS.UNIT * WP) and (idx[:, n_units * WP :] == SKIP).all()
        chunk_pools = torch.zeros(bucket // 4, 128, dtype=torch.bfloat16)
        chunk_pools[: -(-nv // 4)] = pools[cs // 4 : cs // 4 + -(-nv // 4)]
        tiles = chunk_pools.reshape(bucket // 128, TILE, WP, TILE).permute(0, 2, 1, 3).reshape(-1, TILE, TILE)
        for r in range(C):
            _tiles_to(stores[r], tiles[: n_units * WP], idx[r])
        # unit u of the chunk on chip (cs / 128 + u) % C only
        for u in range(n_units):
            gu = cs // DS.UNIT + u
            for r in range(C):
                on = (idx[r, u * WP : (u + 1) * WP] != SKIP).all()
                off = (idx[r, u * WP : (u + 1) * WP] == SKIP).all()
                assert on if r == gu % C else off, (u, r)
    tp_local = sm.local_pool_rows(max_seq)
    for r in range(C):
        want = sm.shard_pools(pools, r, tp_local)
        got = stores[r][slot * pps_local * WP : (slot + 1) * pps_local * WP]
        got = got.reshape(pps_local, WP, TILE, TILE).permute(0, 2, 1, 3).reshape(pps_local * TILE, 128)[:tp_local]
        n = sm.n_closed_local(prompt - 1, r) + (1 if sm.owner(prompt - 1) == r and (prompt % 4) else 0)
        assert torch.equal(got[:n], want[:n]), (C, r)
    # the PACKED prefill: two segments [o, o + nv) on 128-row boundaries == the chunk path's tables shifted by o
    t, segs = 2048, [(0, 300, 1), (384, 1000, 5)]
    bps = max_seq // BLOCK
    rows = {s: _phys_row(sm, bps, -(-nv // BLOCK), seed=10 + s) for _, nv, s in segs}
    lat = sm.packed_lat_idx(t, [(o, nv, rows[s]) for o, nv, s in segs])
    pool = sm.packed_pool_idx(t, segs, pps_local)
    for o, nv, s in segs:
        li = sm.prefill_lat_idx(
            rows[s][: -(-nv // TILE) * TILE // BLOCK + 1][: (-(-nv // TILE) * TILE + BLOCK - 1) // BLOCK], nv
        )
        n = -(-nv // TILE) * W
        assert torch.equal(lat[:, o // TILE * W : o // TILE * W + n], li[:, :n])
        n_u = -(-nv // DS.UNIT)
        pi = sm.prefill_pool_idx(n_u * DS.UNIT, 0, nv, s, pps_local)
        assert torch.equal(pool[:, o // DS.UNIT * WP : o // DS.UNIT * WP + pi.shape[1]], pi)
    # rows between / past the segments: nowhere
    assert (lat[:, 300 // TILE * W + W : 384 // TILE * W] == SKIP).all()
    assert (lat[:, (384 + 1000) // TILE * W + W :] == SKIP).all()


@pytest.mark.parametrize("C", [2, 8])
@pytest.mark.parametrize("t", [2048, 4096, 32768])
def test_owner_aware_reads_rebuild_the_replicated_key_store(C, t):
    """local page_gather (per-chip local tables) -> all_gather in rank order -> gather_table == the replicated gather"""
    sm = DS.ShardMap(C, BLOCK)
    bps = t // BLOCK
    cap = bps * BLOCK
    bpl = sm.local_cap(cap) // BLOCK
    prompt = t - 100
    row = _phys_row(sm, bps + 4, -(-prompt // BLOCK), seed=3 + C)
    rows = torch.randn(cap, 512).to(torch.bfloat16)  # the replicated store's content at every position of the window
    # per-chip paged caches (page = 64 rows), filled through the owner rule
    nbl = int(row.max()) // C + 2
    caches = [torch.zeros(nbl, BLOCK, 512, dtype=torch.bfloat16) for _ in range(C)]
    for j in range(-(-prompt // BLOCK)):
        g = int(row[j])
        caches[g % C][g // C] = rows[j * BLOCK : (j + 1) * BLOCK]
    # the replicated reference: page_gather over the global row
    ref = torch.cat([rows[j * BLOCK : (j + 1) * BLOCK] if j < -(-prompt // BLOCK) else torch.zeros(BLOCK, 512, dtype=rows.dtype)
                     for j in range(bps)])  # fmt: skip
    # the sharded read: each chip gathers its local pages through its local table, blocks concatenated in rank order
    lt = sm.page_tables_local(row.reshape(1, -1), bpl)
    t_local = sm.local_cap(t)
    loc = [torch.cat([caches[r][int(lt[r, 0, lb])] for lb in range(t_local // BLOCK)]) for r in range(C)]
    gathered = torch.cat(loc)  # [C x t_local, 512]
    g2l = sm.gather_table(t // BLOCK, t_local // BLOCK, DS.UNIT // BLOCK)[0].tolist()
    got = torch.cat([gathered[p * BLOCK : (p + 1) * BLOCK] for p in g2l])
    n = -(-prompt // BLOCK) * BLOCK
    assert torch.equal(got[:n], ref[:n]), (C, t)
    # the pooled keys: unit pages (ppu = 1)
    p_local = t_local // 4
    pools = torch.randn(t // 4, 128).to(torch.bfloat16)
    pstores = [sm.shard_pools(pools, r, p_local) for r in range(C)]
    pg = torch.cat(pstores)
    g2l = sm.gather_table(t // 4 // 32, p_local // 32, 1)[0].tolist()
    got = torch.cat([pg[p * 32 : (p + 1) * 32] for p in g2l])
    assert torch.equal(got, pools)


def test_null_row_and_b32_local_tables_at_131k():
    C, B = 8, 32
    sm = DS.ShardMap(C, BLOCK)
    bps = 131072 // BLOCK + 2
    bpl = sm.local_cap(bps * BLOCK) // BLOCK
    null = sm.null_row(bps)
    assert torch.equal(null % C, (torch.arange(bps) // 2) % C) and (null // C == 0).all()
    pt = torch.stack([_phys_row(sm, bps, 100 + 60 * s, seed=s) for s in range(B)])
    pt[5] = null  # an idle slot
    lt = sm.page_tables_local(pt, bpl)
    assert lt.shape == (C, B, bpl)
    # == the per-column reference
    for r in range(C):
        ref = torch.zeros(B, bpl, dtype=torch.int64)
        for j in range(bpl):
            jg = ((j // 2) * C + r) * 2 + j % 2
            if jg < bps:
                ref[:, j] = pt[:, jg] // C
        assert torch.equal(lt[r].to(torch.int64), ref)
    assert (lt[:, 5] == 0).all()  # the idle slot points every chip at its null block
    bad = pt.clone()
    bad[7, 4] = bad[7, 4] + 1  # off its owner
    with pytest.raises(ValueError):
        sm.page_tables_local(bad, bpl)


@pytest.mark.parametrize("C", [2, 8])
@pytest.mark.parametrize("bf16_stats", [False, True])
def test_bq_split_k_merge_matches_the_full_attention(C, bf16_stats):
    """lever B-q (tt/prefill_tp.py MlaPrefill._bq, sparse_sdpa_stats): each chip attends only the selected keys it
    OWNS (expand_rows SHARD mode: local row ids, a count, a dummy key for an empty row) and the owner merge of the C
    partials (m, l, o) equals the one-chip attention over the full selection. fp32 stats: bitwise-class (PCC >= 0.99999,
    max rel err < 1e-5); bf16 stats (the kernel's bf16 running tiles): PCC >= 0.99999 still (reported)"""
    torch.manual_seed(C)
    sm = DS.ShardMap(C, BLOCK)
    H, D, scale = 64, 512, 512 ** -0.5
    T = 4096
    keys = torch.randn(T, D).to(torch.bfloat16)
    rows = [100, 2050, 2051, 3000, 4095]  # dense rows (unit 0 only at p = 100: chips 1.. own nothing), sparse rows
    worst_pcc, worst_rel = 1.0, 0.0
    for p in rows:
        n_closed = (p + 1) // 4
        nsel = min(n_closed, 512)
        pools = torch.randperm(n_closed)[:nsel] if n_closed else torch.zeros(0, dtype=torch.int64)
        ids = torch.cat([(4 * pools[:, None] + torch.arange(4)).reshape(-1), torch.arange(4 * n_closed, p + 1)])
        q = torch.randn(H, D).to(torch.bfloat16)
        # the reference: one chip, every selected key
        o_ref, _, _ = DS.ShardMap.bq_stats(q, keys[ids], scale)
        # the sharded form: per chip the owned keys as local rows of its slice
        lrows = [sm.shard_rows(keys, r, T) for r in range(C)]
        row_ids = torch.full((1, 2304), DS.SKIP, dtype=torch.int64)
        row_ids[0, : ids.numel()] = ids
        parts = []
        for r in range(C):
            loc, cnt = sm.bq_local_ids(row_ids, r)
            n = int(cnt[0])
            sel = loc[0, : max(1, n)]
            o_c, m_c, l_c = DS.ShardMap.bq_stats(q, lrows[r][sel], scale, bf16_stats)
            parts.append((o_c, m_c, l_c, torch.full((H,), n > 0)))
            if n:  # the local ids address exactly the owned global keys
                g = ids[(ids // DS.UNIT) % C == r]
                assert torch.equal(lrows[r][sel], keys[g])
        assert sum(int(v[0]) for _, _, _, v in parts) >= 1
        o = DS.ShardMap.bq_merge(parts, scale)
        pcc = DS.pcc(o, o_ref)
        rel = float((o - o_ref).abs().max() / o_ref.abs().max())
        worst_pcc, worst_rel = min(worst_pcc, pcc), max(worst_rel, rel)
    assert worst_pcc >= 0.99999, (C, bf16_stats, worst_pcc, worst_rel)
    if not bf16_stats:
        assert worst_rel < 1e-5, (C, worst_rel)
    print(f"BQ_MERGE C={C} bf16_stats={bf16_stats} min_pcc={worst_pcc:.8f} max_rel={worst_rel:.2e}")
