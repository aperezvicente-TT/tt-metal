"""Host tests of the LMCache hit page-in over the page-DMA kernel (tt/page_pool.py LatentHitDma, GLM53_LMCACHE_PAGE_DMA):
the staging ring's tilized pages + LOAD descriptors land each row-major host page in its device block's 32 tiles of
every DSA layer's latent cache (a host model of the caches executes the descriptors), across a GiB boundary too."""

import numpy as np
import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import page_pool as PP

BLOCKS = 40


class FakeLatent:
    def __init__(self, addr):
        self.addr, self.shape = addr, (BLOCKS, 1, 64, 512)

    def buffer_address(self):
        return self.addr


def _caches_model(pool):
    """[layer][tile] uint16 [1,024] host model of the 11 latent caches + the descriptor executor"""
    dev = np.zeros((PP.DSA_LAYERS, BLOCKS * PP.LATENT_TILES_PER_PAGE, 1024), dtype=np.uint16)

    def apply(desc):
        for d in desc:
            assert d["op"] == PP.LOAD
            off = int(d["slab"]) * PP.GIB + int(d["slab_off"])
            n = int(d["n_tiles"])
            src = pool.array[off : off + n * PP.TILE_BYTES].view(np.uint16).reshape(n, 1024)
            dev[int(d["tensor"]), int(d["tile0"]) : int(d["tile0"]) + n] = src

    return dev, apply


def _page_of(dev, layer, block):
    t = dev[layer, block * PP.LATENT_TILES_PER_PAGE : (block + 1) * PP.LATENT_TILES_PER_PAGE].reshape(-1)
    return PP.untilize_page(t)


def test_tilize_pages_is_tilize_page_per_page():
    x = np.random.default_rng(0).integers(0, 1 << 16, size=(3, 64, 512), dtype=np.uint16)
    out = PP.tilize_pages(x)
    for i in range(3):
        assert np.array_equal(out[i], PP.tilize_page(x[i]))
        assert np.array_equal(PP.untilize_page(out[i]), x[i])


@pytest.mark.parametrize("gib, head", [(1, 0), (2, None)])
def test_hit_pages_land_in_their_blocks(gib, head):
    rng = np.random.default_rng(1)
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    holder = {}
    dma = PP.LatentHitDma(None, caches, gib=gib, impl=PP.RecordingPageDma(None), backing="anon")
    dev, apply = _caches_model(dma.pool)
    dma.dma.impl.apply = apply
    dma.start()
    blocks = [7, 3, 12, 30, 31]
    if head is None:  # place the submit so its pages span the first GiB boundary (two slabs)
        dma.head = PP.GIB // PP.LATENT_PAGE_BYTES - 2 * len(blocks) - 2
    pages = [rng.integers(0, 1 << 16, size=(len(blocks), 64, 512), dtype=np.uint16) for _ in range(PP.DSA_LAYERS)]
    n = dma.submit(pages, blocks)
    dma.wait()
    assert n == PP.DSA_LAYERS * len(blocks)  # page-sized ring slots never straddle a GiB (16,384 pages per GiB)
    slabs = {int(s) for s in dma.dma.impl.batches[-1]["slab"]}
    assert slabs == ({0} if head is not None else {0, 1})
    for l in range(PP.DSA_LAYERS):
        for p, b in enumerate(blocks):
            assert np.array_equal(_page_of(dev, l, b), pages[l][p]), (l, b)
    untouched = [b for b in range(BLOCKS) if b not in blocks]
    assert not dev[:, [t for b in untouched for t in range(32 * b, 32 * b + 32)]].any()
    assert dma.stats["pages"] == len(blocks) and dma.stats["descriptors"] == n
    # a second submit reuses the ring after the first landed
    pages2 = [rng.integers(0, 1 << 16, size=(1, 64, 512), dtype=np.uint16) for _ in range(PP.DSA_LAYERS)]
    dma.submit(pages2, [5])
    dma.wait()
    assert all(np.array_equal(_page_of(dev, l, 5), pages2[l][0]) for l in range(PP.DSA_LAYERS))
    dma.close()


def test_hit_dma_refuses_oversized_and_wrong_layers():
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    dma = PP.LatentHitDma(None, caches, gib=1, impl=PP.RecordingPageDma(None), backing="anon")
    with pytest.raises(ValueError, match="layers"):
        dma.submit([np.zeros((1, 64, 512), np.uint16)] * 3, [1])
    too_many = dma.slots // PP.DSA_LAYERS + 1
    with pytest.raises(ValueError, match="staging ring"):
        dma.submit([np.zeros((1, 64, 512), np.uint16)] * PP.DSA_LAYERS, list(range(too_many)))
    dma.close()


def test_hit_pages_go_to_their_owner_chip_only():
    """The sequence-sharded pool: owners [n] -> each page's descriptors are queued on its owner chip only (local blocks);
    a host model of every chip's caches shows each page on its owner and nothing anywhere else."""
    chips = 4
    rng = np.random.default_rng(5)
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    dma = PP.LatentHitDma(None, caches, gib=1, impl=PP.RecordingPageDma(None), backing="anon", chips=chips)
    dev = np.zeros((chips, PP.DSA_LAYERS, BLOCKS * PP.LATENT_TILES_PER_PAGE, 1024), dtype=np.uint16)

    def apply(desc, on):
        assert len(on) == 1, "a sharded page goes to exactly one chip"
        for d in desc:
            off = int(d["slab"]) * PP.GIB + int(d["slab_off"])
            n = int(d["n_tiles"])
            dev[on[0], int(d["tensor"]), int(d["tile0"]) : int(d["tile0"]) + n] = (
                dma.pool.array[off : off + n * PP.TILE_BYTES].view(np.uint16).reshape(n, 1024))

    dma.dma.impl.apply, dma.dma.impl.apply_chips = apply, True
    local = [2, 2, 5, 9, 0]  # local blocks (two pages share local block 2 on different owners)
    owners = [0, 3, 1, 3, 2]
    pages = [rng.integers(0, 1 << 16, size=(len(local), 64, 512), dtype=np.uint16) for _ in range(PP.DSA_LAYERS)]
    n = dma.submit(pages, local, owners=owners)
    dma.wait()
    assert n == PP.DSA_LAYERS * len(local)
    assert sorted(c[0] for c in dma.dma.impl.chips) == sorted(set(owners))  # one load per owner
    for l in range(PP.DSA_LAYERS):
        for p, (b, o) in enumerate(zip(local, owners)):
            t = dev[o, l, b * 32 : (b + 1) * 32].reshape(-1)
            assert np.array_equal(PP.untilize_page(t), pages[l][p]), (l, p)
            for other in range(chips):
                if other != o and not any(ob == other and bb == b for bb, ob in zip(local, owners)):
                    assert not dev[other, l, b * 32 : (b + 1) * 32].any(), (l, p, other)
    with pytest.raises(ValueError, match="owners"):
        dma.submit(pages, local, owners=owners[:2])
    dma.close()


def test_read_pages_saves_from_the_device_and_untilizes():
    """the store direction: read_pages SAVEs the pages from the device model into the ring and returns them row-major;
    a LOAD round trip (submit, then read) is the identity; untilize_pages inverts tilize_pages"""
    rng = np.random.default_rng(11)
    rows = rng.integers(0, 1 << 16, size=(5, 64, 512), dtype=np.uint16)
    assert np.array_equal(PP.untilize_pages(PP.tilize_pages(rows)), rows)
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    dma = PP.LatentHitDma(None, caches, gib=1, impl=PP.RecordingPageDma(None), backing="anon")
    dev, load = _caches_model(dma.pool)

    def apply(desc):
        for d in desc:
            off = int(d["slab"]) * PP.GIB + int(d["slab_off"])
            n, t0, l = int(d["n_tiles"]), int(d["tile0"]), int(d["tensor"])
            if int(d["op"]) == PP.LOAD:
                load([d])
            else:
                dma.pool.array[off : off + n * PP.TILE_BYTES] = dev[l, t0 : t0 + n].view(np.uint8).reshape(-1)

    dma.dma.impl.apply = apply
    dev[:] = rng.integers(0, 1 << 16, size=dev.shape, dtype=np.uint16)
    blocks = [7, 3, 39]
    got = dma.read_pages(blocks)
    assert len(got) == PP.DSA_LAYERS and got[0].shape == (3, 64, 512)
    for l in range(PP.DSA_LAYERS):
        for i, b in enumerate(blocks):
            assert np.array_equal(got[l][i], _page_of(dev, l, b)), (l, b)
    assert all(int(d["op"]) == PP.SAVE for d in dma.dma.impl.batches[-1]) and dma.dma.impl.chips[-1] == []
    pages = [rng.integers(0, 1 << 16, size=(2, 64, 512), dtype=np.uint16) for _ in range(PP.DSA_LAYERS)]
    dma.submit(pages, [10, 11])
    dma.wait()
    back = dma.read_pages([10, 11])
    assert all(np.array_equal(back[l], pages[l]) for l in range(PP.DSA_LAYERS))
    assert dma.stats["reads"] == 2 and dma.stats["read_pages"] == 5
    dma.close()


@pytest.mark.parametrize("owned", [False, True])
def test_load_many_batches_a_hit_larger_than_the_ring(owned):
    """a hit past the staging ring (a 1M-token user: 16,384 pages x 11 layers) -> load_many: batches of half the ring,
    double-buffered on the host completion slots; every page lands in its block (owners: on its chip only, here a
    4-chip model where the device model is shared and the chips list is checked); the descriptors are page_desc_array's
    (vectorized) = page_descriptors' rows"""
    rng = np.random.default_rng(13)
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    dma = PP.LatentHitDma(None, caches, gib=1, impl=PP.RecordingPageDma(None, apply_chips=owned), backing="anon",
                          chips=4 if owned else 1)  # fmt: skip
    dev, load = _caches_model(dma.pool)
    dma.dma.impl.apply = (lambda d, chips: load(d)) if owned else load
    dma.slots = PP.DSA_LAYERS * 8  # a small ring: half = 44 slots = 4 pages per batch
    blocks = list(range(3, 23))  # 20 pages = 5 batches
    owners = [i % 4 for i in range(20)] if owned else None
    pages = [rng.integers(0, 1 << 16, size=(20, 64, 512), dtype=np.uint16) for _ in range(PP.DSA_LAYERS)]
    n = dma.load_many(pages, blocks, owners=owners)
    dma.wait()
    assert n == 20 * PP.DSA_LAYERS and dma.stats["large_loads"] == 1
    for l in range(PP.DSA_LAYERS):
        for i, b in enumerate(blocks):
            assert np.array_equal(_page_of(dev, l, b), pages[l][i]), (l, b)
    if owned:
        assert {tuple(c) for c in dma.dma.impl.chips} == {(0,), (1,), (2,), (3,)}
    ref = PP.page_descriptors(PP.LOAD, dma.pool, 5 * PP.LATENT_PAGE_BYTES, 2, 7, tag=1)
    got = PP.page_desc_array(PP.LOAD, dma.pool, 4 * PP.LATENT_PAGE_BYTES, 2, [6, 7])[1]
    assert tuple(got.tolist()[:7]) == tuple(ref[0][:7])
    dma.close()


# ------------------------------------------------------------- Stage 12: pooled-key unit pages (GLM53_KV_POOLED_PAGES)
def _tilize_tile(x: np.ndarray) -> np.ndarray:
    """[32, 32] logical -> the tile's 1,024 elements in face order (0,0) (0,1) (1,0) (1,1)"""
    return np.ascontiguousarray(x.reshape(2, 16, 2, 16).swapaxes(1, 2)).reshape(1024)


def test_pooled_halves_are_the_pages_row_halves():
    """a [32, 128] pooled-key unit page = 4 tiles; its 64-token half h (POOLED_PAGE_BYTES, the LMCache page's block) =
    rows 16 h .. 16 h + 15, i.e. faces 2 h, 2 h + 1 of every tile; pooled_units inverts it"""
    rng = np.random.default_rng(11)
    page = rng.integers(0, 1 << 16, size=(32, 128), dtype=np.uint16)
    native = np.concatenate([_tilize_tile(page[:, 32 * c : 32 * c + 32]) for c in range(4)]).view(np.uint8)[None]
    halves = PP.pooled_halves(native)
    assert halves.shape == (1, 2, PP.POOLED_PAGE_BYTES)
    for h in range(2):
        want = np.concatenate([_tilize_tile(np.concatenate([page[16 * h : 16 * h + 16, 32 * c : 32 * c + 32],
                                                            np.zeros((16, 32), np.uint16)]))[:512]
                               for c in range(4)])  # fmt: skip
        assert np.array_equal(halves[0, h].view(np.uint16), want), h
    assert np.array_equal(PP.pooled_units(halves), native.reshape(1, -1))


class FakePoolStore:
    def __init__(self, addr, pages):
        self.addr, self.shape, self.padded_shape, self.dtype = addr, (pages, 1, 32, 128), (pages, 1, 32, 128), "bf16"

    def buffer_address(self):
        return self.addr


def _pooled_dma(chips, pages, cap=None, monkeypatch=None):
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    stores = [FakePoolStore(0x900000 + 0x10000 * l, pages) for l in range(PP.DSA_LAYERS)]
    dma = PP.LatentHitDma(None, caches, gib=1, impl=PP.RecordingPageDma(None), backing="anon", chips=chips,
                          pool_tensors=stores)  # fmt: skip
    dev = np.zeros((chips, PP.DSA_LAYERS, pages * PP.POOLED_UNIT_TILES, PP.TILE_BYTES), np.uint8)

    def apply(desc, on):
        for d in desc:
            l = int(d["tensor"]) - PP.DSA_LAYERS
            assert 0 <= l < PP.DSA_LAYERS, "a pooled descriptor addresses a pooled-key store"
            off = int(d["slab"]) * PP.GIB + int(d["slab_off"])
            t0, n = int(d["tile0"]), int(d["n_tiles"])
            for ch in on or ([0] if int(d["op"]) == PP.SAVE else range(chips)):
                if int(d["op"]) == PP.LOAD:
                    dev[ch, l, t0 : t0 + n] = dma.pool.array[off : off + n * PP.TILE_BYTES].reshape(n, -1)
                else:
                    dma.pool.array[off : off + n * PP.TILE_BYTES] = dev[ch, l, t0 : t0 + n].reshape(-1)

    dma.dma.impl.apply, dma.dma.impl.apply_chips = apply, True
    if cap is not None:
        monkeypatch.setattr(dma, "_pooled_cap", lambda: cap)
    return dma, dev


@pytest.mark.parametrize("cap", [None, 2])
def test_pooled_units_load_and_read_on_their_owner_chips(cap, monkeypatch):
    """LatentHitDma(pool_tensors=): load_pooled writes each unit page on its owner chip only (chip None: every chip),
    read_pooled reads it back from there; in ring-sized batches too (cap 2); an out-of-store page is refused"""
    chips, pages = 4, 10
    rng = np.random.default_rng(12)
    dma, dev = _pooled_dma(chips, pages, cap, monkeypatch)
    assert dma.pool_ids == list(range(PP.DSA_LAYERS, 2 * PP.DSA_LAYERS))
    targets = [(0, 3), (2, 3), (1, 9), (3, 0), (None, 5)]
    data = [rng.integers(0, 256, (len(targets), PP.POOLED_UNIT_BYTES), dtype=np.uint8) for _ in range(PP.DSA_LAYERS)]
    dma.load_pooled(data, targets)
    dma.wait()
    for l in range(PP.DSA_LAYERS):
        for k, (ch, pg) in enumerate(targets):
            for c in range(chips) if ch is None else [ch]:
                assert np.array_equal(dev[c, l, 4 * pg : 4 * pg + 4].reshape(-1), data[l][k]), (l, k, c)
        assert not dev[1, l, 4 * 3 : 4 * 3 + 4].any(), "chip 1 never got page 3"
    got = dma.read_pooled([(2, 3), (0, 3), (None, 5), (1, 9)])
    order = [1, 0, 4, 2]
    for l in range(PP.DSA_LAYERS):
        assert np.array_equal(got[l], data[l][order]), l
    with pytest.raises(ValueError, match="outside"):
        dma.read_pooled([(0, pages)])
    assert dma.stats["pooled_load_units"] == 5 and dma.stats["pooled_read_units"] == 4
    dma.close()


def _reference_block_pages(rows):
    """the hit rows' padded block pages as _load_prefix_pooled built them before load_pooled_blocks (per layer)"""
    return [np.concatenate([PP.block_page_of_half(PP.pooled_halves(u[l]).reshape(2 * len(u[l]), -1)[:n])
                            for u, n in rows]) for l in range(PP.DSA_LAYERS)]  # fmt: skip


@pytest.mark.parametrize("cap", [None, 3, 1])
def test_load_pooled_blocks_is_bitwise_the_reference_layout(cap, monkeypatch):
    """Stage 12 (2026-10-26T17:50:09Z WARM): load_pooled_blocks builds each hit block's padded page in the ring (block j
    = half j % 2 of unit j // 2 in rows 0-15, rows 16-31 zero) -- the same device bytes as load_pooled of
    block_page_of_half(pooled_halves(...)), for rows of odd and even block counts, a unit's unused padding half, ring
    batches that split a row at odd and even blocks (cap 3, 1), owner chips and a replicated target; a ring left
    holding other bytes (a previous latent page-in) does not leak into rows 16-31"""
    chips, pages = 4, 12
    rng = np.random.default_rng(21)
    rows = [([rng.integers(0, 256, (u, PP.POOLED_UNIT_BYTES), dtype=np.uint8) for _ in range(PP.DSA_LAYERS)], n)
            for u, n in [(3, 5), (2, 4), (1, 1), (4, 7)]]  # fmt: skip
    nb = sum(n for _u, n in rows)
    targets = [(None, pages - 1) if k == 6 else (k % chips, k // chips) for k in range(nb)]
    got, dev = _pooled_dma(chips, pages, cap, monkeypatch)
    got.pool.array[: 4 << 20] = 0xA5  # stale ring bytes
    assert got.load_pooled_blocks(rows, targets) == PP.DSA_LAYERS * nb  # one descriptor per block per layer
    got.wait()
    want, ref = _pooled_dma(chips, pages, cap, monkeypatch)
    want.load_pooled(_reference_block_pages(rows), targets)
    want.wait()
    assert dev.any() and np.array_equal(dev, ref)
    # rows 16-31 of every loaded page are zero
    for c, pg in targets:
        for ch in range(chips) if c is None else [c]:
            assert not dev[ch, :, 4 * pg : 4 * pg + 4].reshape(PP.DSA_LAYERS, 4, 2, 1024)[:, :, 1].any(), (ch, pg)
    assert got.stats["pooled_load_units"] == nb and got.stats["pooled_load_stage_s"] > 0
    with pytest.raises(ValueError, match="targets"):
        got.load_pooled_blocks(rows, targets[:-1])
    with pytest.raises(ValueError, match="unit pages"):
        got.load_pooled_blocks([(rows[0][0], 7)], targets[:7])
    assert got.load_pooled_blocks([], []) == 0
    got.close()
    want.close()


def test_pooled_block_loads_vs_block_page_of_half_at_a_32k_hit():
    """the 32K shape (249 units, 497 blocks: an odd hit) in one ring batch: bitwise the reference block pages"""
    rng = np.random.default_rng(22)
    u = [rng.integers(0, 256, (249, PP.POOLED_UNIT_BYTES), dtype=np.uint8) for _ in range(PP.DSA_LAYERS)]
    ref = _reference_block_pages([(u, 497)])
    dest = np.full((497, PP.POOLED_UNIT_TILES, 2, PP.TILE_BYTES // 4), 0x5A5A, np.int16)
    for l in range(PP.DSA_LAYERS):
        dest[:, :, 1] = 0
        PP._stage_halves(dest, u[l].view(np.int16).reshape(-1, PP.POOLED_UNIT_TILES, 2, PP.TILE_BYTES // 4), 0, 497)
        assert np.array_equal(dest.view(np.uint8).reshape(497, -1), ref[l]), l
    ro = u[0].view(np.int16).reshape(-1, PP.POOLED_UNIT_TILES, 2, PP.TILE_BYTES // 4)
    ro.flags.writeable = False  # a read-only object view takes numpy's copy: the same bytes
    PP._stage_halves(dest, ro, 1, 496)
    assert np.array_equal(dest[:495].view(np.uint8).reshape(495, -1), ref[0][1:496])


def test_pooled_ops_need_the_stores():
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    dma = PP.LatentHitDma(None, caches, gib=1, impl=PP.RecordingPageDma(None), backing="anon")
    with pytest.raises(RuntimeError, match="no pooled-key stores"):
        dma.read_pooled([(0, 1)])
    with pytest.raises(RuntimeError, match="no pooled-key stores"):
        dma.load_pooled([np.zeros((1, PP.POOLED_UNIT_BYTES), np.uint8)] * PP.DSA_LAYERS, [(0, 1)])
    with pytest.raises(RuntimeError, match="no pooled-key stores"):
        dma.load_pooled_blocks([([np.zeros((1, PP.POOLED_UNIT_BYTES), np.uint8)] * PP.DSA_LAYERS, 1)], [(0, 1)])
    with pytest.raises(ValueError, match="pooled-key stores"):
        PP.LatentHitDma(None, caches, gib=1, impl=PP.RecordingPageDma(None), backing="anon",
                        pool_tensors=[FakePoolStore(0x900000, 4)])  # fmt: skip
    dma.close()


def test_block_page_of_half_is_rows_0_to_15():
    """GLM53_DSA_PAGED_POOLS (padded): a 64-token half lands in rows 0-15 of its block's page (faces 0, 1 of every
    tile), rows 16-31 zero; half_of_block_page inverts it"""
    rng = np.random.default_rng(13)
    page = np.zeros((32, 128), np.uint16)
    page[:16] = rng.integers(0, 1 << 16, size=(16, 128), dtype=np.uint16)
    native = np.concatenate([_tilize_tile(page[:, 32 * c : 32 * c + 32]) for c in range(4)]).view(np.uint8)[None]
    half = PP.half_of_block_page(native)
    assert np.array_equal(PP.block_page_of_half(half), native)
    assert np.array_equal(half, PP.pooled_halves(native)[:, 0])  # the same bytes as a unit page's first half


def test_read_pages_tiled_defers_the_untilize_to_result():
    """Stage 12 W4 F3a: read_pages(untilize=False) returns the ring's tile-order bytes (a copy: the ring is reused by the
    next read) whose result() equals read_pages(untilize=True) bit for bit, as the store worker's [n, page bytes] rows"""
    import torch

    rng = np.random.default_rng(12)
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    dma = PP.LatentHitDma(None, caches, gib=1, impl=PP.RecordingPageDma(None), backing="anon")
    dev, _load = _caches_model(dma.pool)

    def apply(desc):
        for d in desc:
            off = int(d["slab"]) * PP.GIB + int(d["slab_off"])
            n, t0, l = int(d["n_tiles"]), int(d["tile0"]), int(d["tensor"])
            dma.pool.array[off : off + n * PP.TILE_BYTES] = dev[l, t0 : t0 + n].view(np.uint8).reshape(-1)

    dma.dma.impl.apply = apply
    dev[:] = rng.integers(0, 1 << 16, size=dev.shape, dtype=np.uint16)
    blocks = [7, 3, 39]
    want = dma.read_pages(blocks)
    tiled = dma.read_pages(blocks, untilize=False)
    assert len(tiled) == PP.DSA_LAYERS and all(isinstance(t, PP.TiledPages) for t in tiled)
    assert tiled[0].nbytes == len(blocks) * PP.LATENT_PAGE_BYTES
    dma.pool.array[:] = 0  # the ring rewritten (the next reads reuse it): the tiled copies keep their bytes
    for l in range(PP.DSA_LAYERS):
        got = tiled[l].result()
        assert got.dtype == torch.int16 and tuple(got.shape) == (len(blocks), 64 * 512)
        assert np.array_equal(got.numpy().view(np.uint16).reshape(len(blocks), 64, 512), want[l]), l
    assert dma.stats["reads_tiled"] == 1
    dma.close()


def test_untilize_pages_threaded_equals_the_numpy_permutation():
    """Stage 12 store bandwidth: untilize_pages' torch copy is the numpy transpose bit for bit (uint16 and int16 views of
    a pool-like byte buffer; dtype kept)"""
    rng = np.random.default_rng(12)
    pool = rng.integers(0, 256, size=7 * PP.LATENT_PAGE_BYTES, dtype=np.uint8)
    for dt in (np.uint16, np.int16):
        flat = pool.view(dt).reshape(7, -1)
        ref = np.ascontiguousarray(flat).reshape(7, 2, 16, 2, 2, 16, 16).transpose(0, 1, 3, 5, 2, 4, 6).reshape(7, 64, 512)
        got = PP.untilize_pages(flat)
        assert got.dtype == dt and got.shape == (7, 64, 512) and np.array_equal(got, ref)


def test_tilize_pages_threaded_and_in_place_equal_the_numpy_permutation():
    """tilize_pages' torch copy (and its out= form into a ring-like buffer) is the numpy transpose bit for bit"""
    rng = np.random.default_rng(13)
    rows = rng.integers(0, 1 << 16, size=(6, 64, 512), dtype=np.uint16)
    ref = rows.reshape(6, 2, 2, 16, 16, 2, 16).transpose(0, 1, 4, 2, 5, 3, 6).reshape(6, -1)
    assert np.array_equal(PP.tilize_pages(rows), ref)
    ring = np.zeros(8 * PP.LATENT_PAGE_BYTES, dtype=np.uint8)
    view = ring[PP.LATENT_PAGE_BYTES : 7 * PP.LATENT_PAGE_BYTES].view(np.uint16).reshape(6, -1)
    PP.tilize_pages(rows, out=view)
    assert np.array_equal(view, ref) and not ring[: PP.LATENT_PAGE_BYTES].any() and not ring[7 * PP.LATENT_PAGE_BYTES :].any()


@pytest.mark.parametrize("owned", [False, True])
def test_vectorized_descriptors_equal_the_per_page_loop(owned):
    """_desc_by_owner (vectorized) == the per-page page_descriptors loop it replaced, owner groups and order included,
    across a GiB boundary of the ring (a page split in two descriptors)"""
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    dma = PP.LatentHitDma(None, caches, gib=2, impl=PP.RecordingPageDma(None), backing="anon")
    n = 5
    base = PP.GIB // PP.LATENT_PAGE_BYTES - 2  # layer 0's pages 2..4 sit past the first GiB
    blocks = [7, 3, 11, 2, 9]
    owners = [0, 0, 1, 1, 0] if owned else None
    want = {}
    for l in range(PP.DSA_LAYERS):
        off = (base + l * n) * PP.LATENT_PAGE_BYTES
        for p, b in enumerate(blocks):
            o = (None if owners is None else owners[p])
            want.setdefault(o, []).extend(PP.page_descriptors(PP.SAVE, dma.pool, off + p * PP.LATENT_PAGE_BYTES,
                                                              dma.dma.ids[l], b, tag=p))
    got = dma._desc_by_owner(PP.SAVE, base, blocks, owners, None)
    assert list(got) == list(want)
    for o in want:
        assert np.array_equal(got[o], np.array(want[o], dtype=PP.DESC))
    dma.close()
