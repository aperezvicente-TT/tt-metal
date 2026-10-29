"""Stage 12 lever (c) step 2: LatentHitDma.submit_resident -- the zero-copy page-in of tile-order LMCache objects that
live in the pool's LMCache region (lmcache_gib): descriptors at the objects' own bytes, no host copy, no ring. Host
model of the 11 latent caches (test_hit_dma's executor): every page lands exactly as submit() of the same row-major
pages lands it -- identity / permuted layer orders, objects of different sizes, a page across a GiB boundary of the
region, the sharded pool's per-owner batches; objects outside the region / not tile-aligned are refused.
save_resident (the zero-copy store, program 2026-10-26T07:25:07Z): the store direction -- the caches' pages SAVEd
straight into the objects' tile pages; a save then a submit of the same objects into other blocks is the identity."""

import numpy as np
import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_hit_dma import BLOCKS, FakeLatent, _caches_model, _page_of
from models.autoports.zai_org_glm_5_3_flash.tt import page_pool as PP

T = PP.TOKENS_PER_PAGE
ROW = PP.PAGE_LAYER_BYTES // T  # 1,088


def _dma(lmcache_gib=1, chips=None):
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    dma = PP.LatentHitDma(None, caches, gib=1, impl=PP.RecordingPageDma(None), backing="anon", lmcache_gib=lmcache_gib,
                          **({"chips": chips} if chips else {}))  # fmt: skip
    dev, apply = _caches_model(dma.pool)
    dma.dma.impl.apply = apply
    dma.start()
    return dma, dev


def _resident_object(dma, at: int, m: int, rng):
    """a tile-order object of m pages written at region byte `at` -> (rows [L, 64 m, 1,088] view, row-major pages
    [L, m, 64, 512] uint16, pooled [L, m, 4,096] uint8)"""
    region = dma.lmcache_tensor()
    nbytes = PP.DSA_LAYERS * m * PP.PAGE_LAYER_BYTES
    rows = region[at : at + nbytes].view(PP.DSA_LAYERS, m * T, ROW)
    lat = rng.integers(0, 1 << 16, size=(PP.DSA_LAYERS, m, T, PP.LATENT_DIM), dtype=np.uint16)
    pooled = rng.integers(0, 256, size=(PP.DSA_LAYERS, m, PP.POOLED_PAGE_BYTES), dtype=np.uint8)
    tiles, pool_v = PP.tile_object_pages(rows)
    for l in range(PP.DSA_LAYERS):
        for p in range(m):
            tiles[l, p].reshape(-1).copy_(torch.from_numpy(PP.tilize_page(lat[l, p]).view(np.uint8).copy()))
            pool_v[l, p].copy_(torch.from_numpy(pooled[l, p]))
    return rows, lat, pooled


@pytest.mark.parametrize("order", ["identity", "reversed"])
def test_resident_objects_land_like_submit(order):
    rng = np.random.default_rng(11)
    a, dev_a = _dma()
    b, dev_b = _dma()
    sizes = [4, 4, 2, 1]
    objs, lat = [], []
    at = 0
    for m in sizes:
        rows, l_, _p = _resident_object(a, at, m, rng)
        objs.append((rows, m))
        lat.append(l_)
        at += PP.DSA_LAYERS * m * PP.PAGE_LAYER_BYTES + 3 * PP.TILE_BYTES  # gaps: objects anywhere tile-aligned
    lat = np.concatenate(lat, axis=1)  # [L, 11 pages, 64, 512]
    blocks = [int(v) for v in rng.permutation(BLOCKS)[: sum(sizes)]]
    layer_order = None if order == "identity" else list(range(PP.DSA_LAYERS))[::-1]
    want = [lat[i] for i in (layer_order or range(PP.DSA_LAYERS))]
    nb = b.submit(want, blocks)
    na = a.submit_resident(objs, blocks, layer_order=layer_order)
    a.wait()
    b.wait()
    assert na == nb and np.array_equal(dev_a, dev_b)
    for j in range(PP.DSA_LAYERS):
        src = (layer_order or list(range(PP.DSA_LAYERS)))[j]
        for k, blk in enumerate(blocks):
            assert np.array_equal(_page_of(dev_a, j, blk), lat[src, k])
    assert a.stats["resident_submits"] == 1 and a.head == 0  # the ring was not touched


def test_a_page_across_a_gib_boundary_of_the_region_is_split():
    rng = np.random.default_rng(12)
    a, dev_a = _dma(lmcache_gib=2)  # region [1, 3) GiB: a GiB boundary at 2 GiB = region byte 1 GiB
    at = PP.GIB - PP.PAGE_LAYER_BYTES - 16 * PP.TILE_BYTES  # layer 0's page 1 straddles the boundary
    rows, lat, _p = _resident_object(a, at, 3, rng)
    blocks = [5, 9, 2]
    n = a.submit_resident([(rows, 3)], blocks)
    a.wait()
    assert n > PP.DSA_LAYERS * 3  # at least one page took two descriptors
    for l in range(PP.DSA_LAYERS):
        for k, blk in enumerate(blocks):
            assert np.array_equal(_page_of(dev_a, l, blk), lat[l, k])


def test_resident_per_owner_batches_like_submit():
    rng = np.random.default_rng(13)
    a, _ = _dma(chips=4)
    b, _ = _dma(chips=4)
    rows, lat, _p = _resident_object(a, 0, 4, rng)
    rows2, lat2, _p2 = _resident_object(a, PP.DSA_LAYERS * 4 * PP.PAGE_LAYER_BYTES, 4, rng)
    blocks = [int(v) for v in rng.permutation(BLOCKS)[:8]]
    owners = [k % 4 for k in range(8)]
    a.submit_resident([(rows, 4), (rows2, 4)], blocks, owners=owners)
    b.submit([np.concatenate([lat[l], lat2[l]]) for l in range(PP.DSA_LAYERS)], blocks, owners=owners)
    ra, rb = a.dma.impl, b.dma.impl
    assert ra.chips == rb.chips  # the same owner chip per batch, in the same order
    key = lambda d: sorted(zip(d["tensor"].tolist(), d["tile0"].tolist(), d["n_tiles"].tolist()))  # noqa: E731
    assert [key(x) for x in ra.batches] == [key(x) for x in rb.batches]  # same device targets per owner


def test_resident_refuses_outside_the_region_and_misaligned():
    rng = np.random.default_rng(14)
    a, _ = _dma()
    outside = torch.zeros(PP.DSA_LAYERS, T, ROW, dtype=torch.uint8)
    with pytest.raises(ValueError, match="outside"):
        a.submit_resident([(outside, 1)], [3])
    rows, _l, _p = _resident_object(a, 0, 1, rng)
    with pytest.raises(ValueError, match="object pages for"):
        a.submit_resident([(rows, 1)], [3, 4])
    mis = a.lmcache_tensor()[64 : 64 + PP.DSA_LAYERS * PP.PAGE_LAYER_BYTES].view(PP.DSA_LAYERS, T, ROW)
    with pytest.raises(ValueError, match="tile-aligned"):
        a.submit_resident([(mis, 1)], [3])
    assert PP.LatentHitDma(None, [FakeLatent(1)] * PP.DSA_LAYERS, gib=1, impl=PP.RecordingPageDma(None),
                           backing="anon").lmcache_tensor() is None  # no region by default  # fmt: skip


def test_tile_object_pages_is_the_page_format():
    rows = torch.arange(2 * 2 * T * ROW, dtype=torch.int64).remainder(251).to(torch.uint8).reshape(2, 2 * T, ROW)
    tiles, pooled = PP.tile_object_pages(rows)
    flat = rows.reshape(2, 2, PP.PAGE_LAYER_BYTES)
    assert tiles.shape == (2, 2, 32, 2048) and pooled.shape == (2, 2, 4096)
    assert torch.equal(tiles.reshape(2, 2, -1), flat[..., : PP.LATENT_PAGE_BYTES])
    assert torch.equal(pooled, flat[..., PP.LATENT_PAGE_BYTES :])


def test_shared_pool_is_one_pool_whoever_asks_first(monkeypatch):
    monkeypatch.setattr(PP, "_SHARED", {})
    # anon backing (no device here): a hugetlbfs pool file of a test process that never closes it would strand 2 x
    # 1 GiB hugepages (2026-10-26 15:38Z: a TP image restore then found 7 of the 8 hugepages it needs)
    monkeypatch.setenv("GLM53_MOCK", "1")
    monkeypatch.setenv("GLM53_LMCACHE_PAGE_DMA_GIB", "1")
    buf = PP.lmcache_external_buffer(PP.GIB)  # LMCache's allocator first (its engine init)
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    dma = PP.LatentHitDma(None, caches, gib=1, impl=PP.RecordingPageDma(None), backing="anon", lmcache_gib=1, shared=True)
    assert dma.pool is PP._SHARED[next(iter(PP._SHARED))]
    assert buf.data_ptr() == dma.lmcache_tensor().data_ptr() and dma.holds(buf[:4096])
    with pytest.raises(ValueError, match="shared pool"):
        PP.shared_pool(1, 2)
    assert dma.pool.backing == "anon" and dma.pool.path is None


def _two_way(pool):
    """the latent caches' host model executing LOAD (pool -> cache) and SAVE (cache -> pool) descriptors"""
    dev = np.zeros((PP.DSA_LAYERS, BLOCKS * PP.LATENT_TILES_PER_PAGE, 1024), dtype=np.uint16)

    def apply(desc):
        for d in desc:
            off = int(d["slab"]) * PP.GIB + int(d["slab_off"])
            n, t0, layer = int(d["n_tiles"]), int(d["tile0"]), int(d["tensor"])
            host = pool.array[off : off + n * PP.TILE_BYTES].view(np.uint16).reshape(n, 1024)
            if int(d["op"]) == PP.LOAD:
                dev[layer, t0 : t0 + n] = host
            else:
                assert int(d["op"]) == PP.SAVE
                host[:] = dev[layer, t0 : t0 + n]

    return dev, apply


@pytest.mark.parametrize("order", ["identity", "reversed"])
def test_save_resident_writes_the_caches_pages_into_the_objects(order):
    rng = np.random.default_rng(21)
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    dma = PP.LatentHitDma(None, caches, gib=1, impl=PP.RecordingPageDma(None), backing="anon", lmcache_gib=1)
    dev, apply = _two_way(dma.pool)
    dma.dma.impl.apply = apply
    dma.start()
    dev[:] = rng.integers(0, 1 << 16, size=dev.shape, dtype=np.uint16)  # the caches' content
    region = dma.lmcache_tensor()
    sizes, objs, at = [4, 1, 3], [], 4096
    for m in sizes:
        rows = region[at : at + PP.DSA_LAYERS * m * PP.PAGE_LAYER_BYTES].view(PP.DSA_LAYERS, m * T, ROW)
        objs.append((rows, m))
        at += PP.DSA_LAYERS * m * PP.PAGE_LAYER_BYTES + 5 * PP.TILE_BYTES
    blocks = [int(v) for v in rng.permutation(BLOCKS)[: sum(sizes)]]
    layer_order = None if order == "identity" else list(range(PP.DSA_LAYERS))[::-1]
    pooled_before = [PP.tile_object_pages(r)[1].clone() for r, _m in objs]
    n = dma.save_resident(objs, blocks, layer_order=layer_order)
    dma.wait()
    assert n == PP.DSA_LAYERS * sum(sizes) and dma.stats["resident_saves"] == 1 and dma.head == 0  # no ring
    assert all(int(d["op"]) == PP.SAVE for d in dma.dma.impl.batches[-1]) and dma.dma.impl.chips[-1] == []
    k = 0
    for (rows, m), pb in zip(objs, pooled_before):
        tiles, pooled = PP.tile_object_pages(rows)
        for li in range(PP.DSA_LAYERS):  # object layer li holds cache layer src
            src = li if layer_order is None else layer_order.index(li)
            for p in range(m):
                blk = blocks[k + p]
                want = dev[src, blk * PP.LATENT_TILES_PER_PAGE : (blk + 1) * PP.LATENT_TILES_PER_PAGE].reshape(-1)
                assert np.array_equal(tiles[li, p].reshape(-1).numpy().view(np.uint16), want)
        assert torch.equal(pooled, pb)  # the pooled halves are the caller's: untouched
        k += m
    # the identity: those objects paged back in (submit_resident) land the same pages in other blocks
    other = [int(v) for v in rng.permutation([b for b in range(BLOCKS) if b not in blocks])[: sum(sizes)]]
    dma.submit_resident(objs, other, layer_order=layer_order)
    dma.wait()
    for i, (b, o) in enumerate(zip(blocks, other)):
        assert np.array_equal(_page_of(dev, 0, o), _page_of(dev, 0, b)), i


def test_save_resident_per_owner_batches_like_read_pages():
    rng = np.random.default_rng(22)
    a, _ = _dma(chips=4)
    rows, _l, _p = _resident_object(a, 0, 4, rng)
    blocks = [3, 9, 1, 6]
    owners = [k % 4 for k in range(4)]
    a.dma.impl.apply = None  # (the LOAD-only cache model: batching is what this checks)
    a.save_resident([(rows, 4)], blocks, owners=owners)
    ri = a.dma.impl
    assert sorted(ri.chips[-4:]) == [[0], [1], [2], [3]]
    for d, ch in zip(ri.batches[-4:], ri.chips[-4:]):
        assert all(int(x["op"]) == PP.SAVE for x in d)
        assert sorted(set(int(t) for t in d["tile0"] // PP.LATENT_TILES_PER_PAGE)) == [blocks[ch[0]]]


def test_save_resident_refuses_like_submit_resident():
    rng = np.random.default_rng(23)
    a, _ = _dma()
    outside = torch.zeros(PP.DSA_LAYERS, T, ROW, dtype=torch.uint8)
    with pytest.raises(ValueError, match="outside"):
        a.save_resident([(outside, 1)], [3])
    rows, _l, _p = _resident_object(a, 0, 1, rng)
    with pytest.raises(ValueError, match="object pages for"):
        a.save_resident([(rows, 1)], [3, 4])
