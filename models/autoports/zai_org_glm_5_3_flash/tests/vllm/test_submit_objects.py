"""Stage 12 lever (c): LatentHitDma.submit_objects -- the hit page-in straight from LMCache memory objects (each a
[layers, tokens, 1,088 B] row block: 1,024 latent + 64 pooled bytes per token), one strided tilizing copy per object
into the ring, no concatenated copy. Host model of the 11 latent caches (test_hit_dma's executor): every page lands in
its block exactly as submit() of the same rows lands it -- identity and permuted layer orders, objects of different
sizes, a ring placed across the GiB boundary, the sharded pool's per-owner batches."""

import numpy as np
import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_hit_dma import BLOCKS, FakeLatent, _caches_model, _page_of
from models.autoports.zai_org_glm_5_3_flash.tt import page_pool as PP

LB, PB, T = 1024, 64, 64


def _objects(rng, pages_per_obj):
    """memory objects [L, 64 m, 1,088] uint8 (random latent + pooled bytes) and their pages' latent rows [m, 64, 512]"""
    objs, lat = [], []
    for m in pages_per_obj:
        o = torch.from_numpy(rng.integers(0, 256, size=(PP.DSA_LAYERS, m * T, LB + PB), dtype=np.uint8))
        objs.append((o[..., :LB], m))  # the connector's slice: latent bytes, row stride 1,088
        lat.append(o[..., :LB].contiguous().view(torch.int16).numpy().view(np.uint16).reshape(PP.DSA_LAYERS, m, T, LB // 2))
    return objs, np.concatenate(lat, axis=1)  # [L, pages, 64, 512] uint16


def _dma(gib=1, chips=None):
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    dma = PP.LatentHitDma(None, caches, gib=gib, impl=PP.RecordingPageDma(None), backing="anon", **({"chips": chips} if chips else {}))
    dev, apply = _caches_model(dma.pool)
    dma.dma.impl.apply = apply
    dma.start()
    return dma, dev


@pytest.mark.parametrize("order", ["identity", "reversed"])
@pytest.mark.parametrize("gib, head", [(1, 0), (2, "boundary")])
def test_objects_land_like_submit(order, gib, head):
    rng = np.random.default_rng(5)
    objs, lat = _objects(rng, [4, 4, 2, 4, 1])  # 15 pages: full 256-token objects, a 128 and a 64
    blocks = [int(b) for b in rng.permutation(BLOCKS)[:15]]
    layer_order = None if order == "identity" else list(range(PP.DSA_LAYERS))[::-1]
    a, dev_a = _dma(gib)
    b, dev_b = _dma(gib)
    if head == "boundary":  # pages straddle the first GiB boundary (two slabs)
        a.head = b.head = PP.GIB // PP.LATENT_PAGE_BYTES - 3 * len(blocks) - 1
    want = [lat[i] for i in (layer_order or range(PP.DSA_LAYERS))]  # cache j <- object layer order[j]
    nb = b.submit(want, blocks)
    na = a.submit_objects(objs, blocks, layer_order=layer_order)
    a.wait()
    b.wait()
    assert na == nb and np.array_equal(dev_a, dev_b)
    for j in range(PP.DSA_LAYERS):
        src = (layer_order or list(range(PP.DSA_LAYERS)))[j]
        for k, blk in enumerate(blocks):
            assert np.array_equal(_page_of(dev_a, j, blk), lat[src, k])
    assert a.stats["object_submits"] == 1 and a.stats["pages"] == 15


def test_objects_per_owner_batches_like_submit():
    """the sharded pool: each page's descriptors only to its owner chip, batched per owner, as submit(owners=)"""
    rng = np.random.default_rng(6)
    objs, lat = _objects(rng, [4, 4])
    blocks = [int(b) for b in rng.permutation(BLOCKS)[:8]]
    owners = [k % 4 for k in range(8)]
    a, _ = _dma(chips=4)
    b, _ = _dma(chips=4)
    a.submit_objects(objs, blocks, owners=owners)
    b.submit([lat[i] for i in range(PP.DSA_LAYERS)], blocks, owners=owners)
    ra, rb = a.dma.impl, b.dma.impl
    assert ra.chips == rb.chips and all(np.array_equal(x, y) for x, y in zip(ra.batches, rb.batches))
    assert np.array_equal(a.pool.array[: 8 * PP.DSA_LAYERS * PP.LATENT_PAGE_BYTES],
                          b.pool.array[: 8 * PP.DSA_LAYERS * PP.LATENT_PAGE_BYTES])  # the same ring bytes


def test_objects_refuse_bad_shapes():
    rng = np.random.default_rng(7)
    objs, _ = _objects(rng, [2])
    a, _ = _dma()
    with pytest.raises(ValueError, match="object pages for"):
        a.submit_objects(objs, [1, 2, 3])
    with pytest.raises(ValueError, match="hold no"):
        a.submit_objects([(objs[0][0][:, :64], 2)], [1, 2])
    with pytest.raises(ValueError, match="permutation"):
        a.submit_objects(objs, [1, 2], layer_order=[0] * PP.DSA_LAYERS)
