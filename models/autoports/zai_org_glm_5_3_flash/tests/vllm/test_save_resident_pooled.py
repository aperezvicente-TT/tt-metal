"""Stage 12 store path P2 (program 2026-10-26T23:48:49Z; doc/ttnn/12-fast-start/store_path/README.md):
LatentHitDma.save_resident_pooled -- the zero-copy store's latent SAVEs into the objects and the pooled-key unit pages'
SAVEs into the ring, pushed together and waited for ONCE. Bitwise the two-wait path it replaces (save_resident + wait,
then read_pooled), with identity and permuted layer orders, and in ring batches (cap 2: the later batches follow in
read_pooled's loop). Host model of the 11 latent caches + the 11 pooled-key stores executing SAVE descriptors."""

import numpy as np
import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_hit_dma import BLOCKS, FakeLatent, FakePoolStore
from models.autoports.zai_org_glm_5_3_flash.tt import page_pool as PP

T = PP.TOKENS_PER_PAGE
ROW = PP.PAGE_LAYER_BYTES // T  # 1,088
UNITS = 12  # pooled-key unit pages per store


def _dma(seed, cap, monkeypatch):
    """a LatentHitDma whose fake device holds random latent pages + pooled units (the same for the same seed) and
    counts its device waits"""
    rng = np.random.default_rng(seed)
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    stores = [FakePoolStore(0x900000 + 0x10000 * l, UNITS) for l in range(PP.DSA_LAYERS)]
    dma = PP.LatentHitDma(None, caches, gib=1, impl=PP.RecordingPageDma(None), backing="anon", lmcache_gib=1,
                          pool_tensors=stores)  # fmt: skip
    lat = rng.integers(0, 1 << 16, size=(PP.DSA_LAYERS, BLOCKS * PP.LATENT_TILES_PER_PAGE, 1024), dtype=np.uint16)
    pooled = rng.integers(0, 256, size=(PP.DSA_LAYERS, UNITS * PP.POOLED_UNIT_TILES, PP.TILE_BYTES), dtype=np.uint8)

    def apply(desc):
        for d in desc:
            off = int(d["slab"]) * PP.GIB + int(d["slab_off"])
            n, t0, t = int(d["n_tiles"]), int(d["tile0"]), int(d["tensor"])
            host = dma.pool.array[off : off + n * PP.TILE_BYTES]
            if int(d["op"]) == PP.LOAD:  # (the threaded test's page-in)
                assert t < PP.DSA_LAYERS
                lat[t, t0 : t0 + n] = host.view(np.uint16).reshape(n, 1024)
                continue
            assert int(d["op"]) == PP.SAVE
            if t < PP.DSA_LAYERS:
                host[:] = lat[t, t0 : t0 + n].reshape(-1).view(np.uint8)
            else:
                host[:] = pooled[t - dma.pool_ids[0], t0 : t0 + n].reshape(-1)

    dma.dma.impl.apply = apply
    waits = []
    inner_wait = dma.dma.wait
    dma.dma.wait = lambda timeout_s, f=inner_wait: (waits.append(1), f(timeout_s))[1]
    if cap is not None:
        monkeypatch.setattr(dma, "_pooled_cap", lambda: cap)
    dma.start()
    return dma, pooled, waits


def _objects(dma, sizes):
    region, objs, at = dma.lmcache_tensor(), [], 4096
    for m in sizes:
        objs.append((region[at : at + PP.DSA_LAYERS * m * PP.PAGE_LAYER_BYTES].view(PP.DSA_LAYERS, m * T, ROW), m))
        at += PP.DSA_LAYERS * m * PP.PAGE_LAYER_BYTES + 3 * PP.TILE_BYTES
    return objs


@pytest.mark.parametrize("cap", [None, 2])
@pytest.mark.parametrize("order", ["identity", "reversed"])
def test_one_wait_and_bitwise_the_two_wait_path(cap, order, monkeypatch):
    layer_order = None if order == "identity" else list(range(PP.DSA_LAYERS))[::-1]
    sizes, blocks = [3, 2], [5, 9, 1, 30, 22]
    targets = [(None, 3), (None, 0), (None, 11), (None, 7), (None, 4)]
    a, pooled, waits_a = _dma(31, cap, monkeypatch)
    b, _pooled_b, waits_b = _dma(31, cap, monkeypatch)
    objs_a, objs_b = _objects(a, sizes), _objects(b, sizes)
    got = a.save_resident_pooled(objs_a, blocks, targets, layer_order=layer_order)
    b.save_resident(objs_b, blocks, layer_order=layer_order)
    b.wait()
    ref = b.read_pooled(targets)
    batches = 1 if cap is None else -(-len(targets) // cap)
    assert len(waits_a) == batches and len(waits_b) == 1 + batches  # one round trip fewer per store
    for l in range(PP.DSA_LAYERS):
        assert np.array_equal(got[l], ref[l]), l
        for k, (_c, page) in enumerate(targets):  # the device's own unit pages
            assert np.array_equal(got[l][k], pooled[l, 4 * page : 4 * page + 4].reshape(-1)), (l, k)
    for (ra, _m), (rb, _m2) in zip(objs_a, objs_b):
        assert torch.equal(ra, rb)  # the objects' latent tiles: the same bytes as save_resident's
    assert a.stats["resident_pooled_saves"] == 1 and a.stats["pooled_read_units"] == len(targets)
    assert not a.pending


def test_refuses_without_pooled_stores():
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    dma = PP.LatentHitDma(None, caches, gib=1, impl=PP.RecordingPageDma(None), backing="anon", lmcache_gib=1)
    with pytest.raises(RuntimeError, match="no pooled-key stores"):
        dma.save_resident_pooled([], [], [(None, 0)])


def test_one_transaction_at_a_time_across_threads(monkeypatch):
    """store path P3: a store worker thread's save and the engine thread's DMA calls never interleave -- a second
    thread's call waits while a save is inside its push (the DMA's transaction lock)"""
    import threading

    a, _pooled, _w = _dma(41, None, monkeypatch)
    objs = _objects(a, [1])
    inside, release, order = threading.Event(), threading.Event(), []
    apply = a.dma.impl.apply

    def slow_apply(desc):
        if not inside.is_set():
            inside.set()
            release.wait(5)
        apply(desc)

    a.dma.impl.apply = slow_apply
    t1 = threading.Thread(target=lambda: (a.save_resident_pooled(objs, [3], [(None, 1)]), order.append("save")))
    t1.start()
    assert inside.wait(5)
    t2 = threading.Thread(target=lambda: (a.wait(), order.append("wait")))
    t2.start()
    t2.join(0.3)
    assert t2.is_alive() and order == []  # blocked behind the save's transaction
    release.set()
    t1.join(5)
    t2.join(5)
    assert order == ["save", "wait"]


def test_after_wait_callbacks_run_on_their_own_thread(monkeypatch):
    """a store worker's wait() never runs the engine thread's after_wait releases; the engine thread's next wait()
    does (the DMA has landed by then)"""
    import threading

    a, _pooled, _w = _dma(42, None, monkeypatch)
    ran = []
    a.after_wait(lambda: ran.append("engine"))
    t = threading.Thread(target=lambda: (a.after_wait(lambda: ran.append("worker")), a.wait()))
    t.start()
    t.join(5)
    assert ran == ["worker"]
    a.wait()
    assert ran == ["worker", "engine"]


def test_lock_order_no_deadlock_with_a_callback_taking_a_worker_held_lock(monkeypatch):
    """14's 02:13:21Z SHOULD (the P3 lock order): the store worker holds an LMCache lock L while it enters the DMA
    (save_resident_pooled inside LMCache's store); the engine thread runs submit_resident + wait() with an after_wait
    callback that takes L (LMCache's ref_count_down). The callback runs only after the engine thread left the DMA lock,
    so the worker gets the DMA and releases L: both finish. (Callbacks run under the DMA lock would deadlock here.)"""
    import threading

    a, _pooled, _w = _dma(43, None, monkeypatch)
    objs = _objects(a, [1, 1])
    L = threading.Lock()  # the LMCache lock
    w_has_l, e_in_cb, done = threading.Event(), threading.Event(), []

    def worker():
        with L:
            w_has_l.set()
            assert e_in_cb.wait(5)  # the engine thread is inside its callback, waiting for L
            a.save_resident_pooled([objs[1]], [7], [(None, 2)])
        done.append("worker")

    def engine():
        assert w_has_l.wait(5)

        def release():
            e_in_cb.set()
            with L:  # (LMCache's allocator lock)
                done.append("callback")

        a.submit_resident([objs[0]], [4])
        a.after_wait(release)
        a.wait()
        done.append("engine")

    tw, te = threading.Thread(target=worker), threading.Thread(target=engine)
    tw.start()
    te.start()
    tw.join(10)
    te.join(10)
    assert not tw.is_alive() and not te.is_alive(), "deadlock"
    assert sorted(done) == ["callback", "engine", "worker"] and done.index("callback") < done.index("engine")


def test_a_nested_wait_runs_callbacks_only_after_the_outer_call(monkeypatch):
    """a callback made ready by a wait nested inside another DMA call (submit_resident's 'if pending: wait') runs when
    the outer call returns, outside the lock -- never under it"""
    a, _pooled, _w = _dma(44, None, monkeypatch)
    objs = _objects(a, [1, 1])
    seen = []
    a.submit_resident([objs[0]], [4])
    a.after_wait(lambda: seen.append(a._lock._is_owned()))
    a.submit_resident([objs[1]], [5])  # waits for the first (pending) inside its own transaction
    assert seen == [False]
    a.wait()
