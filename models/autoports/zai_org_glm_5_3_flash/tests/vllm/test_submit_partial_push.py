"""Stage 12 S1 (program 2026-10-26T16:12:35Z): the three lever (c) submits (submit_objects, submit_resident,
save_resident) count their pending descriptors PER OWNER GROUP. A ring-space TimeoutError on a later group's push must
leave every group already queued counted, so the next wait() (the first thing any later submit does) still waits for
their bytes instead of returning at once with pending 0."""

import numpy as np
import pytest

from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_hit_dma import BLOCKS
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_submit_resident import T, _dma, _resident_object
from models.autoports.zai_org_glm_5_3_flash.tt import page_pool as PP


class _RingFull:
    """wraps the recording impl: the k-th push (1-based) raises the ring-space TimeoutError; waits are counted"""

    def __init__(self, impl, fail_at: int):
        self.impl, self.fail_at, self.pushes, self.waits = impl, fail_at, 0, 0

    def _push(self, fn, desc, timeout_s, chips=()):
        self.pushes += 1
        if self.pushes == self.fail_at:
            raise TimeoutError("page-DMA ring has no space (test)")
        return fn(desc, timeout_s, chips) if chips else fn(desc, timeout_s)

    def load(self, desc, timeout_s, chips=()):
        return self._push(self.impl.load, desc, timeout_s, chips)

    def save(self, desc, timeout_s, chips=()):
        return self._push(self.impl.save, desc, timeout_s, chips)

    def wait(self, timeout_s):
        self.waits += 1
        return self.impl.wait(timeout_s)

    def __getattr__(self, k):
        return getattr(self.impl, k)


def _two_group_objects(dma, rng):
    rows, _lat, _p = _resident_object(dma, 0, 4, rng)
    rows2, _l2, _p2 = _resident_object(dma, PP.DSA_LAYERS * 4 * PP.PAGE_LAYER_BYTES, 4, rng)
    blocks = [int(v) for v in rng.permutation(BLOCKS)[:8]]
    owners = [0, 0, 0, 0, 1, 1, 1, 1]  # two owner groups: chip 0, then chip 1
    return [(rows, 4), (rows2, 4)], blocks, owners


@pytest.mark.parametrize("op", ["submit_resident", "save_resident", "submit_objects"])
def test_a_ring_timeout_on_a_later_group_keeps_the_queued_group_pending(op):
    rng = np.random.default_rng(5)
    a, _dev = _dma(chips=4)
    objs, blocks, owners = _two_group_objects(a, rng)
    if op == "submit_objects":  # row views of the same objects: the 1,024 latent bytes per token row
        objs = [(rows[:, :, : PP.LATENT_PAGE_BYTES // T], m) for rows, m in objs]
    if op == "save_resident":  # the host cache model applies LOADs only; this test checks the accounting
        a.dma.impl.apply = None
    a.dma.impl = _RingFull(a.dma.impl, fail_at=2)
    with pytest.raises(TimeoutError, match="no space"):
        getattr(a, op)(objs, blocks, owners=owners)
    first = 11 * 4  # group 1: 4 pages x 11 layers, one descriptor each
    # group 1 counted although the call raised, and the failing group 2 too (a C++ push can publish part of a
    # group before it throws: 14's repro, test_s14_partial_first_group.py)
    assert a.pending == 2 * first, a.pending
    a.wait()
    assert a.dma.impl.waits == 1 and a.pending == 0  # the wait did wait for group 1


def test_a_clean_two_group_submit_counts_both_groups():
    rng = np.random.default_rng(6)
    a, _dev = _dma(chips=4)
    objs, blocks, owners = _two_group_objects(a, rng)
    n = a.submit_resident(objs, blocks, owners=owners)
    assert n == a.pending == 2 * 11 * 4
    a.wait()
    assert a.pending == 0


def test_after_a_partial_push_the_wait_resyncs_the_tails_to_the_completions():
    """14's half-published group: the host tails are unknown after the throw; the next wait() (which drains everything
    published) resyncs them to the completion slots, so a later _wait_targets compares like with like"""
    from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_s14_partial_first_group import _HalfThenFull

    rng = np.random.default_rng(8)
    a, _dev = _dma(chips=4)
    rows, _lat, _p = _resident_object(a, 0, 4, rng)
    blocks = [int(v) for v in rng.permutation(BLOCKS)[:4]]
    a.dma.impl = _HalfThenFull(a.dma.impl)
    with pytest.raises(RuntimeError, match="stayed full"):
        a.submit_resident([(rows, 4)], blocks)
    a.wait()
    assert a.tails == a.done_counts() and a.pending == 0


@pytest.mark.parametrize("op", ["load_pooled_blocks", "load_pooled", "read_pooled"])
def test_pooled_pushes_count_per_owner_group(op, monkeypatch):
    """14's 18:06:16Z SHOULD (the S1 class on 598738f874c): the pooled-key LOADs (load_pooled, load_pooled_blocks) and
    SAVEs (read_pooled) push per owner group through _push_groups, so a ring-space throw on the second group leaves the
    first group, and the throwing one, counted; the next wait() drains them"""
    from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_hit_dma import _pooled_dma

    rng = np.random.default_rng(9)
    a, _dev = _pooled_dma(4, 12, None, monkeypatch)
    targets = [(0, 1), (0, 2), (1, 1), (1, 2)]  # two owner groups: chip 0, then chip 1
    units = [rng.integers(0, 256, (2, PP.POOLED_UNIT_BYTES), dtype=np.uint8) for _ in range(PP.DSA_LAYERS)]
    a.dma.impl = _RingFull(a.dma.impl, fail_at=2)
    with pytest.raises(TimeoutError, match="no space"):
        if op == "load_pooled_blocks":
            a.load_pooled_blocks([(units, 4)], targets)
        elif op == "load_pooled":
            a.load_pooled([np.concatenate([u, u]) for u in units], targets)
        else:
            a.read_pooled(targets)
    group = PP.DSA_LAYERS * 2  # 2 unit pages x 11 layers, one descriptor each
    assert a.pending == 2 * group, a.pending
    a.wait()
    assert a.dma.impl.waits == 1 and a.pending == 0
    a.close()


def test_a_clean_pooled_load_counts_both_groups(monkeypatch):
    from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_hit_dma import _pooled_dma

    rng = np.random.default_rng(10)
    a, _dev = _pooled_dma(4, 12, None, monkeypatch)
    targets = [(0, 1), (0, 2), (1, 1), (None, 3)]
    units = [rng.integers(0, 256, (2, PP.POOLED_UNIT_BYTES), dtype=np.uint8) for _ in range(PP.DSA_LAYERS)]
    assert a.load_pooled_blocks([(units, 4)], targets) == a.pending == PP.DSA_LAYERS * 4
    a.wait()
    assert a.pending == 0
    assert a.read_pooled(targets[:3])[0].shape == (3, PP.POOLED_UNIT_BYTES) and a.pending == 0
    a.close()
