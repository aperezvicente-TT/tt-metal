"""Stage 14 re-review of S1 (1c2d1fc20f6): the C++ PageDma push publishes descriptors run by run and throws 'ring stayed
full' PART-WAY through a group. Model that: the ONLY group (owners None) pushes half its descriptors, then raises."""
import numpy as np
import pytest

from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_hit_dma import BLOCKS
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_submit_resident import _dma, _resident_object
from models.autoports.zai_org_glm_5_3_flash.tt import page_pool as PP


class _HalfThenFull:
    def __init__(self, impl):
        self.impl, self.waits, self.queued = impl, 0, 0

    def load(self, desc, timeout_s, chips=()):
        n = len(desc) // PP.DESC.itemsize
        half = desc[: (n // 2) * PP.DESC.itemsize]
        (self.impl.load(half, timeout_s, chips) if chips else self.impl.load(half, timeout_s))
        self.queued += n // 2
        raise RuntimeError("page-DMA ring stayed full")

    def wait(self, timeout_s):
        self.waits += 1
        return self.impl.wait(timeout_s)

    def __getattr__(self, k):
        return getattr(self.impl, k)


def test_only_group_partial_push_still_waits_before_releasing():
    rng = np.random.default_rng(7)
    a, _dev = _dma(chips=4)
    rows, _lat, _p = _resident_object(a, 0, 4, rng)
    blocks = [int(v) for v in rng.permutation(BLOCKS)[:4]]
    a.dma.impl = _HalfThenFull(a.dma.impl)
    released = []
    with pytest.raises(RuntimeError, match="stayed full"):
        a.submit_resident([(rows, 4)], blocks)  # owners None: ONE group
    a.after_wait(lambda: released.append(a.dma.impl.waits))  # the caller's hold, released at the next wait()
    assert a.dma.impl.queued > 0  # descriptors ARE in flight
    a.wait()
    # the hold must be released only after the device wait drained the queued descriptors
    assert released == [1], (a.pending, a.dma.impl.waits, released)
