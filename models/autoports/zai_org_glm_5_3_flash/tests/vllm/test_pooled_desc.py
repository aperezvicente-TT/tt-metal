"""Stage 12 zero-copy store follow-up: LatentHitDma._pooled_desc vectorized (the per-unit loop was 42 ms of a 32K
store's 105) -- the same descriptors, in the same layer-major / target order, grouped by chip, as the per-unit
reference (_pooled_desc_split); out-of-range pages refused; a GiB-crossing base takes the reference split."""

import numpy as np
import pytest

from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_hit_dma import FakeLatent, FakePoolStore
from models.autoports.zai_org_glm_5_3_flash.tt import page_pool as PP


def _dma(pages=600, chips=4):
    caches = [FakeLatent(0x1000 * (l + 1)) for l in range(PP.DSA_LAYERS)]
    stores = [FakePoolStore(0x900000 + 0x10000 * l, pages) for l in range(PP.DSA_LAYERS)]
    return PP.LatentHitDma(None, caches, gib=2, impl=PP.RecordingPageDma(None), backing="anon", chips=chips,
                           pool_tensors=stores)  # fmt: skip


@pytest.mark.parametrize("op", [PP.SAVE, PP.LOAD])
@pytest.mark.parametrize("base", [0, 8192 * 37])
def test_vectorized_equals_the_reference(op, base):
    dma = _dma()
    rng = np.random.default_rng(3)
    for chips in ([None], [0, 1, 2, 3], [None, 2]):
        targets = [(chips[int(rng.integers(len(chips)))], int(p)) for p in rng.integers(0, 600, size=257)]
        got, want = dma._pooled_desc(op, targets, base), dma._pooled_desc_split(op, targets, base)
        assert list(got) == list(want)
        for c in want:
            assert np.array_equal(got[c], want[c]), c


def test_a_base_crossing_a_gib_takes_the_split_reference():
    dma = _dma()
    targets = [(None, 5), (None, 7)]
    base = PP.GIB - 4096  # not unit-aligned: the reference splits at the GiB boundary
    got = dma._pooled_desc(PP.SAVE, targets, base)
    assert np.array_equal(got[None], dma._pooled_desc_split(PP.SAVE, targets, base)[None])
    assert len(got[None]) > PP.DSA_LAYERS * 2


def test_out_of_range_pages_are_refused():
    dma = _dma(pages=10)
    with pytest.raises(ValueError, match="outside the layer-0 store"):
        dma._pooled_desc(PP.SAVE, [(None, 3), (None, 10)], 0)
    assert dma._pooled_desc(PP.SAVE, [], 0) == {}
