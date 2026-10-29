# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TPMlaShard._cb_model_check against the REAL L1 floor (program 2026-10-27T15:39:32Z (3); 9's k4d4 d0 mock died at
(8, 2): CB end 1,258,624 vs an L1 buffer at 1,256,576): the default is the served build's final floor, the live allocator
view tightens it, and every later build re-checks the lane-core ends registered earlier."""

from types import SimpleNamespace

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import tp_dsa_shard as TDS


class _MV:
    def __init__(self, alloc):
        self.total_bytes_allocated_per_bank = alloc
        self.largest_contiguous_bytes_free_per_bank = 0


@pytest.fixture
def l1(monkeypatch):
    state = {"alloc": 0}
    monkeypatch.setattr(TDS.ttnn, "get_memory_view", lambda mesh, kind: _MV(state["alloc"]), raising=False)
    monkeypatch.delenv("GLM53_L1_FLOOR", raising=False)
    monkeypatch.delenv("GLM53_CB_BASE", raising=False)
    TDS._CB_ENDS.clear()
    yield state
    TDS._CB_ENDS.clear()


def _shard(mesh, lane_bytes, monkeypatch, link_lanes=True):
    lanes = [(8, 0), (8, 2)]
    obj = SimpleNamespace(mesh=mesh, lanes=lanes, lane_core={"rs2": (8, 0), "rs1": (8, 2)}, link_lanes=link_lanes,
                          _cb_model_printed={})
    monkeypatch.setattr(TDS, "cb_region_bytes", lambda prog, cores: {c: lane_bytes[c] for c in cores})
    return obj


def _check(obj, t=32):
    TDS.TPMlaShard._cb_model_check(obj, prog=None, r=0, c=0, t=t)


V2 = {(8, 0): 958_464, (8, 2): 966_656}  # lanecore v2 (ends 1,250,432 / 1,258,624)
FOUR = {(8, 0): 917_504, (8, 2): 917_504}  # the 4-lane AR lane (end 1,209,472)


def test_constants_are_the_calibrated_ones():
    assert TDS.L1_ALLOC_TOP == 1_256_576 + 313_472
    assert TDS.L1_FLOOR_SERVED == TDS.L1_ALLOC_TOP - 328_704 == 1_241_344


def test_live_floor_reads_the_allocator(l1):
    l1["alloc"] = 313_472
    assert TDS.live_l1_floor(object()) == 1_256_576


def test_the_k4d4_clash_now_raises_host_side(l1, monkeypatch):
    l1["alloc"] = 261_120  # early in the build: the live floor is high, the served final floor binds
    with pytest.raises(ValueError, match="exceed the L1 floor 1241344"):
        _check(_shard(object(), V2, monkeypatch))


def test_four_lane_form_passes_the_whole_build(l1, monkeypatch):
    mesh = object()
    for alloc in range(261_120, 328_705, 4_224):  # the layer-by-layer growth of the served build
        l1["alloc"] = alloc
        _check(_shard(mesh, FOUR, monkeypatch))


def test_live_floor_below_the_constant_binds(l1, monkeypatch):
    l1["alloc"] = 340_000  # a construction with more persistent L1 than the served build
    with pytest.raises(ValueError, match="live 1230048"):
        _check(_shard(object(), {(8, 0): 940_000, (8, 2): 940_000}, monkeypatch))  # ends 1,231,968 > 1,230,048


def test_a_later_build_rechecks_earlier_ends(l1, monkeypatch):
    monkeypatch.setenv("GLM53_L1_FLOOR", "1300000")  # a harness bar above the served one: only the live view binds
    mesh = object()
    l1["alloc"] = 300_000  # live floor 1,270,048
    _check(_shard(mesh, {(8, 0): 970_000, (8, 2): 970_000}, monkeypatch))  # ends 1,261,968: fits now
    l1["alloc"] = 310_000  # later layers: live floor 1,260,048 -> the earlier program no longer clears it
    with pytest.raises(ValueError, match="under an earlier shard program"):
        _check(_shard(mesh, FOUR, monkeypatch), t=16)


def test_unreadable_allocator_falls_back_to_the_constant(l1, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no view")

    monkeypatch.setattr(TDS.ttnn, "get_memory_view", boom, raising=False)
    assert TDS.live_l1_floor(object()) is None
    _check(_shard(object(), FOUR, monkeypatch))
    with pytest.raises(ValueError, match="exceed the L1 floor 1241344"):
        _check(_shard(object(), V2, monkeypatch))
