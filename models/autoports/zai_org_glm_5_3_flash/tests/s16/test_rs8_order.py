# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host test (no device) of the RS8 block / destination orders (tt/fabric_stream.py RS8_ORDERS, GLM53_PMK_RS8_ORDER):
every readiness index a producer raises is the consumer's list position, raised exactly once, on every mesh column;
the default order reproduces the ascending indices the kernels were built with."""
import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import fabric_stream as FS

C, B2, NL1, NR1 = 4, 8, 2, 8  # the 2x4: row of 4, 8 blocks per owner, 2 stage-1 lanes, 8 stage-1 reducers


def _rs8(order: str):
    """the order logic of FabricStreamRS8 / RSR / RSL without tensors (their list methods are pure)"""
    rs8 = object.__new__(FS.FabricStreamRS8)
    rs8.C, rs8.b2 = C, B2
    tab = FS.RS8_ORDERS[order]
    rs8.dst_order = (lambda c_: list(tab[c_])) if tab is not None else (lambda c_: list(range(C)))
    rs1 = object.__new__(FS.FabricStreamRSR)
    rs1.bl, rs1.lanes, rs1.reducers = C * B2, [(0, i) for i in range(NL1)], [(1, i) for i in range(NR1)]
    rs1.block_order = (
        (lambda c_: sorted(range(C * B2), key=lambda j: (rs8.dst_order(c_).index(j // B2), j % B2))) if tab else None
    )
    rs2 = object.__new__(FS.FabricStreamRSL)
    rs2.nc, rs2.b2, rs2.dst_order = C, B2, (rs8.dst_order if tab else None)
    rs8.rs1, rs8.rs2 = rs1, rs2
    return rs8


def _stage2_lane_list(rs2, me, h):  # FabricStreamRSL._build's lane list: (dst, b) in send order
    return [(d, b) for d in rs2.dsts(me) for b in range(B2) if b % 2 == h]


def _r1rdy_index(rs8, c, cp, b):  # FabricStreamRS8.signals' stage-2 send index for r1 block (cp, b), cp != c
    return rs8.rs2.dsts(c).index(cp) * (B2 // 2) + b // 2


@pytest.mark.parametrize("order", ["", "alt"])
def test_every_index_is_the_consumers_position_once(order):
    rs8 = _rs8(order)
    rs1, rs2 = rs8.rs1, rs8.rs2
    for c in range(C):
        # stage 1: the producers (P7 members) raise LREADY + lane position / OWNRDY + reducer position (pmk_assembly)
        lane_idx = {h: sorted(rs1.lane_list(c, b % NL1).index(b) for b in range(C * B2) if b % NL1 == h) for h in range(NL1)}
        red_idx = {k: sorted(rs1.red_list(c, b % NR1).index(b) for b in range(C * B2) if b % NR1 == k) for k in range(NR1)}
        assert all(v == list(range(C * B2 // NL1)) for v in lane_idx.values())
        assert all(v == list(range(C * B2 // NR1)) for v in red_idx.values())
        # stage 1 -> 2: each stage-1 reducer signals its blocks in its list order; every stage-2 lane send index gets
        # exactly one +1 and it names the block that the lane sends at that position
        got = {h: {} for h in range(2)}
        own = {}
        for k in range(NR1):
            for j in rs1.red_list(c, k):
                cp, b = divmod(j, B2)
                if cp == c:
                    own[b] = own.get(b, 0) + 1
                else:
                    i = _r1rdy_index(rs8, c, cp, b)
                    assert i not in got[b % 2]
                    got[b % 2][i] = (cp, b)
        assert own == {b: 1 for b in range(B2)}
        for h in range(2):
            lst = _stage2_lane_list(rs2, c, h)
            assert len(lst) <= 12 and {i: v for i, v in enumerate(lst)} == got[h]
            # an owner counts a source's blocks in b order: each destination's blocks contiguous and ascending
            for d in set(x for x, _ in lst):
                bs = [b for x, b in lst if x == d]
                assert bs == sorted(bs)


def test_default_order_is_the_ascending_one():
    rs8 = _rs8("")
    for c in range(C):
        for b in range(C * B2):
            assert rs8.rs1.lane_list(c, b % NL1).index(b) == b // NL1
            assert rs8.rs1.red_list(c, b % NR1).index(b) == b // NR1
        for cp in range(C):
            if cp != c:
                assert rs8.rs2.dsts(c).index(cp) == (cp if cp < c else cp - 1)


def test_alt_starts_every_column_with_the_far_side():
    tab = FS.RS8_ORDERS["alt"]
    assert all(tab[c][-1] == c and sorted(tab[c]) == list(range(C)) for c in range(C))
    assert all(tab[C - 1 - c] == tuple(C - 1 - d for d in tab[c]) for c in range(C))  # mirror-symmetric
    assert tab[0][0] == 3 and tab[3][0] == 0
