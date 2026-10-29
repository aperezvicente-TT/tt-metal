# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host model of the PfStream PARK ring addressing in program A (tt/pmk_assembly.py _qbase, Stage 14 2026-10-23): the
streamer (pfs_stream.cpp) lands launch-global block q = QBASE + v at the published PARK pointer (the publisher's write
pointer at the phase start, slot w0) + (q mod P) slots; the member's ring holds block v at slot (w0 + v) mod P. The
block counts are program A's at TP2 (qkv 768, lowrank 32, f_b / g_b 64, o_proj in its own program) and TP8 (192, 32,
16, 16), P = 4 at TP2 and 3 at TP8 with the resident pin (PARK2 = P before the fix, 4 after)."""
import math

import pytest

PARK29, PARK2 = 29, 47


def phases(tp: int, park: int, park2: int):
    """(name, blocks, P, cb, program) in launch order"""
    if tp == 2:
        b = dict(qkv=768, lowrank=32, fb=64, gb=64, o=128)
    else:
        b = dict(qkv=192, lowrank=32, fb=16, gb=16, o=32)
    return [("qkv", b["qkv"], park, PARK29, "A2"), ("lowrank", b["lowrank"], park, PARK29, "A2"),
            ("fb", b["fb"], park2, PARK2, "A2"), ("gb", b["gb"], park2, PARK2, "A2"), ("o_proj", b["o"], park, PARK29, "A4")]


def qbases(ph, align: bool):
    """the assembly's QBASE per phase: running (the previous build) or aligned (_qbase)"""
    out, qbase, pushed, prog = {}, 0, {}, None
    for name, blocks, P, cb, pr in ph:
        if pr != prog:
            pushed, prog = {}, pr  # a launch resets the CB pointers (qbase keeps running, as the assembly does)
        w0 = pushed.get(cb, 0) % P
        if align:
            assert w0 == 0, (name, w0)
            step = P * 2 // math.gcd(P, 2)
            qbase = -(-qbase // step) * step
        out[name] = (qbase, w0, P, blocks)
        pushed[cb] = pushed.get(cb, 0) + blocks
        qbase += blocks
    return out


def mismatches(q):
    """blocks whose streamer slot != the member's slot, or written past the CB end"""
    bad = {}
    for name, (qbase, w0, P, blocks) in q.items():
        n = sum(1 for v in range(blocks) if (w0 + (qbase + v) % P) != (w0 + v) % P or w0 + (qbase + v) % P >= P)
        if n:
            bad[name] = n
    return bad


def test_tp2_was_aligned_and_stays_identical():
    old, new = qbases(phases(2, 4, 4), False), qbases(phases(2, 4, 4), True)
    assert mismatches(old) == {} and mismatches(new) == {}
    assert {k: v[0] for k, v in old.items()} == {k: v[0] for k, v in new.items()}  # the TP2 programs do not change


def test_tp8_resident_previous_build_misaligned():
    bad = mismatches(qbases(phases(8, 3, 3), False))
    assert set(bad) == {"fb", "gb", "o_proj"}, bad  # f_b: f = inf (04:09Z); qkv / lowrank aligned (u / lr correct)
    assert bad["fb"] == 16


def test_tp8_resident_fixed():
    q = qbases(phases(8, 3, 4), True)
    assert mismatches(q) == {}
    assert q["fb"][0] % 4 == 0 and q["o_proj"][0] % 6 == 0


def test_pair_park3_reproduces():
    """the pair with pfs_park 3 (PARK2 = 3 in the previous build): the same misalignment (22:43Z park3 row: x-hat_f NaN)"""
    assert "fb" in mismatches(qbases(phases(2, 3, 3), False))
    assert mismatches(qbases(phases(2, 3, 4), True)) == {}


def test_park2_3_with_alignment_refused():
    with pytest.raises(AssertionError):
        qbases(phases(8, 3, 3), True)  # g_b would start on slot 1: the assembly asserts instead of mis-streaming
