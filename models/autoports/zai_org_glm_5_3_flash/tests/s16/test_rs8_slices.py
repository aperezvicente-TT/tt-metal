# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host test (no device) of the column-sliced TP8 RS8 (PMK_RS8_QSLICE; doc pmk/w2_a15prof/rs_tail_design.md §4-5.1):
a Python replica of the kernels' loops and of the producers' increments / consumers' thresholds AS CODED
(kernels/fabric_stream/rs_read / rs_send / red_read / red_write / rsl_read / rsl_send / rsl_red_read.cpp FS_SLICES and
kernels/pfs/pfs_member_wr.cpp PFS_SIG_SLICES), for Q in {1, 2, 4, 8} x order {"", alt} x stage 2 {fp32, bf16}:

  A. loop structure per kernel: each unit (block, slice s) touches exactly slice s's tiles; the readers' push order is
     the writers' order (the compute kernels see the same NB-tile batches: per tile the same arithmetic -> bitwise); one
     fused counter increment per unit, on its last packet; the P7 writer raises Q times, after the slice is stored;
  B. the protocol on the 2x4 over 3 launches, randomly interleaved: every consumer wait for unit (i, s) passes only after
     that unit's data of THIS launch is written (versioned buffers), no deadlock, and every counter ends each launch at
     exactly Q x its units;
  C. Q = 1 reproduces today's thresholds.
"""
import random

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import fabric_stream as FS

R, C, B2, NL1, NR1 = 2, 4, 8, 2, 8  # the 2x4: 8 blocks per owner, 2 stage-1 lanes, 8 stage-1 reducers
BL = C * B2  # 32 blocks per mesh row
TPB, NB1, NB2, NB_O, NBLK_O = 128, 16, 8, 4, 32  # tiles per block; batch widths; o_proj N-block / N-blocks
PER1, PER2 = BL // NL1, B2 // 2  # stage-1 blocks per lane, stage-2 blocks per (source, lane)
QS = (1, 2, 4, 8)


def _rs8(order: str):
    """the order logic of FabricStreamRS8 / RSR / RSL without tensors (their list methods are pure; test_rs8_order)"""
    rs8 = object.__new__(FS.FabricStreamRS8)
    rs8.C, rs8.b2 = C, B2
    tab = FS.RS8_ORDERS[order]
    rs8.dst_order = (lambda c_: list(tab[c_])) if tab is not None else (lambda c_: list(range(C)))
    rs1 = object.__new__(FS.FabricStreamRSR)
    rs1.bl, rs1.lanes, rs1.reducers = BL, [(0, i) for i in range(NL1)], [(1, i) for i in range(NR1)]
    rs1.block_order = (
        (lambda c_: sorted(range(BL), key=lambda j: (rs8.dst_order(c_).index(j // B2), j % B2))) if tab else None
    )
    rs2 = object.__new__(FS.FabricStreamRSL)
    rs2.nc, rs2.b2, rs2.dst_order = C, B2, (rs8.dst_order if tab else None)
    rs8.rs1, rs8.rs2 = rs1, rs2
    return rs8


def _slice_cols(s: int, q: int) -> set:
    return set(range(s * TPB // q, (s + 1) * TPB // q))


# ---- the thresholds as coded (FS_SLICES branch; Q = 1 = the #else branch, checked equal in test C) ----
def need_ready(g, s, q):  # LREADY / OWNRDY / R1RDY + i (rs_read, red_read, rsl_read, rsl_red_read)
    return (g - 1) * q + s + 1


def need_arr(g, s, q, per, idx):  # the in-order landed counters (red_read, rsl_red_read; rs_read's lanes-reduce branch)
    return ((g - 1) * q + s) * per + idx + 1


def need_row_barrier(g, q, per):  # rsl_read's per-hop drain
    return g * per * q


# ---- A. the kernels' loops (page / packet sequences per unit) ----
def rs_read_sends(q: int, js: list) -> list:
    """rs_read.cpp FS_SLICES: [(unit (i, s), page)] in push order (q == 1: the #else loop)"""
    out = []
    if q == 1:
        for i, j in enumerate(js):
            for b in range(TPB // NB1):
                out += [((i, 0), j * TPB + b * NB1 + t) for t in range(NB1)]
        return out
    bps = TPB // NB1 // q
    for s in range(q):
        for i, j in enumerate(js):
            for b in range(s * bps, (s + 1) * bps):
                out += [((i, s), j * TPB + b * NB1 + t) for t in range(NB1)]
    return out


def rs_send_packets(q: int, rows: list) -> list:
    """rs_send.cpp FS_SLICES: [(unit, (page, page + 1) in the peer's r, last)] (2-tile scatter packets)"""
    out = []
    bps = TPB // NB1 // q
    for s in range(q):
        for i, row in enumerate(rows):
            for b in range(s * bps, (s + 1) * bps):
                for t in range(0, NB1, 2):
                    pg = row * TPB + b * NB1 + t
                    last = (b + 1 == (s + 1) * bps and t + 2 == NB1) if q > 1 else (b + 1 == TPB // NB1 and t + 2 == NB1)
                    out.append(((i, s), (pg, pg + 1), last))
    return out


def red_read_pushes(q: int, blk: list) -> list:
    """red_read.cpp FS_SLICES: [(unit, own P page, landed r page)] in push order; blk = [(jo, jl)]"""
    out = []
    bps = TPB // NB1 // q
    for s in range(q):
        for i, (jo, jl) in enumerate(blk):
            for b in range(s * bps, (s + 1) * bps):
                out += [((i, s), jo * TPB + b * NB1 + t, jl * TPB + b * NB1 + t) for t in range(NB1)]
    return out


def red_write_pages(q: int, rows: list, nb: int) -> list:
    """red_write.cpp FS_SLICES (DRAM path): [(unit, y page, signal after it)]"""
    out = []
    bps = TPB // nb // q
    for s in range(q):
        for i, row in enumerate(rows):
            for b in range(s * bps, (s + 1) * bps):
                for t in range(nb):
                    out.append(((i, s), row * TPB + b * nb + t, b + 1 == (s + 1) * bps and t + 1 == nb))
    return out


def rsl_read_pushes(q: int, rows: list) -> list:
    out = []
    bps = TPB // NB2 // q
    for s in range(q):
        for i, row in enumerate(rows):
            for b in range(s * bps, (s + 1) * bps):
                out += [((i, s), row * TPB + b * NB2 + t) for t in range(NB2)]
    return out


def rsl_send_packets(q: int, rows: list, bf16: bool) -> list:
    """rsl_send.cpp FS_SLICES: fp32 one tile per packet, bf16 two-tile scatter packets"""
    out = []
    bps = TPB // NB2 // q
    for s in range(q):
        for i, row in enumerate(rows):
            for b in range(s * bps, (s + 1) * bps):
                if bf16:
                    for t in range(0, NB2, 2):
                        pg = row * TPB + b * NB2 + t
                        out.append(((i, s), (pg, pg + 1), b + 1 == (s + 1) * bps and t + 2 == NB2))
                else:
                    for t in range(NB2):
                        out.append(((i, s), (row * TPB + b * NB2 + t,), b + 1 == (s + 1) * bps and t + 1 == NB2))
    return out


def rsl_red_read_pushes(q: int, rows4: list) -> list:
    """rsl_red_read.cpp FS_SLICES (one block, n = 1): [(unit, (page of term c) x 4)] in push order"""
    out = []
    bps = TPB // NB2 // q
    for s in range(q):
        for bb in range(s * bps, (s + 1) * bps):
            for t in range(NB2):
                out.append(((0, s), tuple(r_ * TPB + bb * NB2 + t for r_ in rows4)))
    return out


def member_wr_raises(q: int) -> list:
    """pfs_member_wr.cpp: the column tiles stored before each H5 raise (PFS_SIG_SLICES: after N-block n with
    (n + 1) % (NBLK / Q) == 0; q == 1: once after the whole row)"""
    stored, raises = set(), []
    for n in range(NBLK_O):
        stored |= set(range(n * NB_O, (n + 1) * NB_O))
        if q > 1 and (n + 1) % (NBLK_O // q) == 0:
            raises.append(set(stored))
    if q == 1:
        raises.append(set(stored))
    return raises


def _units_cover(seq, q, page_of, n_units):
    """every unit (i, s) touches exactly the columns of slice s, once each"""
    got = {}
    for unit, pg in ((u, page_of(x)) for u, x in seq):
        got.setdefault(unit, []).append(pg % TPB)
    assert len(got) == n_units * q
    for (i, s), cols in got.items():
        assert sorted(cols) == sorted(_slice_cols(s, q)), (i, s)


@pytest.mark.parametrize("q", QS)
def test_kernel_loops(q):
    assert TPB % (q * NB1) == 0 and TPB % (q * NB2) == 0  # FS.RS8_SLICES: whole batches per slice at both widths
    js = [5, 7, 9]
    rd = rs_read_sends(q, js)
    _units_cover([(u, p) for u, p in rd], q, lambda p: p, len(js))
    # rs_send consumes the reader's pushes in order: the packet pages are the reader's pages re-based to the landing rows
    pk = rs_send_packets(q, js)
    assert [p for _, pp, _ in pk for p in pp] == [p for _, p in rd]
    assert [u for u, _, _ in pk if _] == sorted({u for u, _, _ in pk}, key=lambda u: (u[1], u[0]))  # one inc per unit
    for u in {u for u, _, _ in pk}:  # ... on the unit's LAST packet
        mine = [last for uu, _, last in pk if uu == u]
        assert mine[-1] and sum(mine) == 1
    # stage-1 reducer: pushes (own, landed) in the order red_write writes r1; same column, same block row
    blk = [(1 * BL + 3, 3), (1 * BL + 11, 11)]
    pu = red_read_pushes(q, blk)
    wr = red_write_pages(q, [3, 11], NB1)
    assert len(pu) == len(wr) and [u for u, _, _ in pu] == [u for u, _, _ in wr]
    for (u, po, pl), (_, py, _) in zip(pu, wr):
        assert po % TPB == pl % TPB == py % TPB and pl // TPB == py // TPB == blk[u[0]][1]
    assert sum(sig for _, _, sig in wr) == q * 2  # FS_RED_SIG: one +1 per unit
    # stage 2
    rows = [0 * B2 + 1, 2 * B2 + 1, 3 * B2 + 1]
    sp = rsl_read_pushes(q, rows)
    _units_cover(sp, q, lambda p: p, len(rows))
    for bf16 in (False, True):
        pk2 = rsl_send_packets(q, rows, bf16)
        assert [p for _, pp, _ in pk2 for p in pp] == [p for _, p in sp]
        for u in {u for u, _, _ in pk2}:
            mine = [last for uu, _, last in pk2 if uu == u]
            assert mine[-1] and sum(mine) == 1
    rows4 = [1, 9, 17, 25]
    p4 = rsl_red_read_pushes(q, rows4)
    w4 = red_write_pages(q, [1], NB2)
    assert len(p4) == len(w4) == TPB and [u for u, _ in p4] == [u for u, _, _ in w4]
    for (u, pgs), (_, py, _) in zip(p4, w4):
        assert len({p % TPB for p in pgs} | {py % TPB}) == 1  # the 4 terms and the y tile: one column
    # the P7 writer: Q raises, raise s after slices 0..s are stored
    rs_ = member_wr_raises(q)
    assert len(rs_) == q
    for s, stored in enumerate(rs_):
        assert stored == set().union(*(_slice_cols(k, q) for k in range(s + 1)))


# ---- B. the protocol simulation (2x4, 3 launches, random interleaving) ----
class _Sim:
    def __init__(self, q: int, order: str, seed: int, launches: int = 3):
        self.q, self.rs8, self.rng, self.G = q, _rs8(order), random.Random(seed), launches
        self.cnt = {}  # (chip, core, word) -> counter
        self.data = {}  # (chip, buf, row, slice) -> launch that wrote it
        self.launch = {ch: 0 for ch in range(R * C)}
        self.started = {ch: 0 for ch in range(R * C)}
        self.actors = []  # [chip, generator, pending predicate]
        self.waits = 0

    def inc(self, ch, core, word, v=1):
        k = (ch, core, word)
        self.cnt[k] = self.cnt.get(k, 0) + v

    def get(self, ch, core, word):
        return self.cnt.get((ch, core, word), 0)

    def write(self, ch, buf, row, s, g):
        self.data[(ch, buf, row, s)] = g

    def read(self, ch, buf, row, s, g, what):
        got = self.data.get((ch, buf, row, s))
        assert got == g, f"{what}: chip {ch} {buf}[{row}] slice {s} holds launch {got}, reader is in launch {g}"

    # actors: generators yielding predicates (a wait); the body after a yield runs once the predicate holds
    def member(self, ch, G, g):
        r, c = divmod(ch, C)
        rs1 = self.rs8.rs1
        rr, b = divmod(G, BL)
        if rr == r:
            core, word = ("red1", b % NR1), ("OWNRDY", rs1.red_list(c, b % NR1).index(b))
        else:
            core, word = ("lane1", b % NL1), ("LREADY", rs1.lane_list(c, b % NL1).index(b))
        for s in range(self.q):  # Q = 1: the whole row, one raise
            yield lambda: True
            self.write(ch, "P", G, s, g)
            self.inc(ch, core, word)

    def lane1(self, ch, h, g):
        r, c = divmod(ch, C)
        p, qq = r, 1 - r
        peer = qq * C + c
        rs1 = self.rs8.rs1
        bs = rs1.lane_list(c, h)
        self.inc(peer, ("lane1", h), ("CREDIT", p))
        yield lambda: self.get(ch, ("lane1", h), ("CREDIT", qq)) >= g
        for s in range(self.q):
            for i, b in enumerate(bs):
                yield lambda i=i, s=s: self.get(ch, ("lane1", h), ("LREADY", i)) >= need_ready(g, s, self.q)
                self.read(ch, "P", qq * BL + b, s, g, "stage-1 lane read of P")
                self.write(peer, "r", b, s, g)  # the unit's packets land, the last one +1s the peer lane's ARR (fused)
                self.inc(peer, ("lane1", h), ("ARR", p * NL1 + h))

    def red1(self, ch, k, g):
        r, c = divmod(ch, C)
        p, qq = r, 1 - r
        rs1 = self.rs8.rs1
        sig = self.signals(c)[k]
        lst = rs1.red_list(c, k)
        for s in range(self.q):
            for i, b in enumerate(lst):
                h = b % NL1
                idx = rs1.lane_list(c, h).index(b)
                yield lambda h=h, idx=idx, s=s: self.get(ch, ("lane1", h), ("ARR", qq * NL1 + h)) >= need_arr(
                    g, s, self.q, PER1, idx
                )
                yield lambda i=i, s=s: self.get(ch, ("red1", k), ("OWNRDY", i)) >= need_ready(g, s, self.q)
                self.read(ch, "P", p * BL + b, s, g, "stage-1 reducer own block")
                self.read(ch, "r", b, s, g, "stage-1 reducer landed block")
                self.write(ch, "r1", b, s, g)
                core, word = sig[i]
                self.inc(ch, core, word)

    def lane2(self, ch, h, g):
        r, me = divmod(ch, C)
        rs2 = self.rs8.rs2
        sends = [(d, b) for d in rs2.dsts(me) for b in range(B2) if b % 2 == h]
        for c_ in range(C):
            if c_ != me:
                self.inc(r * C + c_, ("lane2", h), ("CREDIT", me))
        yield lambda: all(self.get(ch, ("lane2", h), ("CREDIT", c_)) >= g for c_ in range(C) if c_ != me)
        for s in range(self.q):
            for i, (d, b) in enumerate(sends):
                yield lambda i=i, s=s: self.get(ch, ("lane2", h), ("R1RDY", i)) >= need_ready(g, s, self.q)
                self.read(ch, "r1", d * B2 + b, s, g, "stage-2 lane read of r1")
                slot = me if me < d else me - 1
                self.write(r * C + d, "r2", slot * B2 + b, s, g)
                self.inc(r * C + d, ("lane2", h), ("ARR", slot * 2 + h))
        # the per-hop drain (rsl_read): every row source landed its per x Q units on this lane
        yield lambda: all(self.get(ch, ("lane2", h), ("ARR", sl * 2 + h)) >= need_row_barrier(g, self.q, PER2)
                          for sl in range(C - 1))  # fmt: skip

    def red2(self, ch, b, g):
        r, me = divmod(ch, C)
        h = b % 2
        for s in range(self.q):
            yield lambda s=s: self.get(ch, ("red2", b), ("R1RDY", 0)) >= need_ready(g, s, self.q)
            for c_ in range(C):
                if c_ == me:
                    continue
                slot = c_ if c_ < me else c_ - 1
                yield lambda slot=slot, s=s: self.get(ch, ("lane2", h), ("ARR", slot * 2 + h)) >= need_arr(
                    g, s, self.q, PER2, b // 2
                )
            self.read(ch, "r1", me * B2 + b, s, g, "stage-2 reducer own term")
            for c_ in range(C):
                if c_ != me:
                    slot = c_ if c_ < me else c_ - 1
                    self.read(ch, "r2", slot * B2 + b, s, g, "stage-2 reducer landed term")
            self.write(ch, "y", b, s, g)

    def signals(self, c):
        """FabricStreamRS8.signals without the tensors: stage-1 reducer k -> [(core, word)] in its list order"""
        rs1, rs2 = self.rs8.rs1, self.rs8.rs2
        out = {}
        for k in range(NR1):
            sig = []
            for j in rs1.red_list(c, k):
                cp, b = divmod(j, B2)
                if cp == c:
                    sig.append((("red2", b), ("R1RDY", 0)))
                else:
                    i = rs2.dsts(c).index(cp) * (B2 // 2) + b // 2
                    sig.append((("lane2", b % 2), ("R1RDY", i)))
            out[k] = sig
        return out

    def start(self, ch):
        g = self.launch[ch] + 1
        self.launch[ch] = g
        self.started[ch] = g
        gens = [self.member(ch, G, g) for G in range(R * BL)] + [self.lane1(ch, h, g) for h in range(NL1)] \
            + [self.red1(ch, k, g) for k in range(NR1)] + [self.lane2(ch, h, g) for h in range(2)] \
            + [self.red2(ch, b, g) for b in range(B2)]  # fmt: skip
        for gen in gens:
            self.actors.append([ch, gen, next(gen, None)])

    def end_check(self, ch, g):
        """a chip's launch g ended: every counter its consumers waited on holds exactly g x its per-launch units (no
        peer of a later launch can have raised it: the launch credits order them)"""
        r, c = divmod(ch, C)
        q = self.q
        for h in range(NL1):
            for i in range(PER1):
                assert self.get(ch, ("lane1", h), ("LREADY", i)) == g * q
            assert self.get(ch, ("lane1", h), ("ARR", (1 - r) * NL1 + h)) == g * q * PER1
        for k in range(NR1):
            for i in range(BL // NR1):
                assert self.get(ch, ("red1", k), ("OWNRDY", i)) == g * q
        for h in range(2):
            for i in range(3 * PER2):
                assert self.get(ch, ("lane2", h), ("R1RDY", i)) == g * q
            for sl in range(C - 1):
                assert self.get(ch, ("lane2", h), ("ARR", sl * 2 + h)) == g * q * PER2
        for b in range(B2):
            assert self.get(ch, ("red2", b), ("R1RDY", 0)) == g * q
            for s in range(q):
                assert self.data[(ch, "y", b, s)] == g

    def run(self):
        for ch in range(R * C):
            self.start(ch)
        live_n = {ch: sum(1 for a in self.actors if a[0] == ch and a[2] is not None) for ch in range(R * C)}
        while True:
            for ch in range(R * C):  # a chip's next launch starts once all its actors of this launch finished
                if live_n[ch] == 0 and self.launch[ch] < self.G:
                    self.end_check(ch, self.launch[ch])
                    self.actors = [a for a in self.actors if a[0] != ch]
                    self.start(ch)
                    live_n[ch] = sum(1 for a in self.actors if a[0] == ch and a[2] is not None)
            self.actors = [a for a in self.actors if a[2] is not None]
            if not self.actors:
                break
            ready = [a for a in self.actors if a[2]()]
            assert ready, f"deadlock: {len(self.actors)} actors waiting (Q {self.q})"
            a = self.rng.choice(ready)
            a[2] = next(a[1], None)
            if a[2] is None:
                live_n[a[0]] -= 1
            self.waits += 1
        for ch in range(R * C):
            self.end_check(ch, self.G)
        return self.waits


@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("order", ["", "alt"])
@pytest.mark.parametrize("s2", ["fp32", "bf16"])
def test_protocol(q, order, s2):
    # stage 2's dtype changes the packets (test_kernel_loops), not the counters: both forms share the protocol
    for seed in range(2):
        _Sim(q, order, seed=hash((q, order, s2, seed)) & 0xFFFF).run()


def test_wrong_threshold_is_caught():
    """the simulation is not vacuous: an off-by-one-slice reader (waiting slice s - 1's count for slice s) reads a
    slice its producer has not stored in some interleaving"""
    global need_ready
    keep = need_ready
    need_ready = lambda g, s, q: (g - 1) * q + max(s, 1)  # noqa: E731 -- slice s passes on slice s - 1's raise
    try:
        with pytest.raises(AssertionError):
            for seed in range(20):
                _Sim(4, "", seed=seed).run()
    finally:
        need_ready = keep


def test_q1_is_todays_thresholds():
    """C. the FS_SLICES formulas at Q = 1 equal the #else (today's) kernels' thresholds"""
    for g in range(1, 6):
        for idx in range(16):
            assert need_ready(g, 0, 1) == g  # rs_read / red_read / rsl_read / rsl_red_read: >= g x FS_RDY_MULT (1)
            assert need_arr(g, 0, 1, PER1, idx) == (g - 1) * PER1 + idx + 1  # red_read
            assert need_arr(g, 0, 1, PER2, idx % 4) == (g - 1) * PER2 + idx % 4 + 1  # rsl_red_read (b / 2)
        assert need_row_barrier(g, 1, PER2) == g * PER2  # rsl_read's drain
    assert FS.slice_defines(1) == [] and FS.slice_defines(4) == [("FS_SLICES", "4")]
    assert member_wr_raises(1) == [set(range(TPB))]


def test_knob():
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM

    assert ASM.rs8_qslice_from_env({}) == 1 and ASM.rs8_qslice_from_env({"PMK_RS8_QSLICE": "0"}) == 1
    assert ASM.rs8_qslice_from_env({"PMK_RS8_QSLICE": "4"}) == 4
    with pytest.raises(AssertionError):
        ASM.rs8_qslice_from_env({"PMK_RS8_QSLICE": "3"})
