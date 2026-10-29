# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The speculative verify on the sequence-sharded DSA pool (Stage 14, dsa/spec_on_shard_design.md; MkMla rev 61): a host
model of the per-chip index / latent state protocol, checked against the PLAIN sharded decode over the committed tokens.

Plain sharded decode, per position p: ring row p % 4 = kg(p, token) on every chip (replicated); the pooled key of pool
p / 4 = F(ring rows 0..3) on the OWNER chip only (owner(p) = (p / 128) % C, local pool row local_pool(p / 4)); the latent
row of p on the owner only (local position local_pos(p)). A row at p reads, on chip r, the local pools 0 ..
n_closed_local(p, r) - 1 and the local latent rows 0 .. n_local_rows(p, r) - 1.

The verify at base c, rows c .. c + K in row order on one index core (ix_dm1.cpp MLA_DSA_ONE_SLOT + MLA_DSA_SHARD): pool
q's ring rows at ring tile rows 4 (q - q0) .. + 3 (q0 = c / 4) on every chip; each row's pooled key / latent row on the
owner of ITS position (a block straddling a 128-token unit splits across two owners); every row's reads after all rows'
writes (att_dsa_shard_dm0.cpp: LAT_SEM / IXR_SEM). ACCEPT (spec_ctl.cpp) moves rows 4 g .. + 3 to 0..3 on every chip.

Checked: every committed verify row reads, on every chip, exactly the plain sharded decode's local pooled keys and local
latent rows, and after every step the per-chip state on the committed prefix equals the plain decode's. The negative
control (a block's rows all written on the owner of its FIRST row) must fail at a unit-straddling step."""

import random

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt.dsa_shard import KPOOL, UNIT, ShardMap


def kg(p, tok):
    return ("kg", p, tok)


def F(rows):
    return ("pool", tuple(rows))


def plain(sm: ShardMap, tokens, upto):
    """the plain sharded decode after positions 0 .. upto - 1: per chip local pools / latent rows"""
    C = sm.C
    ring, pools, lat = [None] * 4, [dict() for _ in range(C)], [dict() for _ in range(C)]
    for p in range(upto):
        ring[p % 4] = kg(p, tokens[p])
        o = sm.owner(p)
        pools[o][sm.local_pool(p // KPOOL)] = F(ring[0:4])
        lat[o][sm.local_pos(p)] = (p, tokens[p])
    return pools, lat


def reads(sm: ShardMap, pools, lat, p):
    """what a row at position p reads on every chip: (local pooled keys, local latent rows)"""
    out = []
    for r in range(sm.C):
        kp = tuple(pools[r].get(lq) for lq in range(sm.n_closed_local(p, r)))
        kl = tuple(lat[r].get(lp) for lp in range(sm.n_local_rows(p, r)))
        out.append((kp, kl))
    return tuple(out)


def run(C, K, steps, rng, mode, *, c0=None, first_row_owner=False):
    sm = ShardMap(C)
    tokens = [rng.randrange(1000) for _ in range(steps * (K + 1) + 2 * UNIT)]
    ref_pools, ref_lat = plain(sm, tokens, len(tokens))
    ring = [("zero",)] * 32
    pools, lat = [dict() for _ in range(C)], [dict() for _ in range(C)]
    c = rng.randrange(0, UNIT) if c0 is None else c0  # the prefill's committed prefix 0 .. c - 1 (plain writes)
    for p in range(c):
        g = p % 4
        ring[g] = kg(p, tokens[p])
        o = sm.owner(p)
        pools[o][sm.local_pool(p // KPOOL)] = F(ring[0:4])
        lat[o][sm.local_pos(p)] = (p, tokens[p])
    straddles = 0
    for _ in range(steps):
        if c + K + 1 >= len(tokens):
            break
        drafts = [tokens[c + j] if (mode == "acc" or (mode == "rand" and rng.random() < 0.7)) else -1 - j
                  for j in range(1, K + 1)]  # fmt: skip
        rows = [tokens[c]] + drafts
        q0 = c // KPOOL
        straddles += sm.owner(c) != sm.owner(c + K)
        for j, tok in enumerate(rows):  # ix_dm1 / lat_dm1, row order
            p = c + j
            g = p // KPOOL - q0
            assert 4 * g + 3 < 32
            ring[4 * g + p % 4] = kg(p, tok)
            o = sm.owner(c) if first_row_owner else sm.owner(p)
            pools[o][sm.local_pool(p // KPOOL)] = F(ring[4 * g : 4 * g + 4])
            lat[o][sm.local_pos(p)] = (p, tok)
        n = 0
        while n < K and drafts[n] == tokens[c + 1 + n]:
            n += 1
        for j in range(n + 1):  # the committed rows' reads, after every row's writes
            p = c + j
            # the plain decode's reads at p: its own state after positions 0 .. p (the latent row of p itself included)
            assert reads(sm, pools, lat, p) == reads(sm, ref_pools, ref_lat, p), (C, K, mode, c, j, "reads")
        g = (c + n + 1) // KPOOL - q0  # ACCEPT on every chip (the ring is replicated)
        if g > 0:
            ring[0:4] = ring[4 * g : 4 * g + 4]
        c = c + n + 1
        for p in range(KPOOL * (c // KPOOL), c):  # the open pool's committed ring rows
            assert ring[p % 4] == kg(p, tokens[p]), (C, K, mode, c, p, "ring row")
        for r in range(C):  # per-chip closed pooled keys / latent rows of the committed prefix
            for lq in range(sm.n_closed_local(c - 1, r)):
                assert pools[r][lq] == ref_pools[r][lq], (C, K, mode, c, r, lq, "closed pooled key")
            for lp in range(sm.n_local_rows(c - 1, r)):
                assert lat[r][lp] == ref_lat[r][lp], (C, K, mode, c, r, lp, "latent row")
    return c, straddles


@pytest.mark.parametrize("C", [2, 8])
@pytest.mark.parametrize("K", [1, 2, 3, 7])
@pytest.mark.parametrize("mode", ["rand", "rej", "acc"])
def test_verify_on_shard_matches_plain_sharded_decode(C, K, mode):
    rng = random.Random(1000 * C + 10 * K + len(mode))
    total = 0
    for _ in range(3):
        _, s = run(C, K, 400, rng, mode)
        total += s
    # blocks at the unit boundaries (128 - K .. 127 mod 128): every start near one
    for j in range(K + 1):
        _, s = run(C, K, 3, rng, mode, c0=UNIT - j)
        total += s
    if mode != "rej" or K > 1:
        assert total > 0, "no unit-straddling verify block exercised"


@pytest.mark.parametrize("C", [2, 8])
def test_first_row_owner_breaks_at_a_straddle(C):
    """negative control: writing a block's rows on the owner of its first row fails the per-chip check"""
    rng = random.Random(7)
    with pytest.raises(AssertionError):
        run(C, 3, 3, rng, "acc", c0=UNIT - 2, first_row_owner=True)


# ---- step 2 (dsa/spec_on_shard_design.md section 6): ONE_SLOT with token lanes and the baton
def run_lanes(C, K, L, steps, rng, *, baton=True, c0=None):
    """the verify's index writes with token lanes: lane l owns rows l, l + L, ... and processes them in its own order;
    the baton (ix_dm1 MLA_DSA_ONE_SLOT_BATON) lets row t's ring read start only after row t-1's writes, any lane; without
    it a random scheduler interleaves the lanes (each row = ring read, then ring write + pooled key; the read and the
    write of different rows may interleave). The reads of every committed row happen after DONE (all rows written)."""
    sm = ShardMap(C)
    tokens = [rng.randrange(1000) for _ in range(steps * (K + 1) + 2 * UNIT)]
    ref_pools, ref_lat = plain(sm, tokens, len(tokens))
    ring = [("zero",)] * 32
    pools, lat = [dict() for _ in range(C)], [dict() for _ in range(C)]
    c = rng.randrange(0, UNIT) if c0 is None else c0
    for p in range(c):
        ring[p % 4] = kg(p, tokens[p])
        o = sm.owner(p)
        pools[o][sm.local_pool(p // KPOOL)] = F(ring[0:4])
        lat[o][sm.local_pos(p)] = (p, tokens[p])
    for _ in range(steps):
        if c + K + 1 >= len(tokens):
            break
        drafts = [tokens[c + j] if rng.random() < 0.7 else -1 - j for j in range(1, K + 1)]
        rows = [tokens[c]] + drafts
        q0 = c // KPOOL
        # each row = two events: "read" (copy the group's ring rows, insert its own) and "write" (ring rows back +
        # pooled key from the copy + latent). Lane queues of rows in lane order; baton = global row order.
        lanes = [[t for t in range(len(rows)) if t % L == l] for l in range(L)]
        pending = {}  # row -> its ring copy (read done, write not yet)
        order = []
        if baton:
            for t in range(len(rows)):
                order += [("r", t), ("w", t)]
        else:
            heads = [0] * L
            state = {}  # lane -> "r" next or "w" next
            while any(heads[l] < len(lanes[l]) for l in range(L)):
                l = rng.choice([l for l in range(L) if heads[l] < len(lanes[l])])
                t = lanes[l][heads[l]]
                if state.get(l, "r") == "r":
                    order.append(("r", t))
                    state[l] = "w"
                else:
                    order.append(("w", t))
                    state[l] = "r"
                    heads[l] += 1
        for ev, t in order:
            p = c + t
            g = p // KPOOL - q0
            if ev == "r":
                cp = list(ring[4 * g : 4 * g + 4])
                cp[p % 4] = kg(p, rows[t])
                pending[t] = cp
            else:
                cp = pending.pop(t)
                ring[4 * g : 4 * g + 4] = cp
                o = sm.owner(p)
                pools[o][sm.local_pool(p // KPOOL)] = F(cp)
                lat[o][sm.local_pos(p)] = (p, rows[t])
        n = 0
        while n < K and drafts[n] == tokens[c + 1 + n]:
            n += 1
        for j in range(n + 1):
            p = c + j
            assert reads(sm, pools, lat, p) == reads(sm, ref_pools, ref_lat, p), (C, K, L, c, j, "reads")
        g = (c + n + 1) // KPOOL - q0
        if g > 0:
            ring[0:4] = ring[4 * g : 4 * g + 4]
        c = c + n + 1
        for p in range(KPOOL * (c // KPOOL), c):
            assert ring[p % 4] == kg(p, tokens[p]), (C, K, L, c, p, "ring row")
        for r in range(C):
            for lq in range(sm.n_closed_local(c - 1, r)):
                assert pools[r][lq] == ref_pools[r][lq], (C, K, L, c, r, lq, "closed pooled key")
    return c


@pytest.mark.parametrize("C", [2, 8])
@pytest.mark.parametrize("K,L", [(3, 2), (7, 2), (7, 4), (7, 8)])
def test_baton_lanes_match_plain_sharded_decode(C, K, L):
    rng = random.Random(77 + 10 * C + K + L)
    for _ in range(3):
        run_lanes(C, K, L, 300, rng, baton=True)
    for j in range(K + 1):
        run_lanes(C, K, L, 3, rng, baton=True, c0=UNIT - j)


@pytest.mark.parametrize("C", [2, 8])
def test_no_baton_breaks(C):
    """negative control: lanes without the baton interleave the ring read-modify-writes of one pool (a lost row)"""
    rng = random.Random(5 + C)
    with pytest.raises(AssertionError):
        for _ in range(20):
            run_lanes(C, 7, 4, 200, rng, baton=False)
