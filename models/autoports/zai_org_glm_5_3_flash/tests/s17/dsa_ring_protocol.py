#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17 DSA-capable verify (tt/mk_mla.py rev 56, kernels/mla/ix_dm1.cpp MLA_DSA_ONE_SLOT + kernels/spec/spec_ctl.cpp
ACCEPT ring move): a host model of the index state protocol, checked against the plain decode over the committed tokens.

Plain decode, per position p: ring row p % 4 = kg(p, token); pooled key p / 4 = F(ring rows 0..3) (final at p % 4 == 3).
The verify at base c with rows c .. c + K (row 0 = the bonus, rows 1.. = drafts), in row order on one index core: pool q's
ring rows at ring tile rows 4 (q - q0) .. + 3 (q0 = c / 4); the pooled key of pool q = F(those 4 rows); every row's reads
(sparse: pooled keys q < (p + 1) / 4) after all rows' writes. ACCEPT: n = leading correct drafts; when (c + n + 1) / 4 > q0
rows 4 g .. + 3 -> rows 0..3. Checked: every committed verify row reads exactly the plain decode's pooled keys, and the
state after every step equals the plain decode's on the committed prefix (ring rows of committed positions in the open
pool, every closed pooled key). Random acceptance; also the worst cases (every draft rejected / accepted).
usage: no-tt-devices python3 -I dsa_ring_protocol.py [--steps 4000] [--seed 0]"""
import argparse
import random


def kg(p, tok):
    return ("kg", p, tok)


def F(rows):
    return ("pool", tuple(rows))


def plain(tokens, upto):
    """the plain decode's state after positions 0 .. upto - 1, and the pooled keys each position reads"""
    ring, pools, reads = [None] * 32, {}, {}
    for p in range(upto):
        ring[p % 4] = kg(p, tokens[p])
        pools[p // 4] = F(ring[0:4])
        reads[p] = tuple(pools[q] for q in range((p + 1) // 4))
    return ring, pools, reads


def run(K, steps, rng, mode):
    tokens = [rng.randrange(1000) for _ in range(steps * (K + 1) + 16)]
    _, ref_pools, ref_reads = plain(tokens, len(tokens))
    ring, pools = [("zero",)] * 32, {}
    c = rng.randrange(0, 8)  # the prefill's committed prefix 0 .. c - 1 (plain writes), the bonus at c
    for p in range(c):
        ring[p % 4] = kg(p, tokens[p])
        pools[p // 4] = F(ring[0:4])
    for _ in range(steps):
        if c + K + 1 >= len(tokens):
            break
        # drafts: correct with prob 0.7 (mode rand), all wrong (rej), all right (acc)
        drafts = [tokens[c + j] if (mode == "acc" or (mode == "rand" and rng.random() < 0.7)) else -1 - j
                  for j in range(1, K + 1)]
        rows = [tokens[c]] + drafts
        q0 = c // 4
        for j, tok in enumerate(rows):  # I2, row order (ix_dm1)
            p = c + j
            g = p // 4 - q0
            assert 4 * g + 3 < 32
            ring[4 * g + p % 4] = kg(p, tok)
            pools[p // 4] = F(ring[4 * g: 4 * g + 4])
        n = 0
        while n < K and drafts[n] == tokens[c + 1 + n]:
            n += 1
        for j in range(n + 1):  # I4 reads of the committed rows (after every row's writes)
            p = c + j
            got = tuple(pools[q] for q in range((p + 1) // 4))
            assert got == ref_reads[p], (K, mode, c, j, "pooled keys read")
        g = (c + n + 1) // 4 - q0  # ACCEPT (spec_ctl.cpp)
        if g > 0:
            ring[0:4] = ring[4 * g: 4 * g + 4]
        c = c + n + 1
        # state on the committed prefix 0 .. c - 1 == the plain decode's
        for p in range(4 * (c // 4), c):
            assert ring[p % 4] == kg(p, tokens[p]), (K, mode, c, p, "ring row")
        for q in range(c // 4):
            assert pools[q] == ref_pools[q], (K, mode, c, q, "closed pooled key")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for K in range(1, 8):
        for mode in ("rand", "rej", "acc"):
            for rep in range(4):
                c = run(K, a.steps // 4, rng, mode)
        print(f"K={K}: ok (rand / all rejected / all accepted, last position {c})", flush=True)
    # without the move (rows of every pool at p % 4, today's indexing) K >= 2 breaks
    broken = []
    for K in range(1, 8):
        ok = True
        tokens = [rng.randrange(1000) for _ in range(4000)]
        _, ref_pools, _ = plain(tokens, len(tokens))
        ring, pools, c = [None] * 4, {}, 0
        for _ in range(900):
            drafts = [tokens[c + j] if rng.random() < 0.7 else -1 for j in range(1, K + 1)]
            for j, tok in enumerate([tokens[c]] + drafts):
                p = c + j
                ring[p % 4] = kg(p, tok)
                pools[p // 4] = F(ring[0:4])
            n = 0
            while n < K and drafts[n] == tokens[c + 1 + n]:
                n += 1
            c += n + 1
            if any(pools[q] != ref_pools[q] for q in range(c // 4)):
                ok = False
                break
        broken.append((K, ok))
    print("p % 4 rows without the move:", ", ".join(f"K={k} {'ok' if ok else 'BROKEN'}" for k, ok in broken))
    print("DSA-RING-PROTOCOL PASS")


if __name__ == "__main__":
    main()
