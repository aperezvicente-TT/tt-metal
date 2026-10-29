# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host emulation of int17's kernels/mk/topk.hpp (fast int16 reject + composite min-heap + finish_row expansion) vs the
exact reference (per row: the K best by bf16 order key, ties -> the lower vocab id). Runs on random rows incl. ties,
all-negative rows, +-0, inf, partial last tiles, several phases; prints mismatches (must be 0)."""
import random
import sys

import numpy as np

INT32_MIN = -(2**31)


def s32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & 0x80000000 else x


def key(v):
    return v ^ (0x8000 | ((-(v >> 15)) & 0x7FFF))


class Reader:
    def __init__(self, K, T, prepass=False):
        self.K, self.T, self.prepass = K, T, prepass
        self.floor = [0] * 32
        self.h = [[0] * (2 * K) for _ in range(32)]
        self.cnt = [0] * 32
        self.ph_vb, self.nph = [], 0
        self.slow = 0

    def sift_down(self, h, m, i, x):
        while True:
            c = 2 * i + 1
            if c >= m:
                break
            cv = h[c]
            if c + 1 < m and h[c + 1] < cv:
                c, cv = c + 1, h[c + 1]
            if cv >= x:
                break
            h[i] = cv
            i = c
        h[i] = x

    def take(self, r, x):
        h, n, K = self.h[r], self.cnt[r], self.K
        self.slow += 1
        if n < K:
            i = n
            while i > 0:
                p = (i - 1) >> 1
                if h[p] <= x:
                    break
                h[i] = h[p]
                i = p
            h[i] = x
            self.cnt[r] = n + 1
            return h[0] if n + 1 == K else 0
        self.sift_down(h, K, 0, x)
        return h[0]

    @staticmethod
    def bound(root):
        return s32((root & 0x7FFF0000) | 0xFFFF) if root >= 0x80000000 else INT32_MIN

    def scan(self, P, tiles, NR, n0, cw, vbase0):
        """tiles: [NR] uint16 arrays [32 rows, 32 cols] (logical layout; the face layout only changes addresses)"""
        self.ph_vb.append(vbase0 + P * cw + n0 * 32)
        if self.prepass and self.nph == 0:
            for r in range(self.T):
                g = []
                for j in range(NR):
                    if (n0 + j) * 32 + 32 > cw or len(g) + 2 > 64:
                        break
                    for f in range(2):
                        mx = 0
                        for c in range(f * 16, f * 16 + 16, 2):
                            word = int(tiles[j][r, c]) | (int(tiles[j][r, c + 1]) << 16)
                            sb = (word >> 15) & 0x10001
                            k = word ^ (0x80008000 | (((sb << 15) - sb) & 0xFFFFFFFF))
                            mx = max(mx, k >> 16, k & 0xFFFF)
                        g.append(mx)
                if len(g) >= self.K:
                    fk = kth_largest(list(g), len(g), self.K)
                    self.floor[r] = ((fk << 16) - 1) if fk else 0
        lbase = 0xFFFF - (self.nph << 12)
        self.nph += 1
        for j in range(NR):
            col0 = (n0 + j) * 32
            if col0 >= cw:
                break
            ncol = min(32, cw - col0)
            lb = lbase - j * 32
            t = tiles[j]
            for r in range(self.T):
                fl = self.floor[r]
                root = self.h[r][0] if self.cnt[r] == self.K else fl
                bnd = self.bound(root)
                if ncol == 32:
                    for c in range(0, 32, 2):
                        word = int(t[r, c]) | (int(t[r, c + 1]) << 16)
                        hi, lo = s32(word), s32(word << 16)
                        if max(hi, lo) <= bnd:
                            continue
                        x = (key(word & 0xFFFF) << 16) | (lb - c)
                        if x > root:
                            root = max(self.take(r, x), fl)
                            bnd = self.bound(root)
                        x = (key(word >> 16) << 16) | (lb - c - 1)
                        if x > root:
                            root = max(self.take(r, x), fl)
                            bnd = self.bound(root)
                else:
                    for c in range(ncol):
                        x = (key(int(t[r, c])) << 16) | (lb - c)
                        if x > root:
                            root = max(self.take(r, x), fl)

    def finish(self):
        out = []
        for r in range(32):
            h, n, K = self.h[r], self.cnt[r], self.K
            for m in range(n, 1, -1):
                worst, last = h[0], h[m - 1]
                h[m - 1] = worst
                self.sift_down(h, m - 1, 0, last)
            for i in range(K - 1, -1, -1):
                if i >= n:
                    h[2 * i], h[2 * i + 1] = 0, 0xFFFFFFFF
                    continue
                x = h[i]
                local = 0xFFFF - (x & 0xFFFF)
                h[2 * i + 1] = self.ph_vb[local >> 12] + (local & 0xFFF)
                h[2 * i] = x >> 16
            out.append([(h[2 * i], h[2 * i + 1]) for i in range(K)])
        return out


def kth_largest(a, n, k):
    """the kernel's Hoare quickselect (int indices): the k-th largest of a[0..n)"""
    lo, hi, want = 0, n - 1, n - k
    while lo < hi:
        pivot = a[(lo + hi) >> 1]
        i, j = lo, hi
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        if want <= j:
            hi = j
        elif want >= i:
            lo = i
        else:
            return a[want]
    return a[want]


def to_bf16_bits(x):
    return (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def gen_row(rng, n, mode):
    if mode == "normal":
        v = rng.normal(0, 3, n)
    elif mode == "neg":
        v = -np.abs(rng.normal(5, 3, n)) - 0.01
    elif mode == "ties":
        v = rng.integers(-4, 5, n).astype(np.float64) * 0.5
    elif mode == "zeros":
        v = rng.choice([0.0, -0.0, 1.0, -1.0, 0.0], n)
    elif mode == "mixed":
        v = rng.normal(-2, 1, n)
        v[rng.integers(0, n, 5)] = rng.normal(3, 1, 5)
    else:
        v = rng.choice([np.inf, -np.inf, 1.0, -0.0, 0.0, 2.0], n)
    b = to_bf16_bits(v)
    if mode == "zeros":
        neg0 = rng.random(n) < 0.5
        b = np.where((b == 0) & neg0, np.uint16(0x8000), b)
    return b


def main(seed=0, trials=200, prepass=False):
    rng = np.random.default_rng(seed)
    bad = rows = slow = vals = 0
    for trial in range(trials):
        K = random.Random(trial).choice([1, 5, 32, 32, 32])
        T = random.Random(trial + 7).choice([1, 4, 8, 32])
        splits = random.Random(trial + 3).choice([1, 2, 4])
        NR = random.Random(trial + 5).choice([1, 3, 6])
        nreaders = 2
        cw = nreaders * NR * 32 - random.Random(trial + 9).choice([0, 0, 7, 31])  # partial / padded last tile
        mode = ["normal", "neg", "ties", "zeros", "mixed", "inf"][trial % 6]
        vbase0 = 1000 * trial
        # logits of the whole shard: [T rows, splits * cw]
        full = np.stack([gen_row(rng, splits * cw, mode) for _ in range(T)])
        for rd_i in range(nreaders):
            rd = Reader(K, T, prepass)
            n0 = rd_i * NR
            ids_all = []
            for P in range(splits):
                tiles = []
                for j in range(NR):
                    t = np.zeros((32, 32), dtype=np.uint16)
                    for c in range(32):
                        col = (n0 + j) * 32 + c
                        if col < cw:
                            t[:T, c] = full[:, P * cw + col]
                            ids_all.append((P, col))
                    tiles.append(t)
                rd.scan(P, tiles, NR, n0, cw, vbase0)
            got = rd.finish()
            slow += rd.slow
            for r in range(T):
                cand = []
                for P in range(splits):
                    for j in range(NR):
                        for c in range(32):
                            col = (n0 + j) * 32 + c
                            if col < cw:
                                v = int(full[r, P * cw + col])
                                cand.append((key(v), vbase0 + P * cw + col))
                vals += len(cand)
                cand.sort(key=lambda kv: (-kv[0], kv[1]))
                exp = cand[:K] + [(0, 0xFFFFFFFF)] * (K - len(cand[:K]))
                rows += 1
                if got[r] != exp:
                    bad += 1
                    if bad <= 3:
                        print("MISMATCH", dict(trial=trial, mode=mode, K=K, r=r), got[r][:6], exp[:6])
    print(f"rows {rows} mismatches {bad} values {vals} slow-path calls {slow} ({slow / max(vals, 1):.3f} per value)")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 0, prepass=len(sys.argv) > 2 and sys.argv[2] == "prepass")
             else 0)
