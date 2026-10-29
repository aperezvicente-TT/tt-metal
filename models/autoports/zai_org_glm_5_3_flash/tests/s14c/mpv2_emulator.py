# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14c host check (CPU only): mhc_post v2 (kernels/mhc_post/mhc_post_v2_sfpu.hpp + the v2 reader's compact
coefficient tile) emulated at plane / lane level vs out_s = y post_s + sum_i x_i C[i][s] (fp64), T tokens.
usage: mpv2_emulator.py [T] [seed]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_emulator import Dst, f32, from_tile, plane_off, rowsum8, sp, tile_off, to_tile  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 8
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
y = np.zeros((32, 32), f32)
y[:T] = rng.standard_normal((T, 32))
x = [np.zeros((32, 32), f32) for _ in range(4)]
for s in range(4):
    x[s][:T] = rng.standard_normal((T, 32))
coef_rows = np.zeros((32, 32), f32)  # mhc_pre's compact coefficient tile: row t = post 0..3 | C at 4 + 4i + s
coef_rows[:T, :20] = rng.standard_normal((T, 20))
# reader v2: the compact coefficient tile
cw = np.zeros(1024, f32)
crow = to_tile(coef_rows)
for t in range(T):
    for k in range(20):
        cw[plane_off(k, 8 * (t & 3) + (t >> 2))] = crow[tile_off(t, k)]
D = Dst()
D.unpack(0, to_tile(y))
for s in range(4):
    D.unpack(1 + s, to_tile(x[s]))
D.unpack(5, cw)
BC, MASK, X, COEF = 6 * 32, 6 * 32 + 24, 32, 5 * 32
for c in range(8):
    m = np.zeros(32, f32)
    m[[8 * i + c for i in range(4)]] = 1.0
    D.d[MASK + c] = m
nb = (T + 3) // 4
for bb in range(8):
    if bb < nb:
        for k in range(20):
            D.d[BC + k] = rowsum8(f32(D.d[COEF + k] * D.d[MASK + bb]))
        for cg in range(4):
            p = sp(bb, cg)
            yv = D.d[p]
            xs = [D.d[X + 32 * i + p] for i in range(4)]
            outs = []
            for s in range(4):
                acc = f32(yv * D.d[BC + s])
                for i in range(4):
                    acc = f32(acc + xs[i] * D.d[BC + 4 + 4 * i + s])
                outs.append(acc)
            for s in range(4):
                D.d[X + 32 * s + p] = outs[s]
    else:
        for cg in range(4):
            for s in range(4):
                D.d[X + 32 * s + sp(bb, cg)] = 0.0
err = 0.0
for s in range(4):
    out = from_tile(D.pack(1 + s))
    ref = y.astype(np.float64) * coef_rows[:, s:s + 1] + sum(
        x[i].astype(np.float64) * coef_rows[:, 4 + 4 * i + s: 5 + 4 * i + s] for i in range(4))
    err = max(err, float(np.abs(out - ref).max()))
print(dict(T=T, max_abs_err=err))
