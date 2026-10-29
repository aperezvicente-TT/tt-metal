# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14c host check (CPU only): an emulator of kda_rec v2 (kernels/kda_rec/kda_rec_v2_sfpu.hpp + the v2 reader) at
the DST plane / lane level — the L1 tile layout (face-major), the plane map (plane_off), the reader's compact vector
scatter (key_off / val_off), the state planes (sp), the per-band broadcasts, the sub-row transpose reductions and
the output row-0 planes — in fp32 with the kernel's operation order, vs kda_rec.reference_step (fp64).
SFPU semantics assumed (ISA SFPSHFT2 mode 3 / SFPTRANSP): ror1 = rotate the 8 SFPU columns of every sub-row by one;
transpose(a, b, c, d) -> out_k sub-row i = in_i sub-row k. usage: v2_emulator.py [B] [H] [seed]"""
import sys

import numpy as np
import torch

f32 = np.float32


def tile_off(r, c):
    return (((r >> 4) << 1) + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)


def plane_off(p, t):
    return (p >> 3) * 256 + (((p & 7) >> 1) * 4 + (t >> 3)) * 16 + 2 * (t & 7) + (p & 1)


def key_off(base, r):
    return plane_off(base + (r >> 5), 8 * (r & 3) + ((r >> 2) & 7))


def val_off(base, j):
    return plane_off(base + (j >> 5), 8 * (2 * ((j >> 4) & 1) + (j & 1)) + ((j >> 1) & 7))


def sp(bb, cg):
    return (2 * (bb >> 2) + (cg >> 1)) * 8 + 2 * (bb & 3) + (cg & 1)


def row0(cg):
    return (cg >> 1) * 8 + (cg & 1)


PLANE_IDX = np.array([[plane_off(p, t) for t in range(32)] for p in range(32)])  # [plane, lane] -> word


def to_tile(m):  # 32x32 -> face-major words
    w = np.zeros(1024, f32)
    for r in range(32):
        for c in range(32):
            w[tile_off(r, c)] = m[r, c]
    return w


def from_tile(w):
    m = np.zeros((32, 32), f32)
    for r in range(32):
        for c in range(32):
            m[r, c] = w[tile_off(r, c)]
    return m


class Dst:
    """8 tiles x 32 planes x 32 lanes (fp32)."""

    def __init__(self):
        self.d = np.zeros((8 * 32, 32), f32)

    def unpack(self, slot, words):
        self.d[32 * slot: 32 * slot + 32] = words[PLANE_IDX]

    def pack(self, slot):
        w = np.zeros(1024, f32)
        w[PLANE_IDX] = self.d[32 * slot: 32 * slot + 32]
        return w


def ror(v):  # rotate the 8 columns of each sub-row
    return np.roll(v.reshape(4, 8), 1, axis=1).reshape(32)


def rowsum8(v):
    v = f32(v + ror(v))
    v = f32(v + ror(ror(v)))
    t = ror(ror(v))
    return f32(v + ror(ror(t)))


def transp(a, b, c, d):
    ins = [x.reshape(4, 8) for x in (a, b, c, d)]
    return [np.stack([ins[i][k] for i in range(4)]).reshape(32) for k in range(4)]


def total(v):
    v = rowsum8(v)
    a, b, c, d = transp(v, v, v, v)
    return f32(f32(a + b) + f32(c + d))


def rsqrt(x):
    return f32(1.0) / np.sqrt(x.astype(np.float64)).astype(f32)


def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z.astype(np.float64)))).astype(f32)


VEC, ACC, BC, OD = 4 * 32, 5 * 32, 6 * 32, 7 * 32
VQ, VK, VG, VV, VGO, VBETA, VKN, VQK = 0, 4, 8, 12, 16, 20, 21, 25


def masks(D):
    for c in range(8):
        m = np.zeros(32, f32)
        m[[8 * i + c for i in range(4)]] = 1.0
        D.d[BC + 16 + c] = m


def item(q, k, g, v, go, beta, S, w_norm, eps, scale):
    """one (user, head) through the v2 dataflow; q / k / g / v / go [128], beta scalar, S [128, 128] -> (on, S')"""
    vec = np.zeros(1024, f32)
    for r in range(128):
        vec[key_off(VQ, r)] = q[r]
        vec[key_off(VK, r)] = k[r]
        vec[key_off(VG, r)] = g[r]
        vec[val_off(VV, r)] = v[r]
        vec[val_off(VGO, r)] = go[r]
    for t in range(32):
        vec[plane_off(VBETA, t)] = beta
    wt = np.zeros(1024, f32)
    for j in range(128):
        wt[val_off(0, j)] = w_norm[j]
    s_tiles = [[to_tile(S[32 * mi: 32 * mi + 32, 32 * ni: 32 * ni + 32]) for ni in range(4)] for mi in range(4)]
    # W1
    D = Dst()
    D.unpack(4, vec)
    for m in range(4):
        D.d[VEC + VG + m] = np.exp(D.d[VEC + VG + m].astype(np.float64)).astype(f32)
    sq = np.zeros(32, f32)
    sk = np.zeros(32, f32)
    for m in range(4):
        qq, kk = D.d[VEC + VQ + m], D.d[VEC + VK + m]
        sq = f32(sq + qq * qq)
        sk = f32(sk + kk * kk)
    iq = f32(rsqrt(f32(total(sq) + f32(1e-6))) * f32(scale))
    ik = rsqrt(f32(total(sk) + f32(1e-6)))
    qk = np.zeros(32, f32)
    for m in range(4):
        qn = f32(D.d[VEC + VQ + m] * iq)
        kn = f32(D.d[VEC + VK + m] * ik)
        d = D.d[VEC + VG + m]
        qk = f32(qk + qn * kn)
        D.d[VEC + VKN + m] = kn
        D.d[VEC + VQ + m] = f32(qn * d)
        D.d[VEC + VK + m] = f32(kn * d)
    D.d[VEC + VQK] = total(qk)
    masks(D)
    for mi in range(4):
        for ni in range(4):
            D.unpack(ni, s_tiles[mi][ni])
        for bb in range(8):
            mk = D.d[BC + 16 + bb]
            D.d[BC + 0 + bb] = rowsum8(f32(D.d[VEC + VK + mi] * mk))
            D.d[BC + 8 + bb] = rowsum8(f32(D.d[VEC + VQ + mi] * mk))
        for ni in range(4):
            for cg in range(4):
                au = np.zeros(32, f32)
                aw = np.zeros(32, f32)
                for bb in range(8):
                    s = D.d[32 * ni + sp(bb, cg)]
                    au = f32(au + s * D.d[BC + bb])
                    aw = f32(aw + s * D.d[BC + 8 + bb])
                if mi == 0:
                    D.d[ACC + 4 * ni + cg] = au
                    D.d[ACC + 16 + 4 * ni + cg] = aw
                else:
                    D.d[ACC + 4 * ni + cg] = f32(D.d[ACC + 4 * ni + cg] + au)
                    D.d[ACC + 16 + 4 * ni + cg] = f32(D.d[ACC + 16 + 4 * ni + cg] + aw)
    beta_v, qk_v = D.d[VEC + VBETA], D.d[VEC + VQK]
    for ni in range(4):
        a, b, c, d = transp(*[D.d[ACC + 4 * ni + x] for x in range(4)])
        u = f32(f32(a + b) + f32(c + d))
        a, b, c, d = transp(*[D.d[ACC + 16 + 4 * ni + x] for x in range(4)])
        w = f32(f32(a + b) + f32(c + d))
        dl = f32(f32(D.d[VEC + VV + ni] - u) * beta_v)
        D.d[OD + ni] = f32(w + qk_v * dl)
        for x, e in enumerate(transp(dl, dl, dl, dl)):
            D.d[OD + 4 + 4 * ni + x] = e
    vec2, delta = D.pack(4), D.pack(7)
    # W2..W5
    out_tiles = [[None] * 4 for _ in range(4)]
    for mi in range(4):
        D = Dst()
        for ni in range(4):
            D.unpack(ni, s_tiles[mi][ni])
        D.unpack(4, vec2)
        D.unpack(7, delta)
        masks(D)
        for bb in range(8):
            mk = D.d[BC + 16 + bb]
            D.d[BC + 0 + bb] = rowsum8(f32(D.d[VEC + VG + mi] * mk))
            D.d[BC + 8 + bb] = rowsum8(f32(D.d[VEC + VKN + mi] * mk))
        for ni in range(4):
            for cg in range(4):
                dl = D.d[OD + 4 + 4 * ni + cg]
                for bb in range(8):
                    p = 32 * ni + sp(bb, cg)
                    s = f32(D.d[p] * D.d[BC + bb])
                    D.d[p] = f32(s + D.d[BC + 8 + bb] * dl)
        for ni in range(4):
            out_tiles[mi][ni] = D.pack(ni)
    # W6
    D = Dst()
    D.unpack(7, delta)
    D.unpack(4, vec2)
    D.unpack(5, wt)
    ss = np.zeros(32, f32)
    for ni in range(4):
        ss = f32(ss + D.d[OD + ni] * D.d[OD + ni])
    inv = rsqrt(f32(f32(total(ss) * f32(1.0 / 128)) + f32(eps)))
    on_rows = np.zeros(128, f32)
    for ni in range(4):
        o = f32(f32(D.d[OD + ni] * inv) * D.d[ACC + ni])
        o = f32(o * sigmoid(D.d[VEC + VGO + ni]))
        for x, e in enumerate(transp(o, o, o, o)):
            D.d[32 * ni + row0(x)] = e
        on_rows[32 * ni: 32 * ni + 32] = from_tile(D.pack(ni))[0]
    S1 = np.zeros((128, 128), f32)
    for mi in range(4):
        for ni in range(4):
            S1[32 * mi: 32 * mi + 32, 32 * ni: 32 * ni + 32] = from_tile(out_tiles[mi][ni])
    return on_rows, S1


def main():
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))
    from models.autoports.zai_org_glm_5_3_flash.tt.kda_rec import reference_step

    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    g = torch.Generator().manual_seed(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    DK = 128
    rnd = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    q, k, v, go = rnd(B, H, DK), rnd(B, H, DK), rnd(B, H, DK), rnd(B, H, DK)
    gg = -torch.rand(B, H, DK, generator=g) * 0.2
    beta = torch.rand(B, H, generator=g)
    S = rnd(B, H, DK, DK) * 0.05
    w = (1.0 + 0.1 * rnd(DK)).to(torch.bfloat16).float()
    ref = reference_step(q, k, v, gg, beta, go, S, w, 1e-6)
    on = torch.zeros(B, H, DK)
    S1 = torch.zeros(B, H, DK, DK)
    for b in range(B):
        for h in range(H):
            o_, s_ = item(*(t[b, h].numpy().astype(f32) for t in (q, k, gg, v, go)), np.float32(beta[b, h].item()),
                          S[b, h].numpy().astype(f32), w.numpy().astype(f32), 1e-6, 1.0 / np.sqrt(DK))  # fmt: skip
            on[b, h] = torch.from_numpy(o_)
            S1[b, h] = torch.from_numpy(s_)

    def pcc(a, b):
        a, b = a.double().flatten(), b.double().flatten()
        a, b = a - a.mean(), b - b.mean()
        return float(a @ b / (a.norm() * b.norm()))

    print(dict(B=B, H=H, on_pcc=pcc(on, ref["on"]), state_pcc=pcc(S1, ref["state"]),
               on_max_rel=float(((on - ref["on"]).abs().max() / ref["on"].abs().max())),
               state_max_abs=float((S1 - ref["state"]).abs().max())))  # fmt: skip


if __name__ == "__main__":
    main()
