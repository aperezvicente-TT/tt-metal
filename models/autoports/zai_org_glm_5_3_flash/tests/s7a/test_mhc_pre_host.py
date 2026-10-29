# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (CPU) tests of the Stage 7a mhc_pre program description (no device):
* `reference_step` (the program's arithmetic) == the model's reference HyperConnection.weights / forward + RMSNorm;
* the plane layout is a bijection of the tile and `to_planes` / `from_planes` / `plane_off` agree;
* an emulation of the program's data movement (K-core ownership of column tiles, stacked re-layout, group / root
  sums, z -> plane re-layout, coefficient extraction into the mailbox and the post / comb output tiles, the K cores'
  column-0 expansions, collapse, norm) written with the kernels' own index functions reproduces `reference_step`
  for both input layouts and both post / comb layouts."""

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.reference.config import Glm5NextTextConfig
from models.autoports.zai_org_glm_5_3_flash.reference.modules import HyperConnection, RMSNorm
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M

HIDDEN, N = 4096, 4


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    fn = (torch.randn(24, N * HIDDEN, generator=g) * 0.02).to(torch.bfloat16).float()
    base = torch.randn(24, generator=g) * 0.5
    scale = torch.tensor([0.9, 1.3, 2.1])
    w = (1 + 0.2 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16).float()
    return fn, base, scale, w


def _per_mix(scale):
    return torch.cat([scale[0].repeat(N), scale[1].repeat(N), scale[2].repeat(N * N)])


def test_oracle_matches_reference_module():
    torch.manual_seed(0)
    fn, base, scale, w = _weights()
    cfg = Glm5NextTextConfig()
    hc = HyperConnection(cfg)
    with torch.no_grad():
        hc.fn.copy_(fn)
        hc.base.copy_(base)
        hc.scale.copy_(scale)
    norm = RMSNorm(HIDDEN, cfg.rms_norm_eps)
    with torch.no_grad():
        norm.weight.copy_(w)
    t = 5
    x = torch.randn(1, t, N, HIDDEN) * 3.0
    with torch.no_grad():
        post_r, comb_r, col_r = hc(x)
        pre_r, _, _ = hc.weights(x)
        hn_r = norm(col_r)
    ref = M.reference_step(
        x.reshape(t, N * HIDDEN), fn.t(), _per_mix(scale), base, w, hc_eps=cfg.hc_eps, x_eps=cfg.rms_norm_eps,
        h_eps=cfg.rms_norm_eps, iters=cfg.hc_sinkhorn_iters, matmul_tf32=False,
    )  # fmt: skip
    torch.testing.assert_close(ref["pre"].float(), pre_r[0].float(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ref["post"].float(), post_r[0].float(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ref["comb"].float(), comb_r[0].float(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(ref["hn"].float(), hn_r[0].float(), rtol=1e-4, atol=1e-4)


def test_plane_layout_bijection():
    offs = {M.plane_off(p, t) for p in range(32) for t in range(32)}
    assert offs == set(range(1024))
    tiles = {M.tile_off(r, c) for r in range(32) for c in range(32)}
    assert tiles == set(range(1024))
    v = torch.randn(24, 7)
    torch.testing.assert_close(M.from_planes(M.to_planes(v), 24, 7), v)
    tile = torch.arange(1024.0).reshape(32, 32)
    mem = M.face_major(tile)
    assert all(mem[M.tile_off(r, c)] == tile[r, c] for r in range(32) for c in range(32))
    # plane p covers rows 4*band.. of face p/8 with column parity p % 2: lane t = 8i + c -> (row, col)
    for p in (0, 1, 7, 8, 9, 23, 24, 31):
        f, band, par = p // 8, (p % 8) // 2, p % 2
        for t in range(32):
            r = 16 * (f // 2) + 4 * band + t // 8
            c = 16 * (f % 2) + 2 * (t % 8) + par
            assert M.plane_off(p, t) == M.tile_off(r, c)


# ------------------------------------------------------------------------------------------------- emulation
def _tiles_flat(x, t):
    """x [t, 16384] -> 512 fp32 tiles in page order s*128 + c (rows t valid)."""
    out = torch.zeros(N * M.COL_TILES, 32, 32)
    for s in range(N):
        for c in range(M.COL_TILES):
            out[s * M.COL_TILES + c, :t] = x[:, s * HIDDEN + 32 * c : s * HIDDEN + 32 * c + 32]
    return out


def _tiles_stacked(x, t):
    """x [t, 16384] -> stacked [t, 1, 4, 4096] tiles: page t*128 + c, row s = stream s."""
    out = torch.zeros(t * M.COL_TILES, 32, 32)
    for tt in range(t):
        for c in range(M.COL_TILES):
            for s in range(N):
                out[tt * M.COL_TILES + c, s] = x[tt, s * HIDDEN + 32 * c : s * HIDDEN + 32 * c + 32]
    return out


def emulate(x, fn_t, scale, bias, w, t, x_mode, pc_mode, eps):
    """The program's data movement with the kernels' index functions; the arithmetic in float64."""
    hc_eps, x_eps, h_eps, iters = eps
    pages = _tiles_flat(x, t) if x_mode == M.X_FLAT else _tiles_stacked(x, t)
    fn_tiles = torch.zeros(512, 32, 32)
    fnp = torch.zeros(N * HIDDEN, 32)
    fnp[:, :24] = fn_t
    for k in range(512):
        fn_tiles[k] = fnp[32 * k : 32 * k + 32]
    parts = {}
    xt_of = {}
    for cx in range(M.GRID_X):
        for cy in range(M.GRID_Y):
            c0 = 2 * (M.GRID_Y * cx + cy)
            xt = torch.zeros(8, 32, 32)
            for s in range(N):
                for j in range(2):
                    if x_mode == M.X_FLAT:
                        xt[s * 2 + j] = pages[s * M.COL_TILES + c0 + j]
                    else:  # stream.cpp: flat (s, j) row tt = stacked page tt*128 + c0 + j row s
                        for tt in range(t):
                            mem_src = M.face_major(pages[tt * M.COL_TILES + c0 + j])
                            mem_dst = M.face_major(xt[s * 2 + j])
                            for half in (0, 16):
                                so, do = M.tile_off(s, half), M.tile_off(tt, half)
                                mem_dst[do : do + 16] = mem_src[so : so + 16]
                            xt[s * 2 + j] = M.from_face_major(mem_dst)
            xt_of[(cx, cy)] = xt
            Mp = sum(M.tf32(xt[i]).double() @ fn_tiles[(i // 2) * M.COL_TILES + c0 + i % 2].double() for i in range(8))
            Sp = sum(xt[i].double() ** 2 for i in range(8))
            parts[(cx, cy)] = (Mp, Sp)
    Mtot = sum(sum(parts[(cx, cy)][0] for cy in range(M.GRID_Y)) for cx in range(M.GRID_X))
    Stot = sum(sum(parts[(cx, cy)][1] for cy in range(M.GRID_Y)) for cx in range(M.GRID_X))
    inv = torch.rsqrt(Stot.sum(1, keepdim=True) / (N * HIDDEN) + x_eps)
    srow, brow = torch.zeros(32), torch.zeros(32)
    srow[:24], brow[:24] = scale, bias
    z = (Mtot * inv) * srow.double() + brow.double()
    # root sync: z -> plane tile
    zmem = M.face_major(z)
    plane = torch.zeros(1024, dtype=torch.float64)
    for v in range(24):
        for tt in range(t):
            plane[M.plane_off(v, tt)] = zmem[M.tile_off(tt, v)]
    # SFPU pass in the plane layout (value p of token tt at plane_off(p, tt))
    P = lambda p: torch.stack([plane[M.plane_off(p, tt)] for tt in range(32)])
    vals = {p: P(p) for p in range(24)}
    for i in range(4):
        vals[i] = torch.sigmoid(vals[i]) + hc_eps
        vals[4 + i] = 2 * torch.sigmoid(vals[4 + i])
    cm = torch.stack([torch.stack([vals[8 + 4 * j + k] for k in range(4)]) for j in range(4)])  # [4,4,32]
    cm = torch.softmax(cm, dim=1) + hc_eps
    cm = cm / (cm.sum(0, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        cm = cm / (cm.sum(1, keepdim=True) + hc_eps)
        cm = cm / (cm.sum(0, keepdim=True) + hc_eps)
    for j in range(4):
        for k in range(4):
            vals[8 + 4 * j + k] = cm[j, k]
    coef = torch.zeros(1024, dtype=torch.float64)
    for p in range(24):
        for tt in range(32):
            coef[M.plane_off(p, tt)] = vals[p][tt]
    # root: mailbox + outputs
    msg = torch.zeros(128, dtype=torch.float64)
    for s in range(4):
        for tt in range(t):
            msg[s * 32 + tt] = coef[M.plane_off(s, tt)]
    if pc_mode == M.PC_STACKED:
        post_t, comb_t = torch.zeros(t, 32, 32), torch.zeros(t, 32, 32)
        for tt in range(t):
            o = torch.zeros(1024)
            for i in range(4):
                o[M.tile_off(i, 0)] = coef[M.plane_off(4 + i, tt)]
            post_t[tt] = M.from_face_major(o)
            o = torch.zeros(1024)
            for j in range(4):
                for k in range(4):
                    o[M.tile_off(j, k)] = coef[M.plane_off(8 + 4 * j + k, tt)]
            comb_t[tt] = M.from_face_major(o)
        post = post_t[:, :4, 0]  # [T,1,4,1] logical
        comb = comb_t[:, :4, :4]
    else:
        post_t = torch.zeros(4, 32, 32)
        for s in range(4):
            o = torch.zeros(1024)
            for tt in range(t):
                o[M.tile_off(tt, 0)] = coef[M.plane_off(4 + s, tt)]
            post_t[s] = M.from_face_major(o)
        o = torch.zeros(1024)
        for tt in range(t):
            for e in range(16):
                o[M.tile_off(tt, e)] = coef[M.plane_off(8 + e, tt)]
        comb_tile = M.from_face_major(o)
        post = post_t[:, :t, 0].t()  # [1,4,T,1] -> [T, 4]
        comb = comb_tile[:t, :16].reshape(t, 4, 4)
    # K cores: expansion, collapse, norm
    hn = torch.zeros(t, HIDDEN, dtype=torch.float64)
    s2_tot = torch.zeros(32, 32, dtype=torch.float64)
    hs = {}
    for (cx, cy), xt in xt_of.items():
        pre_tiles = torch.zeros(4, 1024, dtype=torch.float64)
        for s in range(4):
            for tt in range(t):
                pre_tiles[s, M.tile_off(tt, 0)] = msg[s * 32 + tt]
        pre_cols = [M.from_face_major(pre_tiles[s])[:, :1] for s in range(4)]  # column 0 broadcast
        h = [sum(xt[s * 2 + j].double() * pre_cols[s] for s in range(4)) for j in range(2)]
        hs[(cx, cy)] = h
        s2_tot += h[0] ** 2 + h[1] ** 2
    inv_h = torch.rsqrt(s2_tot.sum(1, keepdim=True) / HIDDEN + h_eps)
    msg2 = torch.zeros(32, dtype=torch.float64)
    msg2[:t] = inv_h[:t, 0]
    inv_tile = torch.zeros(32, 1, dtype=torch.float64)
    inv_tile[:t, 0] = msg2[:t]
    for (cx, cy), h in hs.items():
        c0 = 2 * (M.GRID_Y * cx + cy)
        for j in range(2):
            wrow = w[32 * (c0 + j) : 32 * (c0 + j) + 32].double()
            hn[:, 32 * (c0 + j) : 32 * (c0 + j) + 32] = (h[j] * inv_tile * wrow)[:t]
    return dict(z=z[:t, :24], pre=torch.stack([vals[i][:t] for i in range(4)], 1), post=post, comb=comb, hn=hn)


@pytest.mark.parametrize("t,x_mode,pc_mode", [(1, M.X_FLAT, M.PC_STACKED), (3, M.X_STACKED, M.PC_STACKED),
                                              (8, M.X_FLAT, M.PC_STREAM_MAJOR), (32, M.X_FLAT, M.PC_STREAM_MAJOR)])
def test_program_emulation(t, x_mode, pc_mode):
    torch.manual_seed(1)
    fn, base, scale, w = _weights(1)
    x = torch.randn(t, N * HIDDEN) * 2.0
    eps = (1e-6, 1e-6, 1e-5, 20)
    ref = M.reference_step(x, fn.t(), _per_mix(scale), base, w, hc_eps=eps[0], x_eps=eps[1], h_eps=eps[2], iters=20)
    em = emulate(x, fn.t(), _per_mix(scale), base, w, t, x_mode, pc_mode, eps)
    torch.testing.assert_close(em["z"], ref["z"], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(em["pre"], ref["pre"], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(em["post"].double(), ref["post"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(em["comb"].double(), ref["comb"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(em["hn"], ref["hn"], rtol=1e-9, atol=1e-9)
