# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (CPU) tests of the Stage 7a mhc_post program description (no device):
* `reference_step` == the model's reference `hc_expand` (new_j = post_j * y + sum_i comb[i, j] * residual_i), with
  the shared-expert summand folded into y;
* an emulation of the program written with the kernels' own index functions — mhc_pre's compact coefficient tile
  (sync.cpp: row t, col v = coefficient plane 4 + v), the reader's column-0 expansion (rows >= T zero) and stacked
  re-layout (flat tile (s, j) row t = stacked page t * 128 + c row s), the compute order (ys = y + y2, then
  ys * post_s + x_0 C[0][s] + ... + x_3 C[3][s], fp32 each step) and both writer layouts — reproduces `reference_step`
  within fp32 rounding, for both residual layouts and both output layouts."""

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.reference.modules import hc_expand
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_post as P
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M

N, D, COLS = 4, 4096, 128


def _inputs(t, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = (torch.randn(t, D, generator=g) * 2).to(torch.bfloat16)
    y2 = (torch.randn(t, D, generator=g) * 0.5).to(torch.bfloat16)
    x = torch.randn(t, N, D, generator=g) * 3
    post = 2 * torch.sigmoid(torch.randn(t, N, generator=g))
    comb = torch.softmax(torch.randn(t, N, N, generator=g), -1) + 1e-6
    comb = comb / comb.sum(-2, keepdim=True)
    return y, y2, x, post, comb


def _coef_tile(post, comb):
    """mhc_pre's compact coefficient tile: the root's coefficient planes (pre 0..3, post 4..7, comb 8..23) in the plane
    layout, then o[tile_off(t, v)] = planes[plane_off(N + v, t)] for v < 20 (sync.cpp), as a [32, 32] row-major tile."""
    t = post.shape[0]
    coef24 = torch.cat([torch.zeros(t, N), post, comb.reshape(t, N * N)], -1)  # [t, 24]
    planes = M.face_major(M.to_planes(coef24.T.contiguous()))  # the root's coefficient tile memory
    tile = torch.zeros(32 * 32)  # the root zeroes its scratch tile before writing
    for tok in range(t):
        for v in range(20):
            tile[M.tile_off(tok, v)] = planes[M.plane_off(N + v, tok)]
    return tile


def _untile(flat):
    """[1024] tile-ordered -> [32, 32] row-major."""
    out = torch.empty(32, 32)
    for r in range(32):
        for c in range(32):
            out[r, c] = flat[M.tile_off(r, c)]
    return out


def _emulate(y, y2, x, coef_flat, *, x_mode, out_mode):
    """The program on host tensors: returns X' [t, 4, 4096] decoded from the written output pages."""
    t = y.shape[0]
    coef = _untile(coef_flat)
    # reader: 20 column-0 tiles, row r = coefficient k of token r (r < T), 0 below
    cvec = torch.zeros(20, 32)
    cvec[:, :t] = coef[:t, :20].T
    # residual tiles in the flat page order: page s * 128 + c holds stream s, columns 32c..32c+31, rows = tokens
    if x_mode == P.X_FLAT:
        xt = torch.zeros(N, COLS, 32, 32)
        xt[:, :, :t, :] = x.permute(1, 0, 2).reshape(N, t, COLS, 32).permute(0, 2, 1, 3)
    else:
        # stacked [T,1,4,4096]: page t * 128 + c holds rows s = streams of token t; flat tile (s, c) row t <- its row s
        stk = torch.zeros(t, COLS, 32, 32)
        stk[:, :, :N, :] = x.reshape(t, N, COLS, 32).permute(0, 2, 1, 3)
        xt = torch.zeros(N, COLS, 32, 32)
        for tok in range(t):
            for s in range(N):
                xt[s, :, tok, :] = stk[tok, :, s, :]
    yt = torch.zeros(COLS, 32, 32)
    yt[:, :t, :] = y.float().reshape(t, COLS, 32).permute(1, 0, 2)
    y2t = torch.zeros(COLS, 32, 32)
    y2t[:, :t, :] = y2.float().reshape(t, COLS, 32).permute(1, 0, 2) if y2 is not None else 0.0
    out = torch.zeros(N, COLS, 32, 32)
    for c in range(COLS):  # core c // 2, j = c % 2 (no cross-core dependence)
        ys = yt[c] + y2t[c] if y2 is not None else yt[c]  # fp32 add
        for s in range(N):
            acc = ys * cvec[s].unsqueeze(-1)  # column broadcast
            for i in range(N):
                acc = acc + xt[i, c] * cvec[N + i * N + s].unsqueeze(-1)
            out[s, c] = acc
    if out_mode == P.OUT_FLAT:
        return out[:, :, :t, :].permute(2, 0, 1, 3).reshape(t, N, D)
    pages = torch.zeros(t, COLS, 32, 32)  # writer: page t * 128 + c, row s = stream s's row t, other rows zero
    for tok in range(t):
        for s in range(N):
            pages[tok, :, s, :] = out[s, :, tok, :]
    assert torch.all(pages[:, :, N:, :] == 0)
    return pages[:, :, :N, :].permute(0, 2, 1, 3).reshape(t, N, D)


def test_reference_matches_hc_expand():
    y, y2, x, post, comb = _inputs(5)
    ys = (y.float() + y2.float()).double()
    ref = hc_expand(ys.unsqueeze(0), x.double().unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0))[0]
    got = P.reference_step(y, x, post, comb, y2)
    assert torch.allclose(got, ref, rtol=1e-12, atol=1e-12)
    ref1 = hc_expand(y.double().unsqueeze(0), x.double().unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0))[0]
    assert torch.allclose(P.reference_step(y, x, post, comb), ref1, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("t,x_mode,out_mode,add", [
    (1, P.X_STACKED, P.OUT_STACKED, False), (4, P.X_STACKED, P.OUT_STACKED, False),
    (1, P.X_STACKED, P.OUT_FLAT, True), (3, P.X_STACKED, P.OUT_FLAT, True),
    (8, P.X_FLAT, P.OUT_FLAT, False), (32, P.X_FLAT, P.OUT_FLAT, True),
])  # fmt: skip
def test_program_emulation_matches_reference(t, x_mode, out_mode, add):
    y, y2, x, post, comb = _inputs(t, seed=t)
    y2 = y2 if add else None
    got = _emulate(y, y2, x, _coef_tile(post, comb), x_mode=x_mode, out_mode=out_mode)
    ref = P.reference_step(y, x, post, comb, y2)
    err = (got.double() - ref).abs().max() / ref.abs().max()
    assert err < 1e-6, err


def _emulate_packed(y, y2, x, coef_flat, *, x_mode, out_mode):
    """The PACKED decode program (kernels/mhc_post/*_p.cpp, t <= 4) on host tensors, with its own index arithmetic:
    per core (column tiles c0, c0 + 1) ONE tile, row r = (s * 2 + j) * t + tok; reader_p builds X_i / Y / COEF from
    whole source pages, compute_p does the same fp32 ops in the same order as compute.cpp, writer_p scatters the rows
    into whole zeroed pages. Returns X' [t, 4, 4096] decoded from the written pages."""
    t = y.shape[0]
    assert 8 * t <= 32
    coef = _untile(coef_flat)
    # the residual source pages as the reader reads them
    if x_mode == P.X_FLAT:
        src = torch.zeros(N, COLS, 32, 32)  # page s * 128 + c: stream s, rows = tokens
        src[:, :, :t, :] = x.permute(1, 0, 2).reshape(N, t, COLS, 32).permute(0, 2, 1, 3)
    else:
        src = torch.zeros(t, COLS, 32, 32)  # page tok * 128 + c: rows = streams
        src[:, :, :N, :] = x.reshape(t, N, COLS, 32).permute(0, 2, 1, 3)
    ysrc = torch.zeros(COLS, 32, 32)
    ysrc[:, :t, :] = y.float().reshape(t, COLS, 32).permute(1, 0, 2)
    y2src = torch.zeros(COLS, 32, 32)
    if y2 is not None:
        y2src[:, :t, :] = y2.float().reshape(t, COLS, 32).permute(1, 0, 2)
    flat_pages = torch.zeros(N, COLS, 32, 32)  # flat output pages s * 128 + c
    stk_pages = torch.zeros(t, COLS, 32, 32)  # stacked output pages tok * 128 + c
    for c0 in range(0, COLS, 2):  # core k: c0 = 2k
        xp = torch.zeros(N, 32, 32)
        yp, y2p, cp = torch.zeros(32, 32), torch.zeros(32, 32), torch.zeros(5, 32)
        for s in range(N):
            for j in range(2):
                for tok in range(t):
                    r = (s * 2 + j) * t + tok
                    for i in range(N):
                        xp[i, r] = src[i, c0 + j, tok] if x_mode == P.X_FLAT else src[tok, c0 + j, i]
                    cp[0, r] = coef[tok, s]
                    for i in range(N):
                        cp[1 + i, r] = coef[tok, N + N * i + s]
                    yp[r], y2p[r] = ysrc[c0 + j, tok], y2src[c0 + j, tok]
        acc = yp + y2p if y2 is not None else yp
        acc = acc * cp[0].unsqueeze(-1)
        for i in range(N):
            acc = acc + xp[i] * cp[1 + i].unsqueeze(-1)
        assert torch.all(acc[8 * t :] == 0)
        for s in range(N):
            for j in range(2):
                for tok in range(t):
                    row = acc[(s * 2 + j) * t + tok]
                    if out_mode == P.OUT_FLAT:
                        flat_pages[s, c0 + j, tok] = row
                    else:
                        stk_pages[tok, c0 + j, s] = row
    if out_mode == P.OUT_FLAT:
        assert torch.all(flat_pages[:, :, t:, :] == 0)
        return flat_pages[:, :, :t, :].permute(2, 0, 1, 3).reshape(t, N, D)
    assert torch.all(stk_pages[:, :, N:, :] == 0)
    return stk_pages[:, :, :N, :].permute(0, 2, 1, 3).reshape(t, N, D)


@pytest.mark.parametrize("t,x_mode,out_mode,add", [
    (1, P.X_STACKED, P.OUT_STACKED, False), (4, P.X_STACKED, P.OUT_STACKED, True),
    (1, P.X_STACKED, P.OUT_FLAT, True), (3, P.X_FLAT, P.OUT_FLAT, False), (2, P.X_FLAT, P.OUT_STACKED, True),
])  # fmt: skip
def test_packed_emulation_bitwise_unpacked(t, x_mode, out_mode, add):
    """The packed program computes every output element with the same fp32 ops in the same order: bitwise the
    unpacked emulation, and it matches reference_step."""
    y, y2, x, post, comb = _inputs(t, seed=10 + t)
    y2 = y2 if add else None
    tile = _coef_tile(post, comb)
    got = _emulate_packed(y, y2, x, tile, x_mode=x_mode, out_mode=out_mode)
    assert torch.equal(got, _emulate(y, y2, x, tile, x_mode=x_mode, out_mode=out_mode))
    ref = P.reference_step(y, x, post, comb, y2)
    assert (got.double() - ref).abs().max() / ref.abs().max() < 1e-6
