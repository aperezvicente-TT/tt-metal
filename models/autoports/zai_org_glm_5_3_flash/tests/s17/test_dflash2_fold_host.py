# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host test of the drafter's block-attention FOLD (tt/dflash2_kind.py TtnnOps.fold_views / sdpa_fold,
GLM53_DF2_OPS_FOLD; program 2026-10-27T00:59:56Z): the index math the device path relies on, emulated in torch
step by step with the op's speculative multi-position semantics (sdpa_decode_nanobind.cpp: batch b's row-tile j is
candidate b T + j, every row of that tile is a q head attending positions [0, cur_pos[b T + j]] of batch b's kv head):
  - the cache view [nblk, nkv, 32, hd] -> [nblk nkv, 1, 32, hd] (block b, kv head g = view block b nkv + g),
  - the page table row g = arange(nblk) nkv + g,
  - Q = the T block rows reshaped to [1, 1, T 32, hd], repeated nkv times,
  - the head mask (row j 32 + h of batch g kept iff h // G == g) + the sum over the batch rows,
  - the output reshaped to [1, T, 32, hd] and padded with zero rows to 32 users,
against TorchOps.sdpa (the host backend's per-row attention) on the T block rows, at TP8's per-chip geometry (nh 4,
nkv 1) and TP2's (nh 16, nkv 4), at a short and a long bound."""
import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_kind as DK

TILE = 32


def _fold_torch(q, K, V, cur, scale, nh, T, Tg):
    """the device path of TtnnOps.sdpa_fold, op by op, in torch (float64): T / Tg calls of Tg candidates"""
    nblk, nkv, tile, hd = K.shape
    K1, V1 = K.reshape(nblk * nkv, 1, tile, hd), V.reshape(nblk * nkv, 1, tile, hd)  # (the aliasing view)
    pt = torch.stack([torch.arange(nblk) * nkv + g for g in range(nkv)])
    G = nh // nkv
    cf = cur[: nkv * Tg]
    parts = []
    for p in range(T // Tg):
        qf = q[:, p * Tg:(p + 1) * Tg].reshape(1, 1, Tg * TILE, hd).repeat(1, nkv, 1, 1)
        out = torch.zeros(1, nkv, Tg * TILE, hd, dtype=torch.float64)
        for b in range(nkv):  # the op: batch b reads ITS page row's blocks (one kv head), every candidate tile j
            Kb = K1[pt[b]].reshape(-1, hd).double()  # [nblk 32, hd] = kv head b's positions in order
            Vb = V1[pt[b]].reshape(-1, hd).double()
            for j in range(Tg):
                n = int(cf[b * Tg + j]) + 1
                rows = qf[0, b, j * TILE:(j + 1) * TILE].double()  # [32 (q heads), hd]
                s = rows @ Kb[:n].T * scale
                out[0, b, j * TILE:(j + 1) * TILE] = torch.softmax(s, -1) @ Vb[:n]
        if nkv > 1:
            mask = torch.zeros(1, nkv, Tg * TILE, hd, dtype=torch.float64)
            for g in range(nkv):
                for j in range(Tg):
                    mask[0, g, j * TILE + g * G: j * TILE + (g + 1) * G] = 1
            out = (out * mask).sum(dim=1, keepdim=True)
        parts.append(out.reshape(1, Tg, TILE, hd))
    return torch.cat(parts + [torch.zeros(1, TILE - T, TILE, hd, dtype=torch.float64)], dim=1)


@pytest.mark.parametrize("Tg", [8, 4])  # (GLM53_DF2_FOLD_TG: one call of 8 candidates / two of 4)
@pytest.mark.parametrize("nh,nkv", [(4, 1), (16, 4)])  # TP8 / TP2 per-chip geometry of the drafter (32 q / 8 kv heads)
@pytest.mark.parametrize("bound", [37, 4 * 1024 - 1])
def test_fold_matches_the_per_row_attention(nh, nkv, bound, Tg):
    torch.manual_seed(nh * 1000 + bound)
    hd, T, nblk = 128, 8, 160
    K = torch.randn(nblk, nkv, TILE, hd, dtype=torch.float64)
    V = torch.randn(nblk, nkv, TILE, hd, dtype=torch.float64)
    q = torch.zeros(1, TILE, TILE, hd, dtype=torch.float64)
    q[:, :, :nh] = torch.randn(1, TILE, nh, hd, dtype=torch.float64)  # 32 users (T real), heads padded to 32
    cur = torch.full((TILE,), bound, dtype=torch.int64)  # DF2_PREP: c' + K in all 32 entries
    scale = hd**-0.5
    got = _fold_torch(q, K, V, cur, scale, nh, T, Tg)
    # the host backend's per-row attention (TorchOps.sdpa: rows attend positions <= cur[row]; its K / V per chip)
    ops = DK.TorchOps.__new__(DK.TorchOps)
    ops._tick = lambda *a, **k: None
    ref = ops.sdpa([q[:, :, :nh].float()], {"t": [K.float()]}, {"t": [V.float()]}, [cur], scale)[0]
    assert torch.allclose(got[0, :T, :nh].float(), ref[0, :T], atol=2e-5, rtol=1e-4), (got[0, :T, :nh] - ref[0, :T]).abs().max()
    if nkv > 1:  # (the mask keeps only each group's own heads: the padded head rows are zero; at nkv 1 they carry the
        # zero query's uniform average, as the unfolded op's do -- concat_heads takes heads [:nh] only)
        assert torch.count_nonzero(got[0, :T, nh:]) == 0
    assert torch.count_nonzero(got[0, T:]) == 0  # (the padding users are zero rows)
