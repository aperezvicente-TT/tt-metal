# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the megakernel weight layout mk-stream-v1 (tt/mm_stream.py): every core's stream, read the way
kernels/mm_stream/wreader.cpp reads it (bank b(c), in-bank pages m0(c) .. m0(c) + K*NT - 1 of the interleaved tensor =
pages b + 8 m), is its columns' K x NT tiles in K-row-major order; the streams of a bank do not overlap; an emulated
stream matmul reproduces x @ W."""

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt import mm_stream as S


@pytest.mark.parametrize("k,n,cores", [(128, 256, 4), (256, 1024, 8), (128, 3072, 12), (4096, 24576, 96)])
def test_stream_layout_roundtrip(k, n, cores):
    g = torch.Generator().manual_seed(k + n)
    w = torch.randn(k, n, generator=g)
    lay = S.stream_layout(k, n, cores)
    tiles = S.stream_tiles(w, lay).reshape(-1, 32, 32)
    kt, nt = lay["kt"], lay["nt"]
    seen = set()
    for c in range(cores):
        b, m0 = c % S.BANKS, lay["m0"][c]
        for k_ in range(0, kt, max(1, kt // 4)):
            for j in range(nt):
                m = m0 + k_ * nt + j
                assert (b, m) not in seen
                seen.add((b, m))
                page = b + S.BANKS * m
                want = w[k_ * 32 : (k_ + 1) * 32, (c * nt + j) * 32 : (c * nt + j + 1) * 32]
                assert torch.equal(tiles[page], want)
        assert m0 + kt * nt <= lay["pages_per_bank"]


def test_stream_matmul_emulation():
    g = torch.Generator().manual_seed(3)
    k, n, cores, kb = 256, 512, 8, 2
    w, x = torch.randn(k, n, generator=g), torch.randn(5, k, generator=g)
    lay = S.stream_layout(k, n, cores)
    tiles = S.stream_tiles(w, lay).reshape(-1, 32, 32)
    kt, nt = lay["kt"], lay["nt"]
    y = torch.zeros(5, n)
    for c in range(cores):
        b, m0 = c % S.BANKS, lay["m0"][c]
        stream = tiles[b + S.BANKS * (m0 + torch.arange(kt * nt))].reshape(kt, nt, 32, 32)
        for blk in range(kt // kb):
            for kk in range(kb):
                k_ = blk * kb + kk
                for j in range(nt):
                    y[:, (c * nt + j) * 32 : (c * nt + j + 1) * 32] += x[:, k_ * 32 : (k_ + 1) * 32] @ stream[k_, j]
    assert torch.allclose(y, x @ w, rtol=1e-4, atol=1e-3)
