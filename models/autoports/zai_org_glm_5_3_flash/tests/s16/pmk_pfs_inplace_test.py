# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host model of PMK_A_INPLACE (pure python, no ttnn): the tiles a PfStream streamer reads for its blocks under (a) the
COPIED mk-stream-v1 layout (tt/mm_stream.stream_layout + stream_tiles: stream st = NW_c columns, bank st mod 8, in-bank
start m0(st), one interleaved page per tile) and (b) the RESIDENT DSLinear layout (DRAM WIDTH_SHARDED (K, n_pad / 8):
shard b = bank b = row-major tiles [K, SHARD_W]; the in-place stream st = bank st, NW_r = SHARD_W, m0 = 0), both through
the streamer kernel's own address arithmetic (kernels/pfs/pfs_stream.cpp: tile (j KB + r) NW + h NB + c of the stream,
blocks in lane order). Asserts, per output column tile, that the K sequence of weight tiles the members accumulate is
IDENTICAL (same (k, n) tiles, same values, same order) -> the member's y is bitwise the same for the same weight dtype."""
import sys
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", ".."))
from models.autoports.zai_org_glm_5_3_flash.tt.mm_stream import stream_layout, stream_tiles  # noqa: E402

TILE, BANKS = 32, 8


def kernel_order(nst, nw, nb, kb, kt):
    """[(st, k_tile, n_tile)] in the delivery order of the members (blocks q = (st NH + h) NBK + j, rows r of a block,
    columns c): lane L sends j = L, L + 2, .. but the members consume q in order"""
    nh, nbk = nw // nb, kt // kb
    out = []
    for st in range(nst):
        for h in range(nh):
            for j in range(nbk):
                for r in range(kb):
                    for c in range(nb):
                        out.append((st, j * kb + r, st * nw + h * nb + c))
    return out


def copied_tiles(w, nst, nw, nb, kb):
    kt, nt = w.shape[0] // TILE, w.shape[1] // TILE
    lay = stream_layout(w.shape[0], w.shape[1], nst)
    assert lay["nt"] == nw
    pages = stream_tiles(w, lay).reshape(-1, TILE, TILE)  # interleaved page p = bank p % 8, in-bank p // 8
    out = {}
    for st, k, n in kernel_order(nst, nw, nb, kb, kt):
        b, m0 = st % BANKS, lay["m0"][st]
        idx = m0 + k * nw + (n - st * nw)  # the kernel: src0 = w_addr + m0 TB; tile (k NW + h NB + c) TB
        out.setdefault(n, []).append((k, pages[b + BANKS * idx]))
    return out


def resident_tiles(w, nb, kb):
    """shard b = columns [b SHARD_W, (b + 1) SHARD_W) as row-major tiles [K, SHARD_W] in bank b"""
    kt, nt = w.shape[0] // TILE, w.shape[1] // TILE
    sw = nt // BANKS
    wt = w.reshape(kt, TILE, nt, TILE).permute(0, 2, 1, 3)
    shards = [wt[:, b * sw:(b + 1) * sw].reshape(kt * sw, TILE, TILE) for b in range(BANKS)]  # in-bank index k sw + j
    out = {}
    for st, k, n in kernel_order(BANKS, sw, nb, kb, kt):
        idx = 0 + k * sw + (n - st * sw)  # m0 = 0, NW = SHARD_W: the same formula over the shard
        out.setdefault(n, []).append((k, shards[st][idx]))
    return out


def check(k, n, nst_c, nw_c, nb_c, kb_c, nb_r, kb_r, name):
    torch.manual_seed(0)
    w = torch.randn(k, n)
    a, b = copied_tiles(w, nst_c, nw_c, nb_c, kb_c), resident_tiles(w, nb_r, kb_r)
    assert set(a) == set(b) == set(range(n // TILE)), name
    for nt_ in a:
        ka = [kk for kk, _ in a[nt_]]
        kb_ = [kk for kk, _ in b[nt_]]
        assert ka == kb_ == list(range(k // TILE)), (name, nt_, ka[:4], kb_[:4])  # the K order (ascending) is identical
        for (_, ta), (_, tb) in zip(a[nt_], b[nt_]):
            assert torch.equal(ta, tb), (name, nt_)  # the same tile values -> the same accumulation inputs
    # the resident read is bank-adjacent by construction: stream st reads bank st only
    print(f"{name}: K {k} N {n}: copied {nst_c} streams x NW {nw_c} (NB {nb_c}, KB {kb_c}) == resident 8 x NW {n // TILE // BANKS} "
          f"(NB {nb_r}, KB {kb_r}): {len(a)} output tiles, {k // TILE} K tiles each, bitwise")


if __name__ == "__main__":
    check(4096, 3072, 3, 32, 4, 16, 4, 16, "TP8 qkv")      # ds_qkv n_pad 3072: shard 12 tiles
    check(1024, 4096, 4, 32, 4, 16, 4, 16, "TP8 o_proj")   # ds_o: shard 16
    check(4096, 512, 1, 16, 4, 16, 2, 16, "TP8/TP2 lowrank")  # ds_lowrank n_pad 512: shard 2 -> NB 2
    check(4096, 12288, 12, 32, 4, 16, 4, 16, "TP2 qkv")    # shard 48
    check(4096, 4096, 4, 32, 4, 16, 4, 16, "TP2 o_proj")
    print("PASS")
