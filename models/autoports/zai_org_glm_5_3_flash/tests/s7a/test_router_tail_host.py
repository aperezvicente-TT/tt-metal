# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the Stage 7a router_tail program description: the writer's integer-key top-8 (emulated with the
kernel's key function and insertion rule) == `router_tail.reference_step` (descending, ties to the lower expert id),
including negative choices, exact ties and the -1e9 padded experts; `reference_step` == the op chain's semantics
(sigmoid, + bias, torch.topk on distinct values)."""

import struct

import torch

from models.autoports.zai_org_glm_5_3_flash.tt import router_tail as R


def sort_key(bits: int) -> int:
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else (bits | 0x80000000)


def emulate_top8(choice_row: torch.Tensor) -> list:
    keys, ids = [], []
    for e in range(R.N_EXPERTS):
        k = sort_key(struct.unpack("<I", struct.pack("<f", float(choice_row[e])))[0])
        if len(keys) == R.TOP_K and k <= keys[-1]:
            continue
        pos = len(keys) if len(keys) < R.TOP_K else R.TOP_K - 1
        if len(keys) < R.TOP_K:
            keys.append(0)
            ids.append(0)
        while pos > 0 and keys[pos - 1] < k:
            keys[pos], ids[pos] = keys[pos - 1], ids[pos - 1]
            pos -= 1
        keys[pos], ids[pos] = k, e
    return ids


def test_top8_emulation_matches_reference():
    g = torch.Generator().manual_seed(0)
    bias = torch.cat([torch.randn(288, generator=g) * 0.3, torch.full((32,), -1e9)])
    for trial in range(20):
        logits = torch.randn(4, 320, generator=g) * 3
        if trial % 4 == 0:  # exact ties: duplicate logits and zero bias on some experts
            logits[:, 10] = logits[:, 200]
            bias[10] = bias[200]
        ids, scores = R.reference_step(logits, bias)
        choice = torch.sigmoid(logits[:, :320]) + bias
        for t in range(4):
            assert emulate_top8(choice[t]) == ids[t].tolist()
        assert scores.dtype == torch.bfloat16 and scores.shape == (4, 288)


def test_reference_matches_chain_semantics():
    g = torch.Generator().manual_seed(1)
    bias = torch.cat([torch.randn(288, generator=g) * 0.3, torch.full((32,), -1e9)])
    logits = torch.randn(8, 384, generator=g) * 2
    ids, scores = R.reference_step(logits, bias)
    s = torch.sigmoid(logits[:, :320])
    _, ref_ids = torch.topk(s + bias, k=8, dim=-1, largest=True, sorted=True)
    assert torch.equal(ids, ref_ids)
    assert torch.equal(scores, s[:, :288].to(torch.bfloat16))
    assert int(ids.max()) < 288


def test_packed_bias():
    bias = torch.arange(320.0)
    t = R.packed_bias(bias)
    assert torch.equal(t[:10].reshape(-1), bias) and float(t[10:].abs().sum()) == 0.0


# ---- GLM53_PMK_RTAIL_BATCH (the PMK P12 copy): kernels/router_tail/batch_dm.cpp / batch_compute.cpp


def tile_off(r: int, c: int) -> int:  # router_tail_common.hpp
    return (((r >> 4) << 1) + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)


def tilize(x: torch.Tensor) -> list:
    """a [32, 32] tile -> its 1,024 words in the device's face order"""
    out = [0] * 1024
    for r in range(32):
        for c in range(32):
            out[tile_off(r, c)] = x[r, c]
    return out


def emulate_batch_rows(choice: torch.Tensor, r0: int) -> list:
    """batch_dm.cpp's scan for rows r0 .. r0 + 15 of a member's natural choice tiles (tile j = experts 32 j .. 32 j + 31):
    tile by tile, each row's state carried across tiles, the face-row walk, the writer's insertion"""
    tiles = [tilize(choice[:, 32 * j : 32 * j + 32]) for j in range(R.BATCH_TILES)]
    st = [([], []) for _ in range(16)]
    for tj, words in enumerate(tiles):
        for i in range(16):
            keys, ids = st[i]
            r = r0 + i
            for h in range(2):
                base = tile_off(r, 16 * h)
                for j in range(16):
                    k = sort_key(struct.unpack("<I", struct.pack("<f", float(words[base + j])))[0])
                    if len(keys) == R.TOP_K and k <= keys[-1]:
                        continue
                    pos = len(keys) if len(keys) < R.TOP_K else R.TOP_K - 1
                    if len(keys) < R.TOP_K:
                        keys.append(0)
                        ids.append(0)
                    while pos > 0 and keys[pos - 1] < k:
                        keys[pos], ids[pos] = keys[pos - 1], ids[pos - 1]
                        pos -= 1
                    keys[pos], ids[pos] = k, 32 * tj + 16 * h + j
    return [ids for _, ids in st]


def test_batch_scan_matches_packed_and_reference():
    g = torch.Generator().manual_seed(2)
    for trial in range(3):
        bias = torch.cat([torch.randn(288, generator=g) * 0.3, torch.full((32,), -1e9)])
        logits = torch.randn(32, 320, generator=g) * 3
        if trial == 1:  # exact ties across tiles and inside one face row; negative choices
            logits[:, 10] = logits[:, 200]
            bias[10] = bias[200]
            logits[:, 33] = logits[:, 34]
            bias[33] = bias[34]
            bias[:288] -= 2.0
        if trial == 2:  # whole-row ties (every expert equal): ids 0 .. 7
            logits[5] = 0.25
            bias[:288] = 0.0
        ids, scores = R.reference_step(logits, bias)
        choice = (torch.sigmoid(logits[:, :320]) + R.batch_bias(bias)).float()
        got = emulate_batch_rows(choice, 0) + emulate_batch_rows(choice, 16)
        for t in range(32):
            assert got[t] == emulate_top8(choice[t]) == ids[t].tolist(), (trial, t)
        # the epilogue's score reads: score tile e // 32, row r, column e % 32 == scores[t, e]
        s_tiles = [tilize(scores[:, 32 * j : 32 * j + 32].float()) for j in range(R.BATCH_TILES)]
        for t in range(32):
            for e in got[t]:
                assert s_tiles[e >> 5][tile_off(t, e & 31)] == float(scores[t, e])


def test_batch_bias():
    bias = torch.arange(320.0)
    t = R.batch_bias(bias)
    assert tuple(t.shape) == (32, 320) and all(torch.equal(t[r], bias) for r in range(32))
    assert R.BATCH_TILES == 9


def _norm(lines: list) -> list:
    return [ln.split("//")[0].strip() for ln in lines if ln.split("//")[0].strip()]


def test_batch_kernel_insertion_is_the_writers():
    """batch_dm.cpp's insertion lines == writer.cpp's (keys / ids / n, the tie rule); the expert id is e0 + j as in the
    PMK_P12_FAST_SCAN walk"""
    import os

    kd = os.path.join(os.path.dirname(R.__file__), "kernels", "router_tail")
    w = open(os.path.join(kd, "writer.cpp")).read().splitlines()
    b = open(os.path.join(kd, "batch_dm.cpp")).read().splitlines()

    def block(lines, first):
        i = next(k for k, ln in enumerate(lines) if ln.strip().startswith(first))
        j = next(k for k in range(i, len(lines)) if lines[k].strip() == "++n;")
        return _norm(lines[i : j + 2])

    wb = block(w, "if (n == TOP_K && k <= keys[TOP_K - 1])")
    bb = block(b, "if (n == TOP_K && k <= keys[TOP_K - 1])")
    assert [ln.replace("ids[pos] = e;", "ids[pos] = e0 + j;") for ln in wb] == bb, (wb, bb)
    assert sum("sort_key(frow[j])" in ln for ln in b) == 1
