# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (CPU) tests of the Stage 7a kda_pre program description: an emulation of the program's data movement
(ROWS mode row-pair packing with the kernels' tile_off offsets, packed taps, the history shift of active users incl.
mixed active / inactive pairs, the writer's row-pair output placement; TILES mode whole tiles) reproduces
`kda_pre.reference_step`, and `reference_step` matches the op chain's arithmetic (FusedKDA.decode, in torch)."""

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt import kda_pre as K

W, QKV = 8192, 24576


def _face_major(tile):
    return tile.reshape(2, 16, 2, 16).permute(0, 2, 1, 3).reshape(1024)


def _from_face_major(flat):
    return flat.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)


def _inputs(b, seed=0):
    g = torch.Generator().manual_seed(seed)
    bf = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).to(torch.bfloat16)
    qkv, lr = bf(b, QKV), bf(b, 320)
    hist = bf(3, b, QKV)
    taps = bf(4, QKV, sc=0.5).float()
    w_fb, w_gb = bf(128, W, sc=0.1), bf(128, W, sc=0.1)
    dt = torch.randn(W, generator=g)
    rate = torch.rand(W, generator=g) + 0.1
    active = torch.tensor([(i * 7 + seed) % 3 != 0 for i in range(b)])
    return qkv, lr, hist, taps, w_fb, w_gb, dt, rate, active


def emulate_rows(qkv, lr, hist, taps, active):
    """ROWS mode conv + history with the kernels' index arithmetic (bf16 tiles in face-major memory)."""
    b = qkv.shape[0]
    pad = lambda x: torch.cat([x, torch.zeros(32 - x.shape[0], x.shape[1], dtype=x.dtype)])
    srcs = [pad(hist[0]), pad(hist[1]), pad(hist[2]), pad(qkv)]  # [32, 24576] each
    tile_mem = lambda x, c: _face_major(x[:, 32 * c : 32 * c + 32].float())
    tp = K.packed_taps(taps).reshape(4, K.N_CORES, 32, 32)
    out = torch.zeros(3, 32, W, dtype=torch.float64)
    hist_mem = {(j, c): tile_mem(srcs[j], c).clone() for j in range(3) for c in range(K.QKV_TILES)}
    for k in range(K.N_CORES):
        c0 = 12 * k
        for u in range((b + 1) // 2):
            packed = [torch.zeros(1024) for _ in range(4)]
            for r in range(12):
                for x in range(4):
                    m = tile_mem(srcs[x], c0 + r)
                    for h in (0, 16):
                        so, do = K.tile_off(2 * u, h), K.tile_off(2 * r, h)
                        packed[x][do : do + 32] = m[so : so + 32]  # 64 B of bf16 = 32 elements = 2 face rows
            pt = [_from_face_major(packed[x]).double() for x in range(4)]
            taps_t = [tp[j, k].double() for j in range(4)]
            acc = sum(pt[j] * taps_t[j] for j in range(4))
            conv = acc * torch.sigmoid(acc)
            cm = _face_major(conv)
            for r in range(12):
                c = c0 + r
                region, page = c // 256, c % 256
                dst = torch.zeros(1024, dtype=torch.float64)
                for h in (0, 16):
                    so, do = K.tile_off(2 * r, h), K.tile_off(2 * u, h)
                    dst[do : do + 32] = cm[so : so + 32]
                tile = _from_face_major(dst)
                for p in range(2):
                    t = 2 * u + p
                    if t < b:
                        out[region, t, 32 * page : 32 * page + 32] = tile[t]
            a0 = bool(active[2 * u])
            a1 = bool(active[2 * u + 1]) if 2 * u + 1 < b else a0
            if a0 or a1:
                for r in range(12):
                    c = c0 + r
                    for j, (new, old) in enumerate(((1, 0), (2, 1), (3, 2))):
                        for h in (0, 16):
                            so, do = K.tile_off(2 * r, h), K.tile_off(2 * u, h)
                            n, o = packed[new][so : so + 32], packed[old][so : so + 32]
                            data = torch.cat([n[:16] if a0 else o[:16], n[16:] if a1 else o[16:]])
                            hist_mem[(j, c)][do : do + 32] = data
    hist_new = torch.stack([torch.cat([_from_face_major(hist_mem[(j, c)])[:b] for c in range(K.QKV_TILES)], 1)
                            for j in range(3)])  # fmt: skip
    return out[0, :b], out[1, :b], out[2, :b], hist_new


@pytest.mark.parametrize("b", [1, 2, 3, 4])
def test_rows_emulation(b):
    qkv, lr, hist, taps, w_fb, w_gb, dt, rate, active = _inputs(b, seed=b)
    ref = K.reference_step(qkv, lr, hist, taps, w_fb, w_gb, dt, rate, -5.0, active)
    q, k, v, hist_new = emulate_rows(qkv, lr, hist, taps, active)
    for name, got in (("q", q), ("k", k), ("v", v)):
        torch.testing.assert_close(got, ref[name], rtol=1e-6, atol=1e-6, msg=name)
    assert torch.equal(hist_new.float(), ref["hist_new"].float())


def test_reference_matches_chain_math():
    """reference_step vs the FusedKDA.decode op chain written in torch (fp32)."""
    b = 3
    qkv, lr, hist, taps, w_fb, w_gb, dt, rate, active = _inputs(b, seed=7)
    ref = K.reference_step(qkv, lr, hist, taps, w_fb, w_gb, dt, rate, -5.0, active)
    m = active.float().reshape(-1, 1)
    prod = hist.float() * taps[:3].reshape(3, 1, -1)
    acc = prod.sum(0) + qkv.float() * taps[3]
    conv = torch.nn.functional.silu(acc)
    f = lr[:, :128].float() @ w_fb.float() + dt
    g = torch.sigmoid(f * rate) * (-5.0 * m)
    beta = torch.sigmoid(lr[:, 256:320].float()) * m
    gate_out = lr[:, 128:256].float() @ w_gb.float()
    torch.testing.assert_close(ref["q"].float(), conv[:, :W], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ref["v"].float(), conv[:, 2 * W :], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ref["g"].float(), g, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ref["beta"].float(), beta, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ref["gate_out"].float(), gate_out, rtol=1e-4, atol=1e-4)


def emulate_rows_pack(qkv, hist, taps, active, heads, spread, pack, rsplit=1):
    """ROWS mode on a TP shard (heads, spread) with KDA_PACK = pack user rows per packed tile: the kernels' unit /
    pair / packed-row-pair arithmetic (reader: pair up = UP u + pp -> packed row pair UP r + pp; writer: back; the
    history write-back per pair, mixed active / inactive pairs included)."""
    b, up_n = qkv.shape[0], pack // 2
    conv_n, qkv_tiles, w_tiles = 12 // spread, 12 * heads, 4 * heads
    pad = lambda x: torch.cat([x, torch.zeros(32 - x.shape[0], x.shape[1], dtype=x.dtype)])
    srcs = [pad(hist[0]), pad(hist[1]), pad(hist[2]), pad(qkv)]
    tile_mem = lambda x, c: _face_major(x[:, 32 * c : 32 * c + 32].float())
    tp = K.packed_taps(taps, heads, spread, pack).reshape(4, heads * spread, 32, 32)
    out = torch.zeros(3, 32, 128 * heads, dtype=torch.float64)
    hist_mem = {(j, c): tile_mem(srcs[j], c).clone() for j in range(3) for c in range(qkv_tiles)}
    pairs_all = (b + 1) // 2
    per = -(-pairs_all // rsplit)
    for kk in range(heads * spread * rsplit):
        k, rg = kk % (heads * spread), kk // (heads * spread)  # column core, row group (KDA_RSPLIT)
        c0 = conv_n * k
        p0 = min(rg * per, pairs_all)
        pairs = min(p0 + per, pairs_all) - p0
        for u in range(-(-pairs // up_n)):
            np_ = min(up_n, pairs - up_n * u)
            packed = [torch.zeros(1024) for _ in range(4)]
            for r in range(conv_n):
                for pp in range(np_):
                    up, rp = p0 + up_n * u + pp, up_n * r + pp
                    for x in range(4):
                        m = tile_mem(srcs[x], c0 + r)
                        for h in (0, 16):
                            so, do = K.tile_off(2 * up, h), K.tile_off(2 * rp, h)
                            packed[x][do : do + 32] = m[so : so + 32]
            pt = [_from_face_major(packed[x]).double() for x in range(4)]
            acc = sum(pt[j] * tp[j, k].double() for j in range(4))
            cm = _face_major(acc * torch.sigmoid(acc))
            for r in range(conv_n):
                c = c0 + r
                region, page = c // w_tiles, c % w_tiles
                for pp in range(np_):
                    up, rp = p0 + up_n * u + pp, up_n * r + pp
                    dst = torch.zeros(1024, dtype=torch.float64)
                    for h in (0, 16):
                        so, do = K.tile_off(2 * rp, h), K.tile_off(2 * up, h)
                        dst[do : do + 32] = cm[so : so + 32]
                    tile = _from_face_major(dst)
                    for p in range(2):
                        t = 2 * up + p
                        if t < b:
                            out[region, t, 32 * page : 32 * page + 32] = tile[t]
            for pp in range(np_):
                up = p0 + up_n * u + pp
                a0 = bool(active[2 * up])
                a1 = bool(active[2 * up + 1]) if 2 * up + 1 < b else a0
                if not (a0 or a1):
                    continue
                for r in range(conv_n):
                    c, rp = c0 + r, up_n * r + pp
                    for j, (new, old) in enumerate(((1, 0), (2, 1), (3, 2))):
                        for h in (0, 16):
                            so, do = K.tile_off(2 * rp, h), K.tile_off(2 * up, h)
                            n, o = packed[new][so : so + 32], packed[old][so : so + 32]
                            data = torch.cat([n[:16] if a0 else o[:16], n[16:] if a1 else o[16:]])
                            hist_mem[(j, c)][do : do + 32] = data
    hist_new = torch.stack([torch.cat([_from_face_major(hist_mem[(j, c)])[:b] for c in range(qkv_tiles)], 1)
                            for j in range(3)])  # fmt: skip
    return out[:, :b], hist_new


@pytest.mark.parametrize("rsplit", [1, 2])
@pytest.mark.parametrize("pack", [2, 4, 10])
@pytest.mark.parametrize("b", [1, 3, 8, 21, 32])
def test_rows_pack_emulation(b, pack, rsplit):
    """KDA_PACK on the TP8 shard (8 heads, spread 4: 3 conv tiles per core): conv output and history = the oracle."""
    heads, spread = 8, 4
    qkv, lr, hist, taps, w_fb, w_gb, dt, rate, active = _inputs(b, seed=b + pack)
    n = 3 * heads * 128
    qkv, hist, taps = qkv[:, :n], hist[:, :, :n], taps[:, :n]
    out, hist_new = emulate_rows_pack(qkv, hist, taps, active, heads, spread, pack, rsplit)
    acc = sum(hist[j].double() * taps[j].double() for j in range(3)) + qkv.double() * taps[3].double()
    conv = acc * torch.sigmoid(acc)
    w = 128 * heads
    for i in range(3):
        torch.testing.assert_close(out[i], conv[:, i * w : (i + 1) * w], rtol=1e-6, atol=1e-6)
    ref = hist.clone()
    ref[0, active], ref[1, active], ref[2, active] = hist[1, active], hist[2, active], qkv[active]
    assert torch.equal(hist_new.float(), ref.float())
