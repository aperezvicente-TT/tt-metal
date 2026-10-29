# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the fused ``ttnn.experimental.deepseek.fused_hyperconnection`` op.

The op implements the ``pre`` / ``post`` / ``comb`` / ``collapsed`` portion of
the mHC hyper-connection (DeepSeek-V4 / GLM-5.3, transformers modeling_glm5_next.py:296-329) given the packed linear
projection ``fused_w`` (shape ``[1, 1, T, (2+H)*H]``). It splits ``fused_w`` into its
``pre_w`` / ``post_w`` / ``comb_w`` slices inside the ``fused_hyperconnection_pre_post``
device kernel; ``pre_w`` / ``post_w`` are consumed in-place and ``comb_w`` is returned
already laid out as the ``[1, T, H, H]`` comb matrix. This test is fully self-contained:
it builds a random ``fused_w`` + biases, runs the op on device, and compares against a
torch reference of exactly that math (no HuggingFace / RMSNorm / matmul involved).
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc


PCC_THRESHOLD = 0.98


def _torch_reference(
    hidden_streams: torch.Tensor,
    fused_w: torch.Tensor,
    pre_b: torch.Tensor,
    post_b: torch.Tensor,
    comb_b: torch.Tensor,
    hc: int,
    iters: int,
    pre_scale: float,
    post_scale: float,
    comb_scale: float,
    eps: float,
):
    """mHC pre / post / comb / collapsed (modeling_glm5_next.py:296-329), in torch float32."""
    b, s, _, d = hidden_streams.shape
    t = b * s

    pre_w, post_w, comb_w = torch.split(fused_w, [hc, hc, hc * hc], dim=-1)
    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + eps  # [1,1,T,H]
    post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)  # [1,1,T,H]

    comb_logits = (comb_w * comb_scale + comb_b).reshape(1, t, hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)  # column
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)  # row
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)  # column

    hs = hidden_streams.reshape(1, t, hc, d)
    pre_col = pre.reshape(1, t, hc, 1)
    collapsed = (hs * pre_col).sum(dim=-2, keepdim=True)  # [1,T,1,D]

    post = post.reshape(b, s, hc, 1)
    comb = comb.reshape(b, s, hc, hc)
    collapsed = collapsed.reshape(b, s, 1, d)
    return post, comb, collapsed


# The T = B * S tokens are independent and are spread across the core grid, so batched decode
# (B > 1) and multi-token prefill (S > 1) run through the same two device ops. T = 40 crosses a
# tile row of fused_w (token t lives in row t % 32 of tile row t / 32), and T = 200 exceeds the
# core grid so each core loops over several tokens and restages fused_w as it crosses tile rows.
@pytest.mark.parametrize("batch_size, seq_len", ((1, 1), (8, 1), (1, 4), (5, 8), (25, 8)))
@pytest.mark.parametrize("sinkhorn_iters", (1, 20))
def test_fused_hyperconnection_op(device, reset_seeds, batch_size, seq_len, sinkhorn_iters):
    hc = 4  # number of streams (hc_mult)
    d = 512  # hidden_size
    t = batch_size * seq_len
    eps = 1.0e-6
    pre_scale, post_scale, comb_scale = 1.0, 1.0, 1.0

    hidden_streams = torch.randn(batch_size, seq_len, hc, d, dtype=torch.float32)
    fused_w = torch.randn(1, 1, t, (2 + hc) * hc, dtype=torch.float32) * 0.5
    pre_b = torch.randn(1, 1, 1, hc, dtype=torch.float32) * 0.1
    post_b = torch.randn(1, 1, 1, hc, dtype=torch.float32) * 0.1
    comb_b = torch.randn(1, 1, 1, hc * hc, dtype=torch.float32) * 0.1

    ref_post, ref_comb, ref_collapsed = _torch_reference(
        hidden_streams,
        fused_w,
        pre_b,
        post_b,
        comb_b,
        hc,
        sinkhorn_iters,
        pre_scale,
        post_scale,
        comb_scale,
        eps,
    )

    def to_tt(x):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    post_tt, comb_tt, collapsed_tt = ttnn.experimental.deepseek.fused_hyperconnection(
        to_tt(hidden_streams),
        fused_w=to_tt(fused_w),
        pre_bias=to_tt(pre_b),
        post_bias=to_tt(post_b),
        comb_bias=to_tt(comb_b),
        num_streams=hc,
        sinkhorn_iters=sinkhorn_iters,
        pre_scale=pre_scale,
        post_scale=post_scale,
        comb_scale=comb_scale,
        eps=eps,
    )

    outputs = {
        "post": (ttnn.to_torch(post_tt).reshape(ref_post.shape).float(), ref_post),
        "comb": (ttnn.to_torch(comb_tt).reshape(ref_comb.shape).float(), ref_comb),
        "collapsed": (ttnn.to_torch(collapsed_tt).reshape(ref_collapsed.shape).float(), ref_collapsed),
    }

    all_pass = True
    msgs = []
    for name, (got, ref) in outputs.items():
        passing, pcc_message = comp_pcc(ref, got, pcc=PCC_THRESHOLD)
        logger.info(f"[fused_hyperconnection:{name}] {comp_allclose(ref, got)}")
        logger.info(f"[fused_hyperconnection:{name}] PCC: {pcc_message}")
        all_pass = all_pass and passing
        if not passing:
            msgs.append(f"{name}: {pcc_message}")

    assert all_pass, (
        f"fused_hyperconnection PCC < {PCC_THRESHOLD} "
        f"(batch={batch_size}, seq={seq_len}, iters={sinkhorn_iters}): {'; '.join(msgs)}"
    )


# fp32 mode: all inputs FLOAT32 -> the pre / post / Sinkhorn path runs in fp32 DST on the SFPU.
@pytest.mark.parametrize("batch_size, seq_len", ((1, 1), (8, 1), (1, 4), (5, 8), (25, 8)))
@pytest.mark.parametrize("sinkhorn_iters", (1, 20))
def test_fused_hyperconnection_op_fp32(device, reset_seeds, batch_size, seq_len, sinkhorn_iters):
    hc = 4
    d = 512
    t = batch_size * seq_len
    eps = 1.0e-6
    pre_scale, post_scale, comb_scale = 0.06, 0.045, 0.09

    hidden_streams = torch.randn(batch_size, seq_len, hc, d, dtype=torch.float32)
    fused_w = torch.randn(1, 1, t, (2 + hc) * hc, dtype=torch.float32) * 4.0
    pre_b = torch.randn(1, 1, 1, hc, dtype=torch.float32) * 4.0
    post_b = torch.randn(1, 1, 1, hc, dtype=torch.float32)
    comb_b = torch.where(torch.eye(hc, dtype=torch.bool), 0.0, -8.0).reshape(1, 1, 1, hc * hc)

    ref_post, ref_comb, ref_collapsed = _torch_reference(
        hidden_streams, fused_w, pre_b, post_b, comb_b, hc, sinkhorn_iters, pre_scale, post_scale, comb_scale, eps
    )

    def to_tt(x):
        return ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    post_tt, comb_tt, collapsed_tt = ttnn.experimental.deepseek.fused_hyperconnection(
        to_tt(hidden_streams),
        fused_w=to_tt(fused_w),
        pre_bias=to_tt(pre_b),
        post_bias=to_tt(post_b),
        comb_bias=to_tt(comb_b),
        num_streams=hc,
        sinkhorn_iters=sinkhorn_iters,
        pre_scale=pre_scale,
        post_scale=post_scale,
        comb_scale=comb_scale,
        eps=eps,
    )
    assert post_tt.dtype == comb_tt.dtype == collapsed_tt.dtype == ttnn.float32

    got_post = ttnn.to_torch(post_tt).reshape(ref_post.shape)
    got_comb = ttnn.to_torch(comb_tt).reshape(ref_comb.shape)
    got_collapsed = ttnn.to_torch(collapsed_tt).reshape(ref_collapsed.shape)
    # post / comb are computed in fp32 on the SFPU; the collapse is an FPU matmul whose source registers keep
    # fewer mantissa bits than fp32, hence the looser bound relative to the output scale.
    for name, got, ref, atol in (
        ("post", got_post, ref_post, 1e-6),
        ("comb", got_comb, ref_comb, 1e-6),
        ("collapsed", got_collapsed, ref_collapsed, 4e-3 * ref_collapsed.abs().max().item()),
    ):
        passing, pcc_message = comp_pcc(ref, got, pcc=0.99999)
        logger.info(f"[fused_hyperconnection_fp32:{name}] {comp_allclose(ref, got)} PCC: {pcc_message}")
        assert passing, f"{name}: PCC {pcc_message}"
        assert torch.allclose(got, ref, rtol=0, atol=atol), f"{name}: max |diff| {(got - ref).abs().max().item()}"


_DTYPES = {"bf16": (ttnn.bfloat16, 0.98), "fp32": (ttnn.float32, 0.99999)}
# max |collapsed - ref| relative to max |ref|: PCC alone is blind to a constant offset (e.g. padding leaking into
# the collapse adds the same amount to every element).
_COLLAPSED_REL_TOL = {ttnn.bfloat16: 0.03, ttnn.float32: 0.005}


def _random_case(batch_size, seq_len, hc, d, fused_w_scale=4.0, comb_bias=None):
    t = batch_size * seq_len
    hidden_streams = torch.randn(batch_size, seq_len, hc, d, dtype=torch.float32)
    fused_w = torch.randn(1, 1, t, (2 + hc) * hc, dtype=torch.float32) * fused_w_scale
    pre_b = torch.randn(1, 1, 1, hc, dtype=torch.float32) * 4.0
    post_b = torch.randn(1, 1, 1, hc, dtype=torch.float32)
    if comb_bias is None:
        comb_bias = torch.where(torch.eye(hc, dtype=torch.bool), 0.0, -8.0)
    return hidden_streams, fused_w, pre_b, post_b, comb_bias.reshape(1, 1, hc, hc)


def _run_and_check(device, tt_dtype, pcc, case, hc, iters, scales, eps, hidden_tt=None, hidden_ref=None):
    hidden_streams, fused_w, pre_b, post_b, comb_b = case
    hidden_ref = hidden_streams if hidden_ref is None else hidden_ref
    ref = _torch_reference(
        hidden_ref, fused_w, pre_b, post_b, comb_b.reshape(1, 1, 1, hc * hc), hc, iters, *scales, eps
    )

    def to_tt(x):
        return ttnn.from_torch(x, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    outs = ttnn.experimental.deepseek.fused_hyperconnection(
        to_tt(hidden_streams) if hidden_tt is None else hidden_tt,
        fused_w=to_tt(fused_w),
        pre_bias=to_tt(pre_b),
        post_bias=to_tt(post_b),
        comb_bias=to_tt(comb_b),
        num_streams=hc,
        sinkhorn_iters=iters,
        pre_scale=scales[0],
        post_scale=scales[1],
        comb_scale=scales[2],
        eps=eps,
    )
    for name, got_tt, want in zip(("post", "comb", "collapsed"), outs, ref):
        got = ttnn.to_torch(got_tt).reshape(want.shape).float()
        assert torch.isfinite(got).all(), f"{name}: non-finite values"
        passing, pcc_message = comp_pcc(want, got, pcc=pcc)
        rel = ((got - want).abs().max() / want.abs().max()).item()
        logger.info(f"[{name}] PCC: {pcc_message} max |err| / max |ref| {rel:.3g}")
        assert passing, f"{name}: PCC {pcc_message}"
        if name == "collapsed":
            assert rel <= _COLLAPSED_REL_TOL[tt_dtype], f"collapsed: max |err| / max |ref| = {rel:.3g}"
    return outs


# Program-cache hit path of both device ops (FusedPrePost, Sinkhorn): repeated calls with freshly allocated
# inputs (earlier tensors are kept alive, so every call sees new buffer addresses) must reuse one program per op
# and still match; a new token count adds exactly one entry per op. The comb bias is passed pre-shaped
# [1,1,H,H], so no reshape program is involved.
@pytest.mark.parametrize("dtype", ("bf16", "fp32"))
def test_fused_hyperconnection_program_cache(device, reset_seeds, dtype):
    tt_dtype, pcc = _DTYPES[dtype]
    hc, d, iters, eps, scales = 4, 512, 20, 1.0e-6, (0.06, 0.045, 0.09)
    keep = []
    for call in range(3):
        keep.append(_run_and_check(device, tt_dtype, pcc, _random_case(2, 1, hc, d), hc, iters, scales, eps))
        keep.append(ttnn.from_torch(torch.zeros(1, 1, 32, 32 * (call + 1)), layout=ttnn.TILE_LAYOUT, device=device))
        assert device.num_program_cache_entries() == 2, device.num_program_cache_entries()
    for _ in range(2):
        keep.append(_run_and_check(device, tt_dtype, pcc, _random_case(8, 1, hc, d), hc, iters, scales, eps))
        assert device.num_program_cache_entries() == 4, device.num_program_cache_entries()


# hidden_streams whose tile padding rows are not zero (an eltwise op on the streams writes exp(0) = 1 there): the
# collapse must only weight the H valid streams.
@pytest.mark.parametrize("dtype", ("bf16", "fp32"))
def test_fused_hyperconnection_nonzero_stream_padding(device, reset_seeds, dtype):
    tt_dtype, pcc = _DTYPES[dtype]
    hc, d, iters, eps, scales = 4, 512, 20, 1.0e-6, (0.06, 0.045, 0.09)
    case = _random_case(8, 1, hc, d)
    x = case[0] * 0.5
    hidden_tt = ttnn.exp(ttnn.from_torch(x, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device))
    padded = hidden_tt.cpu().to_torch_with_padded_shape().float()
    assert padded.shape[-2] == 32 and padded[..., hc:, :].abs().max() > 0.5, "expected non-zero stream padding"
    _run_and_check(device, tt_dtype, pcc, case, hc, iters, scales, eps, hidden_tt=hidden_tt, hidden_ref=torch.exp(x))


# Every comb logit far below 0 (row max < -88): exp of the tile padding must not overflow (0 - rowmax) into
# inf * 0 = NaN in the masked Sinkhorn passes.
@pytest.mark.parametrize("dtype", ("bf16", "fp32"))
def test_fused_hyperconnection_very_negative_comb_logits(device, reset_seeds, dtype):
    tt_dtype, pcc = _DTYPES[dtype]
    hc, d, iters, eps, scales = 4, 512, 20, 1.0e-6, (0.06, 0.045, 0.09)
    comb_bias = torch.where(torch.eye(hc, dtype=torch.bool), -100.0, -104.0)
    case = _random_case(4, 1, hc, d, comb_bias=comb_bias)
    _run_and_check(device, tt_dtype, pcc, case, hc, iters, scales, eps)
