# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the fused ``ttnn.experimental.deepseek.mix_streams`` op.

The op implements the mHC stream mix (DeepSeek-V4 / GLM-5.3 ``hc_expand``, transformers
modeling_glm5_next.py:1335-1346)::

    new_streams = post[..., None] * sublayer_out[..., None, :] + comb.T @ streams

as a single device kernel. This test is self-contained: it builds random inputs,
runs the op on device and compares against a torch reference of exactly that math.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

PCC_THRESHOLD = 0.999


def _torch_reference(
    post: torch.Tensor, comb: torch.Tensor, sublayer_out: torch.Tensor, streams: torch.Tensor
) -> torch.Tensor:
    placement = post * sublayer_out  # [B,S,hc,1] * [B,S,1,D] -> [B,S,hc,D]
    mixed = torch.matmul(comb.transpose(-1, -2), streams)
    return placement + mixed


@pytest.mark.parametrize("batch_size, seq_len", ((1, 1), (2, 1), (1, 4)))
@pytest.mark.parametrize("hc", (4, 32))
@pytest.mark.parametrize("d", (512, 4096))
def test_mix_streams_op(device, reset_seeds, batch_size, seq_len, hc, d):
    post = torch.rand(batch_size, seq_len, hc, 1, dtype=torch.float32) * 2.0
    comb = torch.softmax(torch.randn(batch_size, seq_len, hc, hc, dtype=torch.float32), dim=-1)
    sublayer_out = torch.randn(batch_size, seq_len, 1, d, dtype=torch.float32)
    streams = torch.randn(batch_size, seq_len, hc, d, dtype=torch.float32)

    reference = _torch_reference(post, comb, sublayer_out, streams)

    def to_tt(x):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_tt = ttnn.experimental.deepseek.mix_streams(to_tt(post), to_tt(comb), to_tt(sublayer_out), to_tt(streams))
    got = ttnn.to_torch(out_tt).float()

    assert tuple(out_tt.shape) == tuple(reference.shape)
    passing, pcc_message = comp_pcc(reference, got, pcc=PCC_THRESHOLD)
    logger.info(f"[mix_streams] {comp_allclose(reference, got)}")
    logger.info(f"[mix_streams] PCC: {pcc_message}")
    assert passing, f"mix_streams PCC < {PCC_THRESHOLD} (B={batch_size}, S={seq_len}, hc={hc}, D={d}): {pcc_message}"


# fp32 inputs: the fp32 fused kernel (bf16-split operands, exact part products, fp32 DST accumulation).
@pytest.mark.parametrize("batch_size, seq_len", ((1, 1), (2, 1), (1, 4)))
@pytest.mark.parametrize("hc", (4, 32))
@pytest.mark.parametrize("d", (512, 4096))
def test_mix_streams_op_fp32(device, reset_seeds, batch_size, seq_len, hc, d):
    post = torch.rand(batch_size, seq_len, hc, 1, dtype=torch.float32) * 2.0
    comb = torch.softmax(torch.randn(batch_size, seq_len, hc, hc, dtype=torch.float32), dim=-1)
    sublayer_out = torch.randn(batch_size, seq_len, 1, d, dtype=torch.float32)
    streams = torch.randn(batch_size, seq_len, hc, d, dtype=torch.float32)

    reference = _torch_reference(post, comb, sublayer_out, streams)

    def to_tt(x):
        return ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out_tt = ttnn.experimental.deepseek.mix_streams(to_tt(post), to_tt(comb), to_tt(sublayer_out), to_tt(streams))
    got = ttnn.to_torch(out_tt).float()

    assert out_tt.dtype == ttnn.float32
    assert tuple(out_tt.shape) == tuple(reference.shape)
    max_diff = (got - reference).abs().max().item()
    logger.info(f"[mix_streams_fp32] {comp_allclose(reference, got)} max |diff| {max_diff}")
    assert torch.allclose(got, reference, rtol=0, atol=1e-5), f"max |diff| {max_diff}"


_DTYPES = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}


def _random_mix_inputs(batch_size, seq_len, hc, d):
    post = torch.rand(batch_size, seq_len, hc, 1, dtype=torch.float32) * 2.0
    comb = torch.softmax(torch.randn(batch_size, seq_len, hc, hc, dtype=torch.float32), dim=-1)
    sublayer_out = torch.randn(batch_size, seq_len, 1, d, dtype=torch.float32)
    streams = torch.randn(batch_size, seq_len, hc, d, dtype=torch.float32)
    return post, comb, sublayer_out, streams


def _run_mix(device, dtype, inputs):
    tt_dtype = _DTYPES[dtype]
    reference = _torch_reference(*inputs)
    to_tt = lambda x: ttnn.from_torch(x, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = ttnn.experimental.deepseek.mix_streams(*(to_tt(x) for x in inputs))
    got = ttnn.to_torch(out_tt).float()
    assert tuple(out_tt.shape) == tuple(reference.shape)
    passing, pcc_message = comp_pcc(reference, got, pcc=PCC_THRESHOLD)
    max_diff = (got - reference).abs().max().item()
    logger.info(f"[mix_streams {dtype}] PCC {pcc_message} max |diff| {max_diff}")
    assert passing, f"PCC {pcc_message}"
    if dtype == "fp32":
        assert max_diff < 1e-5, f"max |diff| {max_diff}"
    return out_tt


# Token counts whose output-tile ranges straddle token boundaries on a core (bf16: 32 cores, e.g. T=5, D=512 ->
# 80 tiles, 2-3 per core; fp32: T=40 -> 640 tiles over the whole grid) and T > 32.
@pytest.mark.parametrize("batch_size, seq_len", ((5, 1), (40, 1), (3, 16)))
@pytest.mark.parametrize("dtype", ("bf16", "fp32"))
def test_mix_streams_op_token_boundaries(device, reset_seeds, batch_size, seq_len, dtype):
    _run_mix(device, dtype, _random_mix_inputs(batch_size, seq_len, 4, 512))


# D not tile aligned -> the composite fallback (bf16: comb^T matmul; fp32: SFPU eltwise broadcasts).
@pytest.mark.parametrize("dtype", ("bf16", "fp32"))
def test_mix_streams_op_composite_fallback(device, reset_seeds, dtype):
    _run_mix(device, dtype, _random_mix_inputs(2, 1, 4, 528))


# Program-cache hit path of the fused kernels: repeated calls with freshly allocated inputs (earlier tensors kept
# alive, so the buffer addresses move) reuse one program and still match; a new token count adds exactly one entry.
@pytest.mark.parametrize("dtype", ("bf16", "fp32"))
def test_mix_streams_op_program_cache(device, reset_seeds, dtype):
    keep = []
    for call in range(3):
        keep.append(_run_mix(device, dtype, _random_mix_inputs(2, 1, 4, 512)))
        keep.append(ttnn.from_torch(torch.zeros(1, 1, 32, 32 * (call + 1)), layout=ttnn.TILE_LAYOUT, device=device))
        assert device.num_program_cache_entries() == 1, device.num_program_cache_entries()
    for _ in range(2):
        keep.append(_run_mix(device, dtype, _random_mix_inputs(8, 1, 4, 512)))
        assert device.num_program_cache_entries() == 2, device.num_program_cache_entries()
