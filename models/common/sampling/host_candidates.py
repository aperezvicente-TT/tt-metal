# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Decode-side candidates for host top-k/top-p sampling.

Instead of the full logits row, a decode step returns each row's top-K logits, their vocabulary ids
and the logsumexp of the temperature-scaled row. A host sampler decides top-k/top-p from these
exactly whenever the kept set lies inside the top K (see vllm-tt-plugin
``host_sampler.sample_from_candidates``) and reads the full logits only for the rare rows it cannot.
"""

from typing import NamedTuple

import torch

import ttnn

HOST_CANDIDATE_K = 256
# Additive mask for padded vocabulary columns. Finite so that scaling by 1/T (T >= 1e-5, vLLM's
# greedy threshold) and exp() stay well defined; far below any real logit.
_PAD_LOGIT = -1.0e30


class DecodeCandidates(NamedTuple):
    """One decode step's candidates. ``logits`` is the bf16 TILE row the candidates were taken from,
    kept on device for host fallback reads; it is overwritten by the next replay of the same trace."""

    values: object  # [1, 1, 32, K] bf16 TILE, raw logits, sorted descending
    indices: object  # [1, 1, 32, K] uint32 TILE
    lse: object  # [1, 1, 32, 1] fp32 TILE, logsumexp(logits / T) over the valid vocabulary
    logits: object


def host_candidates_supported(mesh_device, k=HOST_CANDIDATE_K) -> bool:
    """ttnn.topk takes the multi-core topk_large_indices route only on Blackhole (64 < k <= 2048);
    elsewhere a full-vocabulary top-256 falls to a single-core kernel (~17 ms at 128K)."""
    return mesh_device.get_num_devices() == 1 and "blackhole" in ttnn.get_arch_name() and 64 < k <= 2048


def make_vocab_mask(mesh_device, vocab_size, width, rows=32):
    """[1, 1, rows, width] bf16 additive mask hiding columns >= vocab_size, or None without padding."""
    if width <= vocab_size:
        return None
    mask = torch.zeros(1, 1, rows, width)
    mask[..., vocab_size:] = _PAD_LOGIT
    return ttnn.from_torch(
        mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def inv_temperature_host(temperature, mesh_device, rows=32):
    """Host tile tensor [1, 1, rows, 1] fp32 of 1/T; rows without a positive T (greedy, padding) get 1."""
    t = torch.as_tensor(temperature, dtype=torch.float32).reshape(-1)
    inv = torch.ones(rows, dtype=torch.float32)
    inv[: t.numel()] = torch.where(t > 0, 1.0 / t.clamp_min(1e-30), torch.ones_like(t))
    return ttnn.from_torch(
        inv.reshape(1, 1, rows, 1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def decode_candidates(logits, inv_temp, k=HOST_CANDIDATE_K, vocab_mask=None):
    """Device ops only, traceable. ``logits`` [1, 1, 32, W] TILE (bf16 or bf8_b), ``inv_temp`` the
    persistent [1, 1, 32, 1] fp32 device tensor from ``inv_temperature_host``."""
    # fp32_dest_acc_en selects ttnn.sum's accurate fp32 (SFPU) accumulation instead of TF32 inputs.
    sum_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    if logits.is_sharded():
        logits = ttnn.to_memory_config(logits, ttnn.DRAM_MEMORY_CONFIG)
    if logits.dtype != ttnn.bfloat16:
        logits = ttnn.typecast(logits, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if vocab_mask is not None:
        logits = ttnn.add(logits, vocab_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    values, indices = ttnn.topk(logits, k=k, dim=-1, largest=True, sorted=True)

    # The top value is the row max; logsumexp(s*x) = s*m + log(sum(exp(s*(x - m)))) for s = 1/T > 0.
    row_max = ttnn.typecast(ttnn.slice(values, [0, 0, 0, 0], [1, 1, values.shape[2], 1]), ttnn.float32)
    x = ttnn.typecast(logits, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.subtract(x, row_max, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.multiply(x, inv_temp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.exp(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    total = ttnn.sum(x, dim=-1, keepdim=True, compute_kernel_config=sum_config)
    ttnn.deallocate(x)
    lse = ttnn.add(ttnn.log(total), ttnn.multiply(row_max, inv_temp))
    return DecodeCandidates(values, indices, lse, logits)


def candidates_to_torch(candidates: DecodeCandidates, rows):
    """Host (or device) candidate tensors to torch: values [rows, K] fp32, indices [rows, K] int64,
    lse [rows] fp32."""

    def first_device(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0])

    values = first_device(candidates.values).float().reshape(-1, candidates.values.shape[-1])[:rows]
    indices = first_device(candidates.indices).to(torch.int64).reshape(-1, candidates.indices.shape[-1])[:rows]
    lse = first_device(candidates.lse).float().reshape(-1, candidates.lse.shape[-1])[:rows, 0]
    return values, indices, lse
