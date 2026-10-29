// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa/sparse_sdpa.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "block_cyclic_layout.hpp"  // ttnn::prim::BlockCyclicLayout (shared)
#include <optional>

namespace ttnn::prim {

// Sparse MLA prefill (DeepSeek DSA).
struct SparseSDPAParams {
    float scale = 1.0f;  // compile-time (folded into the program hash)
    uint32_t v_dim;      // width of V (= leading v_dim cols of the K_DIM-wide KV cache); the output width
    transformer::SparseKVFormat kv_format;
    uint32_t k_chunk_size = 128;
    DeviceComputeKernelConfig compute_kernel_config;
    // Indexed KV cache: when set, kv is a [B,1,T,K_DIM] shared cache and this selects the batch slot to
    // attend to (the gather page ids are offset by cache_batch_idx * T). It is a DYNAMIC runtime arg
    // (excluded from the program hash, re-applied every dispatch), so changing it does NOT recompile.
    std::optional<uint32_t> cache_batch_idx = std::nullopt;
    bool has_indexed_kv_cache() const { return cache_batch_idx.has_value(); }
    // The remap configuration is compile-time; T is part of the program hash for this path.
    std::optional<BlockCyclicLayout> block_cyclic = std::nullopt;
    bool has_block_cyclic() const { return block_cyclic.has_value(); }
    bool has_scaled_kv() const { return kv_format == transformer::SparseKVFormat::SCALED_FP8; }
    // Split-K attention across devices (the sequence-sharded DSA prefill): also emit the per-(token, head) running
    // max m (the UNSCALED score max the kernel subtracts) and the finalised row-sum l = sum exp((s - m) scale) of the
    // keys this call attended, as a second output [1, 1, S, 2 H] fp32 ROW_MAJOR (row = [m_0 .. m_H-1, l_0 .. l_H-1]).
    // The owner of a head merges the partials of C devices with M = max_c m_c, w_c = exp((m_c - M) scale) l_c /
    // sum_c' exp((m_c' - M) scale) l_c', o = sum_c w_c o_c. Compile-time (in the program hash); the kernels are
    // unchanged when off.
    bool emit_stats = false;
};

struct SparseSDPAInputs {
    Tensor q;        // [1, H, S, K_DIM] bf16/fp8_e4m3 ROW_MAJOR  (K_DIM = head dim, e.g. 576)
    Tensor kv;       // Plain [B,1,T,K_DIM] or packed scaled-FP8 rows; format is explicit in SparseSDPAParams
    Tensor indices;  // [1, 1, S, TOPK] uint32 ROW_MAJOR  (0xFFFFFFFF = masked)
    // [1, 1, 1, H] ROW_MAJOR BF16; scaled like QK.
    std::optional<Tensor> attention_sink;
};

}  // namespace ttnn::prim
