// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::kda {

/**
 * Token-by-token Kimi Delta Attention recurrence (GLM-5.3-Flash `recurrent_kda`) for decode (T = 1) and
 * speculative verify (T = K + 1). One work item per (b, h) with the fp32 state resident in L1 for the step.
 *
 *   q, k  [B, T, H, K] or [B, T, H*K]   (L2-normalised in-kernel when use_qk_l2norm)
 *   v     [B, T, H, V] or [B, T, H*V]
 *   g     [B, T, H, K] or [B, T, H*K]   per-key-channel log decay (<= 0); exp(g) is taken in-kernel
 *   beta  [B, T, H]
 *   initial_state [B, H, K, V] (zeros when absent)
 *
 * Returns o [B, T, H, V] fp32 and, when requested, the state: [B, T, H, K, V] (output_per_token_state, the state
 * after every token) or [B, H, K, V] (output_final_state).
 *
 * Decode layout: q/k/v/g [1, 1, B, H*D] (users as rows, the model's natural decode layout) with beta [1, 1, B, H]
 * is accepted as B users with T = 1 (no head-split / user-split relayout before the call).
 * inplace_state: write the final state into `initial_state`'s buffer (fp32 TILE, required) and return it, so a
 * persistent per-user state needs no copy-back (trace-friendly decode). Each (b, h) item reads its state once
 * before writing it, and items are disjoint, so in-place is race-free.
 */
std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> fused_recurrent_kda(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    bool output_final_state = false,
    bool output_per_token_state = false,
    bool use_qk_l2norm = true,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    bool inplace_state = false);

}  // namespace ttnn::experimental::kda
