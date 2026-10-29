// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fused_recurrent_kda.hpp"

#include <bit>
#include <cmath>
#include <functional>

#include "device/fused_recurrent_kda_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::kda {

namespace {

ttnn::Tensor as_f32_tile(const ttnn::Tensor& x) {
    ttnn::Tensor t = x.layout() == Layout::TILE ? x : ttnn::to_layout(x, Layout::TILE);
    return t.dtype() == DataType::FLOAT32 ? t : ttnn::typecast(t, DataType::FLOAT32);
}

}  // namespace

std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> fused_recurrent_kda(
    const ttnn::Tensor& q_in,
    const ttnn::Tensor& k_in,
    const ttnn::Tensor& v_in,
    const ttnn::Tensor& g_in,
    const ttnn::Tensor& beta_in,
    std::optional<float> scale_opt,
    const std::optional<ttnn::Tensor>& initial_state,
    bool output_final_state,
    bool output_per_token_state,
    bool use_qk_l2norm,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    bool inplace_state) {
    const auto& qs = q_in.logical_shape();
    const auto& bs = beta_in.logical_shape();
    TT_FATAL(qs.rank() == 3 || qs.rank() == 4, "fused_recurrent_kda: q must be [B,T,H,K] or [B,T,H*K], got {}", qs);
    // Decode layout [1, 1, B, H*D] / beta [1, 1, B, H]: B users, one token each (every operand is addressed by its own
    // geometry, so no relayout is needed).
    const bool decode_rows = qs.rank() == 4 && bs.rank() == 4 && qs[0] == 1 && qs[1] == 1 && bs[0] == 1 && bs[1] == 1;
    TT_FATAL(
        bs.rank() == 3 || decode_rows,
        "fused_recurrent_kda: beta must be [B,T,H] (or [1,1,B,H] with q [1,1,B,H*K]), got {}",
        bs);
    const uint32_t B = decode_rows ? qs[2] : qs[0];
    const uint32_t T = decode_rows ? 1u : qs[1];
    const uint32_t H = decode_rows ? bs[3] : bs[2];
    TT_FATAL(
        decode_rows ? bs[2] == B : (bs[0] == B && bs[1] == T),
        "fused_recurrent_kda: beta {} does not match q {} in B, T",
        bs,
        qs);
    if (inplace_state) {
        TT_FATAL(
            initial_state.has_value() && output_final_state && !output_per_token_state,
            "fused_recurrent_kda: inplace_state needs initial_state and output_final_state (not per-token states)");
        TT_FATAL(
            initial_state->dtype() == DataType::FLOAT32 && initial_state->layout() == Layout::TILE,
            "fused_recurrent_kda: inplace_state needs an fp32 TILE initial_state (it is the output buffer)");
    }
    const uint64_t rows = static_cast<uint64_t>(B) * T * H;
    TT_FATAL(qs.volume() % rows == 0, "fused_recurrent_kda: q {} is not B*T*H = {} rows", qs, rows);
    TT_FATAL(v_in.logical_shape().volume() % rows == 0, "fused_recurrent_kda: v is not B*T*H = {} rows", rows);
    const uint32_t K = qs.volume() / rows;
    const uint32_t V = v_in.logical_shape().volume() / rows;

    const float scale = scale_opt.value_or(1.0f / std::sqrt(static_cast<float>(K)));
    auto* dev = q_in.device();
    ttnn::Tensor s0 =
        initial_state.has_value()
            ? as_f32_tile(*initial_state)
            : ttnn::zeros(
                  ttnn::Shape({B, H, K, V}), DataType::FLOAT32, Layout::TILE, std::ref(*dev), ttnn::DRAM_MEMORY_CONFIG);

    const auto kernel_cfg = init_device_compute_kernel_config(
        dev->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);

    const bool want_state = output_final_state || output_per_token_state;
    const prim::FusedRecurrentKdaParams params{
        .batch = B,
        .tokens = T,
        .heads = H,
        .key_dim = K,
        .value_dim = V,
        .scale_bits = std::bit_cast<uint32_t>(scale),
        .use_qk_l2norm = use_qk_l2norm,
        .output_state = want_state,
        .output_per_token_state = output_per_token_state,
        .state_in_place = inplace_state,
        .output_mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        .compute_kernel_config = kernel_cfg,
    };
    auto res = prim::fused_recurrent_kda(
        as_f32_tile(q_in), as_f32_tile(k_in), as_f32_tile(v_in), as_f32_tile(g_in), as_f32_tile(beta_in), s0, params);
    std::optional<ttnn::Tensor> state;
    if (want_state) {
        state = res[1];
    }
    return {res[0], state};
}

}  // namespace ttnn::experimental::kda
