// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fp32 comb / Sinkhorn-Knopp compute, one independent [H,H] tile per token, entirely in fp32 DST on the SFPU:
//
//   comb = softmax(comb_w * comb_scale + comb_bias, dim=-1) + eps
//   comb = comb / (sum(comb, dim=-2) + eps)                     (initial column pass)
//   repeat sinkhorn_iters-1 times: row pass (dim=-1) then column pass (dim=-2)
//
// Row / column sums and the row max use the SFPU fp32 reduce, the normalisations multiply by the SFPU
// column / row broadcast of the reciprocal, so no value passes through the FPU source registers (which
// hold fewer mantissa bits than fp32). The softmax subtracts the max of the H valid columns (the reader puts
// -FLT_MAX in the bias padding columns of the valid rows, so the logits carry it and exp() underflows there)
// and divides by the plain row sum, matching the reference; the padding rows get +1 in that denominator so
// they stay 0 instead of 0 * inf. Every pass keeps the padding at exactly 0.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t kComb = 0;
constexpr uint32_t kTmp = 1;
constexpr uint32_t kConst = 2;

FORCE_INLINE void load(uint32_t cb, uint32_t dst) {
    copy_init(cb);
    copy_tile(cb, 0, dst);
}

FORCE_INLINE void add_const(uint32_t cb, uint32_t dst) {
    load(cb, kConst);
    add_binary_tile_init();
    add_binary_tile(dst, kConst, dst);
}

// comb /= (sum over `dim` of comb) + eps.
template <ReduceDim dim>
FORCE_INLINE void normalize(uint32_t eps_bits) {
    copy_dest_values_init();
    copy_dest_values<DataFormat::Float32>(kComb, kTmp);
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, dim>(kTmp);
    binop_with_scalar_tile_init();
    add_unary_tile(kTmp, eps_bits);
    recip_tile_init();
    recip_tile(kTmp);
    if constexpr (dim == ReduceDim::REDUCE_ROW) {
        sfpu_bcast_col_init();
        sfpu_mul_bcast_col(kComb, kTmp);
    } else {
        sfpu_bcast_row_init();
        sfpu_mul_bcast_row(kComb, kTmp);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_comb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_comb_bias = get_compile_time_arg_val(1);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(2);
    constexpr uint32_t cb_eps_mask = get_compile_time_arg_val(3);
    constexpr uint32_t cb_pad_one = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t sinkhorn_iters = get_compile_time_arg_val(6);
    constexpr uint32_t comb_scale_bits = get_compile_time_arg_val(7);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(8);

    compute_kernel_hw_startup(cb_comb_w, cb_out);

    for (const uint32_t cb : {cb_comb_bias, cb_mask, cb_eps_mask, cb_pad_one}) {
        cb_wait_front(cb, 1);
    }

    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        cb_wait_front(cb_comb_w, 1);
        tile_regs_acquire();

        // logits = comb_w * comb_scale + comb_bias.
        load(cb_comb_w, kComb);
        binop_with_scalar_tile_init();
        mul_unary_tile(kComb, comb_scale_bits);
        add_const(cb_comb_bias, kComb);

        // softmax over the H valid columns.
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(kComb, kTmp);
        sfpu_reduce_init<PoolType::MAX, DataFormat::Float32>();
        sfpu_reduce<PoolType::MAX, DataFormat::Float32, ReduceDim::REDUCE_ROW>(kTmp);
        sfpu_bcast_col_init();
        sfpu_sub_bcast_col(kComb, kTmp);
        exp_tile_init();
        exp_tile(kComb);
        load(cb_mask, kConst);
        mul_binary_tile_init();
        mul_binary_tile(kComb, kConst, kComb);
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(kComb, kTmp);
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(kTmp);
        add_const(cb_pad_one, kTmp);
        recip_tile_init();
        recip_tile(kTmp);
        sfpu_bcast_col_init();
        sfpu_mul_bcast_col(kComb, kTmp);
        add_const(cb_eps_mask, kComb);

        // initial column normalisation, then Sinkhorn-Knopp row / column passes.
        normalize<ReduceDim::REDUCE_COL>(eps_bits);
        for (uint32_t i = 1; i < sinkhorn_iters; ++i) {
            normalize<ReduceDim::REDUCE_ROW>(eps_bits);
            normalize<ReduceDim::REDUCE_COL>(eps_bits);
        }
        tile_regs_commit();

        cb_pop_front(cb_comb_w, 1);

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(kComb, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}
