// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// AD-1A, phase D2 (the sparse_dsa layer program, program 2026-10-27; doc pmk/merge/README.md): the MLA input norms
// of the member's OWN 32-row tile row, in exact fp32 on the SFPU (the decode MK's kernels/mla/rmsnorm.hpp arithmetic):
//   q_r    = RMSNorm_1536(qk[:, :1536]) * qa_norm   -> CB_OUT (bf16 tiles, 48)
//   latent = RMSNorm_512(qk[:, 1536:])  * kva_norm  -> CB_OUT (bf16 tiles, 16)
//   k      = LayerNorm_128(kg[:, :128]) * k_w + k_b -> CB_RM  (fp32, untilized: 4 tile-width row strips)
//   g      = kg[:, 128:256]                         -> CB_RM  (fp32, untilized: 4 strips, a copy)
// Inputs (pushed by dsa_norm_dm.cpp D2_READ): CB_X = the 64 bf16 qk tiles of the row, CB_KG = the 8 fp32 kg tiles, CB_W
// = the 72 weight tiles in consumption order (row 0 = the weights: qa_norm 48, kva_norm 16, then (k_w, k_b) per k
// tile). CB_INV (fp32) and CB_KG are unpacked straight to DEST (UnpackToDestFp32, set by the host): no tf32 operand
// truncation. CT at PH_CT0: [cb_x, cb_kg, cb_w, cb_inv, cb_out, cb_rm, qt, kt, lt, gt, inv_q_bits, inv_k_bits,
// inv_l_bits, eps_bits, eps_ln_bits] (+ the writer / reader accessors). RT at ph_rt0: [run (0 = skip), ...].
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/rmsnorm.hpp"

namespace d2c {
constexpr uint32_t CB_X = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CB_KG = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_W = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t CB_INV = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(PH_CT0 + 4);
constexpr uint32_t CB_RM = get_compile_time_arg_val(PH_CT0 + 5);
constexpr uint32_t QT = get_compile_time_arg_val(PH_CT0 + 6);
constexpr uint32_t KT = get_compile_time_arg_val(PH_CT0 + 7);
constexpr uint32_t LT = get_compile_time_arg_val(PH_CT0 + 8);
constexpr uint32_t GT = get_compile_time_arg_val(PH_CT0 + 9);
constexpr uint32_t INV_Q = get_compile_time_arg_val(PH_CT0 + 10);
constexpr uint32_t INV_K = get_compile_time_arg_val(PH_CT0 + 11);
constexpr uint32_t INV_L = get_compile_time_arg_val(PH_CT0 + 12);
constexpr uint32_t EPS_RMS = get_compile_time_arg_val(PH_CT0 + 13);
constexpr uint32_t EPS_LN = get_compile_time_arg_val(PH_CT0 + 14);

// CB_X tiles [base, base + N) (already waited for) -> one fp32 tile, column 0 = rsqrt(sum_j x_j^2 / n + eps) per row
template <uint32_t N>
FORCE_INLINE void norm_inv_from(uint32_t base, uint32_t inv_n_bits) {
    cb_reserve_back(CB_INV, 1);
    tile_regs_acquire();
    mlan::load(CB_X, base, 0);
    square_tile_init();
    square_tile(0);
    for (uint32_t j = 1; j < N; ++j) {
        mlan::load(CB_X, base + j, 1);
        square_tile_init();
        square_tile(1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_n_bits);
    add_unary_tile(0, EPS_RMS);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_INV);
    pack_tile(0, CB_INV);
    tile_regs_release();
    cb_push_back(CB_INV, 1);
}

// tiles [base, base + N) of CB_X * inv * w (one weight tile each, consumed from CB_W) -> CB_OUT bf16
template <uint32_t N>
FORCE_INLINE void norm_rows(uint32_t base) {
    cb_wait_front(CB_INV, 1);
    for (uint32_t j = 0; j < N; ++j) {
        cb_wait_front(CB_W, 1);
        cb_reserve_back(CB_OUT, 1);
        tile_regs_acquire();
        mlan::norm_tile(CB_X, base + j, CB_INV, CB_W, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_OUT);
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, 1);
        cb_pop_front(CB_W, 1);
    }
    cb_pop_front(CB_INV, 1);
}
}  // namespace d2c

void kernel_main() {
    using namespace d2c;
    if (get_arg_val<uint32_t>(ph_rt0) == 0) {
        return;
    }
#if !defined(MK_PHASED) || defined(PMK_FIRST_COMPUTE)  // the composite's first compute body: full start-up
    compute_kernel_hw_startup(CB_X, CB_OUT);
#else
    reconfig_data_format_srca(CB_X);  // the previous part left the unpackers / packer on its own CBs
    pack_reconfig_data_format(CB_OUT);
#endif
    // ---- q_r / latent: the two RMSNorms of the qk row (q_a = tiles [0, QT), kv_a = [QT, QT + KT))
    cb_wait_front(CB_X, QT + KT);
    norm_inv_from<QT>(0, INV_Q);
    norm_rows<QT>(0);
    norm_inv_from<KT>(QT, INV_K);
    norm_rows<KT>(QT);
    cb_pop_front(CB_X, QT + KT);
    // ---- k = LayerNorm(kg[:, :128]): mean, then rsqrt(mean of the centred squares + eps) -> CB_INV pages 0 / 1
    cb_wait_front(CB_KG, LT + GT);
    cb_reserve_back(CB_INV, 1);
    tile_regs_acquire();
    mlan::load(CB_KG, 0, 0);
    for (uint32_t j = 1; j < LT; ++j) {
        mlan::load(CB_KG, j, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_L);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_INV);
    pack_tile(0, CB_INV);
    tile_regs_release();
    cb_push_back(CB_INV, 1);
    cb_wait_front(CB_INV, 1);
    cb_reserve_back(CB_INV, 1);
    tile_regs_acquire();
    mlan::load(CB_INV, 0, 2);  // the mean (column 0)
    mlan::load(CB_KG, 0, 0);
    sfpu_sub_bcast_col_init();
    sfpu_sub_bcast_col(0, 2);
    square_tile_init();
    square_tile(0);
    for (uint32_t j = 1; j < LT; ++j) {
        mlan::load(CB_KG, j, 1);
        sfpu_sub_bcast_col_init();
        sfpu_sub_bcast_col(1, 2);
        square_tile_init();
        square_tile(1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_L);
    add_unary_tile(0, EPS_LN);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_INV);
    pack_tile(0, CB_INV);
    tile_regs_release();
    cb_push_back(CB_INV, 1);
    cb_wait_front(CB_INV, 2);
    // ---- k (normalised, * k_w + k_b) then g (copied): fp32 rows, one tile-width strip per DST block (BW 1)
    pack_reconfig_data_format(CB_RM);
    pack_untilize_dest_init<1, 1>(CB_RM);
    for (uint32_t j = 0; j < LT; ++j) {
        cb_wait_front(CB_W, 2);
        cb_reserve_back(CB_RM, 1);
        tile_regs_acquire();
        mlan::load(CB_KG, j, 0);
        mlan::load(CB_INV, 0, 1);
        mlan::load(CB_INV, 1, 2);
        sfpu_bcast_col_init();
        sfpu_normalize_bcast_col(0, 1, 2);
        mlan::load(CB_W, 0, 1);
        sfpu_mul_bcast_row_init();
        sfpu_mul_bcast_row(0, 1);
        mlan::load(CB_W, 1, 1);
        sfpu_add_bcast_row_init();
        sfpu_add_bcast_row(0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_untilize_dest<1, 1>(CB_RM, 1, 0);
        tile_regs_release();
        cb_push_back(CB_RM, 1);
        cb_pop_front(CB_W, 2);
    }
    for (uint32_t j = 0; j < GT; ++j) {
        cb_reserve_back(CB_RM, 1);
        tile_regs_acquire();
        mlan::load(CB_KG, LT + j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_untilize_dest<1, 1>(CB_RM, 1, 0);
        tile_regs_release();
        cb_push_back(CB_RM, 1);
    }
    pack_untilize_uninit(CB_RM);
    cb_pop_front(CB_KG, LT + GT);
    cb_pop_front(CB_INV, 2);
}
