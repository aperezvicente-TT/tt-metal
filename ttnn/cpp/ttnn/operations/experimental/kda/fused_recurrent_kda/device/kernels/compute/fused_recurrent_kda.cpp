// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// One KDA recurrence step per (item, token), entirely in fp32 DST on the SFPU. Every CB this kernel reads is
// unpacked straight to DST (UnpackToDestFp32), so no operand passes through the 19-bit source registers.
// q, k and g arrive as column tiles (element 32 j + r at row r, column 0 of tile j), v and beta as row tiles (row 0).
// Per token, with S the [K,V] state as Kt x Vt tiles (Kt <= 4):
//   prep:  qcol = q / sqrt(sum q^2 + 1e-6) * scale, kcol = k / sqrt(sum k^2 + 1e-6), kfull = kcol broadcast over the
//          32 columns, dcol = exp(g) (three DST acquires)
//   then for each 32-column block ni of S (every quantity below stays inside the block), in one DST acquire:
//          S'     = S *col dcol                       (row r of S scaled by exp(g[r]))
//          delta  = (v - colsum(S' *col kcol)) * beta  (colsum = SFPU column reduce into row 0)
//          S      = S' + kfull *row delta              (rank-1 update k delta^T)
//          o      = colsum(S *col qcol)
// DST runs in full-sync mode: 8 fp32 tiles = the block's Kt state tiles + 4 operand / product tiles; each SFPU init is
// issued once per group of tiles.
// The state stays in L1 between the tokens of an item (cb_snew ping-pongs between its two halves); T > 1 is the
// speculative-verify recurrence.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4, cb_s0 = 5;
constexpr uint32_t cb_qcol = 6, cb_kcol = 7, cb_kfull = 8, cb_dcol = 9, cb_snew = 12;
constexpr uint32_t cb_o = 13, cb_st = 14;

constexpr uint32_t T = get_compile_time_arg_val(0);
constexpr uint32_t Kt = get_compile_time_arg_val(1);
constexpr uint32_t Vt = get_compile_time_arg_val(2);
constexpr bool L2NORM = get_compile_time_arg_val(3) != 0;
constexpr bool WANT_STATE = get_compile_time_arg_val(4) != 0;
constexpr bool PER_TOKEN = get_compile_time_arg_val(5) != 0;
constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(6);
constexpr uint32_t KV = Kt * Vt;
constexpr uint32_t EPS_BITS = 0x358637BD;  // 1e-6f
static_assert(Kt >= 1 && Kt <= 4, "a state column block must fit DST slots 0..3");

// DST[out] = DST[data] * DST[col][r][0] (column broadcast, out of place).
ALWI void mul_bcast_col_to(uint32_t data, uint32_t col, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_sfpu_binary_bcast_full_tile_,
        (ckernel::BinaryOp::MUL, ckernel::BroadcastType::COL),
        data,
        col,
        out,
        VectorMode::None)));
}

// SFPU elementwise ops restricted to the faces that carry data: VectorMode::C (faces 0, 2) for column-vector tiles,
// VectorMode::R (faces 0, 1) for row-vector tiles.
template <ckernel::BinaryOp OP>
ALWI void binary_faces(uint32_t a, uint32_t b, uint32_t out, VectorMode mode) {
    if constexpr (OP == ckernel::BinaryOp::MUL) {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary_mul,
            (APPROX, OP, 8, DST_ACCUM_MODE),
            a,
            b,
            out,
            mode)));
    } else {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary,
            (APPROX, OP, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
            a,
            b,
            out,
            mode)));
    }
}

template <int OP>
ALWI void scalar_faces(uint32_t idst, uint32_t bits, VectorMode mode) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar, (APPROX, OP, 8, DST_ACCUM_MODE), idst, mode, bits));
}

ALWI void sqrt_faces(uint32_t idst, VectorMode mode) {
    MATH(
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sqrt, (APPROX, 8, DST_ACCUM_MODE, false), idst, mode));
}

// Column tiles x[0..Kt) in DST[0..Kt) (a second copy in DST[4..4+Kt) is consumed): DST[0..Kt) <- x / sqrt(sum x^2 +
// eps), the 1 / norm going through DST[4] row 0, column 0.
ALWI void normalize_in_place() {
    mul_binary_tile_init();
    for (uint32_t j = 0; j < Kt; j++) {
        binary_faces<ckernel::BinaryOp::MUL>(4 + j, 4 + j, 4 + j, VectorMode::C);
    }
    add_binary_tile_init();
    for (uint32_t j = 1; j < Kt; j++) {
        binary_faces<ckernel::BinaryOp::ADD>(4, 4 + j, 4, VectorMode::C);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(4);
    binop_with_scalar_tile_init();
    scalar_faces<ADD_UNARY>(4, EPS_BITS, VectorMode::R);
    sqrt_tile_init();
    sqrt_faces(4, VectorMode::R);
    recip_tile_init();
    recip_tile(4, VectorMode::R);
    sfpu_mul_bcast_row_init();
    for (uint32_t j = 0; j < Kt; j++) {
        sfpu_mul_bcast_row(j, 4);
    }
}

// Column tiles of cb_x into DST[0..Kt) (and DST[4..4+Kt) when normalising).
ALWI void load_vector(uint32_t cb_x) {
    copy_init(cb_x);
    for (uint32_t j = 0; j < Kt; j++) {
        copy_tile(cb_x, j, j);
        if constexpr (L2NORM) {
            copy_tile(cb_x, j, 4 + j);
        }
    }
}

// qcol, kcol, kfull, dcol of one token.
ALWI void prep() {
    cb_reserve_back(cb_qcol, Kt);
    tile_regs_acquire();
    load_vector(cb_q);
    if constexpr (L2NORM) {
        normalize_in_place();
    }
    binop_with_scalar_tile_init();
    for (uint32_t j = 0; j < Kt; j++) {
        scalar_faces<MUL_UNARY>(j, SCALE_BITS, VectorMode::C);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Kt; j++) {
        pack_tile(j, cb_qcol);
    }
    tile_regs_release();
    cb_push_back(cb_qcol, Kt);

    cb_reserve_back(cb_kcol, Kt);
    cb_reserve_back(cb_kfull, Kt);
    tile_regs_acquire();
    load_vector(cb_k);
    if constexpr (L2NORM) {
        normalize_in_place();
    }
    fill_tile_init();
    for (uint32_t j = 0; j < Kt; j++) {
        fill_tile(4 + j, 0.0f);
    }
    sfpu_add_bcast_col_init();
    for (uint32_t j = 0; j < Kt; j++) {
        sfpu_add_bcast_col(4 + j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Kt; j++) {
        pack_tile(j, cb_kcol);
        pack_tile(4 + j, cb_kfull);
    }
    tile_regs_release();
    cb_push_back(cb_kcol, Kt);
    cb_push_back(cb_kfull, Kt);

    cb_reserve_back(cb_dcol, Kt);
    tile_regs_acquire();
    copy_init(cb_g);
    for (uint32_t j = 0; j < Kt; j++) {
        copy_tile(cb_g, j, j);
    }
    exp_tile_init<false>();
    for (uint32_t j = 0; j < Kt; j++) {
        exp_tile<false>(j, VectorMode::C);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Kt; j++) {
        pack_tile(j, cb_dcol);
    }
    tile_regs_release();
    cb_push_back(cb_dcol, Kt);
}

// DST[4] row 0 <- colsum over the block of DST[0..Kt) *col x (x given as Kt column tiles in cb_xcol).
ALWI void contract(uint32_t cb_xcol) {
    copy_init(cb_xcol);
    for (uint32_t mi = 0; mi < Kt; mi++) {
        copy_tile(cb_xcol, mi, 4 + mi);
    }
    sfpu_mul_bcast_col_init();
    for (uint32_t mi = 0; mi < Kt; mi++) {
        mul_bcast_col_to(mi, 4 + mi, 4 + mi);
    }
    add_binary_tile_init();
    for (uint32_t mi = 1; mi < Kt; mi++) {
        add_binary_tile(4, 4 + mi, 4);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(4);
}

// One token for all Vt column blocks: cb_cur -> cb_snew (+ cb_st when emit), o -> cb_o. DST: the block's state
// tiles in 0..Kt, operands / products in 4..7 (delta in 5 during the rank-1 update).
ALWI void step(uint32_t cb_cur, bool emit) {
    constexpr uint32_t D_DELTA = 5;
    constexpr uint32_t free_slots[3] = {4, 6, 7};
    cb_wait_front(cb_cur, KV);
    cb_wait_front(cb_qcol, Kt);
    cb_wait_front(cb_kcol, Kt);
    cb_wait_front(cb_kfull, Kt);
    cb_wait_front(cb_dcol, Kt);
    cb_wait_front(cb_v, Vt);
    cb_wait_front(cb_beta, 1);
    cb_reserve_back(cb_snew, KV);
    if (emit) {
        cb_reserve_back(cb_st, KV);
    }
    cb_reserve_back(cb_o, Vt);
    for (uint32_t ni = 0; ni < Vt; ni++) {
        tile_regs_acquire();
        copy_init(cb_cur);
        for (uint32_t mi = 0; mi < Kt; mi++) {
            copy_tile(cb_cur, mi * Vt + ni, mi);
            copy_tile(cb_dcol, mi, 4 + mi);
        }
        sfpu_mul_bcast_col_init();
        for (uint32_t mi = 0; mi < Kt; mi++) {
            sfpu_mul_bcast_col(mi, 4 + mi);
        }
        contract(cb_kcol);
        copy_init(cb_v);
        copy_tile(cb_v, ni, D_DELTA);
        copy_tile(cb_beta, 0, 6);
        sub_binary_tile_init();
        binary_faces<ckernel::BinaryOp::SUB>(D_DELTA, 4, D_DELTA, VectorMode::R);
        mul_binary_tile_init();
        binary_faces<ckernel::BinaryOp::MUL>(D_DELTA, 6, D_DELTA, VectorMode::R);
        for (uint32_t m0 = 0; m0 < Kt; m0 += 3) {
            const uint32_t n = (Kt - m0 < 3) ? (Kt - m0) : 3;
            copy_init(cb_kfull);
            for (uint32_t j = 0; j < n; j++) {
                copy_tile(cb_kfull, m0 + j, free_slots[j]);
            }
            sfpu_mul_bcast_row_init();
            for (uint32_t j = 0; j < n; j++) {
                sfpu_mul_bcast_row(free_slots[j], D_DELTA);
            }
            add_binary_tile_init();
            for (uint32_t j = 0; j < n; j++) {
                add_binary_tile(m0 + j, free_slots[j], m0 + j);
            }
        }
        contract(cb_qcol);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t mi = 0; mi < Kt; mi++) {
            pack_tile<true>(mi, cb_snew, mi * Vt + ni);
            if (emit) {
                pack_tile<true>(mi, cb_st, mi * Vt + ni);
            }
        }
        pack_tile(4, cb_o);
        tile_regs_release();
    }
    cb_push_back(cb_snew, KV);
    if (emit) {
        cb_push_back(cb_st, KV);
    }
    cb_push_back(cb_o, Vt);
    cb_pop_front(cb_cur, KV);
    cb_pop_front(cb_qcol, Kt);
    cb_pop_front(cb_kcol, Kt);
    cb_pop_front(cb_kfull, Kt);
    cb_pop_front(cb_dcol, Kt);
    cb_pop_front(cb_v, Vt);
    cb_pop_front(cb_beta, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t items = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_q, cb_q, cb_o);

    for (uint32_t it = 0; it < items; it++) {
        for (uint32_t t = 0; t < T; t++) {
            cb_wait_front(cb_q, Kt);
            cb_wait_front(cb_k, Kt);
            cb_wait_front(cb_g, Kt);
            {
                DeviceZoneScopedN("KDA-prep");
                prep();
            }
            cb_pop_front(cb_q, Kt);
            cb_pop_front(cb_k, Kt);
            cb_pop_front(cb_g, Kt);
            {
                DeviceZoneScopedN("KDA-step");
                step(t == 0 ? cb_s0 : cb_snew, PER_TOKEN || (WANT_STATE && t == T - 1));
            }
        }
        cb_wait_front(cb_snew, KV);
        cb_pop_front(cb_snew, KV);
    }
}
