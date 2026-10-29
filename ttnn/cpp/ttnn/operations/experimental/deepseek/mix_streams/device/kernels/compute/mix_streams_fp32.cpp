// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

// fp32 variant of the fused mix, one output tile per iteration:
//   out = comb^T @ streams + post (x) sublayer_out
// The FPU source registers hold fewer mantissa bits than fp32, so every fp32 operand tile is first split on the SFPU
// into three bf16-exact parts, x = hi + mid + lo (hi = rne_bf16(x), mid = rne_bf16(x - hi), lo = x - hi - mid, all
// subtractions exact). The FPU's accumulation over K inside one tile matmul is not fp32 exact, but a single product
// of bf16 parts is, and so is the accumulation into fp32 DST across matmul calls: comb^T therefore arrives as hc
// column-isolated tiles (one non-zero K term per output element), and the post tile holds only column 0. The partial
// products below 2^-16 relative (mid*lo, lo*mid, lo*lo) are dropped; smallest terms accumulate first.
namespace {

constexpr uint32_t kHi = 0;
constexpr uint32_t kMid = 1;
constexpr uint32_t kLo = 2;
constexpr uint32_t kNumParts = 3;
constexpr uint32_t kFp32 = static_cast<uint32_t>(DataFormat::Float32);
constexpr uint32_t kBf16 = static_cast<uint32_t>(DataFormat::Float16_b);

// Pops one fp32 tile of cb_in and pushes its (hi, mid, lo) bf16 parts to cb_parts.
FORCE_INLINE void split3(uint32_t cb_in, uint32_t cb_parts) {
    cb_wait_front(cb_in, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_in);
    copy_init(cb_in);
    copy_tile(cb_in, 0, kLo);
    copy_dest_values_init();
    copy_dest_values<DataFormat::Float32>(kLo, kHi);
    typecast_tile_init<kFp32, kBf16>();
    typecast_tile<kFp32, kBf16>(kHi);
    sub_binary_tile_init();
    sub_binary_tile(kLo, kHi, kLo);
    copy_dest_values_init();
    copy_dest_values<DataFormat::Float32>(kLo, kMid);
    typecast_tile_init<kFp32, kBf16>();
    typecast_tile<kFp32, kBf16>(kMid);
    sub_binary_tile_init();
    sub_binary_tile(kLo, kMid, kLo);
    tile_regs_commit();
    cb_pop_front(cb_in, 1);

    cb_reserve_back(cb_parts, kNumParts);
    tile_regs_wait();
    pack_reconfig_data_format(cb_parts);
    pack_tile(kHi, cb_parts);
    pack_tile(kMid, cb_parts);
    pack_tile(kLo, cb_parts);
    tile_regs_release();
    cb_push_back(cb_parts, kNumParts);
}

// Kept (a_part, b_part) pairs by magnitude class: |a_i b_j| <= 2^-(8 (i + j)) |a b| for parts 0 = hi, 1 = mid,
// 2 = lo. All classes of every operand pair are accumulated smallest class first, so the small partial products sum
// among themselves before they meet the large ones.
constexpr uint32_t kNumClasses = 3;
constexpr uint32_t kClassBegin[kNumClasses + 1] = {0, 3, 5, 6};
constexpr uint32_t kPairA[6] = {kLo, kMid, kHi, kMid, kHi, kHi};
constexpr uint32_t kPairB[6] = {kHi, kMid, kLo, kHi, kMid, kHi};

// DST[0] += the class-`cls` part products of a @ b, `a` being the part triple at tile a_base of cb_a (and b at tile
// 0 of cb_b); matmul_init must be current for (cb_a, cb_b).
FORCE_INLINE void matmul_class(uint32_t cb_a, uint32_t a_base, uint32_t cb_b, uint32_t cls) {
    for (uint32_t p = kClassBegin[cls]; p < kClassBegin[cls + 1]; ++p) {
        matmul_tiles(cb_a, cb_b, a_base + kPairA[p], kPairB[p], 0);
    }
}

FORCE_INLINE void matmul_reinit(uint32_t cb_a, uint32_t cb_b) {
    reconfig_data_format<SrcOrder::Reverse>(cb_a, cb_b);
    matmul_init(cb_a, cb_b);
}

}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_comb = get_compile_time_arg_val(0);
    constexpr uint32_t cb_post = get_compile_time_arg_val(1);
    constexpr uint32_t cb_streams = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sub = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t cb_comb_parts = get_compile_time_arg_val(6);
    constexpr uint32_t cb_post_parts = get_compile_time_arg_val(7);
    constexpr uint32_t cb_streams_parts = get_compile_time_arg_val(8);
    constexpr uint32_t cb_sub_parts = get_compile_time_arg_val(9);
    constexpr bool sub_fp32 = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t hc = get_compile_time_arg_val(11);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_comb_parts, cb_streams_parts, cb_out);

    const uint32_t end_tile = start_tile + num_tiles;
    uint32_t tile = start_tile;
    while (tile < end_tile) {
        const uint32_t t = tile / n_tiles;
        const uint32_t group_end = (end_tile < (t + 1) * n_tiles) ? end_tile : (t + 1) * n_tiles;

        for (uint32_t i = 0; i < hc; ++i) {
            split3(cb_comb, cb_comb_parts);
        }
        split3(cb_post, cb_post_parts);
        cb_wait_front(cb_comb_parts, kNumParts * hc);
        cb_wait_front(cb_post_parts, kNumParts);

        for (uint32_t page = tile; page < group_end; ++page) {
            split3(cb_streams, cb_streams_parts);
            if constexpr (sub_fp32) {
                split3(cb_sub, cb_sub_parts);
                cb_wait_front(cb_sub_parts, kNumParts);
            } else {
                cb_wait_front(cb_sub, 1);
            }
            cb_wait_front(cb_streams_parts, kNumParts);

            tile_regs_acquire();
            for (uint32_t cls = 0; cls < kNumClasses; ++cls) {
                matmul_reinit(cb_comb_parts, cb_streams_parts);
                for (uint32_t i = 0; i < hc; ++i) {
                    matmul_class(cb_comb_parts, kNumParts * i, cb_streams_parts, cls);
                }
                if constexpr (sub_fp32) {
                    matmul_reinit(cb_post_parts, cb_sub_parts);
                    matmul_class(cb_post_parts, 0, cb_sub_parts, cls);
                } else {
                    // bf16 sublayer_out is a single exact part: post part `2 - cls` is the class-`cls` term.
                    matmul_reinit(cb_post_parts, cb_sub);
                    matmul_tiles(cb_post_parts, cb_sub, kLo - cls, 0, 0);
                }
            }
            tile_regs_commit();

            cb_pop_front(cb_streams_parts, kNumParts);
            if constexpr (sub_fp32) {
                cb_pop_front(cb_sub_parts, kNumParts);
            } else {
                cb_pop_front(cb_sub, 1);
            }

            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }

        cb_pop_front(cb_comb_parts, kNumParts * hc);
        cb_pop_front(cb_post_parts, kNumParts);
        tile = group_end;
    }
}
