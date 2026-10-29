// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {

// Fill a single tile with the reduce scaler value (bf16 hi-half) in the standard layout
// expected by reduce_tile: the first column of each of the 4 faces.
FORCE_INLINE void generate_reduce_scaler(uint32_t cb_id, uint16_t scaler) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = 0;
    }
    if (scaler != 0) {
        for (uint32_t k = 0; k < 4; ++k) {
            const uint32_t idx = k << 8;
            for (uint32_t j = 0; j < 16; ++j) {
                ptr[idx + j] = scaler;
            }
        }
    }
    cb_push_back(cb_id, 1);
}

// Fill a single tile with `value` (bf16 hi-half) inside the valid HxH block and 0 elsewhere,
// honouring the 4x(16x16)-face tile layout. H <= 32.
FORCE_INLINE void generate_hxh_mask(uint32_t cb_id, uint32_t h, uint16_t value) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = 0;
    }
    for (uint32_t r = 0; r < h; ++r) {
        for (uint32_t c = 0; c < h; ++c) {
            const uint32_t face = ((r >= 16) ? 2u : 0u) + ((c >= 16) ? 1u : 0u);
            const uint32_t idx = face * 256u + (r & 15u) * 16u + (c & 15u);
            ptr[idx] = value;
        }
    }
    cb_push_back(cb_id, 1);
}

// fp32 constant tile: `inside` on the HxH block, `pad_cols` on the padding columns of the H valid rows,
// `pad_rows` on the padding rows.
FORCE_INLINE void generate_fp32_block_tile(
    uint32_t cb_id, uint32_t h, uint32_t inside, uint32_t pad_cols, uint32_t pad_rows) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t face = ((r >= 16) ? 2u : 0u) + ((c >= 16) ? 1u : 0u);
            const uint32_t idx = face * 256u + (r & 15u) * 16u + (c & 15u);
            ptr[idx] = (r < h) ? ((c < h) ? inside : pad_cols) : pad_rows;
        }
    }
    cb_push_back(cb_id, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t comb_w_addr = get_arg_val<uint32_t>(0);
    const uint32_t comb_bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_comb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_comb_bias = get_compile_time_arg_val(1);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(4);
    constexpr uint32_t num_streams = get_compile_time_arg_val(5);
    constexpr uint32_t cb_eps_mask = get_compile_time_arg_val(6);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(7);
    constexpr uint32_t element_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t cb_pad_one = get_compile_time_arg_val(9);

    constexpr auto comb_w_args = TensorAccessorArgs<10>();
    constexpr auto comb_bias_args = TensorAccessorArgs<comb_w_args.next_compile_time_args_offset()>();

    const auto comb_w = TensorAccessor(comb_w_args, comb_w_addr);
    const auto comb_bias = TensorAccessor(comb_bias_args, comb_bias_addr);

    if constexpr (element_bytes == 4) {
        // fp32 SFPU path: ones mask, eps mask, and 1.0 on the padding rows (keeps the softmax denominator of
        // empty rows non-zero).
        generate_fp32_block_tile(cb_mask, num_streams, scaler_bits, 0u, 0u);
        generate_fp32_block_tile(cb_eps_mask, num_streams, eps_bits, 0u, 0u);
        generate_fp32_block_tile(cb_pad_one, num_streams, 0u, 0u, scaler_bits);
    } else {
        // Reduce scaler is a constant 1.0 (plain sum / max); only the bf16 hi-half is used.
        generate_reduce_scaler(cb_scaler, static_cast<uint16_t>(scaler_bits >> 16));
        // Ones mask delimiting the valid HxH block (1.0 inside, 0 in the tile padding).
        generate_hxh_mask(cb_mask, num_streams, static_cast<uint16_t>(scaler_bits >> 16));
        // The same mask scaled by eps, so that "comb += eps then re-mask" is a single masked add.
        generate_hxh_mask(cb_eps_mask, num_streams, static_cast<uint16_t>(eps_bits >> 16));
    }

    Noc noc;
    CircularBuffer cb_w(cb_comb_w);
    CircularBuffer cb_b(cb_comb_bias);

    constexpr uint32_t one_tile = 1;

    // The bias is the same for every token: read it once and never pop it, so the compute
    // kernel can re-wait on the same tile for each of its tiles.
    // The padding columns of the H valid rows get -FLT_MAX, so the logits carry it: the row max is taken over
    // the valid columns only and exp() of the padding underflows to 0 instead of overflowing for a very negative
    // row (0 - rowmax). The padding rows stay 0 and are masked after exp.
    cb_b.reserve_back(one_tile);
    noc.async_read(comb_bias, cb_b, cb_b.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    if constexpr (element_bytes == 4) {
        volatile tt_l1_ptr uint32_t* bias = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_b.get_write_ptr());
        for (uint32_t r = 0; r < num_streams; ++r) {
            for (uint32_t c = num_streams; c < 32; ++c) {
                const uint32_t face = ((r >= 16) ? 2u : 0u) + ((c >= 16) ? 1u : 0u);
                bias[face * 256u + (r & 15u) * 16u + (c & 15u)] = 0xFF7FFFFFu;
            }
        }
    } else {
        volatile tt_l1_ptr uint16_t* bias = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_b.get_write_ptr());
        for (uint32_t r = 0; r < num_streams; ++r) {
            for (uint32_t c = num_streams; c < 32; ++c) {
                const uint32_t face = ((r >= 16) ? 2u : 0u) + ((c >= 16) ? 1u : 0u);
                bias[face * 256u + (r & 15u) * 16u + (c & 15u)] = 0xFF7Fu;
            }
        }
    }
    cb_b.push_back(one_tile);

    // comb_w is [1,T,H,H]: one page per token, and this core owns [start_tile, start_tile+num_tiles).
    const uint32_t comb_w_tile_size = cb_w.get_tile_size();
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_w.reserve_back(one_tile);
        noc.async_read(comb_w, cb_w, comb_w_tile_size, {.page_id = start_tile + i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_w.push_back(one_tile);
    }
}
