// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer: row 0 of each o tile into the (b, t, h) row of o [B,T,H,V] (two 64 B face rows per tile; the other rows
// of those tiles belong to other items), and the [K,V] state tiles of each item: per token into [B,T,H,K,V] or once
// into [B,H,K,V].

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {

constexpr uint32_t cb_o = 13, cb_st = 14;

constexpr uint32_t FACE_BYTES = 16 * 16 * 4;
constexpr uint32_t FACE_ROW_BYTES = 16 * 4;

}  // namespace

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Vt = get_compile_time_arg_val(3);
    constexpr bool WANT_STATE = get_compile_time_arg_val(4) != 0;
    constexpr bool PER_TOKEN = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t o_cols = get_compile_time_arg_val(6);
    constexpr uint32_t o_rows = get_compile_time_arg_val(7);
    constexpr uint32_t o_row_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t o_col_tiles = get_compile_time_arg_val(9);
    constexpr auto o_a = TensorAccessorArgs<10>();
    constexpr auto st_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    constexpr uint32_t V = Vt * 32;
    constexpr uint32_t KV = Kt * Vt;

    const uint32_t item_start = get_arg_val<uint32_t>(0);
    const uint32_t item_count = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t st_addr = get_arg_val<uint32_t>(3);

    const uint32_t tb = get_tile_size(cb_o);
    const auto o_acc = TensorAccessor(o_a, o_addr, tb);
    const auto st_acc = TensorAccessor(st_a, st_addr, tb);

    Noc noc;
    CircularBuffer co(cb_o), cst(cb_st);

    auto write_state = [&](uint32_t base) {
        cst.wait_front(KV);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cst);
        for (uint32_t i = 0; i < KV; i++) {
            noc.async_write(src, st_acc, tb, {.offset_bytes = i * tb}, {.page_id = base + i});
        }
        noc.async_write_barrier();
        cst.pop_front(KV);
    };

    for (uint32_t it = 0; it < item_count; it++) {
        const uint32_t item = item_start + it;  // b * H + h
        const uint32_t b = item / H;
        const uint32_t h = item % H;
        for (uint32_t t = 0; t < T; t++) {
            const uint32_t row = (b * T + t) * H + h;
            const uint32_t e = row * V;
            const uint32_t col = e % o_cols;
            const uint32_t rest = e / o_cols;
            const uint32_t r = rest % o_rows;
            const uint32_t page = ((rest / o_rows) * o_row_tiles + r / 32) * o_col_tiles + col / 32;
            const uint32_t dst = ((r % 32) / 16) * 2 * FACE_BYTES + (r % 16) * FACE_ROW_BYTES;

            co.wait_front(Vt);
            auto src = use<CircularBuffer::AddrSelector::READ_PTR>(co);
            for (uint32_t j = 0; j < Vt; j++) {
                noc.async_write(
                    src, o_acc, FACE_ROW_BYTES, {.offset_bytes = j * tb}, {.page_id = page + j, .offset_bytes = dst});
                noc.async_write(
                    src,
                    o_acc,
                    FACE_ROW_BYTES,
                    {.offset_bytes = j * tb + FACE_BYTES},
                    {.page_id = page + j, .offset_bytes = dst + FACE_BYTES});
            }
            noc.async_write_barrier();
            co.pop_front(Vt);

            if constexpr (PER_TOKEN) {
                write_state(((b * T + t) * H + h) * KV);
            }
        }
        if constexpr (WANT_STATE && !PER_TOKEN) {
            write_state(item * KV);
        }
    }
}
