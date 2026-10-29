// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// WS-O MEMBER BRISC: the output store (tt/pf_wo.PfStreamWO). Per chunk c: the member's COLS tiles of row block c ->
// today's y (DRAM tile-interleaved [rows, N]) at pages c NTT + COL0 + i (pfs_member_wr.cpp's layout), a write barrier
// (the tiles landed), then +1 on block c's H5 word (RT_H5 table: the RS8 readiness word of block c, or the unit's
// stand-in) -- 64 raises per block per launch, one per member (+1 at QSLICE 1; at PMK_RS8_QSLICE the member's slice digit
// value, RT_H5 + 1 + MT).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pfwo_common.hpp"

namespace pfwo_w {
constexpr uint32_t MT = get_compile_time_arg_val(PH_CT0 + pfwo::CT_MT);
constexpr uint32_t COLS = get_compile_time_arg_val(PH_CT0 + pfwo::CT_COLS);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(PH_CT0 + pfwo::CT_CB_OUT);
}  // namespace pfwo_w

void kernel_main() {
    using namespace pfwo_w;
    constexpr auto y_args = TensorAccessorArgs<PH_CT0 + pfwo::CT_ACC>();
    const auto yy = TensorAccessor(y_args, get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_YADDR));
    const uint32_t ntt = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_NTT);
    const uint32_t col0 = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_COL0);
    const uint32_t tb = get_tile_size(CB_OUT);
    const uint32_t h5base = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_H5);
    // the raise value (RT_H5 + 1 + MT): 1 at QSLICE 1; at PMK_RS8_QSLICE Q, (holders per slice + 1)^s for this member's
    // column slice s (the RS readers' FS_SLICE_DIGITS digit wait: the slices of a block complete in any order)
    const uint32_t inc = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_H5 + 1 + MT);
    for (uint32_t c = 0; c < MT; c++) {
        cb_wait_front(CB_OUT, COLS);
        const uint32_t rp = get_read_ptr(CB_OUT);
        for (uint32_t i = 0; i < COLS; i++) {
            noc_async_write_page(c * ntt + col0 + i, yy, rp + i * tb);
        }
        noc_async_write_barrier();  // block c's tiles landed before its H5 raise
        cb_pop_front(CB_OUT, COLS);
        const uint32_t hw = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_H5 + 1 + c);  // x | y << 8 | offset << 16
        noc_semaphore_inc(get_noc_addr(hw & 0xFFu, (hw >> 8) & 0xFFu, h5base + (hw >> 16)), inc);
    }
    noc_async_atomic_barrier();
}
