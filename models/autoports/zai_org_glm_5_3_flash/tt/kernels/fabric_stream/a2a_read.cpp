// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 16 FabricStreamA2A2D (tt/fabric_a2a.py), lane NCRISC: streams this lane's RUN LIST (host-built: every run =
// RUN consecutive tile pages of the source x or of this chip's relay store) into the staging CB in NB-page batches,
// in the order a2a_send.cpp writes / sends them. A relay run (stage B forwarding what the column partner's A lane
// relayed here in stage A) first waits for this core's RELAY counter: the partner's A lane L sends its relay runs in
// exactly this lane's read order, each ending with a fused atomic +1 of RELAY, so the k-th relay run of launch g is in
// DRAM once RELAY >= (g - 1) relay_per + k (`need` = k, monotonic across launches).
// Compile-time: NB, CB_STAGE, RUN, (the arg count, hash only), then the TensorAccessorArgs of x and of the relay store.
// Runtime: sync, x address, relay address, relay runs per launch, n runs, then per run (src_sel | need << 1, page).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fs_common.hpp"
#include "a2a_common.hpp"

constexpr uint32_t NB = get_compile_time_arg_val(0);  // pages per staging batch
constexpr uint32_t CB_STAGE = get_compile_time_arg_val(1);
constexpr uint32_t RUN = get_compile_time_arg_val(2);  // pages per run
constexpr uint32_t TB = 2048;
static_assert(RUN % NB == 0, "a batch never straddles a run");

void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto r_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    size_t a = 0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t r_addr = get_arg_val<uint32_t>(a++);
    const uint32_t relay_per = get_arg_val<uint32_t>(a++);
    const uint32_t nruns = get_arg_val<uint32_t>(a++);
    const auto x = TensorAccessor(x_args, x_addr);
    const auto rl = TensorAccessor(r_args, r_addr);
    volatile tt_l1_ptr uint32_t* gen = fs::word(sync, a2a::GEN_NC);
    invalidate_l1_cache();
    const uint32_t g = *gen + 1;
    for (uint32_t k = 0; k < nruns; k++) {
        const uint32_t w0 = get_arg_val<uint32_t>(a++);
        const uint32_t page0 = get_arg_val<uint32_t>(a++);
        const bool relay = (w0 & 1u) != 0;
        const uint32_t need = w0 >> 1;
        if (need != 0) {
            fs::wait_ge(fs::word(sync, a2a::RELAY), (g - 1) * relay_per + need);
        }
        for (uint32_t b = 0; b < RUN / NB; b++) {
            cb_reserve_back(CB_STAGE, NB);
            const uint32_t wp = get_write_ptr(CB_STAGE);
            if (relay) {
                for (uint32_t i = 0; i < NB; i++) {
                    noc_async_read_page(page0 + b * NB + i, rl, wp + i * TB);
                }
            } else {
                for (uint32_t i = 0; i < NB; i++) {
                    noc_async_read_page(page0 + b * NB + i, x, wp + i * TB);
                }
            }
            noc_async_read_barrier();
            cb_push_back(CB_STAGE, NB);
        }
    }
    *gen = g;
}
