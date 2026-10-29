// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// WS-O unit only (tests/s16/jobs/pmk_wso_unit.py): the H5 STAND-IN, NCRISC of the probe core (the RS8 reader's place).
// Block c is complete when its H5 word reached RAISES (1: today's PfStream member writer, one raise per row; 64: WS-O,
// one per member = the RS8's FS_RDY_MULT 64). Waits the blocks in order (the RS8's ascending order at QSLICE 1) and
// stamps DONE[c] (absolute, this core's clock = the clock kernel's), then zeroes the words (EPOCH: every raise awaited).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t NCH = get_compile_time_arg_val(0);
constexpr uint32_t RAISES = get_compile_time_arg_val(1);
constexpr uint32_t W = 16, P_DONE = 65, P_H5 = 129, P_DL = 193;
constexpr uint64_t BUDGET = 2700000000ull;  // cycles (~2 s): a missing raise records P_DL and proceeds

void kernel_main() {
    const uint32_t probe = get_arg_val<uint32_t>(0);
    auto w = [&](uint32_t i) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(probe + i * W); };
    for (uint32_t c = 0; c < NCH; c++) {
        uint64_t elapsed = 0;
        uint32_t last = get_timestamp_32b();
        while (true) {
            invalidate_l1_cache();
            if (*w(P_H5 + c) >= RAISES) {
                break;
            }
            const uint32_t now = get_timestamp_32b(), d = now - last;
            elapsed += d < (1u << 30) ? d : 0;
            last = now;
            if (elapsed >= BUDGET) {
                *w(P_DL) = 0xDEAD0000u | c;
                break;
            }
        }
        *w(P_DONE + c) = get_timestamp_32b();
    }
    for (uint32_t c = 0; c < NCH; c++) {
        *w(P_H5 + c) = 0;
    }
}
