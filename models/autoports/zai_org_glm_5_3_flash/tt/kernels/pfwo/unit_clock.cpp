// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// WS-O unit only (tests/s16/jobs/pmk_wso_unit.py): the SYNTHETIC SCAN CLOCK, BRISC of the probe core. Chunk c is
// released at t0 + c PERIOD cycles (the `on` rows are pre-filled in DRAM): +1 on SEM_ON of every target (the members --
// the served scan leader's H4 semantics -- and, for WS-O, the two lanes), as the KDA scan raises them once per chunk.
// Probe words (16 B, the probe tensor on this core): T0 = the absolute start stamp, REL[c] = c's release (cycles after
// T0).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t NCH = get_compile_time_arg_val(0);
constexpr uint32_t PERIOD = get_compile_time_arg_val(1);  // cycles between chunk releases
constexpr uint32_t SEM_ON = get_compile_time_arg_val(2);
constexpr uint32_t NT = get_compile_time_arg_val(3);  // targets
constexpr uint32_t W = 16, P_T0 = 0, P_REL = 1;

void kernel_main() {
    const uint32_t probe = get_arg_val<uint32_t>(0);
    const uint32_t sem_on = get_semaphore(SEM_ON);
    const uint32_t t0 = get_timestamp_32b();
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(probe + P_T0 * W) = t0;
    for (uint32_t c = 0; c < NCH; c++) {
        while (static_cast<uint32_t>(get_timestamp_32b() - t0) < c * PERIOD) {
        }
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(probe + (P_REL + c) * W) = get_timestamp_32b() - t0;
        for (uint32_t t = 0; t < NT; t++) {
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(1 + 2 * t), get_arg_val<uint32_t>(2 + 2 * t), sem_on), 1);
        }
    }
    noc_async_atomic_barrier();
}
