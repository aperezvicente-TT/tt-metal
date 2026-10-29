// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PMK assembly: the ONE-TARGET signal part (pmk_sig.cpp with the semaphore id at compile time): after this RISC's
// writes landed, +1 on program semaphore SEM of one core. For composites at the 256 runtime-arg limit (M1's TP2 H10 on
// the lane BRISC: each lane +1 lane 0's SEM_P15 after its y15 rows, 3 args instead of pmk_sig's 5). Compile time at
// PH_CT0: [sem_id]. Runtime args at ph_rt0: [active, x, y]; active = 0 -> nothing.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a = ph_rt0;
    constexpr uint32_t sem = get_compile_time_arg_val(PH_CT0 + 0);
    if (get_arg_val<uint32_t>(a + 0) == 0) {
        return;
    }
    noc_async_write_barrier();  // every write this RISC issued has landed before the counter moves
    noc_async_read_barrier();
    const uint32_t x = get_arg_val<uint32_t>(a + 1), y = get_arg_val<uint32_t>(a + 2);
    noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem)), 1);
    noc_async_atomic_barrier();
}
