// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRAM-image pull: copy byte runs from pinned host memory (read over PCIe by this core) into DRAM banks.
//
// Runtime args: src_lo, src_hi (64-bit PCIe address of the pinned slot), n_runs, then n_runs x {src_off, bank,
// dst_addr, len}: run r copies len bytes from slot offset src_off to DRAM bank `bank` at bank offset dst_addr. All
// offsets / lengths are multiples of 64 B. Two L1 staging buffers of `stage` bytes (CB 0): the PCIe read of block k
// overlaps the DRAM write of block k - 1.
//
// The PCIe source is a raw 64-bit address (Blackhole pinned memory: IOVA in the non-iATU range), so reads use the
// coordinate overload of noc_read_with_state, which programs NOC_TARG_ADDR_MID. MID is sticky on read_cmd_buf: it is
// cleared before the kernel returns, or the next on-chip read on this core would be routed to host memory.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t pcie_xy_enc = get_compile_time_arg_val(0);
    constexpr uint32_t stage = get_compile_time_arg_val(1);  // L1 staging buffer bytes (multiple of burst)
    constexpr uint32_t burst = get_compile_time_arg_val(2);  // <= NOC_MAX_BURST_SIZE

    const uint64_t src_base = (static_cast<uint64_t>(get_arg_val<uint32_t>(1)) << 32) | get_arg_val<uint32_t>(0);
    const uint32_t n_runs = get_arg_val<uint32_t>(2);
    const uint32_t l1_base = get_write_ptr(0);

    uint32_t k = 0;
    for (uint32_t r = 0; r < n_runs; r++) {
        const uint32_t src_off = get_arg_val<uint32_t>(3 + 4 * r);
        const uint32_t bank = get_arg_val<uint32_t>(4 + 4 * r);
        const uint32_t dst_addr = get_arg_val<uint32_t>(5 + 4 * r);
        const uint32_t len = get_arg_val<uint32_t>(6 + 4 * r);
        for (uint32_t off = 0; off < len; off += stage, k++) {
            const uint32_t n = (len - off) < stage ? (len - off) : stage;
            const uint32_t slot = l1_base + (k & 1) * stage;
            if (k >= 2) {
                noc_async_writes_flushed();  // the slot's previous block (k - 2) has left L1
            }
            const uint64_t src = src_base + src_off + off;
            for (uint32_t b = 0; b < n; b += burst) {
                const uint32_t m = (n - b) < burst ? (n - b) : burst;
                noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
                    noc_index, pcie_xy_enc, src + b, slot + b, m);
            }
            noc_async_read_barrier();
            noc_async_write(slot, get_noc_addr_from_bank_id<true>(bank, dst_addr + off), n);
        }
    }
    noc_async_write_barrier();
    noc_async_read_clear_pcie_state();
}
