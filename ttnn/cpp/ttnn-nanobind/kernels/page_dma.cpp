// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Persistent page-DMA kernel (one per service core): moves pages between pinned host slabs and interleaved DRAM
// buffers as the host's descriptor ring asks. Protocol: page_dma_proto.h; logic: page_dma_body.h (shared with the
// host emulator).
//
// Runtime args: window (L1 base of this core's PdmaCtrl / tables / ring / staging), pcie_xy_enc (the PCIe core as
// the chip's NoC addresses it, PinnedMemory::NocAddr::pcie_xy_enc).
//
// Host memory is addressed with raw 64-bit IOVAs (Blackhole pinned memory, no iATU), so PCIe reads and writes use the
// coordinate overloads that program NOC_TARG_ADDR_MID / NOC_RET_ADDR_MID. MID is sticky and plain transfers do not
// reprogram it: every PCIe batch is closed with the matching clear before any on-chip transfer is issued.

#include <cstddef>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "page_dma_body.h"

struct DeviceIO {
    uint32_t pcie_xy_enc;

    FORCE_INLINE uint32_t ld(uint32_t a) {
        invalidate_l1_cache();  // the host writes the ring / tail over PCIe
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a);
    }
    FORCE_INLINE void st(uint32_t a, uint32_t v) { *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a) = v; }

    FORCE_INLINE void host_read(uint64_t src, uint32_t l1, uint32_t n) {
        noc_read_init_state<read_cmd_buf>(noc_index);  // response-marked reads (else they never retire)
        for (uint32_t b = 0; b < n; b += NOC_MAX_BURST_SIZE) {
            const uint32_t m = n - b < NOC_MAX_BURST_SIZE ? n - b : NOC_MAX_BURST_SIZE;
            noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
                noc_index, pcie_xy_enc, src + b, l1 + b, m);
        }
        noc_async_read_clear_pcie_state();
    }

    FORCE_INLINE void host_write(uint32_t l1, uint64_t dst, uint32_t n) {
        noc_write_init_state<write_cmd_buf>(noc_index, NOC_UNICAST_WRITE_VC);
        for (uint32_t b = 0; b < n; b += NOC_MAX_BURST_SIZE) {
            const uint32_t m = n - b < NOC_MAX_BURST_SIZE ? n - b : NOC_MAX_BURST_SIZE;
            noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
                noc_index, l1 + b, pcie_xy_enc, dst + b, m, 1);
        }
        noc_async_write_clear_pcie_state(noc_index, write_cmd_buf);
    }

    FORCE_INLINE void dram_read(uint32_t bank, uint32_t addr, uint32_t l1, uint32_t n) {
        noc_async_read(get_noc_addr_from_bank_id<true>(bank, addr), l1, n);
    }
    FORCE_INLINE void dram_write(uint32_t l1, uint32_t bank, uint32_t addr, uint32_t n) {
        noc_async_write(l1, get_noc_addr_from_bank_id<true>(bank, addr), n);
    }

    FORCE_INLINE void read_barrier() { noc_async_read_barrier(); }
    FORCE_INLINE void writes_flushed() { noc_async_writes_flushed(); }
    FORCE_INLINE void write_barrier() { noc_async_write_barrier(); }
    FORCE_INLINE void end_pcie() {
        noc_async_read_clear_pcie_state();
        noc_async_write_clear_pcie_state(noc_index, write_cmd_buf);
    }
    FORCE_INLINE void idle() {}
    FORCE_INLINE uint32_t clock() { return get_timestamp_32b(); }
};

void kernel_main() {
    const uint32_t window = get_arg_val<uint32_t>(0);
    DeviceIO io{get_arg_val<uint32_t>(1)};
    PdmaKernel<DeviceIO> k{io, window};
    k.run();
}
