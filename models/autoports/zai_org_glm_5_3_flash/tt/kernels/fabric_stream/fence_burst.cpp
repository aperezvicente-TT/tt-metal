// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The row fence's CLASS REPRODUCER (tt/row_fence.FenceBurst; 14's spec, program 2026-10-26 20:09:45Z), lane BRISC
// on the line-end chip (position 0) only: a RELAYED late packet. NPKT 4-KiB packets as LINE MULTICASTS over the whole
// line (range nc - 1: every middle chip relays them in its forwarding channel) into a DRAM landing buffer, then ONE
// multicast +1 of LAND on the fence lane core of every other chip. Nothing waits for any of it: this program ends at
// the hop-1 drain, so the burst's tail is still in the middle chips' routers when the next program (the fence + its
// LAND check) starts. Other chips (and any lane with range 0) return at once.
#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "fs_common.hpp"

constexpr uint32_t F_LAND = 4;  // row_fence.cpp
constexpr uint32_t CB_SRC = get_compile_time_arg_val(0);
constexpr uint32_t PAGE = 4096;
constexpr uint32_t NHDR = 4;

using Hdr = tt::tt_fabric::LowLatencyPacketHeader;

void kernel_main() {
    using namespace tt::tt_fabric;
    constexpr auto dst_args = TensorAccessorArgs<1>();
    size_t a = 0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);  // the fence's sync shard (LAND on its lane core)
    const uint32_t lx = get_arg_val<uint32_t>(a++);    // the fence lane core, NoC 0
    const uint32_t ly = get_arg_val<uint32_t>(a++);
    const uint32_t range = get_arg_val<uint32_t>(a++);  // 0: not the line-end sender -> nothing to do
    const uint32_t dst_addr = get_arg_val<uint32_t>(a++);
    const uint32_t npkt = get_arg_val<uint32_t>(a++);
    const uint32_t npages = get_arg_val<uint32_t>(a++);
    a = get_arg_val<uint32_t>(a);  // fab_at
    if (range == 0) {
        return;
    }
    const auto dst = TensorAccessor(dst_args, dst_addr);
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    PacketHeaderPool::reset();
    conn.open();
    volatile tt_l1_ptr Hdr* hd[NHDR];
    for (uint32_t k = 0; k < NHDR; k++) {
        hd[k] = reinterpret_cast<volatile tt_l1_ptr Hdr*>(PacketHeaderPool::allocate_header());
        hd[k]->to_chip_multicast(MulticastRoutingCommandHeader{1, static_cast<uint8_t>(range)});
    }
    const uint32_t src = get_write_ptr(CB_SRC);  // payload: whatever the CB holds (the landing content is not checked)
    for (uint32_t i = 0; i < npkt; i++) {
        volatile tt_l1_ptr Hdr* h = hd[i % NHDR];
        if (i % NHDR == 0) {
            noc_async_writes_flushed();  // the headers about to be reused have left L1
        }
        h->to_noc_unicast_write(NocUnicastCommandHeader{dst.get_noc_addr(i % npages, 0, 0)}, PAGE);
        conn.wait_for_empty_write_slot();
        conn.send_payload_without_header_non_blocking_from_address(src, PAGE);
        conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
    }
    noc_async_writes_flushed();
    volatile tt_l1_ptr Hdr* l = hd[0];
    l->to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(lx, ly, sync + F_LAND * fs::WORD, 0), 1});
    conn.wait_for_empty_write_slot();
    conn.send_payload_flush_non_blocking_from_address((uint32_t)l, sizeof(Hdr));
    noc_async_writes_flushed();
    fs::drain_close(conn);  // hop 1 only: the relayed tail is still in flight
    noc_async_full_barrier();
}
