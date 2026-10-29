// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// THE ROW FENCE (tt/row_fence.RowFence; program 2026-10-26 20:06:53Z, 14's pre-review 20:09:45Z), lane BRISC: one
// lane core per fabric link h of the line axis, on every chip. A launch of this program on a chip starts only after
// that chip's previous program finished, so every packet that program WAITED has landed here; what can remain is
// RELAYED traffic -- another chip's multi-hop packets still queued in this chip's onward router (forwarding channel),
// because on the FABRIC_1D mesh first-level ack is off and a sender's drain_close certifies hop 1 only. FENCE_MODE 1
// (mcast, the fix): every chip sends ONE token per direction as a LINE MULTICAST to the line end, a +1 of TOK on the
// same lane core of every chip it passes. At the source it rides this lane's worker channel; at every relay it rides
// the forwarding channel, i.e. FIFO behind every earlier packet from that source on this link at every hop. A lane
// waits TOK >= g x (nc - 1): every other chip's token has reached it, so every packet those chips sent on link h before
// their fence is delivered here, and passed this chip's router.
// FENCE_MODE 0 (hop1, the A/B arm = the 10-23 row barrier): 1-hop tokens, low -> high (BAR_L) then high -> low (BAR_R).
// A relay's own token leaves through its worker channel and can overtake relayed packets in its forwarding channel.
// FENCE_MODE 2 (none): no tokens (the reproducer's control).
// FENCE_CHECK (the reproducer, tt/row_fence.FenceBurst): after the fence, LAND >= g is expected on every destination of
// the burst (the burst's last packet +1s LAND); a short LAND counts LEAK += 1 and keeps the shortfall (SEEN = g -
// LAND). Probes: T0 (start), T1 (tokens sent), T2 (passed) = the low 32 bits of the cycle counter of the LAST launch.
#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "fs_common.hpp"

// the fence's own sync shard (16 words x 16 B per lane core; tt/row_fence.py F_* mirrors these)
constexpr uint32_t F_GEN = 0;
constexpr uint32_t F_TOK = 1;
constexpr uint32_t F_BAR_L = 2;
constexpr uint32_t F_BAR_R = 3;
constexpr uint32_t F_LAND = 4;
constexpr uint32_t F_LEAK = 5;
constexpr uint32_t F_SEEN = 6;
constexpr uint32_t F_CHECKS = 7;
constexpr uint32_t F_T0 = 8;
constexpr uint32_t F_T1 = 9;
constexpr uint32_t F_T2 = 10;

constexpr uint32_t MODE = get_compile_time_arg_val(0);  // 0 hop1 | 1 mcast | 2 none

using Hdr = tt::tt_fabric::LowLatencyPacketHeader;

struct FDir {
    tt::tt_fabric::WorkerToFabricEdmSender conn;
    volatile tt_l1_ptr Hdr* hdr;
    uint32_t range = 0;
};

void kernel_main() {
    using namespace tt::tt_fabric;
    size_t a = 0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    const uint32_t my_x = get_arg_val<uint32_t>(a++);  // this lane core, NoC 0 (the peers' lane h is the same core)
    const uint32_t my_y = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t nc = get_arg_val<uint32_t>(a++);
    const uint32_t check = get_arg_val<uint32_t>(a++);  // FENCE_CHECK on this lane / chip (a burst destination)
    a = get_arg_val<uint32_t>(a);                       // fab_at: where connect() appended the fabric args
    volatile tt_l1_ptr uint32_t* gen = fs::word(sync, F_GEN);
    const uint32_t g = *gen + 1;
    fs::word(sync, F_T0)[0] = get_timestamp_32b();
    FDir dirs[2];  // 0 = toward lower positions, 1 = toward higher
    dirs[0].range = me;
    dirs[1].range = nc - 1 - me;
    if constexpr (MODE != 2) {
        for (uint32_t d = 0; d < 2; d++) {
            if (dirs[d].range) {
                dirs[d].conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
            }
        }
        PacketHeaderPool::reset();
        for (uint32_t d = 0; d < 2; d++) {
            if (dirs[d].range) {
                dirs[d].conn.open();
                dirs[d].hdr = reinterpret_cast<volatile tt_l1_ptr Hdr*>(PacketHeaderPool::allocate_header());
            }
        }
    }
    auto send = [&](FDir& d, uint8_t hops, bool mcast, uint32_t word) {
        if (mcast) {
            d.hdr->to_chip_multicast(MulticastRoutingCommandHeader{1, hops});
        } else {
            d.hdr->to_chip_unicast(hops);
        }
        d.hdr->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(my_x, my_y, sync + word * fs::WORD, 0), 1});
        d.conn.wait_for_empty_write_slot();
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)d.hdr, sizeof(Hdr));
        noc_async_writes_flushed();
    };
    if constexpr (MODE == 1) {
        for (uint32_t d = 0; d < 2; d++) {
            if (dirs[d].range) {
                send(dirs[d], static_cast<uint8_t>(dirs[d].range), true, F_TOK);  // to the line end, every chip +1
            }
        }
        fs::word(sync, F_T1)[0] = get_timestamp_32b();
        fs::wait_ge(fs::word(sync, F_TOK), g * (nc - 1));
    } else if constexpr (MODE == 0) {
        fs::word(sync, F_T1)[0] = get_timestamp_32b();
        if (dirs[0].range) {
            fs::wait_ge(fs::word(sync, F_BAR_L), g);  // the chips below entered
        }
        if (dirs[1].range) {
            send(dirs[1], 1, false, F_BAR_L);
            fs::wait_ge(fs::word(sync, F_BAR_R), g);  // the chips above entered
        }
        if (dirs[0].range) {
            send(dirs[0], 1, false, F_BAR_R);
        }
    } else {
        fs::word(sync, F_T1)[0] = get_timestamp_32b();
    }
    fs::word(sync, F_T2)[0] = get_timestamp_32b();
    if (check) {
        invalidate_l1_cache();
        const uint32_t land = fs::word(sync, F_LAND)[0];
        fs::word(sync, F_CHECKS)[0] = fs::word(sync, F_CHECKS)[0] + 1;
        if (static_cast<int32_t>(land - g) < 0) {
            fs::word(sync, F_LEAK)[0] = fs::word(sync, F_LEAK)[0] + 1;
            const uint32_t short_by = g - land;
            if (short_by > fs::word(sync, F_SEEN)[0]) {
                fs::word(sync, F_SEEN)[0] = short_by;
            }
        }
    }
    *gen = g;
    if constexpr (MODE != 2) {
        for (uint32_t d = 0; d < 2; d++) {
            if (dirs[d].range) {
                fs::drain_close(dirs[d].conn);
            }
        }
    }
    noc_async_full_barrier();
}
