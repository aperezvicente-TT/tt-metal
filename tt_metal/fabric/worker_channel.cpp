// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/worker_channel.hpp>

#include <thread>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

namespace tt::tt_fabric::experimental {

WorkerChannelReclaim reclaim_worker_channel(
    uint32_t physical_chip_id,
    const WorkerChannel& ch,
    std::chrono::microseconds drain,
    std::chrono::microseconds timeout) {
    namespace ci = tt::tt_fabric::connection_interface;
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const tt_cxy_pair core(physical_chip_id, ch.router_noc);
    auto read_word = [&](uint32_t addr) {
        uint32_t v = 0;
        cluster.read_core(&v, sizeof(v), core, addr);
        return v;
    };
    WorkerChannelReclaim r;
    r.handshake_before = read_word(ch.handshake_addr);
    if (r.handshake_before == ci::unused_connection_value) {
        return r;
    }
    std::this_thread::sleep_for(drain);
    if (r.handshake_before == ci::open_connection_value) {
        const uint32_t req = ci::close_connection_request_value;
        cluster.write_core(&req, sizeof(req), core, ch.handshake_addr);
        cluster.l1_barrier(physical_chip_id);
    }
    const auto t0 = std::chrono::steady_clock::now();
    while (read_word(ch.handshake_addr) != ci::unused_connection_value) {
        if (std::chrono::steady_clock::now() - t0 > timeout) {
            r.timed_out = true;
            return r;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    r.closed = true;
    r.read_counter = read_word(ch.worker_location_info_addr + offsetof(EDMChannelWorkerLocationInfo, edm_read_counter));
    SenderChannelProducerCursor cursor{};
    cursor.write_counter = r.read_counter;
    cursor.write_index = ch.num_buffers ? r.read_counter % ch.num_buffers : 0;
    cluster.write_core(&cursor, sizeof(cursor), core, ch.producer_cursor_addr);
    cluster.l1_barrier(physical_chip_id);
    return r;
}

}  // namespace tt::tt_fabric::experimental
