// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_fabric::experimental {

// A router's VC0 worker channel as fabric_connection_info_t describes it.
struct WorkerChannel {
    tt::tt_metal::CoreCoord router_noc;  // virtual NoC coordinates of the router's ETH core
    uint32_t handshake_addr = 0;         // edm_connection_handshake_addr
    uint32_t worker_location_info_addr = 0;
    uint32_t producer_cursor_addr = 0;  // buffer_index_semaphore_id
    uint32_t num_buffers = 0;
};

struct WorkerChannelReclaim {
    uint32_t handshake_before = 0;  // 0 unused, 1 open, 2 close requested
    bool closed = false;            // the router acknowledged a close this call requested or found pending
    bool timed_out = false;         // the router did not acknowledge: leave the channel alone
    uint32_t read_counter = 0;      // the router's read counter the producer cursor now holds
};

// Host-side reclaim of a worker channel whose producer died without closing it (its core was reset
// with the connection open). Waits `drain` for packets the producer already handed to the router,
// requests a close as the producer would, waits up to `timeout` for the router's acknowledgement,
// then sets the router-side producer cursor to the router's read counter, so the next producer
// starts at the router's next slot. A channel found unused is left untouched. The router's teardown
// acknowledgement increments the dead producer's teardown word in its core's L1: call this before
// anything new runs on that core. Chips are physical chip ids of chips this process opened.
WorkerChannelReclaim reclaim_worker_channel(
    uint32_t physical_chip_id,
    const WorkerChannel& channel,
    std::chrono::microseconds drain = std::chrono::milliseconds(10),
    std::chrono::microseconds timeout = std::chrono::milliseconds(10));

}  // namespace tt::tt_fabric::experimental
