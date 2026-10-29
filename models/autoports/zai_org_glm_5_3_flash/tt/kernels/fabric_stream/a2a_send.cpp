// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 16 FabricStreamA2A2D (tt/fabric_a2a.py, doc pmk/merge/a2a_fs_design.md), lane BRISC: the SPA q / o all-to-all
// as host-built page-run lists (every run = RUN consecutive tile pages, page-contiguous on both sides). One lane = one
// worker connection per (link, direction) on ONE mesh axis: a B lane (the row / the line, link L, both directions) or,
// on the 2x4, an A lane (axis 0, the column partner, link L). Per launch:
//   1. CREDITS: one +1 of CREDIT + (this chip's axis position) on every peer this lane sends to (its same-role lane,
//      unicast): "this chip started the launch" -- a peer writes into this chip's y / relay store only after it (the
//      op's init barrier; the previous launch's consumers on this chip are done when its next program starts).
//   2. RUNS (fs_send.cpp FS_SCATTER's packets): per run, the credit of its destination, then every staged page PAIR
//      (a2a_read.cpp) as one unicast scatter packet (to_chip_unicast(hops)) into the destination's y or relay store
//      (or two local page writes: the own block); a signalled run's last pair is a fused scatter write + atomic +1
//      (flush) of the destination's counter: the waiter's YLAND + j (landed in y) or the forwarding B lane's RELAY.
//   3. DONE: +1 on this chip's waiter (B lane 0) after the write barrier; a non-waiter drains and closes here.
//   4. The WAITER: every YLAND word at g x its per-launch count, DONE at g x lanes, then THE PER-HOP DRAIN row barrier
//      of 1-hop tokens (fs_common.hpp; any multi-hop line), drain + close; GEN_B = g.
// Compile-time: NB, CB_STAGE, RUN, CONN_AT (the fabric connection args' index), then the TensorAccessorArgs of y and of
// the relay store. Runtime: sync, y, relay, waiter x, waiter y, up range, down range, is_waiter, barrier, lanes (DONE
// per launch), n_credit + n_credit x (dir | hops << 2, x | y << 8, word), n_wait + n_wait x (word, per launch), n_runs +
// n_runs x (word 0 (a2a_common.hpp), destination page, signal x | y << 8 | word << 16), zero padding to CONN_AT, then
// the fabric connections (up first, then down).
#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "fs_common.hpp"
#include "a2a_common.hpp"

constexpr uint32_t NB = get_compile_time_arg_val(0);
constexpr uint32_t CB_STAGE = get_compile_time_arg_val(1);
constexpr uint32_t RUN = get_compile_time_arg_val(2);
constexpr uint32_t CONN_AT = get_compile_time_arg_val(3);
constexpr uint32_t TB = 2048;
constexpr uint32_t NHDR = 4;  // data-header ring per direction (<= 12 per RISC: tt/fabric_a2a.py asserts the budget)
static_assert(RUN % NB == 0 && NB % 2 == 0, "page pairs never straddle a batch or a run");

using Hdr = tt::tt_fabric::LowLatencyPacketHeader;

struct Dir {
    tt::tt_fabric::WorkerToFabricEdmSender conn;
    volatile tt_l1_ptr Hdr* data[NHDR];
    uint32_t range = 0;
    uint32_t next = 0;
};

void kernel_main() {
    using namespace tt::tt_fabric;
    constexpr auto y_args = TensorAccessorArgs<4>();
    constexpr auto r_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    size_t a = 0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    const uint32_t y_addr = get_arg_val<uint32_t>(a++);
    const uint32_t r_addr = get_arg_val<uint32_t>(a++);
    const uint32_t wx = get_arg_val<uint32_t>(a++);  // this chip's waiter core (B lane 0), NoC 0
    const uint32_t wy = get_arg_val<uint32_t>(a++);
    const uint32_t up = get_arg_val<uint32_t>(a++);
    const uint32_t dn = get_arg_val<uint32_t>(a++);
    const uint32_t is_waiter = get_arg_val<uint32_t>(a++);
    const uint32_t barrier = get_arg_val<uint32_t>(a++);
    const uint32_t n_lanes = get_arg_val<uint32_t>(a++);
    const uint32_t n_credit = get_arg_val<uint32_t>(a++);
    const uint32_t credit0 = a;
    a += 3 * n_credit;
    const uint32_t n_wait = get_arg_val<uint32_t>(a++);
    const uint32_t wait0 = a;
    a += 2 * n_wait;
    const uint32_t n_runs = get_arg_val<uint32_t>(a++);
    const uint32_t run0 = a;
    a += 3 * n_runs;
    ASSERT(a <= CONN_AT);
    a = CONN_AT;
    const auto y = TensorAccessor(y_args, y_addr);
    const auto rl = TensorAccessor(r_args, r_addr);
    const uint32_t ranges[2] = {up, dn};
    Dir dirs[2];
    for (uint32_t d = 0; d < 2; d++) {
        dirs[d].range = ranges[d];
        if (ranges[d]) {
            dirs[d].conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        }
    }
    for (uint32_t d = 0; d < 2; d++) {
        if (dirs[d].range) {
            dirs[d].conn.open();
            for (uint32_t k = 0; k < NHDR; k++) {
                dirs[d].data[k] = reinterpret_cast<volatile tt_l1_ptr Hdr*>(PacketHeaderPool::allocate_header());
            }
        }
    }
    volatile tt_l1_ptr uint32_t* gen = fs::word(sync, a2a::GEN_B);
    invalidate_l1_cache();
    const uint32_t g = *gen + 1;
    auto hdr = [&](Dir& d) {
        if (d.next == NHDR) {
            noc_async_writes_flushed();  // the ring wraps: the headers' previous packets have left L1
            d.next = 0;
        }
        return d.data[d.next++];
    };
    // 1. credits: this chip started launch g
    for (uint32_t k = 0; k < n_credit; k++) {
        const uint32_t w0 = get_arg_val<uint32_t>(credit0 + 3 * k);
        const uint32_t xy = get_arg_val<uint32_t>(credit0 + 3 * k + 1);
        const uint32_t word = get_arg_val<uint32_t>(credit0 + 3 * k + 2);
        Dir& d = dirs[w0 & 3u];
        ASSERT((w0 & 3u) < 2 && d.range >= ((w0 >> 2) & 0xFu));
        volatile tt_l1_ptr Hdr* h = hdr(d);
        h->to_chip_unicast(static_cast<uint8_t>((w0 >> 2) & 0xFu));
        h->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
            safe_get_noc_addr(xy & 0xFFu, (xy >> 8) & 0xFFu, sync + word * a2a::WORD, 0), 1});
        d.conn.wait_for_empty_write_slot();
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
    }
    // 2. the runs
    auto send = [&](Dir& d, uint32_t src, uint64_t d0, uint64_t d1, bool signal, uint64_t sig_noc, uint32_t hops) {
        volatile tt_l1_ptr Hdr* h = hdr(d);
        h->to_chip_unicast(static_cast<uint8_t>(hops));
        d.conn.wait_for_empty_write_slot();
        if (signal) {
            h->to_noc_fused_unicast_scatter_write_atomic_inc(
                NocUnicastScatterAtomicIncFusedCommandHeader({d0, d1}, sig_noc, {static_cast<uint16_t>(TB)}, 1, true),
                2 * TB);
        } else {
            h->to_noc_unicast_scatter_write(
                NocUnicastScatterCommandHeader({d0, d1}, {static_cast<uint16_t>(TB)}), 2 * TB);
        }
        d.conn.send_payload_without_header_non_blocking_from_address(src, 2 * TB);
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
    };
    uint32_t credited = 0;  // positions whose credit this launch was already seen
    for (uint32_t k = 0; k < n_runs; k++) {
        const uint32_t w0 = get_arg_val<uint32_t>(run0 + 3 * k);
        const uint32_t page0 = get_arg_val<uint32_t>(run0 + 3 * k + 1);
        const uint32_t w2 = get_arg_val<uint32_t>(run0 + 3 * k + 2);
        const uint32_t route = w0 & 3u, hops = (w0 >> 2) & 0xFu, to_relay = (w0 >> 6) & 1u;
        const uint32_t cpos1 = (w0 >> 8) & 0xFu;
        const bool sig = ((w0 >> 12) & 1u) != 0;
        if (cpos1 != 0 && !(credited & (1u << cpos1))) {
            fs::wait_ge(fs::word(sync, a2a::CREDIT + cpos1 - 1), g);
            credited |= 1u << cpos1;
        }
        ASSERT(route == a2a::ROUTE_LOCAL || (route < 2 && hops > 0 && dirs[route].range >= hops));
        const uint64_t sig_noc =
            sig ? safe_get_noc_addr(w2 & 0xFFu, (w2 >> 8) & 0xFFu, sync + (w2 >> 16) * a2a::WORD, 0) : 0;
        for (uint32_t b = 0; b < RUN / NB; b++) {
            cb_wait_front(CB_STAGE, NB);
            const uint32_t buf = get_read_ptr(CB_STAGE);
            for (uint32_t i = 0; i < NB; i += 2) {
                const uint32_t idx = b * NB + i;
                const uint32_t pg = page0 + idx;
                const uint32_t src = buf + i * TB;
                if (route == a2a::ROUTE_LOCAL) {
                    noc_async_write_page(pg, y, src);
                    noc_async_write_page(pg + 1, y, src + TB);
                } else {
                    const uint64_t d0 = to_relay ? rl.get_noc_addr(pg, 0, 0) : y.get_noc_addr(pg, 0, 0);
                    const uint64_t d1 = to_relay ? rl.get_noc_addr(pg + 1, 0, 0) : y.get_noc_addr(pg + 1, 0, 0);
                    send(dirs[route], src, d0, d1, sig && idx + 2 == RUN, sig_noc, hops);
                }
            }
            noc_async_writes_flushed();  // the local writes and the packets' payload reads have left the batch
            cb_pop_front(CB_STAGE, NB);
        }
    }
    noc_async_write_barrier();
    // 3. DONE (a non-waiter closes first: its link's next holder may open)
    if (!is_waiter) {
        for (uint32_t d = 0; d < 2; d++) {
            if (dirs[d].range) {
                fs::drain_close(dirs[d].conn);
            }
        }
    }
    noc_semaphore_inc(safe_get_noc_addr(wx, wy, sync + a2a::DONE * a2a::WORD, 0), 1);
    noc_async_atomic_barrier();
    // 4. the waiter
    if (is_waiter) {
        for (uint32_t k = 0; k < n_wait; k++) {
            const uint32_t word = get_arg_val<uint32_t>(wait0 + 2 * k), per = get_arg_val<uint32_t>(wait0 + 2 * k + 1);
            fs::wait_ge(fs::word(sync, word), g * per);
        }
        fs::wait_ge(fs::word(sync, a2a::DONE), g * n_lanes);
        if (barrier) {  // THE PER-HOP DRAIN: no packet of this launch is in transit when any chip of the line ends
            auto token = [&](Dir& d, uint32_t word) {  // a data header reused: every data send has been flushed
                noc_async_writes_flushed();
                volatile tt_l1_ptr Hdr* th = d.data[0];
                th->to_chip_unicast(1);
                th->to_noc_unicast_atomic_inc(
                    NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(wx, wy, sync + word * a2a::WORD, 0), 1});
                d.conn.wait_for_empty_write_slot();
                d.conn.send_payload_flush_non_blocking_from_address((uint32_t)th, sizeof(Hdr));
                noc_async_writes_flushed();
            };
            if (dirs[1].range) {
                fs::wait_ge(fs::word(sync, a2a::BAR_L), g);  // the chips below landed everything
            }
            if (dirs[0].range) {
                token(dirs[0], a2a::BAR_L);
                fs::wait_ge(fs::word(sync, a2a::BAR_R), g);  // the chips above landed everything
            }
            if (dirs[1].range) {
                token(dirs[1], a2a::BAR_R);
            }
        }
        for (uint32_t d = 0; d < 2; d++) {
            if (dirs[d].range) {
                fs::drain_close(dirs[d].conn);
            }
        }
    }
    *gen = g;
    noc_async_full_barrier();
}
