// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SP-row sparse attention v2, the device-side UNION + gather (tt/spa_union.py), lane BRISC (spa_send.cpp's v1 protocol
// plus the bitmap exchange): after member 0 delivered this chip's union bitmap (S_LOCAL), its bitmap slot goes to every
// peer's slot of this chip's position (unicast writes, <= 4 KiB per packet) + one atomic increment of the peer's ARR_BM + p
// (flush: the bitmap landed first). A peer starts sending into this chip's compact buffer only after it has this chip's
// bitmap of the SAME launch, so the exchange is also the handshake that this chip left its previous launch (and the
// attention that read the compact buffer). Then per staged pool (spa_ur.cpp) the 4 rows go to compact rows 4 slot .. + 3
// of chip pos: a local write or ONE 4 x 1 KiB scatter packet; the end marker (slot SKIP) stops the loop. Then one atomic
// increment of every peer's ARR_DATA + p, the wait ARR_DATA + q >= GEN + 1 for every peer, and GEN + 1.
// The 2x4 (tt/spa_union.py, two stages) runs this kernel on BOTH lanes: the row lane (bitmap slot p -> the row-mates'
// slot p, then the stage-B pools into their compact buffers) and the column lane (its row's OR, slot 0 -> the column
// partner's slot 1, then the stage-A pools into the partner's relay rows; `extra` = one more increment after the data:
// the partner's ROW lane word ARR_DATA_A, which its stage-B walk waits for before reading relay rows).
// The bitmap slots are a CB of this program (CB_SLOTS: same L1 address on every chip, live only while the program runs):
// at its start every lane sends each peer one increment of the peer's RDY + p (its slots are live for this launch), and
// writes a peer's slot only after the peer's RDY of the same generation -- no persistent L1 beyond the sync words.
// Compile-time: CB_STAGE, CB_META, WORDS, CB_SLOTS, then the destination tensor's TensorAccessorArgs. Runtime: sync, the
// bitmap source slot, the destination slot (on the peers), the semaphore id to wait for before the bitmap send, p, n, up
// range, down range, destination address, extra x, extra y, extra address (0: none), then the fabric connections (up
// first, then down).
#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/spa/spa_stage.hpp"
#if defined(SPA_WALK_BOUND)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/spa/spa_bound.hpp"
#endif
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

constexpr uint32_t CB_STAGE = get_compile_time_arg_val(0);
constexpr uint32_t CB_META = get_compile_time_arg_val(1);
constexpr uint32_t WORDS = get_compile_time_arg_val(2);
constexpr uint32_t CB_SLOTS = get_compile_time_arg_val(3);
constexpr uint32_t RB = 1024;
constexpr uint32_t NHDR = 4;
constexpr uint32_t PKT = 4096;  // bitmap bytes per packet
constexpr uint32_t SKIP = 0xFFFFFFFFu;
constexpr uint32_t GEN = 0, ARR_BM = 1;  // sync words: GEN, ARR_BM + source position, ARR_DATA = 1 + n + source position

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
    size_t a = 0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    cb_reserve_back(CB_SLOTS, 1);
    const uint32_t slots = get_write_ptr(CB_SLOTS);
    const uint32_t bm_src = slots + get_arg_val<uint32_t>(a++) * WORDS * 4;
    const uint32_t bm_dst = slots + get_arg_val<uint32_t>(a++) * WORDS * 4;
    const uint32_t wait_sem = get_arg_val<uint32_t>(a++);
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t nchips = get_arg_val<uint32_t>(a++);
    const uint32_t up = get_arg_val<uint32_t>(a++);
    const uint32_t dn = get_arg_val<uint32_t>(a++);
    const uint32_t y_addr = get_arg_val<uint32_t>(a++);
    const uint32_t extra_x = get_arg_val<uint32_t>(a++);
    const uint32_t extra_y = get_arg_val<uint32_t>(a++);
    const uint32_t extra_addr = get_arg_val<uint32_t>(a++);
#if defined(SPA_WALKERS)
    // A0 (GLM53_SPA_WALKERS=1, the ROW lane only): the members' count + NoC (x, y), before the fabric connections
    const uint32_t mw_m = get_arg_val<uint32_t>(a++);
    const uint32_t mxy0 = a;
    a += 2 * mw_m;
#endif
#if defined(SPA_A2B)
    // A2b (GLM53_SPA_A2B=1): the lane NCRISC is a second sender on fabric link 1 (spa_ur.cpp) for the slots of the ODD
    // members whose destination lies in a direction it connected (a2b_up / a2b_dn); this BRISC sends every other slot.
    // a2b_mask: the peers whose NCRISC reaches this chip (their link-1 arrivals: sync words ARR2 + q, generation GEN2)
    const uint32_t a2b_up = get_arg_val<uint32_t>(a++);
    const uint32_t a2b_dn = get_arg_val<uint32_t>(a++);
    const uint32_t a2b_mask = get_arg_val<uint32_t>(a++);
#endif
    const auto y = TensorAccessor(y_args, y_addr);
    const uint32_t ARR_DATA = 1 + nchips, RDY = 3 + 2 * nchips;
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* gen = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + GEN * 4);
    const uint32_t g = *gen + 1;
#if defined(SPA_A2B)
    constexpr uint32_t GEN2 = 24, ARR2 = 25;  // the link-1 arrivals' own generation (A2b launches only) + one word per peer
    volatile tt_l1_ptr uint32_t* gen2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + GEN2 * 4);
    const uint32_t g2 = *gen2 + 1;
#endif
    SPA_STAMP(sync, 16);
#if defined(SPA_WALK_BOUND)
    // GLM53_SPA_WALK_BOUND (spa_bound.hpp; the ROW lane only: SPA_WALKERS): this lane's waits bounded; an expiry records
    // and the protocol goes on (the member sweep ends; the arrivals, GEN, drain and close still run). Stash: CB_MSLOT's tail
    spab::State sb;
    sb.sync = sync;
    sb.lx = my_x[noc_index];
    sb.ly = my_y[noc_index];
    sb.g = g;
    cb_reserve_back(SPA_CB_MSLOT, 1);
#if defined(SPA_WALK_HDRB)
    sb.stash = get_write_ptr(SPA_CB_MSLOT) + 2 * mw_m * (4096 + 16) + 48 * mw_m;
#else
    sb.stash = get_write_ptr(SPA_CB_MSLOT) + 2 * mw_m * (4096 + 16);
#endif
#endif
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
    auto hdr = [&](Dir& d) {
        if (d.next == NHDR) {
            noc_async_writes_flushed();  // the ring wraps: the headers' previous packets have left L1
            d.next = 0;
        }
        return d.data[d.next++];
    };
    // RDY: this lane's slots are live for this launch (one increment of every peer's RDY + p)
    for (uint32_t q = 0; q < nchips; ++q) {
        if (q == p) {
            continue;
        }
        Dir& d = dirs[q > p ? 0 : 1];
        volatile tt_l1_ptr Hdr* h = hdr(d);
        h->to_chip_unicast(static_cast<uint8_t>(q > p ? q - p : p - q));
        h->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
            safe_get_noc_addr(my_x[noc_index], my_y[noc_index], sync + (RDY + p) * 4, 0), 1});
        d.conn.wait_for_empty_write_slot();
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
    }
    // the bitmap exchange (this chip's slot -> every peer's slot of position p, once the peer's slots are live) + one
    // ARR_BM increment per peer
#if defined(SPA_WALK_BOUND)
    {
        spab::Clock c;
        volatile tt_l1_ptr uint32_t* ws = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(wait_sem));
        while (!spab::wait_min(ws, 1, c)) {
            SPAB_EXPIRE(sb, spab::US_SEM, *ws, 1, wait_sem, c.el);
            SPAB_RESUME(c);
        }
    }
#else
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(wait_sem)), 1);
#endif
    const uint32_t own = bm_src;
    for (uint32_t q = 0; q < nchips; ++q) {
        if (q == p) {
            continue;
        }
        Dir& d = dirs[q > p ? 0 : 1];
        const uint32_t hops = q > p ? q - p : p - q;
        volatile tt_l1_ptr uint32_t* rdy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + (RDY + q) * 4);
#if defined(SPA_WALK_BOUND)
        {
            spab::Clock c;
            while (!spab::wait_min(rdy, g, c)) {
                SPAB_EXPIRE(sb, spab::US_RDY, *rdy, g, q, c.el);
                SPAB_RESUME(c);
            }
        }
#else
        while (*rdy < g) {
            invalidate_l1_cache();
        }
#endif
        for (uint32_t o = 0; o < WORDS * 4; o += PKT) {
            const uint32_t len = WORDS * 4 - o < PKT ? WORDS * 4 - o : PKT;
            volatile tt_l1_ptr Hdr* h = hdr(d);
            h->to_chip_unicast(static_cast<uint8_t>(hops));
            h->to_noc_unicast_write(NocUnicastCommandHeader{safe_get_noc_addr(my_x[noc_index], my_y[noc_index], bm_dst + o, 0)}, len);
            d.conn.wait_for_empty_write_slot();
            d.conn.send_payload_without_header_non_blocking_from_address(own + o, len);
            d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
        }
        volatile tt_l1_ptr Hdr* h = hdr(d);
        h->to_chip_unicast(static_cast<uint8_t>(hops));
        h->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
            safe_get_noc_addr(my_x[noc_index], my_y[noc_index], sync + (ARR_BM + p) * 4, 0), 1});
        d.conn.wait_for_empty_write_slot();
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
    }
    SPA_STAMP(sync, 17);  // the bitmaps sent
    // one staged batch of spa::NB pools from the lane NCRISC's CB stream (the end marker: done = true)
    auto cb_batch = [&](bool& done) {
        uint32_t k = 0;
        for (; k < spa::NB; ++k) {
            cb_wait_front(CB_META, k + 1);
            volatile tt_l1_ptr uint32_t* mt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_META) + k * spa::MB);
            const uint32_t slot = mt[0], pos = mt[1];
            if (slot == spa::SKIP) {  // the end marker: pushed alone after a closed batch (k == 0)
                done = true;
                break;
            }
            if (slot == spa::DUMMY) {
                continue;
            }
            cb_wait_front(CB_STAGE, k + 1);
            const uint32_t src = get_read_ptr(CB_STAGE) + k * spa::PB;
            if (pos == p) {
                for (uint32_t j = 0; j < 4; ++j) {
                    noc_async_write_page(4 * slot + j, y, src + j * RB);
                }
            } else {
                Dir& d = dirs[pos > p ? 0 : 1];
                const uint32_t hops = pos > p ? pos - p : p - pos;
                volatile tt_l1_ptr Hdr* h = hdr(d);
                h->to_chip_unicast(static_cast<uint8_t>(hops));
                h->to_noc_unicast_scatter_write(
                    NocUnicastScatterCommandHeader(
                        {y.get_noc_addr(4 * slot, 0, 0), y.get_noc_addr(4 * slot + 1, 0, 0), y.get_noc_addr(4 * slot + 2, 0, 0),
                         y.get_noc_addr(4 * slot + 3, 0, 0)},
                        {static_cast<uint16_t>(RB), static_cast<uint16_t>(RB), static_cast<uint16_t>(RB)}),
                    4 * RB);
                d.conn.wait_for_empty_write_slot();
                d.conn.send_payload_without_header_non_blocking_from_address(src, 4 * RB);
                d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
            }
        }
        if (done) {
            cb_pop_front(CB_META, 1);
            return;
        }
        noc_async_writes_flushed();  // the batch's staged rows have left (local writes and packet payloads)
        cb_wait_front(CB_STAGE, spa::NB);  // (a batch's DUMMY pages too: pushed with it)
        cb_pop_front(CB_STAGE, spa::NB);
        cb_pop_front(CB_META, spa::NB);
    };
#if defined(SPA_WALKERS)
    // A0: sweep the members' 2 slots each (spa_mw.cpp: 4 KiB data, then (slot, d, valid)); a full slot -> the same 4 KiB
    // scatter packet as a staged pool; after the sweep's packets left (one flush), clear the valid words and +1 each
    // owner's slot-free semaphore. Ends when every member counted DONE (after its last slot write landed) and a sweep
    // made after that observation found nothing; with the 2x4 relay pass (SPA_WALK_STAGE) also the CB end marker.
    {
        cb_reserve_back(SPA_CB_MSLOT, 1);
        const uint32_t mbase = get_write_ptr(SPA_CB_MSLOT);
        volatile tt_l1_ptr uint32_t* donep = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SPA_S_DONE));
        volatile tt_l1_ptr uint32_t* wready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + SPA_W_READY * 4);
#if defined(SPA_WALK_BOUND)
        {
            spab::Clock c;  // the lane NCRISC cleared the slots' valid words of this launch
            while (!spab::wait_min(wready, g, c)) {
                SPAB_EXPIRE(sb, spab::US_WREADY, *wready, g, 0, c.el);
                SPAB_RESUME(c);
            }
        }
#else
        while (*wready < g) {  // the lane NCRISC cleared the slots' valid words of this launch
            invalidate_l1_cache();
        }
#endif
#if defined(SPA_WALK_STAGE)
        bool cb_done = false;
#else
        bool cb_done = true;  // no lane stream: CB_STAGE / CB_META are not allocated
        (void)cb_batch;
#endif
        uint16_t freed[128];
#if defined(SPA_WALK_BOUND)
        spab::Clock sc;  // no progress (no slot sent, no DONE change, no lane batch) for the bound -> expiry
        uint32_t done_last = 0xFFFFFFFFu;
#endif
        for (;;) {
            invalidate_l1_cache();
            const uint32_t done_seen = *donep;
            uint32_t nf = 0;
            for (uint32_t i = 0; i < 2 * mw_m; ++i) {
                const uint32_t sl = mbase + i * (spa::PB + 16);
                volatile tt_l1_ptr uint32_t* mt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sl + spa::PB);
                if (mt[2] != 1) {
                    continue;
                }
                const uint32_t slot = mt[0], pos = mt[1];
#if defined(SPA_A2B)
                if (((i / 2) & 1u) && (pos > p ? a2b_up : a2b_dn)) {
                    continue;  // the lane NCRISC's slot (link 1)
                }
#endif
                Dir& d = dirs[pos > p ? 0 : 1];
                const uint32_t hops = pos > p ? pos - p : p - pos;
                volatile tt_l1_ptr Hdr* h = hdr(d);
                h->to_chip_unicast(static_cast<uint8_t>(hops));
                h->to_noc_unicast_scatter_write(
                    NocUnicastScatterCommandHeader(
                        {y.get_noc_addr(4 * slot, 0, 0), y.get_noc_addr(4 * slot + 1, 0, 0), y.get_noc_addr(4 * slot + 2, 0, 0),
                         y.get_noc_addr(4 * slot + 3, 0, 0)},
                        {static_cast<uint16_t>(RB), static_cast<uint16_t>(RB), static_cast<uint16_t>(RB)}),
                    4 * RB);
                d.conn.wait_for_empty_write_slot();
                d.conn.send_payload_without_header_non_blocking_from_address(sl, 4 * RB);
                d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
                freed[nf++] = static_cast<uint16_t>(i);
            }
            if (nf) {
                noc_async_writes_flushed();  // the sweep's payloads have left the slots
                for (uint32_t f = 0; f < nf; ++f) {
                    const uint32_t i = freed[f];
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mbase + i * (spa::PB + 16) + spa::PB)[2] = 0;
                    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(mxy0 + 2 * (i / 2)),
                                                   get_arg_val<uint32_t>(mxy0 + 2 * (i / 2) + 1),
                                                   get_semaphore(SPA_S_FREE0 + (i % 2))),
                                      1);
                }
            }
#if defined(SPA_WALK_STAGE)
            if (!cb_done && cb_pages_available_at_front(CB_META, 1)) {
                cb_batch(cb_done);
            }
#endif
            if (nf == 0 && done_seen >= mw_m && cb_done) {
                break;
            }
#if defined(SPA_WALK_BOUND)
            if (nf || done_seen != done_last) {
                sc.reset();
                done_last = done_seen;
            } else if (!sc.ok()) {  // seen = the DONE count, want = the members, ctx = the lane stream's end seen
                SPAB_EXPIRE(sb, spab::US_SWEEP, done_seen, mw_m, cb_done ? 1u : 0u, sc.el);
                SPAB_RESUME(sc);
            }
#endif
        }
    }
#else
    for (bool done = false; !done;) {  // one staged batch of spa::NB pools per pass (spa_stage.hpp)
        cb_batch(done);
    }
#endif
    noc_async_write_barrier();
    SPA_STAMP(sync, 18);  // every pool sent
    // one arrival increment per peer (flush: the peer's writes from this chip landed first)
    for (uint32_t q = 0; q < nchips; ++q) {
        if (q == p) {
            continue;
        }
        Dir& d = dirs[q > p ? 0 : 1];
        const uint32_t hops = q > p ? q - p : p - q;
        volatile tt_l1_ptr Hdr* h = hdr(d);
        h->to_chip_unicast(static_cast<uint8_t>(hops));
        h->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
            safe_get_noc_addr(my_x[noc_index], my_y[noc_index], sync + (ARR_DATA + p) * 4, 0), 1});
        d.conn.wait_for_empty_write_slot();
        d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
        if (extra_addr != 0) {  // the partner's row lane: its relay rows from this chip landed (same route: after the data)
            volatile tt_l1_ptr Hdr* h2 = hdr(d);
            h2->to_chip_unicast(static_cast<uint8_t>(hops));
            h2->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(extra_x, extra_y, extra_addr, 0), 1});
            d.conn.wait_for_empty_write_slot();
            d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h2, sizeof(Hdr));
        }
    }
    noc_async_writes_flushed();
    // the waiter: every peer's rows landed in this chip's compact buffer
    for (uint32_t q = 0; q < nchips; ++q) {
        if (q != p) {
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + (ARR_DATA + q) * 4);
#if defined(SPA_WALK_BOUND)
            spab::Clock c;
            while (!spab::wait_min(w, g, c)) {
                SPAB_EXPIRE(sb, spab::US_ARR_DATA, *w, g, q, c.el);
                SPAB_RESUME(c);
            }
#else
            while (*w < g) {
                invalidate_l1_cache();
            }
#endif
        }
    }
#if defined(SPA_A2B)
    for (uint32_t q = 0; q < nchips; ++q) {  // and every link-1 sender's (its arrival came after its data on link 1)
        if (q != p && ((a2b_mask >> q) & 1u)) {
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + (ARR2 + q) * 4);
#if defined(SPA_WALK_BOUND)
            spab::Clock c;
            while (!spab::wait_min(w, g2, c)) {
                SPAB_EXPIRE(sb, spab::US_ARR2, *w, g2, q, c.el);
                SPAB_RESUME(c);
            }
#else
            while (*w < g2) {
                invalidate_l1_cache();
            }
#endif
        }
    }
#endif
    SPA_STAMP(sync, 19);  // every peer's data arrived
    *gen = g;
#if defined(SPA_A2B)
    *gen2 = g2;
#endif
    for (uint32_t d = 0; d < 2; d++) {
        if (dirs[d].range) {  // drain (every slot free again) then close, as fs_common.hpp drain_close
            while (dirs[d].conn.get_num_free_write_slots() != dirs[d].conn.num_buffers_per_channel) {
            }
            dirs[d].conn.close();
        }
    }
    noc_async_full_barrier();
}
