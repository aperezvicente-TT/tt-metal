// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SP-row sparse attention v2, the device-side UNION + gather (tt/spa_union.py), lane NCRISC (the row lane on the 2x4):
//   1. waits for this chip's union bitmap (member 0 -> the bitmap slot of this chip's position, S_LOCAL);
//   2. its exclusive word prefix counts -> CB_PFX [bitmap | prefix] (the members' remap reads it), S_PFX on every member;
//      the union size > cap is flagged in the sync word STATUS (the pools past cap are not gathered);
//   3. waits for every line peer's bitmap (the peers' lane BRISCs write them into the slots of their positions and bump
//      ARR_BM + q; generation GEN + 1);
//   TWO (the 2x4): 3b. the row's OR of the 4 bitmaps -> the column lane's OR slot 0 (+ S_ORMINE there: its BRISC sends it
//      to the column partner, whose stage A pushes the pools THIS row needs from it into this chip's relay rows), and the
//      relay prefix table: relay index of pool p = its rank among the row-OR pools in the PARTNER's words;
//   4. walks each destination's bitmap (this chip first): word w = the 128-token unit w, owned by chip rank w % N; for
//      every pool this chip owns, its 4 latent rows (local row (w / N) 128 + 4 b .. + 3) -> CB_STAGE and (slot = its rank
//      in the destination's union, destination position) -> CB_META;
//   TWO: 4b. waits ARR_DATA_A (the partner's stage A landed), then walks again for the pools the partner owns: rows
//      4 ridx .. + 3 of the relay tensor;
//   then an end marker (slot SKIP).
// Compile-time: CB_STAGE, CB_META, CB_PFX, WORDS, TWO, CB_ROWOR, CB_RP, then the TensorAccessorArgs of the local store and
// of the relay tensor (the local store again when not TWO). Runtime: sync, bitmaps (n slots of WORDS words, same address
// on every chip), store, p, n, cap, members, N (the owner modulus: every chip), k (this chip's rank), kp (the column
// partner's rank), relay address, column lane x, y, its OR slot 0 address, relay cap (pools), then the members' NoC (x, y).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/spa/spa_stage.hpp"
#if defined(SPA_WALK_BOUND)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/spa/spa_bound.hpp"
#endif
#if defined(SPA_A2B)
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#endif

constexpr uint32_t CB_STAGE = get_compile_time_arg_val(0);
constexpr uint32_t CB_META = get_compile_time_arg_val(1);
constexpr uint32_t CB_PFX = get_compile_time_arg_val(2);
constexpr uint32_t WORDS = get_compile_time_arg_val(3);
constexpr uint32_t TWO = get_compile_time_arg_val(4);
constexpr uint32_t CB_ROWOR = get_compile_time_arg_val(5);
constexpr uint32_t CB_RP = get_compile_time_arg_val(6);
constexpr uint32_t CB_SLOTS = get_compile_time_arg_val(7);  // the bitmap slots (this lane's CB, n x WORDS words)
constexpr uint32_t CB_ORS = get_compile_time_arg_val(8);    // TWO: the column lane's OR slots (same address there)
constexpr uint32_t RB = 1024;   // one latent row: 512 bf16
constexpr uint32_t UNIT = 128;  // tokens per shard unit = one bitmap word
constexpr uint32_t SKIP = 0xFFFFFFFFu;
constexpr uint32_t S_PFX = 1, S_LOCAL = 2, S_ORMINE = 3;  // semaphore ids (tt/spa_union.py)
constexpr uint32_t GEN = 0, ARR_BM = 1;  // sync words: GEN, ARR_BM + position, ARR_DATA (spa_us.cpp), STATUS, ARR_DATA_A

inline uint32_t popc(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}

void kernel_main() {
    constexpr auto lat_args = TensorAccessorArgs<9>();
    constexpr auto rel_args = TensorAccessorArgs<lat_args.next_compile_time_args_offset()>();
#if defined(SPA_A2B)
    constexpr auto yc_args = TensorAccessorArgs<rel_args.next_compile_time_args_offset()>();  // the compact store (A2b)
#endif
    size_t a = 0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    a++;  // (the bitmap slots: CB_SLOTS)
    cb_reserve_back(CB_SLOTS, 1);
    const uint32_t bms = get_write_ptr(CB_SLOTS);
    const uint32_t lat_addr = get_arg_val<uint32_t>(a++);
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t cap = get_arg_val<uint32_t>(a++);
    const uint32_t m = get_arg_val<uint32_t>(a++);
    const uint32_t nall = get_arg_val<uint32_t>(a++);
    const uint32_t k = get_arg_val<uint32_t>(a++);
    const uint32_t kp = get_arg_val<uint32_t>(a++);
    const uint32_t rel_addr = get_arg_val<uint32_t>(a++);
    const uint32_t col_x = get_arg_val<uint32_t>(a++);
    const uint32_t col_y = get_arg_val<uint32_t>(a++);
    a++;  // (the column lane's OR slot 0: CB_ORS, the same address on the column lane)
    uint32_t col_or0 = 0;
    if constexpr (TWO) {
        cb_reserve_back(CB_ORS, 1);
        col_or0 = get_write_ptr(CB_ORS);
    }
    const uint32_t rcap = get_arg_val<uint32_t>(a++);
    const uint32_t xy0 = a;
    const auto lat = TensorAccessor(lat_args, lat_addr);
    const auto rel = TensorAccessor(rel_args, rel_addr);
    volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync);
    invalidate_l1_cache();  // GEN / ARR / the bitmap slots are written by the BRISC, member 0 and the peers' fabric writes
    const uint32_t g = sw[GEN] + 1;  // read before anything (the BRISC bumps GEN at the very end of the launch)
    SPA_STAMP(sync, 20);
    const uint32_t status = 1 + 2 * n, arr_a = 2 + 2 * n;
#if defined(SPA_WALK_BOUND)
    // GLM53_SPA_WALK_BOUND (spa_bound.hpp): this lane's waits bounded; an expiry records and the walk goes on (the end
    // protocol still runs: the peers and the program complete). The stash: CB_MSLOT's tail + 64 (the BRISC's at + 0)
    spab::State sb;
    sb.sync = sync;
    sb.lx = my_x[noc_index];
    sb.ly = my_y[noc_index];
    sb.g = g;
    cb_reserve_back(SPA_CB_MSLOT, 1);
#if defined(SPA_WALK_HDRB)
    sb.stash = get_write_ptr(SPA_CB_MSLOT) + 2 * m * (4096 + 16) + 48 * m + 64;
#else
    sb.stash = get_write_ptr(SPA_CB_MSLOT) + 2 * m * (4096 + 16) + 64;
#endif
#endif

    // 1. this chip's union bitmap
#if defined(SPA_WALK_BOUND)
    {
        spab::Clock c;
        volatile tt_l1_ptr uint32_t* sl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(S_LOCAL));
        while (!spab::wait_min(sl, 1, c)) {
            SPAB_EXPIRE(sb, spab::UR_LOCAL, *sl, 1, 0, c.el);
            SPAB_RESUME(c);
        }
    }
#else
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(S_LOCAL)), 1);
#endif
    invalidate_l1_cache();
    SPA_STAMP(sync, 21);  // this chip's union bitmap
    // 2. the prefix table for the members' remap
    cb_reserve_back(CB_PFX, 1);
    volatile tt_l1_ptr uint32_t* ubm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_PFX));
    volatile tt_l1_ptr uint32_t* pfx = ubm + WORDS;
    volatile tt_l1_ptr uint32_t* own = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bms + p * WORDS * 4);
    uint32_t run = 0;
    for (uint32_t w = 0; w < WORDS; ++w) {
        const uint32_t x = own[w];
        ubm[w] = x;
        pfx[w] = run;
        run += popc(x);
    }
    sw[status] = run > cap ? run : 0;
    const uint32_t s_pfx = get_semaphore(S_PFX);
    for (uint32_t i = 0; i < m; ++i) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(xy0 + 2 * i), get_arg_val<uint32_t>(xy0 + 2 * i + 1), s_pfx), 1);
    }
    // 3. every line peer's bitmap
    for (uint32_t q = 0; q < n; ++q) {
        if (q != p) {
#if defined(SPA_WALK_BOUND)
            spab::Clock c;
            while (!spab::wait_min(sw + ARR_BM + q, g, c)) {
                SPAB_EXPIRE(sb, spab::UR_PEER_BM, sw[ARR_BM + q], g, q, c.el);
                SPAB_RESUME(c);
            }
#else
            while (sw[ARR_BM + q] < g) {
                invalidate_l1_cache();
            }
#endif
        }
    }
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* rowor = nullptr;
    volatile tt_l1_ptr uint32_t* rp = nullptr;
    if constexpr (TWO) {  // 3b. the row's OR -> the column lane; the relay prefix table over the partner's words
        cb_reserve_back(CB_ROWOR, 1);
        cb_reserve_back(CB_RP, 1);
        const uint32_t ro_l1 = get_write_ptr(CB_ROWOR);
        rowor = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ro_l1);
        rp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_RP));
        uint32_t rr = 0;
        for (uint32_t w = 0; w < WORDS; ++w) {
            uint32_t x = 0;
            for (uint32_t q = 0; q < n; ++q) {
                x |= reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bms + q * WORDS * 4)[w];
            }
            rowor[w] = x;
            rp[w] = rr;
            if (w % nall == kp) {
                rr += popc(x);
            }
        }
        noc_async_write(ro_l1, get_noc_addr(col_x, col_y, col_or0), WORDS * 4);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(col_x, col_y, get_semaphore(S_ORMINE)), 1);
    }
#if defined(SPA_WALKERS)
    // A0 (GLM53_SPA_WALKERS=1): pass 0 runs on the M members (spa_mw.cpp on their BRISC). The member slots' valid words
    // on this lane are cleared first and the clear published in sync word SPA_W_READY (= g: the lane BRISC sweeps only
    // after it), then every member gets its header in CB_MW (one descriptor over the lane + the members: the same
    // address there) = [CB_SLOTS base, the member-slot region base, per destination the popcount of its words before the
    // member's range [i WORDS / M, (i + 1) WORDS / M)], then S_WALK
    {
        cb_reserve_back(SPA_CB_MW, 1);
        cb_reserve_back(SPA_CB_MSLOT, 1);
        const uint32_t mw_l1 = get_write_ptr(SPA_CB_MW);
        const uint32_t mbase = get_write_ptr(SPA_CB_MSLOT);
        for (uint32_t i = 0; i < 2 * m; ++i) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mbase + i * (4096 + 16) + 4096)[2] = 0;
        }
        sw[SPA_W_READY] = g;
        uint32_t run[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint32_t wpos = 0;
#if defined(SPA_WALK_HDRB)
        // header batching (GLM53_SPA_WALK_HDRB=1): every member's header from its own 48-B scratch after the member slots
        // in CB_MSLOT (lane only), all writes in flight, ONE write barrier, then the M increments, ONE atomic barrier
        // (the serial form: write, barrier, increment per member = ur_prefix_peers 5.6 -> 33 us on the d0 pair)
        const uint32_t hscr = mbase + 2 * m * (4096 + 16);
#endif
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t lo = i * WORDS / m;
            for (; wpos < lo; ++wpos) {
                for (uint32_t d = 0; d < n; ++d) {
                    run[d] += popc(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bms + d * WORDS * 4)[wpos]);
                }
            }
#if defined(SPA_WALK_HDRB)
            const uint32_t hl1 = hscr + i * 48;
#else
            const uint32_t hl1 = mw_l1;
#endif
            volatile tt_l1_ptr uint32_t* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hl1);
            h[0] = bms;
            h[1] = mbase;
            for (uint32_t d = 0; d < n; ++d) {
                h[2 + d] = run[d];
            }
            h[10] = TWO ? reinterpret_cast<uint32_t>(rp) : 0;  // A0r: the relay prefix / row-OR tables (CB_RP / CB_ROWOR)
            h[11] = TWO ? reinterpret_cast<uint32_t>(rowor) : 0;
            const uint32_t mx = get_arg_val<uint32_t>(xy0 + 2 * i), my = get_arg_val<uint32_t>(xy0 + 2 * i + 1);
            noc_async_write(hl1, get_noc_addr(mx, my, mw_l1), 12 * 4);
#if !defined(SPA_WALK_HDRB)
            noc_async_write_barrier();  // (the header scratch is reused for the next member)
            noc_semaphore_inc(get_noc_addr(mx, my, get_semaphore(SPA_S_WALK)), 1);
#endif
        }
#if defined(SPA_WALK_HDRB)
        noc_async_write_barrier();  // every header landed before any member's S_WALK
        for (uint32_t i = 0; i < m; ++i) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(xy0 + 2 * i), get_arg_val<uint32_t>(xy0 + 2 * i + 1), get_semaphore(SPA_S_WALK)),
                1);
        }
#endif
        noc_async_atomic_barrier();  // (without the 2x4 relay this NCRISC ends right here: no pending atomics at exit)
#if defined(SPA_WALK_RELAY) && !defined(SPA_A2B)
        if constexpr (TWO) {  // A0r: the members take pass 1 too, once the partner's stage A landed in the relay rows
#if defined(SPA_WALK_BOUND)
            spab::Clock c;
            while (!spab::wait_min(sw + arr_a, g, c)) {
                SPAB_EXPIRE(sb, spab::UR_ARR_A, sw[arr_a], g, 0, c.el);
                SPAB_RESUME(c);
            }
#else
            while (sw[arr_a] < g) {
                invalidate_l1_cache();
            }
#endif
            for (uint32_t i = 0; i < m; ++i) {
                noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(xy0 + 2 * i), get_arg_val<uint32_t>(xy0 + 2 * i + 1),
                                               get_semaphore(SPA_S_RELAY)),
                                  1);
            }
            noc_async_atomic_barrier();
        }
#endif
    }
#if defined(SPA_WALK_RELAY)
    constexpr uint32_t PASS0 = 2;  // A0r: the members took both passes (no lane stream)
#else
    constexpr uint32_t PASS0 = 1;  // the members took pass 0
#endif
#else
    constexpr uint32_t PASS0 = 0;
#endif
    // 4. the owned pools of every destination's union (this chip first: local writes overlap the first remote reads);
    // TWO: then the partner's pools out of the relay rows, after its stage A landed
    SPA_STAMP(sync, 22);  // the prefix + every peer's bitmap
    spa::Stager<CB_STAGE, CB_META> st;
    for (uint32_t pass = PASS0; pass < (TWO ? 2u : 1u); ++pass) {
        const uint32_t owner = pass == 0 ? k : kp;
        if (pass == 1) {
            st.close();  // pass 0's last pools go out before the wait
#if defined(SPA_WALK_BOUND)
            spab::Clock c;
            while (!spab::wait_min(sw + arr_a, g, c)) {
                SPAB_EXPIRE(sb, spab::UR_ARR_A_PASS1, sw[arr_a], g, 1, c.el);
                SPAB_RESUME(c);
            }
#else
            while (sw[arr_a] < g) {
                invalidate_l1_cache();
            }
#endif
            invalidate_l1_cache();
        }
        // one destination's owned pools of word w (rank = the pools of d's union in the words before w)
        auto emit = [&](uint32_t d, uint32_t w, uint32_t x, uint32_t rank) {
            uint32_t r = rank;
            for (uint32_t b = 0; b < 32; ++b) {
                if (!((x >> b) & 1u)) {
                    continue;
                }
                if (r < cap) {
                    if (pass == 0) {
                        st.add(lat, (w / nall) * UNIT + 4 * b, r, d);
                    } else {
                        const uint32_t ridx = rp[w] + popc(rowor[w] & ((1u << b) - 1u));
                        if (ridx < rcap) {  // past the relay rows: not delivered (the column lane flags it)
                            st.add(rel, 4 * ridx, r, d);
                        }
                    }
                }
                ++r;
            }
        };
#if defined(SPA_INTERLEAVE)
        // A2'(a): word-outer, destination-inner -- consecutive staged pools alternate the destinations, so the sender's
        // up and down channels (spa_us.cpp dirs[0] / dirs[1]) carry packets concurrently; same (slot, d) per pool
        constexpr uint32_t NMAX = 8;
        uint32_t ranks[NMAX] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint32_t w = 0; w < WORDS; ++w) {
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t d = (p + i) % n;
                const uint32_t x = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bms + d * WORDS * 4)[w];
                if (x != 0 && w % nall == owner) {
                    emit(d, w, x, ranks[i]);
                }
                ranks[i] += popc(x);
            }
        }
#else
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t d = (p + i) % n;
            volatile tt_l1_ptr uint32_t* bm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bms + d * WORDS * 4);
            uint32_t rank = 0;
            for (uint32_t w = 0; w < WORDS; ++w) {
                const uint32_t x = bm[w];
                if (x != 0 && w % nall == owner) {
                    emit(d, w, x, rank);
                }
                rank += popc(x);
            }
        }
#endif
    }
#if defined(SPA_WALKERS)
#if !defined(SPA_WALK_RELAY)
    if constexpr (TWO) {  // the relay stream's end marker (no lane stream at all without TWO: no CB_STAGE / CB_META)
        st.end();
    }
#endif
#else
    st.end();
#endif
#if defined(SPA_A2B)
    // A2b (GLM53_SPA_A2B=1; the pair, or the 2x4 under A0r -- this NCRISC is otherwise idle once the members walk): a
    // SECOND sender on fabric link 1. It sweeps the slots of the ODD members whose destination lies in a direction it
    // connected (nc_up / nc_dn: <= 3 fabric connections per core, the BRISC holds its link-0 ones) -- the BRISC's sweep
    // skips exactly those (spa_us.cpp) -- with the same scatter packet per pool (same (slot, d): the compact stores are
    // bitwise the BRISC-only form's), frees them the same way, and ends when every member counted DONE and a sweep after
    // that found none of its slots. Then one increment of every reached peer's ARR2 + p (sync word 25 + p, after its data
    // on the same link-1 channel); the peers' BRISCs wait them against their own GEN2 (word 24, bumped by A2b launches only)
    {
        using namespace tt::tt_fabric;
        using Hdr = LowLatencyPacketHeader;
        constexpr uint32_t NHDR = 4, ARR2 = 25;
        struct Dir {
            WorkerToFabricEdmSender conn;
            volatile tt_l1_ptr Hdr* data[NHDR];
            uint32_t range = 0;
            uint32_t next = 0;
        };
        size_t fa = xy0 + 2 * m;
        const uint32_t yc_addr = get_arg_val<uint32_t>(fa++);
        const uint32_t nc_up = get_arg_val<uint32_t>(fa++);
        const uint32_t nc_dn = get_arg_val<uint32_t>(fa++);
        const auto yc = TensorAccessor(yc_args, yc_addr);
        const uint32_t ranges[2] = {nc_up ? n - 1 - p : 0u, nc_dn ? p : 0u};
        Dir dirs[2];
        for (uint32_t d = 0; d < 2; d++) {
            dirs[d].range = ranges[d];
            if (ranges[d]) {
                dirs[d].conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(fa);
            }
        }
        for (uint32_t d = 0; d < 2; d++) {
            if (dirs[d].range) {
                dirs[d].conn.open();
                for (uint32_t k2 = 0; k2 < NHDR; k2++) {
                    dirs[d].data[k2] = reinterpret_cast<volatile tt_l1_ptr Hdr*>(PacketHeaderPool::allocate_header());
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
        cb_reserve_back(SPA_CB_MSLOT, 1);
        const uint32_t mbase = get_write_ptr(SPA_CB_MSLOT);
        volatile tt_l1_ptr uint32_t* donep = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SPA_S_DONE));
        uint16_t freed[128];
#if defined(SPA_WALK_RELAY)
        bool relay_go = !TWO;  // A0r + A2b: S_RELAY from inside the sweep (a blocking wait would stall the odd members)
#endif
#if defined(SPA_WALK_BOUND)
        spab::Clock sc;  // no progress (no slot sent, no DONE change) for the bound -> expiry
        uint32_t done_last = 0xFFFFFFFFu;
#endif
        for (;;) {
            invalidate_l1_cache();
#if defined(SPA_WALK_RELAY)
            if (!relay_go && sw[arr_a] >= g) {  // the partner's stage A landed in the relay rows: the members' pass 1
                for (uint32_t i = 0; i < m; ++i) {
                    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(xy0 + 2 * i), get_arg_val<uint32_t>(xy0 + 2 * i + 1),
                                                   get_semaphore(SPA_S_RELAY)),
                                      1);
                }
                noc_async_atomic_barrier();
                relay_go = true;
            }
#endif
            const uint32_t done_seen = *donep;
            uint32_t nf = 0;
            for (uint32_t i = 0; i < 2 * m; ++i) {
                if (!((i / 2) & 1u)) {
                    continue;  // an even member's slot: the BRISC's
                }
                const uint32_t sl = mbase + i * (spa::PB + 16);
                volatile tt_l1_ptr uint32_t* mt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sl + spa::PB);
                if (mt[2] != 1) {
                    continue;
                }
                const uint32_t slot = mt[0], pos = mt[1];
                if (!(pos > p ? nc_up : nc_dn)) {
                    continue;  // a direction this RISC did not connect: the BRISC's
                }
                Dir& d = dirs[pos > p ? 0 : 1];
                const uint32_t hops = pos > p ? pos - p : p - pos;
                volatile tt_l1_ptr Hdr* h = hdr(d);
                h->to_chip_unicast(static_cast<uint8_t>(hops));
                h->to_noc_unicast_scatter_write(
                    NocUnicastScatterCommandHeader({yc.get_noc_addr(4 * slot, 0, 0), yc.get_noc_addr(4 * slot + 1, 0, 0),
                                                    yc.get_noc_addr(4 * slot + 2, 0, 0), yc.get_noc_addr(4 * slot + 3, 0, 0)},
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
                    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(xy0 + 2 * (i / 2)), get_arg_val<uint32_t>(xy0 + 2 * (i / 2) + 1),
                                                   get_semaphore(SPA_S_FREE0 + (i % 2))),
                                      1);
                }
            }
            if (nf == 0 && done_seen >= m) {
                break;
            }
#if defined(SPA_WALK_BOUND)
            if (nf || done_seen != done_last) {
                sc.reset();
                done_last = done_seen;
            } else if (!sc.ok()) {  // seen = the DONE count, want = the members
                SPAB_EXPIRE(sb, spab::UR_A2B_SWEEP, done_seen, m, 0, sc.el);
                SPAB_RESUME(sc);
            }
#endif
        }
        noc_async_write_barrier();
        for (uint32_t q = 0; q < n; ++q) {  // the link-1 arrivals (each after this RISC's data on the same channel)
            if (q == p || !(q > p ? nc_up : nc_dn)) {
                continue;
            }
            Dir& d = dirs[q > p ? 0 : 1];
            volatile tt_l1_ptr Hdr* h = hdr(d);
            h->to_chip_unicast(static_cast<uint8_t>(q > p ? q - p : p - q));
            h->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(my_x[0], my_y[0], sync + (ARR2 + p) * 4, 0), 1});
            d.conn.wait_for_empty_write_slot();
            d.conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(Hdr));
        }
        noc_async_writes_flushed();
        for (uint32_t d = 0; d < 2; d++) {
            if (dirs[d].range) {  // drain (every slot free again) then close
                while (dirs[d].conn.get_num_free_write_slots() != dirs[d].conn.num_buffers_per_channel) {
                }
                dirs[d].conn.close();
            }
        }
        noc_async_atomic_barrier();
    }
#endif
    SPA_STAMP(sync, 23);  // the walk staged its last pool
#if defined(SPA_WALKERS)
    noc_async_full_barrier();  // no NoC transaction outstanding at the kernel's end (the watcher's inter-kernel race check)
#endif
}
