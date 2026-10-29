// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SP-row sparse attention v2, A0 PARALLEL WALKERS (GLM53_SPA_WALKERS=1, tt/spa_union.py; 16's plan 2026-10-27 21:4xZ),
// member BRISC (the member NCRISC keeps spa_um.cpp): pass 0 of the lane walker (spa_ur.cpp step 4), split over the M
// members by CONTIGUOUS word ranges [me WORDS / M, (me + 1) WORDS / M).
//   1. waits S_WALK: the lane NCRISC wrote this member's header into CB_MW (one descriptor over the lane + the members:
//      the same L1 address on both) = [the lane's CB_SLOTS base, the lane's member-slot region base, then per
//      destination d the rank of this range's first pool in d's union (the popcount of d's words before the range)];
//   2. NoC-reads its word slice of every destination's union bitmap from the lane's CB_SLOTS;
//   3. walks the slice exactly as spa_ur.cpp pass 0 (the owned words w % nall == k, rank r per pool, r < cap only): each
//      selected pool's 4 latent rows (local row (w / nall) UNIT + 4 b ..) are read into this member's staging, then
//        d == p (this chip's own compact rows): written straight into compact rows 4 r .. 4 r + 3 (no lane);
//        d != p: one of this member's 2 dedicated slots on the lane (free semaphores S_FREE0 / S_FREE1, the lane +1s a
//        slot's semaphore after its packet left): 4 KiB -> the slot, (r, d) -> its meta, a write barrier, then the
//        meta's valid word (an inline write) = the lane may send it;
//   4. after its last pool: a write barrier, then +1 the lane's S_DONE.
// Every pool goes to the same (slot r, destination d) as the lane walker's: the compact stores are bitwise today's.
// Compile-time: WORDS, RMAX (the longest slice: ceil(WORDS / M); CB_MW = (12 + 8 RP) words, RP below), then the
// TensorAccessorArgs of the local latent store,
// of the compact store and of the relay tensor. Runtime: lat, compact, lane x, lane y, me, M, n, p, nall, k, cap, relay,
// kp (the column partner's rank), relay cap. SPA_WALK_RELAY (GLM53_SPA_WALKERS=2, the 2x4): pass 1 on the members too;
// CB_MW then holds 10 regions of RP words (8 bitmap slices, the rp and row-OR slices).
// Defines: SPA_CB_MW, SPA_CB_MST, SPA_S_WALK, SPA_S_FREE0 (S_FREE1 = + 1), SPA_S_DONE (ids from tt/spa_union.py).
// SPA_WALK_BOUND (GLM53_SPA_WALK_BOUND, spa_bound.hpp): the S_WALK / free-slot / S_RELAY waits bounded; runtime arg 14 =
// the row lane's sync words (the expiry record); CB_MST + 64 B (the record stash).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#if defined(SPA_WALK_BOUND)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/spa/spa_bound.hpp"
#endif

constexpr uint32_t WORDS = get_compile_time_arg_val(0);
constexpr uint32_t RMAX = get_compile_time_arg_val(1);
constexpr uint32_t CB_MW = SPA_CB_MW;
constexpr uint32_t CB_MST = SPA_CB_MST;
constexpr uint32_t RB = 1024;      // one latent row: 512 bf16
constexpr uint32_t PB = 4 * RB;    // one pool
constexpr uint32_t UNIT = 128;     // tokens per shard unit = one bitmap word
constexpr uint32_t HDR = 12;  // header words: slots base, member-slot base, 8 start ranks (n <= 8), CB_RP / CB_ROWOR bases
// a destination's slice region: the slice is read from the 16-B-aligned word at or below its first word (a BH L1 NoC
// read needs source and destination equal mod 16: NOC_L1_READ_ALIGNMENT_BYTES, the watcher's DebugSanitizeNocAlignment),
// so up to 3 lead words + RMAX + up to 3 tail words, rounded to 4 words: every region starts 16-B aligned
constexpr uint32_t RP = (RMAX + 6 + 3) & ~3u;
constexpr uint32_t SLOTB = PB + 16;  // one member slot on the lane: 4 KiB data, then the meta (slot, d, valid, -)

inline uint32_t popc(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}

void kernel_main() {
    constexpr auto lat_args = TensorAccessorArgs<2>();
    constexpr auto y_args = TensorAccessorArgs<lat_args.next_compile_time_args_offset()>();
    constexpr auto rel_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();  // (the local store w/o the 2x4)
    size_t a = 0;
    const uint32_t lat_addr = get_arg_val<uint32_t>(a++);
    const uint32_t y_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lane_x = get_arg_val<uint32_t>(a++);
    const uint32_t lane_y = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t m = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t nall = get_arg_val<uint32_t>(a++);
    const uint32_t k = get_arg_val<uint32_t>(a++);
    const uint32_t cap = get_arg_val<uint32_t>(a++);
    const uint32_t rel_addr = get_arg_val<uint32_t>(a++);
    const uint32_t kp = get_arg_val<uint32_t>(a++);
    const uint32_t rcap = get_arg_val<uint32_t>(a++);
#if defined(SPA_WALK_BOUND)
    spab::State sb;  // GLM53_SPA_WALK_BOUND (spa_bound.hpp): the row lane's sync words take the record
    sb.sync = get_arg_val<uint32_t>(a++);
    sb.lx = lane_x;
    sb.ly = lane_y;
#endif
    const auto lat = TensorAccessor(lat_args, lat_addr);
    const auto y = TensorAccessor(y_args, y_addr);
    const auto rel = TensorAccessor(rel_args, rel_addr);
    (void)rel;
    (void)kp;
    (void)rcap;

    cb_reserve_back(CB_MW, 1);
    cb_reserve_back(CB_MST, 1);
    const uint32_t mw_l1 = get_write_ptr(CB_MW);
    const uint32_t st_l1 = get_write_ptr(CB_MST);  // [3 x 4 KiB staging: slot 0, slot 1, local] [2 x 16 B meta]
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mw_l1);
#if defined(SPA_WALK_BOUND)
    sb.stash = st_l1 + 3 * PB + 32;  // CB_MST's 64-B tail (tt/spa_union.py adds it under the knob)
#endif
    volatile tt_l1_ptr uint32_t* fr[2] = {reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SPA_S_FREE0)),
                                          reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SPA_S_FREE0 + 1))};

    // 1. the header (the lane wrote it before S_WALK)
#if defined(SPA_WALK_BOUND)
    {
        spab::Clock c;
        volatile tt_l1_ptr uint32_t* swk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SPA_S_WALK));
        while (!spab::wait_min(swk, 1, c)) {
            SPAB_EXPIRE(sb, spab::MW_WALK, *swk, 1, me, c.el);
            SPAB_RESUME(c);
        }
    }
#else
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SPA_S_WALK)), 1);
#endif
    invalidate_l1_cache();
    const uint32_t slots_addr = hdr[0], mbase = hdr[1];
    const uint32_t lo = me * WORDS / m, hi = (me + 1) * WORDS / m, len = hi - lo;
    // 2. this range's words of every destination's union bitmap (aligned down to 4 words: lead = the offset of word lo
    // in its 16-B group, the same for every d when WORDS % 4 == 0, else per d)
    uint32_t lead[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#if defined(SPA_WALK_BOUND)
    if (len && !sb.bail) {
#else
    if (len) {
#endif
        for (uint32_t d = 0; d < n; ++d) {
            const uint32_t w0 = d * WORDS + lo, al = w0 & ~3u;
            lead[d] = w0 - al;
            noc_async_read(get_noc_addr(lane_x, lane_y, slots_addr + al * 4), mw_l1 + (HDR + d * RP) * 4,
                           ((lead[d] + len + 3) & ~3u) * 4);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
    }
    const uint32_t loc_buf = st_l1 + 2 * PB;
    bool loc_pending = false;  // the local buffer's previous writes not yet flushed
    // one pool = 4 rows of acc from row0 -> (slot r, destination d): d == p straight into this chip's compact rows
    // 4 r .. 4 r + 3 (no lane), else one of this member's 2 slots on the lane
    auto put_pool = [&](const auto& acc, uint32_t row0, uint32_t r, uint32_t d) {
#if defined(SPA_WALK_BOUND)
        if (sb.bail) {
            return;  // bailed: no further NoC traffic
        }
#endif
        if (d == p) {
            if (loc_pending) {
                noc_async_writes_flushed();  // the local buffer's previous rows have left
            }
            for (uint32_t j = 0; j < 4; ++j) {
                noc_async_read_page(row0 + j, acc, loc_buf + j * RB);
            }
            noc_async_read_barrier();
            for (uint32_t j = 0; j < 4; ++j) {
                noc_async_write_page(4 * r + j, y, loc_buf + j * RB);
            }
            loc_pending = true;
            return;
        }
        uint32_t s = 2;
#if defined(SPA_WALK_BOUND)
        spab::Clock c;
#endif
        while (s == 2) {  // a free slot of the two (the lane frees them in any order)
            invalidate_l1_cache();
            if (*fr[0] >= 1) {
                s = 0;
            } else if (*fr[1] >= 1) {
                s = 1;
            }
#if defined(SPA_WALK_BOUND)
            else if (!c.ok()) {  // seen = the two free words, ctx = destination << 24 | slot rank
                SPAB_EXPIRE(sb, spab::MW_FREE, (*fr[0] << 16) | (*fr[1] & 0xFFFFu), 1, (d << 24) | (r & 0xFFFFFFu), c.el);
#if defined(SPA_WALK_BOUND_REPORT)
                c.reset();
#else
                return;
#endif
            }
#endif
        }
        *fr[s] = 0;
        const uint32_t buf = st_l1 + s * PB;
        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_read_page(row0 + j, acc, buf + j * RB);
        }
        noc_async_read_barrier();
        const uint32_t slot_l1 = mbase + (me * 2 + s) * SLOTB;
        volatile tt_l1_ptr uint32_t* mt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st_l1 + 3 * PB + s * 16);
        mt[0] = r;
        mt[1] = d;
        noc_async_write(buf, get_noc_addr(lane_x, lane_y, slot_l1), PB);
        noc_async_write(st_l1 + 3 * PB + s * 16, get_noc_addr(lane_x, lane_y, slot_l1 + PB), 8);
        noc_async_write_barrier();  // the rows + (slot, d) landed before the valid word
        noc_inline_dw_write(get_noc_addr(lane_x, lane_y, slot_l1 + PB + 8), 1);
    };
    // 3. pass 0 (spa_ur.cpp pass 0's order per destination; the order does not matter: fixed (slot, d) per pool): the
    // words this chip owns (w % nall == k), rows of the local latent store
#if defined(SPA_WALK_BOUND)
    for (uint32_t i = 0; i < n && len && !sb.bail; ++i) {
#else
    for (uint32_t i = 0; i < n && len; ++i) {
#endif
        const uint32_t d = (p + i) % n;
        volatile tt_l1_ptr uint32_t* sl =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mw_l1 + (HDR + d * RP + lead[d]) * 4);
        uint32_t rank = hdr[2 + d];
        for (uint32_t w = lo; w < hi; ++w) {
            const uint32_t x = sl[w - lo];
            if (x != 0 && w % nall == k) {
                uint32_t r = rank;
                for (uint32_t b = 0; b < 32; ++b) {
                    if ((x >> b) & 1u) {
                        if (r < cap) {
                            put_pool(lat, (w / nall) * UNIT + 4 * b, r, d);
                        }
                        ++r;
                    }
                }
            }
            rank += popc(x);
        }
    }
#if defined(SPA_WALK_RELAY)
    // A0r (GLM53_SPA_WALKERS=2 on the 2x4): pass 1 = the column partner's words (w % nall == kp), rows of the RELAY tensor
    // (its stage A pushed the pools this row needs: relay index = rp[w] + the row-OR bits below b), after the lane NCRISC
    // saw ARR_DATA_A and bumped S_RELAY; this range's rp / row-OR words come from the lane's CB_RP / CB_ROWOR (header
    // words 10 / 11), aligned like the bitmap slices
#if defined(SPA_WALK_BOUND)
    if (!sb.bail) {
        spab::Clock c;
        volatile tt_l1_ptr uint32_t* srl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SPA_S_RELAY));
        while (!spab::wait_min(srl, 1, c)) {
            SPAB_EXPIRE(sb, spab::MW_RELAY, *srl, 1, me, c.el);
            SPAB_RESUME(c);
        }
    }
    invalidate_l1_cache();
    if (len && !sb.bail) {
#else
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SPA_S_RELAY)), 1);
    invalidate_l1_cache();
    if (len) {
#endif
        const uint32_t al = lo & ~3u, ld = lo - al, nb = ((ld + len + 3) & ~3u) * 4;
        noc_async_read(get_noc_addr(lane_x, lane_y, hdr[10] + al * 4), mw_l1 + (HDR + 8 * RP) * 4, nb);
        noc_async_read(get_noc_addr(lane_x, lane_y, hdr[11] + al * 4), mw_l1 + (HDR + 9 * RP) * 4, nb);
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* rpw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mw_l1 + (HDR + 8 * RP + ld) * 4);
        volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mw_l1 + (HDR + 9 * RP + ld) * 4);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t d = (p + i) % n;
            volatile tt_l1_ptr uint32_t* sl =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mw_l1 + (HDR + d * RP + lead[d]) * 4);
            uint32_t rank = hdr[2 + d];
            for (uint32_t w = lo; w < hi; ++w) {
                const uint32_t x = sl[w - lo];
                if (x != 0 && w % nall == kp) {
                    uint32_t r = rank;
                    for (uint32_t b = 0; b < 32; ++b) {
                        if ((x >> b) & 1u) {
                            if (r < cap) {
                                const uint32_t ridx = rpw[w - lo] + popc(row[w - lo] & ((1u << b) - 1u));
                                if (ridx < rcap) {  // past the relay rows: not delivered (the column lane flags it)
                                    put_pool(rel, 4 * ridx, r, d);
                                }
                            }
                            ++r;
                        }
                    }
                }
                rank += popc(x);
            }
        }
    }
#endif
    // 4. done: every write of this member landed, then the lane's DONE count
    noc_async_write_barrier();
#if defined(SPA_WALK_BOUND)
    if (!sb.bail) {  // bailed: no DONE (the lane's sweep expires in turn and runs its end protocol)
        noc_semaphore_inc(get_noc_addr(lane_x, lane_y, get_semaphore(SPA_S_DONE)), 1);
    }
#else
    noc_semaphore_inc(get_noc_addr(lane_x, lane_y, get_semaphore(SPA_S_DONE)), 1);
#endif
    noc_async_atomic_barrier();
}
