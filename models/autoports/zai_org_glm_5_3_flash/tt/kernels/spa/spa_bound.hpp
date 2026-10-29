// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GLM53_SPA_WALK_BOUND=<cycles> (16, program [session 10] 2026-10-28 04:03:56Z; doc pmk/merge/walkbound): the
// SpaUnion walker path's waits as BOUNDED spins (BAIL mode: the unit form; REPORT mode: the served default, see SPAB_EXPIRE). Included only under SPA_WALK_BOUND (the
// knob-off kernels are today's text). A wait that sees no progress for SPA_WALK_BOUND cycles of WALL_CLOCK_L EXPIRES:
//   WAYPOINT "SBX" + ASSERT at the call site (SPAB_EXPIRE: watched, the watcher names the core and the line), then
//   (unwatched) the RECORD: 8 words, first writer wins, into the row lane's sync words 16..23 (the GLM53_SPA_STAMPS
//   words: the knobs are exclusive, tt/spa_union.py) + one atomic increment of sync word 29 (the expiry count, every
//   core); the expiring core keeps its own copy in its stash (members: CB_MST's tail; the lane: CB_MSLOT's tail).
// Record (tt/spa_union.py SpaUnion.bound_records decodes it):
//   [0] MAGIC | FAIL | wait id    [1] risc << 16 | NoC x << 8 | NoC y    [2] seen    [3] want    [4] ctx (the loop
//   index: member / peer / slot)  [5] the launch generation g (GEN + 1; 0 on the members)  [6] cycles waited (low 32)
//   [7] xor of [0..6]
// After an expiry the core BAILS: a member stops its walk (no further NoC traffic toward the lane, no DONE); the lane
// leaves the member sweep and still runs its end protocol (arrivals, GEN, drain, close) so the peers and the program
// complete. The sync words' generations are then undefined: the host must not launch the SpaUnion again (the unit stops).
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace spab {

constexpr uint32_t REC = 16;  // sync words 16..23: the record
constexpr uint32_t CNT = 29;  // sync word 29: the expiry count (free at n <= 4: ARR2 25..28, W_READY 30, DROP 31)
constexpr uint32_t MAGIC = 0x5BB00000u;
constexpr uint32_t FAIL = 0x8000u;
// wait ids (tt/spa_union.py BOUND_WAITS)
constexpr uint32_t MW_WALK = 1;          // spa_mw: S_WALK (the lane NCRISC's header)
constexpr uint32_t MW_FREE = 2;          // spa_mw: put_pool's free-slot spin (the lane frees after its packet left)
constexpr uint32_t MW_RELAY = 3;         // spa_mw: S_RELAY (A0r: the partner's stage A landed)
constexpr uint32_t UR_LOCAL = 4;         // spa_ur: S_LOCAL (member 0's union bitmap)
constexpr uint32_t UR_PEER_BM = 5;       // spa_ur: a line peer's bitmap (ARR_BM + q)
constexpr uint32_t UR_ARR_A = 6;         // spa_ur: ARR_DATA_A (the partner's stage A) before the members' S_RELAY
constexpr uint32_t UR_ARR_A_PASS1 = 7;   // spa_ur: ARR_DATA_A before the lane's own pass 1
constexpr uint32_t UR_A2B_SWEEP = 8;     // spa_ur: the A2b sweep (odd members' slots, DONE count)
constexpr uint32_t US_SEM = 9;           // spa_us: the bitmap-send semaphore (S_LOCAL on the row lane)
constexpr uint32_t US_RDY = 10;          // spa_us: a peer's RDY + q (its slots live)
constexpr uint32_t US_WREADY = 11;       // spa_us: W_READY (the lane NCRISC cleared the slots' valid words)
constexpr uint32_t US_SWEEP = 12;        // spa_us: the member-slot sweep (valid words, DONE count)
constexpr uint32_t US_ARR_DATA = 13;     // spa_us: a peer's ARR_DATA + q (its rows landed here)
constexpr uint32_t US_ARR2 = 14;         // spa_us: a peer's ARR2 + q (A2b: its link-1 rows landed here)

#if defined(COMPILE_FOR_NCRISC)
constexpr uint32_t RISC = 1;
#else
constexpr uint32_t RISC = 0;
#endif

struct State {
    uint32_t sync = 0;        // the row lane's sync words (L1)
    uint32_t lx = 0, ly = 0;  // the row lane's NoC (x, y): the record and the count go there
    uint32_t stash = 0;       // 64 B, 16-B aligned: this core's record (32 B) + the read-back of words 16..19 (16 B)
    uint32_t g = 0;           // the launch generation (0 where unknown)
    bool bail = false;
};

// the budget counts forward deltas of WALL_CLOCK_L only (pfs_common.hpp wait_ge_budget: the H latch the L read arms is
// shared by the core's RISCs, a 64-bit stamp can pair one L with a later H); reset() = progress seen
struct Clock {
    uint64_t el = 0;
    uint32_t last = 0;
    Clock() : last(get_timestamp_32b()) {}
    void reset() {
        el = 0;
        last = get_timestamp_32b();
    }
    bool ok() {
        const uint32_t now = get_timestamp_32b();
        const uint32_t d = now - last;
        if (d < (1u << 30)) {
            el += d;
        }
        last = now;
        return el < static_cast<uint64_t>(SPA_WALK_BOUND);
    }
};

// *p >= v (noc_semaphore_wait_min's test) within the bound: true = satisfied; false = expired (c.el = the cycles waited)
inline bool wait_min(volatile tt_l1_ptr uint32_t* p, uint32_t v, Clock& c) {
    for (;;) {
        invalidate_l1_cache();
        if (*p >= v) {
            return true;
        }
        if (!c.ok()) {
            return false;
        }
    }
}

// the record (after the call site's WAYPOINT + ASSERT): stash it, then first writer wins at the row lane (one 32-B packet
// per writer: a race leaves one whole record), then +1 the expiry count
inline void record(State& s, uint32_t id, uint32_t seen, uint32_t want, uint32_t ctx, uint64_t el) {
#if !defined(SPA_WALK_BOUND_REPORT)
    s.bail = true;
#endif
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s.stash);
    r[0] = MAGIC | FAIL | (id & 0xFFu);
    r[1] = (RISC << 16) | ((static_cast<uint32_t>(my_x[noc_index]) & 0xFFu) << 8) | (static_cast<uint32_t>(my_y[noc_index]) & 0xFFu);
    r[2] = seen;
    r[3] = want;
    r[4] = ctx;
    r[5] = s.g;
    r[6] = static_cast<uint32_t>(el);
    r[7] = r[0] ^ r[1] ^ r[2] ^ r[3] ^ r[4] ^ r[5] ^ r[6];
    const uint64_t dst = get_noc_addr(s.lx, s.ly, s.sync + REC * 4);  // 16-B aligned (an L1 buffer + 64)
    noc_async_read(dst, s.stash + 32, 16);
    noc_async_read_barrier();
    invalidate_l1_cache();
    if ((r[8] & 0xFFFF0000u) != MAGIC) {
        noc_async_write(s.stash, dst, 32);
        noc_async_write_barrier();
    }
    noc_semaphore_inc(get_noc_addr(s.lx, s.ly, s.sync + CNT * 4), 1);
    noc_async_atomic_barrier();
}

}  // namespace spab

#if defined(SPA_WALK_BOUND_REPORT)
// REPORT mode (GLM53_SPA_WALK_BOUND_MODE=report, the k4d8s-line default: program [session 10] 2026-10-28 04:28:04Z "a
// bounded spin that reports beats a silent hang"): an expiry records (first writer wins) and +1s the count, then the wait
// RESUMES -- no ASSERT, no bail, so the results are unchanged; a real hang still hangs, with the record in the row lane's
// sync words 16..23 and a count (word 29) that grows by one per bound of no progress (a heartbeat for the triage read)
#define SPAB_EXPIRE(st, id, seen, want, ctx, el) \
    do {                                          \
        WAYPOINT("SBR");                          \
        spab::record((st), (id), (seen), (want), (ctx), (el)); \
    } while (0)
#define SPAB_RESUME(c) (c).reset()
#else
// BAIL mode (the unit form): the watcher names this line (ASSERT is a no-op unwatched), then the record and the bail
#define SPAB_EXPIRE(st, id, seen, want, ctx, el) \
    do {                                          \
        WAYPOINT("SBX");                          \
        ASSERT(false);                            \
        spab::record((st), (id), (seen), (want), (ctx), (el)); \
    } while (0)
#define SPAB_RESUME(c) break
#endif
