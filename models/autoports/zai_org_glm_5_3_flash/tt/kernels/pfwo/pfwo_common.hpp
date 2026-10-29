// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// WS-O (GLM53_PMK_WS_O; program [session 10] 2026-10-28 05:24:26Z GO; doc/ttnn/16-prefill-production/plan_1m/
// l5_ws_oproj_design.md): program A's P7 o_proj WEIGHT-STATIONARY under the KDA scan. The 64 members (ONE exact 8 x 8
// multicast rectangle) are the receivers: member m holds N columns [64 m, 64 m + 64) = COLS (2) column tiles x KT (32)
// K tiles of the o_proj weight resident in L1 (CB_WS, read once in place from the image's DRAM WIDTH_SHARDED DSLinear
// weight). The scan finishes the `on` rows chunk by chunk (chunk c = row tile c = 32 rows x KT tiles, DRAM; SEM_ON
// counts the chunks); TWO lanes OUTSIDE the box -- lane 0 on NoC 0, lane 1 on NoC 1, exactly one multicaster per NoC
// (the stage-1c hang class: never two concurrent multicasters on one NoC / VC 4 into the box, never a sender inside it)
// -- take the even / odd chunks: wait their SEM_ON >= c + 1, read the chunk from DRAM and multicast it into ring slot
// c mod S on the members as single packets (MC 1: writes_flushed per packet, from the start; MC 2: a barrier each),
// then the VALID SET (= the lane's chunk count) behind the data. Per chunk each member computes y(c, n) over K
// ascending in COLS fp32 DST tiles (the PfStream member's per-tile MAC sequence: bitwise), packs once, stores the tiles
// at today's y positions and raises block c's H5 word (the RS8 readiness: 64 raises per block per launch =
// FS_RDY_MULT 64 on the RS readers, QSLICE 1).
// Credits (per launch: the EPOCH form): a member credits chunk c (+1 on lane (c mod 2)'s READY[c mod S]) once ring
// slot c mod S is free (chunk c - S consumed); a later chunk of the same lane in the same slot c' >= c + S is credited
// only after c' - S >= c is consumed (in-order), i.e. after c was sent: the lane's wait READY[w] >= NDEST x (its chunks
// so far in slot w) means every member credited this chunk. Zeroed at the end of every launch (every credit awaited).
#include <cstdint>

#if !defined(MK_PHASED)  // standalone program (the unit); the PMK composite defines both per phase namespace
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#endif

namespace pfwo {
// ONE compile-time arg list (every kernel of the members / the lanes): get_compile_time_arg_val(PH_CT0 + CT_x)
constexpr uint32_t CT_KT = 0;       // K tiles (32: o_proj K = 1,024 per chip at TP8)
constexpr uint32_t CT_MT = 1;       // row tiles = chunks (64)
constexpr uint32_t CT_S = 2;        // ring slots (<= 8)
constexpr uint32_t CT_COLS = 3;     // column tiles per member (2)
constexpr uint32_t CT_TBW = 4;      // weight tile bytes (bf8 1,088)
constexpr uint32_t CT_SW = 5;       // the resident weight's shard width in tiles (n_pad / 32 / 8)
constexpr uint32_t CT_XTB = 6;      // x tile bytes (bf16 2,048)
constexpr uint32_t CT_NDEST = 7;    // members (every one credits every chunk)
constexpr uint32_t CT_MC = 8;       // 1: writes_flushed per multicast packet; 2: a write barrier per packet
constexpr uint32_t CT_CB_WS = 9;    // the resident weight slice (KT x COLS tiles, tile (k, c) at k COLS + c)
constexpr uint32_t CT_CB_XR = 10;   // the ring (S chunks of KT x tiles)
constexpr uint32_t CT_CB_OUT = 11;  // the output tiles (2 chunks x COLS)
constexpr uint32_t CT_CB_STG = 12;  // the lane's staging (2 chunks)
constexpr uint32_t CT_SEM_V0 = 13;  // the members' VALID of lane L = SEM_V0 + L (PfStream's 10 / 11 in program A)
constexpr uint32_t CT_SEM_ON = 14;  // the lanes' SEM_ON: chunks the scan finished (kda SEM_ON 7: the scan leader)
constexpr uint32_t CT_NJ = 15;      // chunks per lane (MT / 2)
constexpr uint32_t CT_QBASE = 16;   // the phase's first launch-global block (even): chunk c = block QBASE + c, so lane
                                    //   L's VALID continues the PfStream phases' numbering of the launch (VALID = Q / 2 +
                                    //   1 = QBASE / 2 + j + 1): the members' VALID semaphores (10 / 11) carry P3's counts
constexpr uint32_t CT_ACC = 17;     // TensorAccessorArgs of y, then of x (the scan's `on` rows, DRAM)
// ONE runtime list per member (reader / writer): ph_rt0 + RT_x
constexpr uint32_t RT_SYNC = 0;     // the lanes' sync base (the same L1 address on both lane cores)
constexpr uint32_t RT_FIRST = 1;    // member 0: publishes the ring base to both lanes before its first credit
constexpr uint32_t RT_WADDR = 2;    // the resident weight's address (every bank)
constexpr uint32_t RT_BANK = 3;     // (bank, first tile column in the bank's shard): the COLS columns are consecutive
constexpr uint32_t RT_YADDR = 5;    // y's address
constexpr uint32_t RT_NTT = 6;      // y's N tiles (the page row stride)
constexpr uint32_t RT_COL0 = 7;     // the member's first output column tile
constexpr uint32_t RT_LXY = 8;      // the 2 lanes' NoC (x, y)
constexpr uint32_t RT_H5 = 12;      // the H5 words' base L1 address, then MT packed words, block c: noc x | noc y << 8 |
                                    //   byte offset from the base << 16 (block c's H5 word: +1 after its tiles landed;
                                    //   the RS8's per-block LREADY / OWNRDY words share one sync base)
                                    //   then the member's raise value at RT_H5 + 1 + MT (pfwo_wr.cpp)
// the lane cores' runtime list
constexpr uint32_t RT_LANE = 1;     // L (= its NoC)
constexpr uint32_t RT_RECT = 2;     // (lx, ly, hx, hy, n): the members' box (NoC coordinates), n = 64 (lane outside)
constexpr uint32_t RT_XADDR = 7;    // x's address
// the lanes' persistent sync words (16 B each; zeroed by the host once)
constexpr uint32_t WORD = 16;
constexpr uint32_t READY = 0;       // READY[0 .. S - 1]: credits of the lane's chunks c with c mod S = w (this launch)
constexpr uint32_t GEN = 8;         // launches done
constexpr uint32_t RINGADDR = 9;    // the members' ring base (member 0 publishes; zeroed per launch)
constexpr uint32_t VAL = 10;        // the VALID multicast source word
constexpr uint32_t DL_WHAT = 11;    // a budgeted wait that expired: 0xDEAD0000 | (what); the launch proceeds (garbage)
constexpr uint32_t DL_SEEN = 12;
constexpr uint32_t P_SPAN = 13;     // the lane's cycles (last launch): span, credit waits, SEM_ON waits, x reads, mcast
constexpr uint32_t P_READY = 14;
constexpr uint32_t P_ON = 15;
constexpr uint32_t P_READ = 16;
constexpr uint32_t P_MC = 17;
constexpr uint32_t SYNC_WORDS = 24;

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
inline volatile tt_l1_ptr uint32_t* word(uint32_t base, uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + i * WORD);
}
#if !defined(PFWO_WAIT_BUDGET)
#define PFWO_WAIT_BUDGET 2700000000ull  // cycles (~2 s at 1.35 GHz): far beyond any legitimate wait
#endif
// a budgeted wait (pfs_common.hpp's wait_ge_budget form: 32-bit WALL_CLOCK_L deltas): true = cond met; false = the
// budget expired -> the deadlock record (when rec != 0), the caller proceeds (that launch's outputs are garbage)
template <typename F>
inline bool wait_budget(F cond, uint32_t rec, uint32_t what, volatile tt_l1_ptr uint32_t* seen) {
    WAYPOINT("PWOW");
    uint64_t elapsed = 0;
    uint32_t last = get_timestamp_32b();
    do {
        invalidate_l1_cache();
        if (cond()) {
            WAYPOINT("PWOD");
            return true;
        }
        const uint32_t now = get_timestamp_32b();
        const uint32_t d = now - last;
        if (d < (1u << 30)) {
            elapsed += d;
        }
        last = now;
    } while (elapsed < PFWO_WAIT_BUDGET);
    if (rec) {
        *word(rec, DL_SEEN) = *seen;
        *word(rec, DL_WHAT) = 0xDEAD0000u | (what & 0xFFFFu);
    }
    WAYPOINT("PWOX");
    return false;
}
#endif
}  // namespace pfwo
