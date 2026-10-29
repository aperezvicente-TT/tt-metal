// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 16 FabricStreamA2A2D (tt/fabric_a2a.py, doc pmk/merge/a2a_fs_design.md): the SPA q / o all-to-all as page-run
// lists. The persistent sync words of every LANE core (one L1 shard per lane core, the same address on every core and
// chip; zeroed once by the host). Every counter is MONOTONIC across launches: launch g waits for counter >= g x its
// per-launch count (or (g - 1) x per + k for the k-th relay run). One `constexpr uint32_t NAME = N;` per line: the host
// parses this header for the word indices (tt/fabric_a2a.py _consts).
#pragma once
#include <cstdint>

namespace a2a {
constexpr uint32_t WORD = 16;    // one counter per 16 B (NoC atomic / L1 alignment)
constexpr uint32_t GEN_B = 0;    // launches completed by this core's BRISC (a2a_send.cpp)
constexpr uint32_t GEN_NC = 1;   // launches completed by this core's NCRISC (a2a_read.cpp)
constexpr uint32_t DONE = 2;     // the WAITER (B lane 0): this chip's lanes finished their sends + local writes (+1 each)
constexpr uint32_t BAR_L = 3;    // the waiter's 1-hop row barrier: low -> high sweep (fs_common.hpp THE PER-HOP DRAIN)
constexpr uint32_t BAR_R = 4;    //   high -> low sweep
constexpr uint32_t RELAY = 5;    // a B lane L: relay runs landed in this chip's relay store from the partner's A lane L
constexpr uint32_t CREDIT = 8;   // CREDIT + q: the peer at axis position q started this launch (its same-role lane)
constexpr uint32_t NCREDIT = 8;  //   (positions 0 .. 7)
constexpr uint32_t YLAND = 16;   // the waiter: YLAND + j = runs landed in y from source lane j (tt/fabric_a2a.py)
constexpr uint32_t NYLAND = 16;
constexpr uint32_t WORDS = 32;   // words per lane core (512 B)
// sender run word 0: route (bits 0-1: 0 up / 1 down / 2 local), hops (2-5), destination (6: 0 y / 1 relay store),
// credit position + 1 (8-11: 0 = none), run-end signal (12)
constexpr uint32_t ROUTE_LOCAL = 2;
static_assert(YLAND + NYLAND <= WORDS && CREDIT + NCREDIT <= YLAND && RELAY < CREDIT, "a2a sync words");
}  // namespace a2a
