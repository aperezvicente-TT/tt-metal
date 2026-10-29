// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// WS-O MEMBER NCRISC (tt/pf_wo.PfStreamWO; protocol: pfwo_common.hpp).
// 1. member 0 publishes the ring base to both lanes (a write barrier before its first credit).
// 2. the resident o_proj slice: KT x COLS tiles read in place from the image's DRAM WIDTH_SHARDED weight (tile (k, j)
//    of bank b's shard at WADDR + (k SW + j) TBW; the member's COLS columns are consecutive in one bank) into CB_WS
//    (tile (k, c) at k COLS + c), overlapped with the first S credits.
// 3. the credit / landing loop in chunk order c (pfs_member_rd.cpp's form, one ring): credit chunk c (+1 on lane
//    (c mod 2)'s READY[c mod S]) once ring slots v .. c are free; push chunk v to the compute once that lane's VALID
//    >= QBASE / 2 + v / 2 + 1 (block QBASE + v in the launch-global numbering of the PfStream phases).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pfwo_common.hpp"

namespace pfwo_r {
constexpr uint32_t KT = get_compile_time_arg_val(PH_CT0 + pfwo::CT_KT);
constexpr uint32_t MT = get_compile_time_arg_val(PH_CT0 + pfwo::CT_MT);
constexpr uint32_t S = get_compile_time_arg_val(PH_CT0 + pfwo::CT_S);
constexpr uint32_t COLS = get_compile_time_arg_val(PH_CT0 + pfwo::CT_COLS);
constexpr uint32_t TBW = get_compile_time_arg_val(PH_CT0 + pfwo::CT_TBW);
constexpr uint32_t SW = get_compile_time_arg_val(PH_CT0 + pfwo::CT_SW);
constexpr uint32_t CB_WS = get_compile_time_arg_val(PH_CT0 + pfwo::CT_CB_WS);
constexpr uint32_t CB_XR = get_compile_time_arg_val(PH_CT0 + pfwo::CT_CB_XR);
constexpr uint32_t SEM_V0 = get_compile_time_arg_val(PH_CT0 + pfwo::CT_SEM_V0);
constexpr uint32_t QBASE = get_compile_time_arg_val(PH_CT0 + pfwo::CT_QBASE);
static_assert(S >= 1 && S <= 8 && COLS >= 1 && COLS <= 4, "ring slots / columns");
}  // namespace pfwo_r

void kernel_main() {
    using namespace pfwo_r;
    const uint32_t sync = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_SYNC);
    auto lx = [](uint32_t L) { return get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_LXY + 2 * L); };
    auto ly = [](uint32_t L) { return get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_LXY + 2 * L + 1); };
    // 1. the ring base (member 0), landed on both lanes before any of its credits
    if (get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_FIRST)) {
        for (uint32_t L = 0; L < 2; L++) {
            noc_inline_dw_write(get_noc_addr(lx(L), ly(L), sync + pfwo::RINGADDR * pfwo::WORD), get_write_ptr(CB_XR));
        }
        noc_async_write_barrier();
    }
    // 2. the resident slice: issue the reads, credit the first S chunks, then land them
    cb_reserve_back(CB_WS, KT * COLS);
    {
        const uint32_t ws = get_write_ptr(CB_WS);
        const uint32_t waddr = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_WADDR);
        const uint32_t bank = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_BANK), j0 = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_BANK + 1);
        for (uint32_t k = 0; k < KT; k++) {  // DRAM src and the CB dst both 64-B aligned (TBW 1,088)
            for (uint32_t cc = 0; cc < COLS; cc++) {
                noc_async_read(get_noc_addr_from_bank_id<true>(bank, waddr + (k * SW + j0 + cc) * TBW),
                               ws + (k * COLS + cc) * TBW, TBW);  // fmt: skip
            }
        }
    }
    uint32_t v = 0, c = 0;
    auto credit = [&]() {  // chunk c -> lane (c mod 2)'s READY[c mod S]
        const uint32_t L = c & 1;
        noc_semaphore_inc(get_noc_addr(lx(L), ly(L), sync + (pfwo::READY + c % S) * pfwo::WORD), 1);
        c++;
    };
    while (c < MT && c < S) {  // the ring is empty at the launch start
        credit();
    }
    noc_async_read_barrier();
    cb_push_back(CB_WS, KT * COLS);
    // 3. credits ahead / landings in chunk order
    while (v < MT) {
        invalidate_l1_cache();  // the compute's pops (tiles_acked) and the lanes' VALID sets
        if (c < MT && c - v < S && cb_pages_reservable_at_back(CB_XR, (c - v + 1) * KT)) {
            credit();
            continue;
        }
        if (c > v) {
            volatile tt_l1_ptr uint32_t* vl =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_V0 + (v & 1)));
            if (*vl >= QBASE / 2 + v / 2 + 1) {  // lane (v mod 2)'s VALID for block QBASE + v
                cb_push_back(CB_XR, KT);
                v++;
            }
        }
    }
    noc_async_atomic_barrier();
}
