// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// WS-O LANE, BRISC, DM_DYNAMIC_NOC (tt/pf_wo.PfStreamWO; protocol: pfwo_common.hpp). Lane L (on NoC L; the only
// multicaster on that NoC into the box) serves chunks c = 2 j + L in order: wait its SEM_ON >= c + 1 (the scan finished
// chunk c's `on` rows; budgeted), read the chunk (row tile c: KT DRAM pages) into a 2-chunk staging ring, wait every
// member's credit (ring slot c mod S free), multicast it to the members' box as single packets (<= NOC_MAX_BURST_SIZE;
// MC 1: writes_flushed each, MC 2: a write barrier each), then the VALID SET (= QBASE / 2 + j + 1: block QBASE + c, PfStream's numbering) behind the data. The lane is
// OUTSIDE the box: n = 64 destinations. EPOCH: READY / RINGADDR zeroed at the end of the launch.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pfwo_common.hpp"

namespace pfwo_l {  // (one copy per phase: the PMK composite includes this body in its own namespace)
constexpr uint32_t KT = get_compile_time_arg_val(PH_CT0 + pfwo::CT_KT);
constexpr uint32_t MT = get_compile_time_arg_val(PH_CT0 + pfwo::CT_MT);
constexpr uint32_t S = get_compile_time_arg_val(PH_CT0 + pfwo::CT_S);
constexpr uint32_t XTB = get_compile_time_arg_val(PH_CT0 + pfwo::CT_XTB);
constexpr uint32_t NDEST = get_compile_time_arg_val(PH_CT0 + pfwo::CT_NDEST);
constexpr uint32_t MC = get_compile_time_arg_val(PH_CT0 + pfwo::CT_MC);
constexpr uint32_t CB_STG = get_compile_time_arg_val(PH_CT0 + pfwo::CT_CB_STG);
constexpr uint32_t SEM_V0 = get_compile_time_arg_val(PH_CT0 + pfwo::CT_SEM_V0);
constexpr uint32_t SEM_ON = get_compile_time_arg_val(PH_CT0 + pfwo::CT_SEM_ON);
constexpr uint32_t NJ = get_compile_time_arg_val(PH_CT0 + pfwo::CT_NJ);
constexpr uint32_t QBASE = get_compile_time_arg_val(PH_CT0 + pfwo::CT_QBASE);
constexpr uint32_t CHB = KT * XTB;  // one chunk = one row tile of `on` (64 KB at K 1,024)
static_assert(MT % 2 == 0 && NJ == MT / 2 && S >= 1 && S <= 8, "lane geometry");
static_assert(CHB % NOC_MAX_BURST_SIZE == 0 || CHB < NOC_MAX_BURST_SIZE, "whole packets per chunk");
static_assert(MC == 1 || MC == 2, "CT_MC");
static_assert(QBASE % 2 == 0, "lane parity continues across the phases of the launch");
}  // namespace pfwo_l

void kernel_main() {
    using namespace pfwo_l;
    const uint32_t sync = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_SYNC);
    // the PMK composite runs this body on EVERY root (idle_ok): a root that is no WS lane has all-zero args -> idle (the
    // Galaxy blk0 06:16Z watcher stop: an idle root multicast 16 KB to box (0, 0)-(0, 0), DRAM 0xffff0010)
    if (sync == 0) {
        return;
    }
    const uint32_t L = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_LANE);
    const uint8_t noc = static_cast<uint8_t>(L);  // exactly one lane per NoC
    const uint32_t lx = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_RECT), ly = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_RECT + 1);
    const uint32_t hx = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_RECT + 2), hy = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_RECT + 3);
    const uint32_t n = get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_RECT + 4);
    constexpr auto y_args = TensorAccessorArgs<PH_CT0 + pfwo::CT_ACC>();
    constexpr auto x_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const auto xx = TensorAccessor(x_args, get_arg_val<uint32_t>(ph_rt0 + pfwo::RT_XADDR));
    const uint32_t t0 = get_timestamp_32b();
    uint64_t t_ready = 0, t_on = 0, t_read = 0, t_mc = 0;
    volatile tt_l1_ptr uint32_t* gen = pfwo::word(sync, pfwo::GEN);
    const uint32_t g = *gen + 1;
    const uint32_t stg = get_write_ptr(CB_STG);
    const uint32_t valid = get_semaphore(SEM_V0 + L);  // the members' VALID of this lane (same id, same address)
    volatile tt_l1_ptr uint32_t* on = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ON));
    volatile tt_l1_ptr uint32_t* val = pfwo::word(sync, pfwo::VAL);
    uint32_t cnt[S];
    for (uint32_t w = 0; w < S; w++) {
        cnt[w] = 0;
    }
    // chunk j's read: wait the scan's SEM_ON >= c + 1 (budgeted), then the row tile's KT pages
    auto issue = [&](uint32_t j) {
        const uint32_t c = 2 * j + L;
        uint32_t tq = get_timestamp_32b();
        pfwo::wait_budget([&] { return *on >= c + 1; }, sync, 0x1000u | c, on);
        t_on += static_cast<uint32_t>(get_timestamp_32b() - tq);
        const uint32_t dst = stg + (j % 2) * CHB;
        for (uint32_t k = 0; k < KT; k++) {
            noc_async_read(xx.get_noc_addr(c * KT + k, 0, noc), dst + k * XTB, XTB, noc);
        }
    };
    issue(0);
    for (uint32_t j = 0; j < NJ; j++) {
        uint32_t tq = get_timestamp_32b();
        noc_async_read_barrier(noc);  // chunk j staged (the only read in flight)
        t_read += static_cast<uint32_t>(get_timestamp_32b() - tq);
        // the next chunk's read now when the scan already finished it (the other staging slot held chunk j - 1, whose
        // multicast ended with a flush / barrier below); else after this chunk's multicast
        bool pre = false;
        if (j + 1 < NJ) {
            invalidate_l1_cache();
            pre = *on >= 2 * (j + 1) + L + 1;
            if (pre) {
                issue(j + 1);
            }
        }
        const uint32_t c = 2 * j + L, slot = c % S;
        cnt[slot]++;
        tq = get_timestamp_32b();
        {
            volatile tt_l1_ptr uint32_t* rw = pfwo::word(sync, pfwo::READY + slot);
            const uint32_t want = NDEST * cnt[slot];
            pfwo::wait_budget([&] { return static_cast<int32_t>(*rw - want) >= 0; }, sync, (slot << 12) | (j & 0xFFFu), rw);
        }
        t_ready += static_cast<uint32_t>(get_timestamp_32b() - tq);
        const uint32_t ring = *pfwo::word(sync, pfwo::RINGADDR);
        ASSERT(ring != 0);  // member 0 published it before its first credit (a write barrier in between)
        const uint32_t dst0 = ring + slot * CHB;
        tq = get_timestamp_32b();
        for (uint32_t off = 0; off < CHB; off += NOC_MAX_BURST_SIZE) {
            const uint32_t len = (CHB - off) < NOC_MAX_BURST_SIZE ? (CHB - off) : NOC_MAX_BURST_SIZE;
            // NoC 0 multicasts go low -> high corner, NoC 1 high -> low (pfs_stream.cpp's convention)
            const uint64_t dst = noc ? get_noc_multicast_addr(hx, hy, lx, ly, dst0 + off, noc)
                                     : get_noc_multicast_addr(lx, ly, hx, hy, dst0 + off, noc);
#if defined(PFS_MC_PKT) && PFS_MC_PKT == 4
            noc_async_write_barrier(noc);  // the tt-metal #15018 rule: no unlinked multicast while an earlier one is un-acked
#endif
            noc_async_write_multicast_one_packet(stg + (j % 2) * CHB + off, dst, len, n, false, noc);
            if constexpr (MC == 2) {
                noc_async_write_barrier(noc);
            } else {
                noc_async_writes_flushed(noc);
            }
        }
        *val = QBASE / 2 + j + 1;  // the launch-global block QBASE + c's VALID (PfStream's Q / 2 + 1)
        const uint64_t vdst = noc ? get_noc_multicast_addr(hx, hy, lx, ly, valid, noc)
                                  : get_noc_multicast_addr(lx, ly, hx, hy, valid, noc);
#if defined(PFS_MC_PKT) && PFS_MC_PKT == 4
        noc_async_write_barrier(noc);  // #15018: the VALID SET is an unlinked multicast too
#endif
        noc_semaphore_set_multicast(reinterpret_cast<uint32_t>(val), vdst, n, false, noc);
        noc_async_writes_flushed(noc);  // the VALID source word is read before the next chunk rewrites it
        t_mc += static_cast<uint32_t>(get_timestamp_32b() - tq);
        if (j + 1 < NJ && !pre) {
            issue(j + 1);
        }
    }
    noc_async_write_barrier(noc);  // every member acked this lane's chunks and VALID sets
    // EPOCH: this launch's credits all landed (every chunk's full count was awaited); the next launch's members cannot
    // write before this program ends
    for (uint32_t w = 0; w < S; w++) {
        *pfwo::word(sync, pfwo::READY + w) = 0;
    }
    *pfwo::word(sync, pfwo::RINGADDR) = 0;
    *pfwo::word(sync, pfwo::P_READY) = static_cast<uint32_t>(t_ready);
    *pfwo::word(sync, pfwo::P_ON) = static_cast<uint32_t>(t_on);
    *pfwo::word(sync, pfwo::P_READ) = static_cast<uint32_t>(t_read);
    *pfwo::word(sync, pfwo::P_MC) = static_cast<uint32_t>(t_mc);
    *pfwo::word(sync, pfwo::P_SPAN) = static_cast<uint32_t>(get_timestamp_32b() - t0);
    *gen = g;
}
