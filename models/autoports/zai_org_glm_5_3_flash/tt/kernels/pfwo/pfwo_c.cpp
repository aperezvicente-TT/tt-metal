// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// WS-O MEMBER TRISC (tt/pf_wo.PfStreamWO). Per chunk c (= row tile c of `on`, in the ring in chunk order): y(c, n) =
// sum over k = 0 .. KT - 1 ascending of x(c, k) W(k, n) for the member's COLS column tiles n, in COLS fp32 DST tiles --
// one matmul_block step per K tile (in0 = the x tile, in1 = the resident weight row k: COLS tiles) -- packed once per
// chunk (so block c's tiles leave as soon as its chunk is done: the scan's pace). Per output tile this is the
// PfStream member's sequence (pfs_member_c.cpp: in0 x, in1 W, every K tile ascending into one DST tile, one pack):
// bitwise.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "pfwo_common.hpp"

namespace pfwo_c {
constexpr uint32_t KT = get_compile_time_arg_val(PH_CT0 + pfwo::CT_KT);
constexpr uint32_t MT = get_compile_time_arg_val(PH_CT0 + pfwo::CT_MT);
constexpr uint32_t COLS = get_compile_time_arg_val(PH_CT0 + pfwo::CT_COLS);
constexpr uint32_t CB_WS = get_compile_time_arg_val(PH_CT0 + pfwo::CT_CB_WS);
constexpr uint32_t CB_XR = get_compile_time_arg_val(PH_CT0 + pfwo::CT_CB_XR);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(PH_CT0 + pfwo::CT_CB_OUT);
static_assert(COLS <= 4, "COLS fp32 tiles in a half-sync DST (4)");
}  // namespace pfwo_c

// PMK_PART_FID (the PMK composite's per-phase fidelity, kernels/pmk/pmk_fid.hpp): this part's matmul at its own fidelity
// under a kernel-wide MATH_FIDELITY of another phase (pfs_member_c.cpp's form)
#if defined(PMK_PART_FID)
#define PFWO_MM_INIT(...) pmk::mm_block_init_fid<PMK_PART_FID>(__VA_ARGS__)
#define PFWO_MM(...) pmk::mm_block_fid<PMK_PART_FID>(__VA_ARGS__)
#else
#define PFWO_MM_INIT(...) matmul_block_init(__VA_ARGS__)
#define PFWO_MM(...) matmul_block(__VA_ARGS__)
#endif

void kernel_main() {
    using namespace pfwo_c;
    // in0 (x) lands in SrcB, in1 (W) in SrcA: pfs_member_c.cpp's start-up; in a composite after other compute phases
    // only the data formats are reconfigured (the previous phase left the unpackers / packer on its own CBs)
#if !defined(MK_PHASED) || defined(PMK_FIRST_COMPUTE)
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_XR, CB_WS, CB_OUT);
#else
    reconfig_data_format(CB_WS, CB_XR);
    pack_reconfig_data_format(CB_OUT);
#endif
    PFWO_MM_INIT(CB_XR, CB_WS, false, COLS, 1, KT);
    cb_wait_front(CB_WS, KT * COLS);
    for (uint32_t c = 0; c < MT; ++c) {
        tile_regs_acquire();
        cb_wait_front(CB_XR, KT);
        for (uint32_t k = 0; k < KT; ++k) {
            PFWO_MM(CB_XR, CB_WS, k, k * COLS, 0, false, COLS, 1, KT);
        }
        cb_pop_front(CB_XR, KT);
        tile_regs_commit();
        cb_reserve_back(CB_OUT, COLS);
        tile_regs_wait();
        for (uint32_t i = 0; i < COLS; ++i) {
            pack_tile(i, CB_OUT);
        }
        tile_regs_release();
        cb_push_back(CB_OUT, COLS);
    }
    cb_pop_front(CB_WS, KT * COLS);
}
#undef PFWO_MM_INIT
#undef PFWO_MM
