// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a PfStream v3 (phase-ready), MEMBER TRISC (tt/pf_stream.PfStreamMM): y[32, N] = x[32, K] @ W[K, N] for the
// member's stationary row tile (CB_X, KT tiles, pushed by the member's NCRISC body). Per N-block (NB columns) the NB
// fp32 outputs stay in DST (half-sync: NB <= 4, the PMK composite's mode; PFS_DST_FULL: NB <= 8) across ALL K-blocks (PARK, KB x NB tiles from the streamers);
// after the last K-block the NB tiles are packed once to CB_OUT (EPI 0), or through the SwiGLU epilogue into the
// next phase's in0 (EPI 1), or untilized into ROW_MAJOR rows (EPI 2). CB_X is popped at the end of the phase.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "pfs_common.hpp"

namespace pfs_c {
constexpr uint32_t NBK = get_compile_time_arg_val(PH_CT0 + pfs::CT_NBK);
constexpr uint32_t KB = get_compile_time_arg_val(PH_CT0 + pfs::CT_KB);
constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + pfs::CT_NB);
constexpr uint32_t KT = get_compile_time_arg_val(PH_CT0 + pfs::CT_KT);
constexpr uint32_t CB_X = get_compile_time_arg_val(PH_CT0 + pfs::CT_X);
constexpr uint32_t CB_PARK = get_compile_time_arg_val(PH_CT0 + pfs::CT_PARK);
constexpr uint32_t NBLK = get_compile_time_arg_val(PH_CT0 + pfs::CT_NBLK);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(PH_CT0 + pfs::CT_OUT);
constexpr uint32_t EPI = get_compile_time_arg_val(PH_CT0 + pfs::CT_EPI);
constexpr uint32_t LIMIT_BITS = get_compile_time_arg_val(PH_CT0 + pfs::CT_LIMIT);
constexpr uint32_t NEG_LIMIT_BITS = LIMIT_BITS ^ 0x80000000u;
constexpr uint32_t RT = get_compile_time_arg_val(PH_CT0 + pfs::CT_RT);  // row tiles per member (DST: RT x NB fp32)
constexpr uint32_t OUT_TILES = RT * (EPI == 1 ? NB / 2 : NB);  // tiles pushed to CB_OUT per N-block
static_assert(EPI != 1 || NB % 2 == 0, "SwiGLU: (gate, up) tile pairs per N-block");
#if defined(PFS_DST_FULL)
static_assert(KT == KB * NBK && RT * NB <= 8, "blocks; RT x NB fp32 tiles in a full-sync DST (8)");
#else
static_assert(KT == KB * NBK && RT * NB <= 4, "blocks; RT x NB fp32 tiles in a half-sync DST (4: the PMK member composite's mode)");
#endif
}  // namespace pfs_c
// PMK_PART_FID (the PMK composite's per-phase fidelity, kernels/pmk/pmk_fid.hpp, included by the composite): this
// part's matmul at its own fidelity under a kernel-wide MATH_FIDELITY of another phase (A3 + A4 in one program)
#if defined(PMK_PART_FID)
#define PFS_MM_INIT(...) pmk::mm_block_init_fid<PMK_PART_FID>(__VA_ARGS__)
#define PFS_MM(...) pmk::mm_block_fid<PMK_PART_FID>(__VA_ARGS__)
#else
#define PFS_MM_INIT(...) matmul_block_init(__VA_ARGS__)
#define PFS_MM(...) matmul_block(__VA_ARGS__)
#endif

void kernel_main() {
    using namespace pfs_c;
    // PMK_FIRST_COMPUTE (tt/pmk_layer.composite_source): this part is the composite's FIRST compute body, so nothing
    // earlier in the kernel started the hardware. A reconfigure-only start would inherit the previous PROGRAM's state,
    // incl. the unpacker config context: the kernel crt re-zeroes unp_cfg_context while the HW CfgContextOffset keeps
    // the previous kernel's last context (AD-3a d0 pair, 2026-10-27: D2's 289 unpacks left context 1, so P18 alone read
    // its first in0 / in1 tiles through stale context-1 base addresses)
#if !defined(MK_PHASED) || defined(PMK_FIRST_COMPUTE)
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_PARK, CB_OUT);
#else
    reconfig_data_format(CB_PARK, CB_X);  // the previous phase left the unpackers / packer on its own CBs
    pack_reconfig_data_format(CB_OUT);
#endif
    // RT row tiles per block: in0 tile (r, k) = r KT + k (kt_dim = KT is the row stride), in1 tile (k, c) = k NB + c, the
    // product (r, c) at DST r NB + c; the LLK reuses in0 across the NB columns (reuse_a = ct >= rt) and streams in1
    PFS_MM_INIT(CB_X, CB_PARK, false, NB, RT, KT);
    if constexpr (EPI == 2) {
        pack_untilize_dest_init<NB, NB>(CB_OUT);
    }
#if defined(PFS_PROBE)
    const uint32_t probe = get_arg_val<uint32_t>(ph_rt0 + pfs::RT_PROBE);
    uint32_t n_thr = 0;
    uint32_t t_c0 = 0;
#endif
    cb_wait_front(CB_X, RT * KT);
    for (uint32_t n = 0; n < NBLK; ++n) {
        if constexpr (EPI == 1) {
            if (n > 0) {  // the SFPU inits of the previous N-block's epilogue: restore the matmul state
                PFS_MM_INIT(CB_X, CB_PARK, false, NB, RT, KT);
            }
        }
        tile_regs_acquire();
        for (uint32_t kb = 0; kb < NBK; ++kb) {
            cb_wait_front(CB_PARK, KB * NB);
#if defined(PFS_PROBE) && defined(ARCH_BLACKHOLE)
            MATH((n_thr += (*ckernel::throttle_ptr) & 1u));  // the FW power throttle flag at this block
            MATH((t_c0 = (n == 0 && kb == 0) ? get_timestamp_32b() : t_c0));
#endif
#if defined(PFS_ZSKIP)
            // GLM53_PMK_A_ZSKIP (cf3h-k4d8a L3, 16's plan_1m/proga_p3p4_mechanism.md): the weight's zero K rows (K tiles
            // outside [PFS_ZK_LO, PFS_ZK_HI)) and zero N columns (N-blocks from tile PFS_ZN_FROM on) run no matmul: their
            // products are exact zeros, so the fp32 DST accumulation is unchanged; a skipped N-block packs the DST the
            // previous release cleared (zeros). The streamed PARK blocks are waited / popped as before (same protocol)
            if (n * NB < PFS_ZN_FROM) {
                for (uint32_t k = 0; k < KB; ++k) {
                    const uint32_t kt_ = kb * KB + k;
                    if (kt_ >= PFS_ZK_LO && kt_ < PFS_ZK_HI) {
                        PFS_MM(CB_X, CB_PARK, kt_, k * NB, 0, false, NB, RT, KT);
                    }
                }
            }
#else
            for (uint32_t k = 0; k < KB; ++k) {
                PFS_MM(CB_X, CB_PARK, kb * KB + k, k * NB, 0, false, NB, RT, KT);
            }
#endif
            cb_pop_front(CB_PARK, KB * NB);
        }
        if constexpr (EPI == 1) {  // DST (g_a, u_a, g_b, u_b, ..): act_p = SiLU(min(g_p, L)) * clamp(u_p, -L, L) in place
            for (uint32_t r = 0; r < RT; ++r) {
                for (uint32_t p = 0; p < NB / 2; ++p) {
                    const uint32_t g = r * NB + 2 * p;
                    unary_min_tile_init();
                    unary_min_tile(g, LIMIT_BITS);
                    silu_tile_init();
                    silu_tile(g);
                    clamp_tile_init();
                    clamp_tile(g + 1, NEG_LIMIT_BITS, LIMIT_BITS);
                    mul_binary_tile_init();
                    mul_binary_tile(g, g + 1, g);
                }
            }
        }
        tile_regs_commit();
        cb_reserve_back(CB_OUT, OUT_TILES);
        tile_regs_wait();
        if constexpr (EPI == 1) {
            for (uint32_t r = 0; r < RT; ++r) {
                for (uint32_t p = 0; p < NB / 2; ++p) {
                    pack_tile(r * NB + 2 * p, CB_OUT);
                }
            }
        } else if constexpr (EPI == 2) {
            pack_untilize_dest<NB, NB>(CB_OUT, RT, 0);  // RT x NB tile-sized pages = RT x 32 rows x NB tiles ROW_MAJOR
        } else {
            for (uint32_t i = 0; i < RT * NB; ++i) {
                pack_tile(i, CB_OUT);
            }
        }
        tile_regs_release();
        cb_push_back(CB_OUT, OUT_TILES);
    }
    if constexpr (EPI == 2) {
        pack_untilize_uninit(CB_OUT);
    }
    cb_pop_front(CB_X, RT * KT);
#if defined(PFS_PROBE)
    if (probe) {
        MATH((*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(probe + pfs::M_THROTTLE * pfs::WORD) = n_thr));
        MATH(
            (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(probe + pfs::M_CSPAN * pfs::WORD) =
                 static_cast<uint32_t>(get_timestamp_32b() - t_c0)));
    }
#endif
}
#undef PFS_MM_INIT
#undef PFS_MM
