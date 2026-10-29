// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Batched router tail (PMK P12, GLM53_PMK_RTAIL_BATCH=1; batch_compute.cpp is the TRISC) — data movement, ONE source for
// both RISCs of a member (rows t0 .. t0 + 31 = the member's tile row of the logits, t0 % 32 == 0):
// - RTB_READER (BRISC, NoC0): the SCORE_TILES natural logits tiles of the tile row (tile j = experts 32 j .. 32 j + 31)
//   + the host-built bias tiles (bias[32 j + c] replicated over 32 rows) -> the TRISC (tile 0 first, the rest in one
//   batch); then the top-8 of rows 16 .. 31 (faces 2 / 3).
// - else (NCRISC, NoC1): the top-8 of rows 0 .. 15 (faces 0 / 1); after the BRISC's rows landed (it +1s CB_IDX: one
//   producer, one consumer) it pops CHOICE / SCORES and sends the early-plan signals.
// The top-8 is router_tail/writer.cpp's (PMK_P12_FAST_SCAN walk): per row the 288 keys sort_key(fp32 choice bits) in
// ascending expert order, the same insertion, ties to the lower expert id -- walked tile by tile as the TRISC pushes
// them (the row's keys / ids carried across tiles on the stack, 16 rows x 8 x 2 words), so the scan overlaps the
// compute. RTAIL_WEIGHTS / RTAIL_RM_OUT as the writer's: w_j = bf16_rne(s_j * scaling / (sum_j s_j + eps)) over the 8
// selected bf16 scores in selection order, the ids / weights rows collected in CB_IDX past its first 64 B (row r at
// + r x 16 B), each RISC storing its own 16 rows (256 B per tensor). RTB_VE_RT (GLM53_PMK_ROW_MASK): rows >=
// valid_end -> the sentinel expert N_EXPERTS, weight 0 (the writer's _rtail_writer_row_mask semantics).
// Compile-time args: TensorAccessorArgs of logits [C, 320] fp32, bias [32, 320] fp32, ids [1, C x 8] uint16, weights
// [1, C x 8] bf16, SCALING_BITS, EPS_BITS. Runtime: logits, bias, the row argument, ids, weights addresses, then
// (RTB_VE_RT) the valid_end address, then (NCRISC, RTAIL_PLAN_SIG_SEM) the signal count + NoC (x, y) per plan core.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/router_tail/router_tail_common.hpp"

using namespace rtail;

#if !defined(RTAIL_ROWTILES) || !defined(RTAIL_WEIGHTS) || !defined(RTAIL_RM_OUT) || RTAIL_KSPLIT != 1
#error "the batched router tail is the PMK P12 form: RTAIL_ROWTILES + RTAIL_WEIGHTS + RTAIL_RM_OUT, one K chunk"
#endif
#if defined(MK_RTAIL_WAIT_SEM) || defined(MK_RTAIL_SIGNAL_SEM)
#error "the batched router tail: no decode megakernel hand-offs"
#endif

namespace {
constexpr uint32_t RTB_ROWS = 16;  // token rows per RISC
#if defined(RTB_READER)
constexpr uint32_t RTB_R0 = 16;  // BRISC: rows 16 .. 31
constexpr uint32_t RTB_VE_OFF = 1152;  // its valid_end word in CB_IDX (past the 64 B + 2 x 512 B rows; 64-B aligned)
#else
constexpr uint32_t RTB_R0 = 0;  // NCRISC: rows 0 .. 15
constexpr uint32_t RTB_VE_OFF = 0;
#endif

FORCE_INLINE float bf16_to_f32(uint16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}
FORCE_INLINE uint16_t f32_to_bf16_rne(float v) {
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    return static_cast<uint16_t>(((bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u) >> 16);
}
}  // namespace

void kernel_main() {
    constexpr auto log_args = TensorAccessorArgs<PH_CT0 + 0>();
    constexpr auto bias_args = TensorAccessorArgs<log_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto wt_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr uint32_t SCALING_BITS = get_compile_time_arg_val(wt_args.next_compile_time_args_offset() + 0);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(wt_args.next_compile_time_args_offset() + 1);
    const uint32_t arg = get_arg_val<uint32_t>(ph_rt0 + 2);
    if (arg == ROW_SKIP) {
        return;  // megakernel: an owner without a token row
    }
    MK_MARK();  // tail probe 0: start
    const auto idx = TensorAccessor(idx_args, get_arg_val<uint32_t>(ph_rt0 + 3));
    const auto weights = TensorAccessor(wt_args, get_arg_val<uint32_t>(ph_rt0 + 4));
    const uint32_t t0 = first_row(arg);
    const uint32_t scratch = get_write_ptr(CB_IDX);  // (before the BRISC's CB_IDX push)
    const uint32_t blk_ids = scratch + 64, blk_wt = blk_ids + 32 * TOP_K * 2;
#if defined(RTB_VE_RT)
    const InterleavedAddrGen<true> ve_g{.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + RTB_VE_RT), .page_size = 4};
    noc_async_read(ve_g.get_noc_addr(0), scratch + RTB_VE_OFF, 4);  // GLM53_PMK_ROW_MASK: the chunk's valid rows
#endif
#if defined(RTB_READER)
    {
        const auto logits = TensorAccessor(log_args, get_arg_val<uint32_t>(ph_rt0 + 0));
        const auto bias = TensorAccessor(bias_args, get_arg_val<uint32_t>(ph_rt0 + 1));
#if defined(RTAIL_PREFILL)
        noc_async_write_barrier();  // PMK P12: this RISC's PfStream writer stored the logits rows just before (program order)
#endif
        cb_reserve_back(CB_LOG, SCORE_TILES);
        cb_reserve_back(CB_BIAS, SCORE_TILES);
        const uint32_t l_l1 = get_write_ptr(CB_LOG), b_l1 = get_write_ptr(CB_BIAS);
        const uint32_t l_sz = get_tile_size(CB_LOG), b_sz = get_tile_size(CB_BIAS);
        noc_async_read_page(log_page(t0, 0), logits, l_l1);
        noc_async_read_page(0, bias, b_l1);
        noc_async_read_barrier();
        cb_push_back(CB_LOG, 1);  // tile 0: the TRISC starts
        cb_push_back(CB_BIAS, 1);
        for (uint32_t j = 1; j < SCORE_TILES; ++j) {
            noc_async_read_page(log_page(t0, j), logits, l_l1 + j * l_sz);
            noc_async_read_page(j, bias, b_l1 + j * b_sz);
        }
        noc_async_read_barrier();
        cb_push_back(CB_LOG, SCORE_TILES - 1);  // (the CBs hold SCORE_TILES pages: the pushes end on the CB end)
        cb_push_back(CB_BIAS, SCORE_TILES - 1);
    }
#endif
#if defined(RTB_VE_RT)
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t valid_end = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + RTB_VE_OFF)[0];
#endif
    uint32_t row_keys[RTB_ROWS][TOP_K];
    uint32_t row_ids[RTB_ROWS][TOP_K];
    uint32_t row_n[RTB_ROWS];
    for (uint32_t i = 0; i < RTB_ROWS; ++i) {
        row_n[i] = 0;
    }
    const uint32_t c_sz = get_tile_size(CB_CHOICE);
    for (uint32_t tj = 0; tj < SCORE_TILES; ++tj) {
        cb_wait_front(CB_CHOICE, tj + 1);  // choice tiles 0 .. tj packed (never popped before the end: no wrap)
        invalidate_l1_cache();             // packer-written data read by this RISC
        volatile tt_l1_ptr uint32_t* choice =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_CHOICE) + tj * c_sz);
        for (uint32_t i = 0; i < RTB_ROWS; ++i) {
            uint32_t* keys = row_keys[i];
            uint32_t* ids = row_ids[i];
            uint32_t n = row_n[i];
            const uint32_t r = RTB_R0 + i;  // the row in the tile: expert 32 tj + 16 h + j at tile_off(r, 16 h) + j
            for (uint32_t h = 0; h < 2; ++h) {
                volatile tt_l1_ptr uint32_t* frow = choice + tile_off(r, 16u * h);
                const uint32_t e0 = 32u * tj + 16u * h;
                for (uint32_t j = 0; j < 16; ++j) {
                    const uint32_t k = sort_key(frow[j]);
                    if (n == TOP_K && k <= keys[TOP_K - 1]) {
                        continue;
                    }
                    uint32_t pos = n < TOP_K ? n : TOP_K - 1;
                    while (pos > 0 && keys[pos - 1] < k) {
                        keys[pos] = keys[pos - 1];
                        ids[pos] = ids[pos - 1];
                        --pos;
                    }
                    keys[pos] = k;
                    ids[pos] = e0 + j;
                    if (n < TOP_K) {
                        ++n;
                    }
                }
            }
            row_n[i] = n;
        }
    }
    MK_MARK();  // the rows' top-8
    cb_wait_front(CB_SCORES, SCORE_TILES);  // (pushed with the choice tiles)
    invalidate_l1_cache();                  // packer-written scores read by this RISC
    {
        const uint32_t s_l1 = get_read_ptr(CB_SCORES), s_sz = get_tile_size(CB_SCORES);
        constexpr float scaling = __builtin_bit_cast(float, SCALING_BITS);
        constexpr float eps = __builtin_bit_cast(float, EPS_BITS);
        for (uint32_t i = 0; i < RTB_ROWS; ++i) {
            const uint32_t r = RTB_R0 + i;
            const uint32_t* ids = row_ids[i];
            volatile tt_l1_ptr uint16_t* out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(blk_ids + r * TOP_K * 2);
            for (uint32_t j = 0; j < TOP_K; ++j) {
#if defined(RTB_VE_RT)
                out[j] = static_cast<uint16_t>(t0 + r >= valid_end ? N_EXPERTS : ids[j]);
#else
                out[j] = static_cast<uint16_t>(ids[j]);
#endif
            }
            // the 8 selected scores (bf16: score tile e / 32, row r, column e % 32), their fp32 sum in selection order
            float sv[TOP_K];
            float sum = 0.0f;
            for (uint32_t j = 0; j < TOP_K; ++j) {
                volatile tt_l1_ptr uint16_t* sc =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(s_l1 + (ids[j] >> 5) * s_sz);
                sv[j] = bf16_to_f32(sc[tile_off(r, ids[j] & 31u)]);
                sum += sv[j];
            }
            const float inv = scaling / (sum + eps);
            volatile tt_l1_ptr uint16_t* wout = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(blk_wt + r * TOP_K * 2);
            for (uint32_t j = 0; j < TOP_K; ++j) {
#if defined(RTB_VE_RT)
                wout[j] = t0 + r >= valid_end ? static_cast<uint16_t>(0) : f32_to_bf16_rne(sv[j] * inv);
#else
                wout[j] = f32_to_bf16_rne(sv[j] * inv);
#endif
            }
        }
    }
    {
        // this RISC's 16 rows, contiguous in each one-page tensor (256 B at row t0 + RTB_R0)
        constexpr uint32_t nb = RTB_ROWS * TOP_K * 2;
        noc_async_write(blk_ids + RTB_R0 * TOP_K * 2, idx.get_noc_addr(0, (t0 + RTB_R0) * TOP_K * 2), nb);
        noc_async_write(blk_wt + RTB_R0 * TOP_K * 2, weights.get_noc_addr(0, (t0 + RTB_R0) * TOP_K * 2), nb);
        noc_async_write_barrier();
    }
#if defined(RTB_READER)
    cb_reserve_back(CB_IDX, 1);  // the BRISC's rows landed and it reads CHOICE / SCORES no more
    cb_push_back(CB_IDX, 1);
#else
    cb_wait_front(CB_IDX, 1);
    cb_pop_front(CB_IDX, 1);
    cb_pop_front(CB_CHOICE, SCORE_TILES);
    cb_pop_front(CB_SCORES, SCORE_TILES);
#if defined(RTAIL_PLAN_SIG_SEM)
    // Stage 14 B2.3 step (2): the member's 32 ids / weights rows landed (both RISCs' write barriers) -> +1 on every
    // early-plan core: the count at ph_rt0 + RTAIL_PLAN_SIG_RT, then (x, y) per plan core
    mk::signal_handoff(RTAIL_PLAN_SIG_SEM, ph_rt0 + RTAIL_PLAN_SIG_RT + 1, get_arg_val<uint32_t>(ph_rt0 + RTAIL_PLAN_SIG_RT));
#endif
#endif
    MK_MARK();  // signalled
}
