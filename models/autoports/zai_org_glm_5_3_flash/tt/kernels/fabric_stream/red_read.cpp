// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream RS, REDUCER core NCRISC (tt/fabric_stream.FabricStreamRSR with reducers): the fp32 reduce of the
// owned blocks is spread over N reducer cores (the 2 lanes only send: their exchange runs at the fabric rate). For each of
// its blocks in order: wait until the peer's copy landed — poll the in-order counter ARR + q LANES + h of the receiving
// lane h over the NoC (>= (g - 1) PER_LANE + idx + 1, idx = the block's position in the peer lane h's send list) — then
// stream own (P) and landed (r) tiles as matched NB-tile batches into CB_A / CB_B for red_reduce (= rs_reduce.cpp).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fs_common.hpp"

#if !defined(MK_PHASED)  // standalone; a composite (tt/pf_stream.composite_source) defines both per phase namespace
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#endif

constexpr uint32_t TPB = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_A = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t CB_B = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t TB = 2048;
static_assert(TPB % NB == 0, "a batch never straddles a block");

#if !defined(FS_RDY_MULT)
#define FS_RDY_MULT 1  // readiness increments per block per launch (P15 streamed: B's combine units per row block)
#endif
#if defined(FS_SLICES)
// PMK_RS8_QSLICE (rs_read.cpp FS_SLICES): units (block i, column slice s) slice-major; the lane's in-order ARR counts
// units (per_lane x FS_SLICES per launch), OWNRDY + i counts slices; the batches of slice s only (rs_reduce.cpp sees
// the same NB-tile batches, in this order: per tile the same exact fp32 pair sum)
static_assert(FS_SLICES > 1 && TPB % (FS_SLICES * NB) == 0, "a slice is whole NB-tile batches");
static_assert(FS_RDY_MULT == 1, "FS_SLICES: one readiness raise per slice (P15's streamed readiness keeps Q = 1)");
#endif

#if defined(FS_SLICE_DIGITS) && !defined(FS_SLICES)
#error "FS_SLICE_DIGITS (WS-O's per-slice holder raises) needs FS_SLICES"
#endif
void kernel_main() {
    constexpr auto p_args = TensorAccessorArgs<PH_CT0 + 4>();
    constexpr auto r_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    size_t a = ph_rt0;
    const uint32_t own_sync = get_arg_val<uint32_t>(a++);  // this reducer's sync shard (its launch counter)
    const uint32_t lane_sync = get_arg_val<uint32_t>(a++); // the lanes' sync shard address (same on every lane)
    const uint32_t per_lane = get_arg_val<uint32_t>(a++);  // blocks per sender lane per launch
    const uint32_t p_addr = get_arg_val<uint32_t>(a++);
    const uint32_t r_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ready = get_arg_val<uint32_t>(a++);  // 1: wait OWNRDY + i >= g (the producer wrote own block i of P)
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t blk = a;  // n x (own P row tile, landed r row tile, lane NoC x, lane NoC y, ARR word, idx)
    const auto pp = TensorAccessor(p_args, p_addr);
    const auto rr = TensorAccessor(r_args, r_addr);
    volatile tt_l1_ptr uint32_t* gen = fs::word(own_sync, fs::GEN_NC);
    const uint32_t g = *gen + 1;
    volatile tt_l1_ptr uint32_t* probe = fs::word(own_sync, fs::GEN_C);  // NoC-read landing word for the polls
#if defined(FS_SLICE_DIGITS)
    // WS-O at PMK_RS8_QSLICE (GLM53_PMK_WS_O; tt/pf_wo.py, kernels/pfwo/pfwo_wr.cpp): a block's readiness word is raised
    // by the FS_SLICE_DIGITS members holding each column slice, a holder of slice s by B^s (B = FS_SLICE_DIGITS + 1), in
    // ANY order. After launch g's raises so far it holds (g - 1) FULL + sum_s n_s B^s, n_s <= FS_SLICE_DIGITS (every
    // earlier launch completed every slice; FULL = the sum at n_s = FS_SLICE_DIGITS): slice s of this launch is complete
    // when its base-B digit is FS_SLICE_DIGITS (exact in uint32 wrap arithmetic)
    auto wait_slice = [&](volatile tt_l1_ptr uint32_t* p, uint32_t s) {
        constexpr uint32_t B = FS_SLICE_DIGITS + 1;
        uint32_t full = 0, pw = 1, ps = 1;
        for (uint32_t k = 0; k < FS_SLICES; k++) {
            if (k == s) {
                ps = pw;
            }
            full += FS_SLICE_DIGITS * pw;
            pw *= B;
        }
        const uint32_t base = (g - 1) * full;
        WAYPOINT("FSSW");
        do {
            invalidate_l1_cache();
        } while (((*p - base) / ps) % B != FS_SLICE_DIGITS);
        WAYPOINT("FSSD");
    };
#endif
#if defined(FS_SLICES)
    constexpr uint32_t BPS = TPB / NB / FS_SLICES;  // batches per slice
    for (uint32_t s = 0; s < FS_SLICES; s++) {
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t jo = get_arg_val<uint32_t>(blk + 6 * i), jl = get_arg_val<uint32_t>(blk + 6 * i + 1);
            const uint64_t cnt = get_noc_addr(get_arg_val<uint32_t>(blk + 6 * i + 2),
                                              get_arg_val<uint32_t>(blk + 6 * i + 3),
                                              lane_sync + get_arg_val<uint32_t>(blk + 6 * i + 4) * fs::WORD);
            const uint32_t need = ((g - 1) * FS_SLICES + s) * per_lane + get_arg_val<uint32_t>(blk + 6 * i + 5) + 1;
            do {
                noc_async_read(cnt, reinterpret_cast<uint32_t>(probe), fs::WORD);
                noc_async_read_barrier();
                invalidate_l1_cache();
            } while (static_cast<int32_t>(*probe - need) < 0);
            if (ready) {
#if defined(FS_SLICE_DIGITS)
                wait_slice(fs::word(own_sync, fs::OWNRDY + i), s);
#else
                fs::wait_ge(fs::word(own_sync, fs::OWNRDY + i), (g - 1) * FS_SLICES + s + 1);
#endif
            }
            for (uint32_t b = s * BPS; b < (s + 1) * BPS; b++) {
                cb_reserve_back(CB_A, NB);
                cb_reserve_back(CB_B, NB);
                const uint32_t wa = get_write_ptr(CB_A), wb = get_write_ptr(CB_B);
                for (uint32_t t = 0; t < NB; t++) {
                    noc_async_read_page(jo * TPB + b * NB + t, pp, wa + t * TB);
                    noc_async_read_page(jl * TPB + b * NB + t, rr, wb + t * TB);
                }
                noc_async_read_barrier();
                cb_push_back(CB_A, NB);
                cb_push_back(CB_B, NB);
            }
        }
    }
#else
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t jo = get_arg_val<uint32_t>(blk + 6 * i), jl = get_arg_val<uint32_t>(blk + 6 * i + 1);
        const uint64_t cnt = get_noc_addr(get_arg_val<uint32_t>(blk + 6 * i + 2), get_arg_val<uint32_t>(blk + 6 * i + 3),
                                          lane_sync + get_arg_val<uint32_t>(blk + 6 * i + 4) * fs::WORD);
        const uint32_t need = (g - 1) * per_lane + get_arg_val<uint32_t>(blk + 6 * i + 5) + 1;
        do {
            noc_async_read(cnt, reinterpret_cast<uint32_t>(probe), fs::WORD);
            noc_async_read_barrier();
            invalidate_l1_cache();
        } while (static_cast<int32_t>(*probe - need) < 0);
        if (ready) {
            fs::wait_ge(fs::word(own_sync, fs::OWNRDY + i), g * FS_RDY_MULT);
        }
        for (uint32_t b = 0; b < TPB / NB; b++) {
            cb_reserve_back(CB_A, NB);
            cb_reserve_back(CB_B, NB);
            const uint32_t wa = get_write_ptr(CB_A), wb = get_write_ptr(CB_B);
            for (uint32_t t = 0; t < NB; t++) {
                noc_async_read_page(jo * TPB + b * NB + t, pp, wa + t * TB);
                noc_async_read_page(jl * TPB + b * NB + t, rr, wb + t * TB);
            }
            noc_async_read_barrier();
            cb_push_back(CB_A, NB);
            cb_push_back(CB_B, NB);
        }
    }
#endif  // FS_SLICES
    *gen = g;
}
