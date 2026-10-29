// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK_COMB_DUAL (14b V5 B2-c, the BRISC half; moe_common.hpp): included in moe_brisc.cpp's `comb` namespace after
// comb_writer.hpp (CT0 / RT0 = the y writer's args). The combine's out_rm gather was ONE NCRISC on NoC1 at the all-core
// NoC1 DRAM rate (d0 bb1 2026-10-26: 244 GB/s, 2.2-3.2 GB/s per core vs NoC0's 380); here the BRISC reads slots [PMK_COMB_DUAL_NC, K) of
// every token on NoC0 into the CIN slot the NCRISC reserved and published (SEM_DUAL_R), raises SEM_DUAL_D, and keeps the
// y write of each unit (comb_writer.hpp's body, H9 included) after its 32 tokens. Same units (the writer's u0 / nu, already
// rebased per rank by pmk_b_window), same token order as comb_reader.hpp. The reader's geometry: PMK_CT_COMBR + [C, K,
// ROW_BYTES, PART_BYTES, PT, H_SPLIT]; out_rm's address: PMK_RT_COMBR + 0. The dest words are the NCRISC's (CB_CIDX past
// the sync words), read by it before the unit's first token is published and kept until the unit's last push.
#if defined(PMK_COMB_DUAL) && defined(PMK_MCAST)
#error "PMK_COMB_DUAL reuses the PMK_MCAST counters SEM_VALID_GU / SEM_VALID_D"
#endif
#if defined(PMK_COMB_DUAL) && !defined(PMK_COMB_DUAL_NC)
#error "PMK_COMB_DUAL needs PMK_COMB_DUAL_NC (the NCRISC's routed-slot count; host default K / 2)"
#endif
FORCE_INLINE void dual_main() {
    constexpr uint32_t PT = get_compile_time_arg_val(CT0 + 0);
    constexpr uint32_t HT = get_compile_time_arg_val(CT0 + 1);
    constexpr uint32_t H_SPLIT = get_compile_time_arg_val(CT0 + 2);
    constexpr uint32_t TILE_BYTES = 2048;
    constexpr uint32_t NT = 32 * PT;
    constexpr uint32_t K = get_compile_time_arg_val(PMK_CT_COMBR + 1);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(PMK_CT_COMBR + 2);
    constexpr uint32_t PART_BYTES = get_compile_time_arg_val(PMK_CT_COMBR + 3);
    static_assert(get_compile_time_arg_val(PMK_CT_COMBR + 4) == PT, "reader / writer PT");
    static_assert(get_compile_time_arg_val(PMK_CT_COMBR + 5) == H_SPLIT, "reader / writer H_SPLIT");
    const uint32_t y_addr = get_arg_val<uint32_t>(RT0 + 0);
    const uint32_t u0 = get_arg_val<uint32_t>(RT0 + 1);
    const uint32_t nu = get_arg_val<uint32_t>(RT0 + 2);
    if (nu == 0) {
        return;
    }
    const uint32_t out_addr = get_arg_val<uint32_t>(PMK_RT_COMBR + 0);
    const InterleavedAddrGen<true> yg{.bank_base_address = y_addr, .page_size = TILE_BYTES};
    const InterleavedAddrGen<true> out_g{.bank_base_address = out_addr, .page_size = ROW_BYTES};
    volatile tt_l1_ptr uint32_t* idx =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_CIDX) + SYNC_WORDS_B);
    volatile tt_l1_ptr uint32_t* dual_r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DUAL_R));
    volatile tt_l1_ptr uint32_t* dual_d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DUAL_D));
    uint32_t n = 0;
#if !defined(PMK_COMB_STRIDE)
#define PMK_COMB_STRIDE 1
#endif
    for (uint32_t k_u = 0; k_u < nu; ++k_u) {
#if defined(PMK_COMB_UNITS)
        const uint32_t u = (u0 + k_u * PMK_COMB_STRIDE) % PMK_COMB_UNITS;
#else
        const uint32_t u = u0 + k_u * PMK_COMB_STRIDE;
#endif
        const uint32_t chunk = u / H_SPLIT;
        const uint32_t part = u - chunk * H_SPLIT;
        for (uint32_t t = 0; t < 32; ++t) {
            ++n;
            uint32_t rv;
            do {  // the NCRISC reserved token n's slot (its dest words for this unit landed before the first one)
                invalidate_l1_cache();
                rv = *dual_r;
            } while ((rv >> DUAL_ADDR_BITS) < n);
            const uint32_t wp = (rv & ((1u << DUAL_ADDR_BITS) - 1u)) << 4;
            for (uint32_t j = PMK_COMB_DUAL_NC; j < K; ++j) {  // the NCRISC reads [0, NC) + y_s on NoC1
                noc_async_read(out_g.get_noc_addr(idx[j * 32 + t], part * PART_BYTES), wp + j * PART_BYTES, PART_BYTES);
            }
            noc_async_read_barrier();
            *dual_d = n;
        }
        // the unit's y write (comb_writer.hpp)
        const uint32_t base = chunk * HT + part * NT;
        cb_wait_front(CB_COUT, NT);
#if defined(PMK_BW_STAGE) && PMK_BW_STAGE == 5
        cb_pop_front(CB_COUT, NT);
        continue;
#endif
        const uint32_t rp = get_read_ptr(CB_COUT);
        for (uint32_t x = 0; x < NT; ++x) {
            noc_async_write(rp + x * TILE_BYTES, yg.get_noc_addr(base + x), TILE_BYTES);
        }
        noc_async_write_barrier();
        PMK_H9_SIGNAL(chunk);
        cb_pop_front(CB_COUT, NT);
    }
}
