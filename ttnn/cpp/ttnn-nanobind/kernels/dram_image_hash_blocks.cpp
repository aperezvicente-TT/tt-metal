// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRAM-image block hashes on device: each run {bank, addr, len, out_page} is read from its DRAM bank into L1 (two
// stages: the read of stage k + 1 overlaps the hashing of stage k) and hashed as 32-bit words w_j (j = word index in
// the run) with two lanes:
//   s1 += rotl32(w_j ^ (j * 0x9E3779B9), 15) * 0x85EBCA6B        s2 += ((w_j + j * 0xC2B2AE35) ^ (w_j >> 13)) *
//   0x27D4EB2F
// (mod 2^32; the host computes the same function over the image file: ttnn-nanobind/dram_image.cpp block_hash).
// The 8-byte result {s1, s2} goes to page out_page (64 B pages) of the interleaved DRAM output buffer.
//
// Runtime args: out_addr, n_runs, then n_runs x {bank, addr, len, out_page}. len is a multiple of 64.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t stage = get_compile_time_arg_val(0);  // bytes per L1 stage (multiple of 64)
    constexpr uint32_t out_page_size = 64;
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_runs = get_arg_val<uint32_t>(1);
    const uint32_t l1_base = get_write_ptr(0);    // 2 * stage bytes
    const uint32_t l1_out = l1_base + 2 * stage;  // 64 B result page
    const InterleavedAddrGen<true> out = {.bank_base_address = out_addr, .page_size = out_page_size};

    for (uint32_t r = 0; r < n_runs; r++) {
        const uint32_t bank = get_arg_val<uint32_t>(2 + 4 * r);
        const uint32_t addr = get_arg_val<uint32_t>(3 + 4 * r);
        const uint32_t len = get_arg_val<uint32_t>(4 + 4 * r);
        const uint32_t page = get_arg_val<uint32_t>(5 + 4 * r);
        uint32_t s1 = 0, s2 = 0, j = 0;
        // prime stage 0
        uint32_t n0 = len < stage ? len : stage;
        noc_async_read(get_noc_addr_from_bank_id<true>(bank, addr), l1_base, n0);
        uint32_t k = 0;
        for (uint32_t off = 0; off < len; off += stage, k++) {
            const uint32_t n = (len - off) < stage ? (len - off) : stage;
            noc_async_read_barrier();
            const uint32_t next = off + stage;
            if (next < len) {
                const uint32_t nn = (len - next) < stage ? (len - next) : stage;
                noc_async_read(get_noc_addr_from_bank_id<true>(bank, addr + next), l1_base + ((k + 1) & 1) * stage, nn);
            }
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base + (k & 1) * stage);
            for (uint32_t i = 0, words = n >> 2; i < words; i++, j++) {
                const uint32_t w = p[i];
                uint32_t x = w ^ (j * 0x9E3779B9u);
                x = (x << 15) | (x >> 17);
                s1 += x * 0x85EBCA6Bu;
                s2 += ((w + j * 0xC2B2AE35u) ^ (w >> 13)) * 0x27D4EB2Fu;
            }
        }
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_out);
        o[0] = s1;
        o[1] = s2;
        noc_async_write(l1_out, out.get_noc_addr(page), out_page_size);
        noc_async_write_barrier();
    }
}
