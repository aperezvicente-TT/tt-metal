// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host test of the rev 8 mask path (kernels/pmk/attn_common.hpp mask_byte_diag / bf16_of_byte / mask_col_row + the
// TRISC expansion of attn_c.cpp / attn_mask_sfpu.hpp, emulated here: T = B_col @ E8, masked <=> floor(T) even, plus
// the causal triangle on the diagonal) against the rev 6 BRISC rule (attn_dm0.cpp rev 6 push_pool_mask: one fp32
// element per (row, column) from the row memberships, the diagonal's causal bound + open-pool tail). For random
// memberships the two must agree on the allowed / masked PATTERN of every element, on context tiles and on chunk
// tiles, diagonal or not, for every row position. Also checks bf16_of_byte against a float conversion. Built and
// run by tests/s16/test_pmk_attn.py::test_mask_row_host (g++ -std=c++17 -I<worktree>).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/attn_common.hpp"

namespace {
using pmkatt::KPOOL;
using pmkatt::POOLS_PER_TILE;

// rev 6 push_pool_mask semantics: in[i][s] = row i selected pool s of the tile (every slot a real pool here)
void rev6_mask(
    bool (*masked)[32], const bool in[32][8], bool chunk_tile, uint32_t chunk_start, uint32_t qt, uint32_t k) {
    for (uint32_t i = 0; i < 32; ++i) {
        const uint32_t p = chunk_start + qt * 32 + i;
        const uint32_t tail0 = p + 1 - ((p + 1) & 3);
        for (uint32_t s = 0; s < POOLS_PER_TILE; ++s) {
            for (uint32_t c = 0; c < KPOOL; ++c) {
                bool ok = in[i][s];
                if (chunk_tile && k == qt) {
                    const uint32_t tok = chunk_start + k * 32 + s * KPOOL + c;
                    ok = tok <= p && (in[i][s] || tok >= tail0);
                }
                masked[i][s * KPOOL + c] = !ok;
            }
        }
    }
}

float bf16_to_float(uint32_t bits) {
    const uint32_t w = bits << 16;
    float f;
    std::memcpy(&f, &w, 4);
    return f;
}
}  // namespace

int main(int argc, char** argv) {
    const uint32_t trials = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 2000;
    for (uint32_t n = 0; n < 256; ++n) {  // the operand's bf16 integers
        if (bf16_to_float(pmkatt::bf16_of_byte(n)) != static_cast<float>(n)) {
            std::printf("bf16_of_byte(%u) = %04x\n", n, pmkatt::bf16_of_byte(n));
            return 1;
        }
    }
    std::mt19937 rng(7);
    uint32_t checked = 0;
    for (uint32_t t = 0; t < trials; ++t) {
        const bool chunk_tile = (t & 1) != 0;
        const uint32_t chunk_start = 128 * (1 + rng() % 64);
        const uint32_t qt = rng() % 64;
        const uint32_t k = chunk_tile ? (t % 3 == 1 ? qt : (qt == 0 ? 0 : rng() % (qt + 1))) : 0;
        const bool diag = chunk_tile && k == qt;
        bool in[32][8];
        uint32_t byte[32];
        for (uint32_t i = 0; i < 32; ++i) {
            byte[i] = 0;
            for (uint32_t s = 0; s < POOLS_PER_TILE; ++s) {
                in[i][s] = (rng() & 1) != 0;
                byte[i] |= (in[i][s] ? 1u : 0u) << s;
            }
        }
        bool m6[32][32];
        rev6_mask(m6, in, chunk_tile, chunk_start, qt, k);
        // the BRISC: the operand tile (column 0 = the byte, the diagonal rule applied), the rest zero
        std::vector<uint32_t> op(512, 0);
        for (uint32_t r = 0; r < 32; ++r) {
            pmkatt::mask_col_row(op.data(), r, diag ? pmkatt::mask_byte_diag(byte[r], r) : byte[r]);
        }
        // the TRISC: T = B_col @ E8 (B_col column 0 of row r = the operand word's low half as bf16), masked <=>
        // floor(T) even, plus the triangle (column > row) on the diagonal tile
        for (uint32_t r = 0; r < 32; ++r) {
            const float b = bf16_to_float(op[pmkatt::col0_word(r)] & 0xFFFFu);
            if (op[pmkatt::col0_word(r)] >> 16) {
                std::printf("column 1 written at row %u\n", r);
                return 1;
            }
            for (uint32_t c = 0; c < 32; ++c) {
                const float tv = b * std::ldexp(1.0f, -static_cast<int>(c / KPOOL));  // exact
                const bool masked8 = (static_cast<uint32_t>(std::floor(tv)) & 1u) == 0 || (diag && c > r);
                if (masked8 != m6[r][c]) {
                    std::printf(
                        "MISMATCH trial %u chunk %d diag %d chunk_start %u qt %u k %u row %u col %u: rev6 %d rev8 %d\n",
                        t,
                        chunk_tile,
                        diag,
                        chunk_start,
                        qt,
                        k,
                        r,
                        c,
                        m6[r][c],
                        masked8);
                    return 1;
                }
                ++checked;
            }
        }
    }
    std::printf(
        "OK %u trials, %u elements compared (rev 6 fp32 rule == rev 8 operand + TRISC expansion)\n", trials, checked);
    return 0;
}
