// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader: per item (b, h) and token the q/k/g vectors, the v row and beta; after the first token's vectors (so the
// compute kernel's per-token prep overlaps it) the item's [K,V] fp32 state.
// A (b, t, h) row of D values sits in one tensor row, i.e. row r % 32 of D/32 consecutive tiles; each tile row is
// two 64 B face rows (faces 0/1 or 2/3). v's face rows go straight into row 0 of its CB tiles. q, k and g are read
// into a scratch tile and scattered into column 0 of their CB tiles (tile j, row r = element 32 j + r), the
// layout the compute kernel broadcasts across state columns. The input CBs are zeroed once, so everything outside
// that row / column stays zero. beta's face row is read into row 0 of the beta tile and the scalar is then
// replicated over the 32 columns of row 0.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4, cb_s0 = 5, cb_scratch = 16;

constexpr uint32_t FACE_BYTES = 16 * 16 * 4;
constexpr uint32_t FACE_ROW_BYTES = 16 * 4;

struct Geometry {
    uint32_t cols, rows, row_tiles, col_tiles;
};

struct Location {
    uint32_t page, row, col;
};

// Tile page and in-tile (row, col) of flat element index e.
Location locate(const Geometry& gm, uint32_t e) {
    const uint32_t col = e % gm.cols;
    const uint32_t rest = e / gm.cols;
    const uint32_t row = rest % gm.rows;
    const uint32_t outer = rest / gm.rows;
    return {(outer * gm.row_tiles + row / 32) * gm.col_tiles + col / 32, row % 32, col % 32};
}

// Byte offset of the 16-value face row holding (row, col) of a 32x32 fp32 tile.
uint32_t face_row_offset(uint32_t row, uint32_t col) {
    return ((row / 16) * 2 + col / 16) * FACE_BYTES + (row % 16) * FACE_ROW_BYTES;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Vt = get_compile_time_arg_val(3);
    constexpr Geometry gq{
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7)};
    constexpr Geometry gk{
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11)};
    constexpr Geometry gv{
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15)};
    constexpr Geometry gg{
        get_compile_time_arg_val(16),
        get_compile_time_arg_val(17),
        get_compile_time_arg_val(18),
        get_compile_time_arg_val(19)};
    constexpr Geometry gb{
        get_compile_time_arg_val(20),
        get_compile_time_arg_val(21),
        get_compile_time_arg_val(22),
        get_compile_time_arg_val(23)};
    constexpr auto q_a = TensorAccessorArgs<24>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    constexpr uint32_t KV = Kt * Vt;

    const uint32_t item_start = get_arg_val<uint32_t>(0);
    const uint32_t item_count = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t g_addr = get_arg_val<uint32_t>(5);
    const uint32_t b_addr = get_arg_val<uint32_t>(6);
    const uint32_t s_addr = get_arg_val<uint32_t>(7);

    const uint32_t tb = get_tile_size(cb_q);
    const auto q_acc = TensorAccessor(q_a, q_addr, tb);
    const auto k_acc = TensorAccessor(k_a, k_addr, tb);
    const auto v_acc = TensorAccessor(v_a, v_addr, tb);
    const auto g_acc = TensorAccessor(g_a, g_addr, tb);
    const auto b_acc = TensorAccessor(b_a, b_addr, tb);
    const auto s_acc = TensorAccessor(s_a, s_addr, tb);

    Noc noc;
    CircularBuffer cq(cb_q), ck(cb_k), cv(cb_v), cg(cb_g), cbeta(cb_beta), cs(cb_s0);

    // Zero every page of the per-token input CBs once; later reads only write row 0.
    auto zero_cb = [&](CircularBuffer& c, uint32_t pages) {
        c.reserve_back(pages);
        noc.async_write_zeros(c, pages * tb);
    };
    zero_cb(cq, 2 * Kt);
    zero_cb(ck, 2 * Kt);
    zero_cb(cv, 2 * Vt);
    zero_cb(cg, 2 * Kt);
    zero_cb(cbeta, 2);
    noc.write_zeros_l1_barrier();

    CircularBuffer cscr(cb_scratch);
    cscr.reserve_back(1);
    const uint32_t scratch = cscr.get_write_ptr();

    // The two face rows of each of the `tiles` tiles holding flat row `row_index`: tile j's to dst + j * stride and
    // dst + j * stride + half.
    auto read_row = [&](const auto& acc,
                        const Geometry& gm,
                        uint32_t row_index,
                        uint32_t tiles,
                        uint32_t dst,
                        uint32_t stride,
                        uint32_t half) {
        const Location loc = locate(gm, row_index * tiles * 32);
        const uint32_t src = face_row_offset(loc.row, 0);
        for (uint32_t j = 0; j < tiles; j++) {
            noc.async_read(
                acc,
                CoreLocalMem<uint32_t>(dst + j * stride),
                FACE_ROW_BYTES,
                {.page_id = loc.page + j, .offset_bytes = src},
                {});
            noc.async_read(
                acc,
                CoreLocalMem<uint32_t>(dst + j * stride + half),
                FACE_ROW_BYTES,
                {.page_id = loc.page + j, .offset_bytes = src + FACE_BYTES},
                {});
        }
    };
    // Scratch vector slot -> column 0 of the CB's Kt reserved tiles.
    auto scatter_columns = [&](uint32_t slot, CircularBuffer& c) {
        volatile tt_l1_ptr uint32_t* src =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + slot * Kt * 2 * FACE_ROW_BYTES);
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c.get_write_ptr());
        for (uint32_t j = 0; j < Kt; j++) {
            for (uint32_t r = 0; r < 32; r++) {
                dst[j * (tb / 4) + (r / 16) * 2 * (FACE_BYTES / 4) + (r % 16) * 16] = src[j * 32 + r];
            }
        }
    };

    for (uint32_t it = 0; it < item_count; it++) {
        const uint32_t item = item_start + it;  // b * H + h
        const uint32_t b = item / H;
        const uint32_t h = item % H;

        for (uint32_t t = 0; t < T; t++) {
            const uint32_t row = (b * T + t) * H + h;
            constexpr uint32_t VEC = 2 * FACE_ROW_BYTES;  // one tile row in scratch
            read_row(q_acc, gq, row, Kt, scratch, VEC, FACE_ROW_BYTES);
            read_row(k_acc, gk, row, Kt, scratch + Kt * VEC, VEC, FACE_ROW_BYTES);
            read_row(g_acc, gg, row, Kt, scratch + 2 * Kt * VEC, VEC, FACE_ROW_BYTES);
            cv.reserve_back(Vt);
            read_row(v_acc, gv, row, Vt, cv.get_write_ptr(), tb, FACE_BYTES);

            const Location bl = locate(gb, row);
            cbeta.reserve_back(1);
            noc.async_read(
                b_acc,
                cbeta,
                FACE_ROW_BYTES,
                {.page_id = bl.page, .offset_bytes = face_row_offset(bl.row, bl.col)},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            cq.reserve_back(Kt);
            ck.reserve_back(Kt);
            cg.reserve_back(Kt);
            scatter_columns(0, cq);
            scatter_columns(1, ck);
            scatter_columns(2, cg);
            volatile tt_l1_ptr uint32_t* bp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbeta.get_write_ptr());
            const uint32_t beta_bits = bp[bl.col % 16];
            for (uint32_t c = 0; c < 16; c++) {
                bp[c] = beta_bits;
                bp[FACE_BYTES / 4 + c] = beta_bits;
            }

            cq.push_back(Kt);
            ck.push_back(Kt);
            cv.push_back(Vt);
            cg.push_back(Kt);
            cbeta.push_back(1);

            if (t == 0) {
                cs.reserve_back(KV);
                for (uint32_t i = 0; i < KV; i++) {
                    noc.async_read(s_acc, cs, tb, {.page_id = item * KV + i}, {.offset_bytes = i * tb});
                }
                noc.async_read_barrier();
                cs.push_back(KV);
            }
        }
    }
}
