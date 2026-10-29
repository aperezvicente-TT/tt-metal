// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// AD-1A, phase D2: the member data-movement parts of the MLA input norms (dsa_norm_c.cpp computes).
//   D2_READ  (member DM0 = the RISC that ran P3's PfStream writer, whose last act is a write barrier on qk / kg: the
//            stores are in DRAM in program order -- no hand-off): the member's tile row of qk (QT + KT bf16 tiles),
//            of kg (LT + GT fp32 tiles), then the 72 weight tiles in the compute's consumption order (a ring).
//   D2_WRITE (member DM1): q_r / latent tiles (bf16, [C, 1536] / [C, 512] tile-interleaved) of the row, then the kg
//            strips as ROW_MAJOR rows of kg_rm [C, 256] fp32 (page = one row; strip j = columns 32 j .. 32 j + 31), and
//            a write barrier: the stores are in DRAM when the phase ends (the op remainder reads them after the
//            launch).
// CT at PH_CT0: [cb_x, cb_kg, cb_w, cb_inv, cb_out, cb_rm, qt, kt, lt, gt, inv_q, inv_k, inv_l, eps, eps_ln] +
// TensorAccessorArgs(qk, kg, w, q_r, latent, kg_rm). RT at ph_rt0: [run (0 = skip), qk, kg, w, q_r, latent, kg_rm,
// row].
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace d2d {
constexpr uint32_t CB_X = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CB_KG = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_W = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(PH_CT0 + 4);
constexpr uint32_t CB_RM = get_compile_time_arg_val(PH_CT0 + 5);
constexpr uint32_t QT = get_compile_time_arg_val(PH_CT0 + 6);
constexpr uint32_t KT = get_compile_time_arg_val(PH_CT0 + 7);
constexpr uint32_t LT = get_compile_time_arg_val(PH_CT0 + 8);
constexpr uint32_t GT = get_compile_time_arg_val(PH_CT0 + 9);
constexpr uint32_t NW = QT + KT + 2 * LT;  // the weight tiles: qa_norm, kva_norm, then (k_w, k_b) per k tile
constexpr auto a_qk = TensorAccessorArgs<PH_CT0 + 15>();
constexpr auto a_kg = TensorAccessorArgs<a_qk.next_compile_time_args_offset()>();
constexpr auto a_w = TensorAccessorArgs<a_kg.next_compile_time_args_offset()>();
constexpr auto a_qr = TensorAccessorArgs<a_w.next_compile_time_args_offset()>();
constexpr auto a_lat = TensorAccessorArgs<a_qr.next_compile_time_args_offset()>();
constexpr auto a_rm = TensorAccessorArgs<a_lat.next_compile_time_args_offset()>();
}  // namespace d2d

void kernel_main() {
    using namespace d2d;
    if (get_arg_val<uint32_t>(ph_rt0) == 0) {
        return;
    }
    const uint32_t row = get_arg_val<uint32_t>(ph_rt0 + 7);  // the member's tile row (its 32 rows of the chunk)
#if defined(D2_READ)
    const uint32_t tb_x = get_tile_size(CB_X), tb_kg = get_tile_size(CB_KG);
    const auto qk = TensorAccessor(a_qk, get_arg_val<uint32_t>(ph_rt0 + 1));  // page = the tensor's own (tile)
    const auto kg = TensorAccessor(a_kg, get_arg_val<uint32_t>(ph_rt0 + 2));
    const auto w = TensorAccessor(a_w, get_arg_val<uint32_t>(ph_rt0 + 3));
    cb_reserve_back(CB_X, QT + KT);
    uint32_t wp = get_write_ptr(CB_X);
    for (uint32_t j = 0; j < QT + KT; ++j) {
        noc_async_read_page(row * (QT + KT) + j, qk, wp + j * tb_x);
    }
    noc_async_read_barrier();
    cb_push_back(CB_X, QT + KT);
    cb_reserve_back(CB_KG, LT + GT);
    wp = get_write_ptr(CB_KG);
    for (uint32_t j = 0; j < LT + GT; ++j) {
        noc_async_read_page(row * (LT + GT) + j, kg, wp + j * tb_kg);
    }
    noc_async_read_barrier();
    cb_push_back(CB_KG, LT + GT);
    for (uint32_t j = 0; j < NW; ++j) {
        cb_reserve_back(CB_W, 1);
        noc_async_read_page(j, w, get_write_ptr(CB_W));
        noc_async_read_barrier();
        cb_push_back(CB_W, 1);
    }
#elif defined(D2_WRITE)
    const auto qr = TensorAccessor(a_qr, get_arg_val<uint32_t>(ph_rt0 + 4));
    const auto lat = TensorAccessor(a_lat, get_arg_val<uint32_t>(ph_rt0 + 5));
    const auto rm = TensorAccessor(a_rm, get_arg_val<uint32_t>(ph_rt0 + 6));
    for (uint32_t j = 0; j < QT; ++j) {
        cb_wait_front(CB_OUT, 1);
        noc_async_write_page(row * QT + j, qr, get_read_ptr(CB_OUT));
        noc_async_writes_flushed();
        cb_pop_front(CB_OUT, 1);
    }
    for (uint32_t j = 0; j < KT; ++j) {
        cb_wait_front(CB_OUT, 1);
        noc_async_write_page(row * KT + j, lat, get_read_ptr(CB_OUT));
        noc_async_writes_flushed();
        cb_pop_front(CB_OUT, 1);
    }
    constexpr uint32_t SEG = 32 * 4;  // one row of a tile-width strip, fp32
    for (uint32_t j = 0; j < LT + GT; ++j) {
        cb_wait_front(CB_RM, 1);
        const uint32_t rp = get_read_ptr(CB_RM);
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_write(rp + r * SEG, rm.get_noc_addr(row * 32 + r, j * SEG), SEG);
        }
        noc_async_writes_flushed();
        cb_pop_front(CB_RM, 1);
    }
    noc_async_write_barrier();
#else
#error "dsa_norm_dm.cpp: define D2_READ or D2_WRITE"
#endif
}
