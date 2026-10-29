// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// One work item per (b, h), B*H items split evenly over the compute grid. The reader places each token's q/k/g
// vectors in column 0 and its v row in row 0 of their tiles (the rest of the tile stays zero) and the item's fp32
// state in L1; the compute kernel runs the whole step in fp32 DST on the SFPU (every CB it reads is unpacked
// straight to DST); the writer stores row 0 of o into the (b, t, h) row of the output and the state tiles.

#include "fused_recurrent_kda_program_factory.hpp"

#include <set>
#include <string>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace cb {
constexpr uint32_t q = tt::CBIndex::c_0;       // [K,1] column tiles of one token
constexpr uint32_t k = tt::CBIndex::c_1;       // [K,1]
constexpr uint32_t v = tt::CBIndex::c_2;       // [1,V] row tiles
constexpr uint32_t g = tt::CBIndex::c_3;       // [K,1] log decay
constexpr uint32_t beta = tt::CBIndex::c_4;    // 1 tile, beta in row 0
constexpr uint32_t s0 = tt::CBIndex::c_5;      // [K,V] initial state of one item
constexpr uint32_t qcol = tt::CBIndex::c_6;    // [K,1] normalised, scaled q in column 0
constexpr uint32_t kcol = tt::CBIndex::c_7;    // [K,1] normalised k in column 0
constexpr uint32_t kfull = tt::CBIndex::c_8;   // [K,32] normalised k broadcast over columns
constexpr uint32_t dcol = tt::CBIndex::c_9;    // [K,1] exp(g) in column 0
constexpr uint32_t snew = tt::CBIndex::c_12;   // 2 x [K,V] updated state, ping-pong (next token's state)
constexpr uint32_t o = tt::CBIndex::c_13;      // [1,V] output row
constexpr uint32_t st = tt::CBIndex::c_14;     // [K,V] state output
constexpr uint32_t scratch = tt::CBIndex::c_16;  // reader-only: q/k/g face rows before the column scatter
}  // namespace cb

namespace {

// Flat-order geometry of a TILE tensor: logical last dim, logical rows of the last two dims, padded tile counts.
void append_geometry(std::vector<uint32_t>& args, const Tensor& t) {
    const auto& s = t.logical_shape();
    const auto& p = t.padded_shape();
    args.push_back(s[-1]);
    args.push_back(s[-2]);
    args.push_back(p[-2] / TILE_HEIGHT);
    args.push_back(p[-1] / TILE_WIDTH);
}

}  // namespace

tt::tt_metal::ProgramDescriptor FusedRecurrentKdaProgramFactory::create_descriptor(
    const FusedRecurrentKdaParams& a, const FusedRecurrentKdaInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t Kt = a.key_dim / TILE_WIDTH;
    const uint32_t Vt = a.value_dim / TILE_WIDTH;
    const uint32_t KV = Kt * Vt;
    const uint32_t items = a.batch * a.heads;
    const uint32_t want_state = a.output_state ? 1u : 0u;
    const uint32_t per_token = (a.output_state && a.output_per_token_state) ? 1u : 0u;

    auto* device = in.q.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t n_cores = std::min(items, static_cast<uint32_t>(grid.x * grid.y));
    const uint32_t per_core = items / n_cores;
    const uint32_t extra = items % n_cores;

    std::vector<CoreCoord> cores_used;
    std::set<CoreRange> ranges;
    for (uint32_t i = 0; i < n_cores; i++) {
        CoreCoord c{i % grid.x, i / grid.x};
        cores_used.push_back(c);
        ranges.insert(CoreRange{c, c});
    }
    CoreRangeSet cores{ranges};

    ProgramDescriptor desc;
    const tt::DataFormat fmt = tt::DataFormat::Float32;
    const uint32_t ts = tt::tile_size(fmt);
    auto add_cb = [&](uint32_t idx, uint32_t n_tiles) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = n_tiles * ts,
            .core_ranges = cores,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = static_cast<uint8_t>(idx), .data_format = fmt, .page_size = ts}}}});
    };
    add_cb(cb::q, 2 * Kt);
    add_cb(cb::k, 2 * Kt);
    add_cb(cb::v, 2 * Vt);
    add_cb(cb::g, 2 * Kt);
    add_cb(cb::beta, 2);
    add_cb(cb::s0, 2 * KV);
    add_cb(cb::qcol, Kt);
    add_cb(cb::kcol, Kt);
    add_cb(cb::kfull, Kt);
    add_cb(cb::dcol, Kt);
    add_cb(cb::snew, 2 * KV);
    add_cb(cb::o, 2 * Vt);
    if (want_state) {
        add_cb(cb::st, 2 * KV);
    }
    add_cb(cb::scratch, 1);

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/kda/fused_recurrent_kda/device/kernels/";

    std::vector<uint32_t> reader_ct = {a.tokens, a.heads, Kt, Vt};
    append_geometry(reader_ct, in.q);
    append_geometry(reader_ct, in.k);
    append_geometry(reader_ct, in.v);
    append_geometry(reader_ct, in.g);
    append_geometry(reader_ct, in.beta);
    TensorAccessorArgs(*in.q.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.v.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.g.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.beta.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.initial_state.buffer()).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = {a.tokens, a.heads, Kt, Vt, want_state, per_token};
    append_geometry(writer_ct, outputs[0]);
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[want_state ? 1 : 0].buffer()).append_to(writer_ct);

    const std::vector<uint32_t> compute_ct = {
        a.tokens, Kt, Vt, a.use_qk_l2norm ? 1u : 0u, want_state, per_token, a.scale_bits};

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_fused_recurrent_kda.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(n_cores);

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_fused_recurrent_kda.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(n_cores);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/fused_recurrent_kda.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = compute_ct;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, unused_dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), a.compute_kernel_config);
    // Every CB the compute kernel reads is consumed by copy_tile only, so all of them unpack straight to fp32 DST.
    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t idx :
         {cb::q, cb::k, cb::v, cb::g, cb::beta, cb::s0, cb::qcol, cb::kcol, cb::kfull, cb::dcol, cb::snew}) {
        unpack_modes[idx] = UnpackToDestMode::UnpackToDestFp32;
    }
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = true,  // the column-block step holds 8 fp32 DST tiles
        .unpack_to_dest_mode = unpack_modes,
        .math_approx_mode = math_approx_mode};
    compute.runtime_args.reserve(n_cores);

    auto* q_buf = in.q.buffer();
    auto* k_buf = in.k.buffer();
    auto* v_buf = in.v.buffer();
    auto* g_buf = in.g.buffer();
    auto* beta_buf = in.beta.buffer();
    auto* s0_buf = in.initial_state.buffer();
    auto* o_buf = outputs[0].buffer();
    auto* st_buf = outputs[want_state ? 1 : 0].buffer();

    uint32_t start = 0;
    for (uint32_t i = 0; i < n_cores; i++) {
        const uint32_t count = per_core + (i < extra ? 1u : 0u);
        const auto& core = cores_used[i];
        reader.emplace_runtime_args(core, {start, count, q_buf, k_buf, v_buf, g_buf, beta_buf, s0_buf});
        writer.emplace_runtime_args(core, {start, count, o_buf, st_buf});
        compute.emplace_runtime_args(core, {count});
        start += count;
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::prim
