// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mix_streams_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <algorithm>

namespace ttnn::operations::experimental::deepseek::mix_streams {

namespace {

constexpr uint32_t kCbCombSrc = tt::CBIndex::c_0;
constexpr uint32_t kCbComb = tt::CBIndex::c_1;
constexpr uint32_t kCbPostSrc = tt::CBIndex::c_2;
constexpr uint32_t kCbPost = tt::CBIndex::c_3;
constexpr uint32_t kCbStreams = tt::CBIndex::c_4;
constexpr uint32_t kCbSub = tt::CBIndex::c_5;
constexpr uint32_t kCbOut = tt::CBIndex::c_6;
// fp32 path: the (hi, mid, lo) bf16 parts the compute kernel splits each fp32 operand tile into.
constexpr uint32_t kCbCombParts = tt::CBIndex::c_7;
constexpr uint32_t kCbPostParts = tt::CBIndex::c_8;
constexpr uint32_t kCbStreamsParts = tt::CBIndex::c_9;
constexpr uint32_t kCbSubParts = tt::CBIndex::c_10;
constexpr uint32_t kNumParts = 3;

// The per-tile work (two single-tile matmuls) is dwarfed by the per-core setup -- rebuilding
// the comb / post tiles and paying the DRAM round trip for them -- so spreading the tiny
// decode shapes over the whole grid costs more than it saves. Measured on a [1,1,4,4096]
// decode step: 18.5 us at 4 cores, 8.1 at 16, 7.3 at 32, 8.1 at 64.
constexpr uint32_t kMaxCores = 32;
// The fp32 kernel does far more work per tile (operand splits, 6 * hc + 3 HiFi4 matmuls), so it spreads wider: 64
// cores, or the whole grid once that leaves >= 8 tiles per core (measured on [B,1,4,4096], bf16 sublayer_out: B=1
// 24 us on 64 cores vs 31 us on 110; B=32 201 us vs 128 us).
constexpr uint32_t kMaxCoresFp32 = 64;
constexpr uint32_t kFp32FullGridTilesPerCore = 8;

constexpr char kReaderKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/mix_streams/device/kernels/dataflow/reader_mix_streams.cpp";
constexpr char kComputeKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/mix_streams/device/kernels/compute/mix_streams.cpp";
constexpr char kFp32ComputeKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/mix_streams/device/kernels/compute/mix_streams_fp32.cpp";
constexpr char kWriterKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/mix_streams/device/kernels/dataflow/writer_mix_streams.cpp";

}  // namespace

MixStreamsProgramFactory::cached_program_t MixStreamsProgramFactory::create(
    const MixStreamsParams& operation_attributes,
    const MixStreamsInputs& tensor_args,
    MixStreamsTensorReturn& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& post = tensor_args.post;
    const auto& comb = tensor_args.comb;
    const auto& sublayer_out = tensor_args.sublayer_out;
    const auto& streams = tensor_args.streams;
    auto& output = tensor_return_value;

    Program program = CreateProgram();

    const DataFormat tile_data_format = datatype_to_dataformat_converter(streams.dtype());
    const bool fp32 = streams.dtype() == DataType::FLOAT32;
    const DataFormat sub_data_format = datatype_to_dataformat_converter(sublayer_out.dtype());
    const bool sub_fp32 = sublayer_out.dtype() == DataType::FLOAT32;

    const auto& streams_shape = streams.logical_shape();
    const uint32_t hc = operation_attributes.num_streams;
    const uint32_t num_tokens = static_cast<uint32_t>(streams_shape[0]) * static_cast<uint32_t>(streams_shape[1]);
    const uint32_t n_tiles = streams.padded_shape()[-1] / constants::TILE_WIDTH;
    // hc <= TILE_HEIGHT, so every token is exactly one tile row of the [T*hc, D] tile grid.
    const uint32_t total_tiles = num_tokens * n_tiles;

    IDevice* device = streams.device();
    const CoreCoord grid_size = device->compute_with_storage_grid_size();
    const uint32_t grid_cores = static_cast<uint32_t>(grid_size.x * grid_size.y);
    const uint32_t core_budget = !fp32                                                      ? kMaxCores
                                 : total_tiles >= kFp32FullGridTilesPerCore * kMaxCoresFp32 ? grid_cores
                                                                                            : kMaxCoresFp32;
    const uint32_t max_cores = std::min<uint32_t>(total_tiles, std::min<uint32_t>(core_budget, grid_cores));
    const CoreRangeSet work_grid = num_cores_to_corerangeset(max_cores, grid_size, /*row_wise=*/true);
    const auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_1, tiles_per_core_2] =
        split_work_to_cores(work_grid, total_tiles, /*row_wise=*/true);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    constexpr uint32_t tile_buffering = 2;
    auto make_cb = [&](uint32_t index, uint32_t num_pages, DataFormat format) {
        const uint32_t page_bytes = tile_size(format);
        CircularBufferConfig config =
            CircularBufferConfig(num_pages * page_bytes, {{index, format}}).set_page_size(index, page_bytes);
        CreateCircularBuffer(program, all_cores, config);
    };
    make_cb(kCbCombSrc, 1, tile_data_format);
    // fp32: the reader emits hc column-isolated comb^T tiles per token (see reader_mix_streams.cpp).
    const uint32_t comb_cb_pages = fp32 ? hc : tile_buffering;
    make_cb(kCbComb, comb_cb_pages, tile_data_format);
    make_cb(kCbPostSrc, 1, tile_data_format);
    make_cb(kCbPost, tile_buffering, tile_data_format);
    make_cb(kCbStreams, tile_buffering, tile_data_format);
    make_cb(kCbSub, tile_buffering, sub_data_format);
    make_cb(kCbOut, tile_buffering, tile_data_format);
    if (fp32) {
        make_cb(kCbCombParts, kNumParts * hc, DataFormat::Float16_b);
        make_cb(kCbPostParts, kNumParts, DataFormat::Float16_b);
        make_cb(kCbStreamsParts, kNumParts, DataFormat::Float16_b);
        if (sub_fp32) {
            make_cb(kCbSubParts, kNumParts, DataFormat::Float16_b);
        }
    }

    std::vector<uint32_t> reader_compile_time_args = {
        kCbCombSrc,
        kCbComb,
        kCbPostSrc,
        kCbPost,
        kCbStreams,
        kCbSub,
        hc,
        n_tiles,
        tile_buffering,
        fp32 ? 4u : 2u,
        comb_cb_pages,
    };
    TensorAccessorArgs(post.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(comb.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(sublayer_out.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(streams.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {kCbOut};
    TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        kCbComb,
        kCbPost,
        kCbStreams,
        kCbSub,
        kCbOut,
        n_tiles,
    };
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32) {
        compute_compile_time_args.insert(
            compute_compile_time_args.end(),
            {kCbCombParts, kCbPostParts, kCbStreamsParts, kCbSubParts, sub_fp32 ? 1u : 0u, hc});
        // fp32 operand tiles are unpacked straight to fp32 DST for the SFPU split.
        for (const uint32_t cb : {kCbComb, kCbPost, kCbStreams}) {
            unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
        }
        if (sub_fp32) {
            unpack_to_dest_mode[kCbSub] = UnpackToDestMode::UnpackToDestFp32;
        }
        // Exact products of the bf16 parts need HiFi4 (HiFi3 / HiFi2 measured 3.6e3 / 1.2e5 ulp off), and the
        // accumulation must be fp32.
        math_fidelity = MathFidelity::HiFi4;
        fp32_dest_acc_en = true;
    }

    const KernelHandle reader_kernel_id =
        CreateKernel(program, kReaderKernelPath, all_cores, ReaderDataMovementConfig(reader_compile_time_args));
    const KernelHandle writer_kernel_id =
        CreateKernel(program, kWriterKernelPath, all_cores, WriterDataMovementConfig(writer_compile_time_args));
    const KernelHandle compute_kernel_id = CreateKernel(
        program,
        fp32 ? kFp32ComputeKernelPath : kComputeKernelPath,
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args});

    const std::vector<CoreCoord> cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
    uint32_t start_tile = 0;
    for (const auto& core : cores) {
        uint32_t tiles_this_core = 0;
        if (core_group_1.contains(core)) {
            tiles_this_core = tiles_per_core_1;
        } else if (core_group_2.contains(core)) {
            tiles_this_core = tiles_per_core_2;
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {post.buffer()->address(),
             comb.buffer()->address(),
             sublayer_out.buffer()->address(),
             streams.buffer()->address(),
             start_tile,
             tiles_this_core});
        SetRuntimeArgs(program, compute_kernel_id, core, {tiles_this_core, start_tile});
        SetRuntimeArgs(program, writer_kernel_id, core, {output.buffer()->address(), start_tile, tiles_this_core});

        start_tile += tiles_this_core;
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id, .writer_kernel_id = writer_kernel_id, .cores = cores}};
}

void MixStreamsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const MixStreamsParams& /*operation_attributes*/,
    const MixStreamsInputs& tensor_args,
    MixStreamsTensorReturn& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const uint32_t post_addr = tensor_args.post.buffer()->address();
    const uint32_t comb_addr = tensor_args.comb.buffer()->address();
    const uint32_t sub_addr = tensor_args.sublayer_out.buffer()->address();
    const uint32_t streams_addr = tensor_args.streams.buffer()->address();
    const uint32_t out_addr = tensor_return_value.buffer()->address();

    auto& reader_args_by_core = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& reader_args = reader_args_by_core[core.x][core.y];
        reader_args[0] = post_addr;
        reader_args[1] = comb_addr;
        reader_args[2] = sub_addr;
        reader_args[3] = streams_addr;

        writer_args_by_core[core.x][core.y][0] = out_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek::mix_streams
