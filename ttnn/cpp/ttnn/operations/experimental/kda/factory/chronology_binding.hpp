// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <optional>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
namespace ttnn::experimental::prim::kda_factory_detail {
// Rank belongs to mesh placement, not to caller-owned tensor data. The mesh
// adapter retains each coordinate's scalar specialization on program-cache hits.
// sequence_parallel_axis == KDA_NO_SEQUENCE_PARALLEL: the sequence is NOT split over the mesh (every chip holds the
// whole chunk, e.g. tensor-parallel heads on a 2-D mesh): every device is rank 0 of 1, whatever its mesh coordinate.
inline constexpr uint32_t KDA_NO_SEQUENCE_PARALLEL = 0xFFFFu;

inline bool kda_sequence_axis_ok(uint32_t sequence_parallel_axis, size_t mesh_dims) {
    return sequence_parallel_axis == KDA_NO_SEQUENCE_PARALLEL || sequence_parallel_axis < mesh_dims;
}

inline ttnn::device_operation::MeshWorkloadArtifacts chronology_workload(
    const ttnn::device_operation::ProgramArtifacts& artifacts,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ttnn::MeshDevice& device,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    const tt::tt_metal::experimental::KernelSpecName& chronology_reader) {
    using namespace tt::tt_metal::experimental;
    TT_FATAL(
        kda_sequence_axis_ok(sequence_parallel_axis, device.shape().dims()), "KDA: invalid sequence_parallel_axis");
    const bool split = sequence_parallel_axis != KDA_NO_SEQUENCE_PARALLEL;
    ttnn::device_operation::MeshWorkloadArtifacts workload;
    for (const auto& coordinate : tensor_coords.coords()) {
        auto spec = artifacts.spec;
        auto reader = std::find_if(spec.kernels.begin(), spec.kernels.end(), [&](const auto& kernel) {
            return kernel.unique_id == chronology_reader;
        });
        TT_FATAL(reader != spec.kernels.end(), "KDA: chronology reader is absent from the program");
        reader->compile_time_args.insert({"sp_rank", split ? coordinate[sequence_parallel_axis] : 0u});
        reader->compile_time_args.insert({"sp_size", split ? device.shape()[sequence_parallel_axis] : 1u});
        reader->compile_time_args.insert({"local_rows", local_rows});
        workload.programs.push_back({
            .range = ttnn::MeshCoordinateRange(coordinate),
            .spec = std::move(spec),
            .run_params = artifacts.run_params,
        });
    }
    return workload;
}

inline void bind_chronology_channel(
    tt::tt_metal::experimental::ProgramSpec& spec,
    tt::tt_metal::experimental::KernelSpec& reader,
    tt::tt_metal::experimental::KernelSpec& compute) {
    using namespace tt::tt_metal::experimental;
    const DFBSpecName channel{"chronology_compute"};
    spec.dataflow_buffers.push_back(
        {.unique_id = channel, .entry_size = 32, .num_entries = 1, .data_format_metadata = tt::DataFormat::UInt32});
    reader.dfb_bindings.push_back(ProducerOf(channel, "chronology_compute"));
    compute.dfb_bindings.push_back(ConsumerOf(channel, "chronology_compute"));
}
inline void bind_chronology(
    tt::tt_metal::experimental::ProgramSpec& spec,
    tt::tt_metal::experimental::ProgramRunArgs& run,
    const Tensor& actual_start,
    tt::tt_metal::experimental::KernelSpec& reader,
    tt::tt_metal::experimental::KernelSpec& compute) {
    using namespace tt::tt_metal::experimental;
    const TensorParamName name{"actual_start"};
    const auto& tensor = actual_start.mesh_tensor();
    spec.tensor_parameters.push_back({.unique_id = name, .spec = tensor.tensor_spec()});
    run.tensor_args.emplace(name, tensor);
    reader.tensor_bindings.push_back({name, "actual_start"});
    bind_chronology_channel(spec, reader, compute);
}
// Leave actual_start unbound when absent; kernels resolve it with get_token_if_present.
inline void bind_chronology(
    tt::tt_metal::experimental::ProgramSpec& spec,
    tt::tt_metal::experimental::ProgramRunArgs& run,
    const std::optional<Tensor>& actual_start,
    tt::tt_metal::experimental::KernelSpec& reader,
    tt::tt_metal::experimental::KernelSpec& compute) {
    if (actual_start) {
        bind_chronology(spec, run, *actual_start, reader, compute);
    } else {
        bind_chronology_channel(spec, reader, compute);
    }
}
inline void bind_actual_end(
    tt::tt_metal::experimental::ProgramSpec& spec,
    tt::tt_metal::experimental::ProgramRunArgs& run,
    const std::optional<Tensor>& actual_end,
    tt::tt_metal::experimental::KernelSpec& reader) {
    using namespace tt::tt_metal::experimental;
    reader.compile_time_args.insert({"has_actual_end", uint32_t(actual_end.has_value())});
    // Leave the parameter unbound when absent; kernels resolve it with get_token_if_present.
    if (!actual_end) {
        return;
    }
    const TensorParamName name{"actual_end"};
    const auto& tensor = actual_end->mesh_tensor();
    spec.tensor_parameters.push_back({.unique_id = name, .spec = tensor.tensor_spec()});
    run.tensor_args.emplace(name, tensor);
    reader.tensor_bindings.push_back({name, "actual_end"});
}
}  // namespace ttnn::experimental::prim::kda_factory_detail
