// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu {

using unified_routed_expert_ffn::RoutedExpertActivation;

struct OperationArguments {
    uint32_t experts_per_chip = 1;
    uint32_t m_tiles = 0;
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    bool read_x_at_offset = false;
    // Active-token band this op owns. An expert whose count falls outside [min, max] is
    // dropped like a zero count -- uniform across the grid, so it costs no CB traffic, no
    // collective round and no semaphore. Wide open by default; a hybrid dispatch narrows it
    // so this op and unified_routed_expert_moe split the experts by load over ONE shared
    // counts vector, with no masked tensors and no host sync.
    uint32_t min_active_tokens = 0;
    uint32_t max_active_tokens = std::numeric_limits<uint32_t>::max();
    RoutedExpertActivation activation = RoutedExpertActivation::Silu;
    // Whether the per-expert gate/up/down biases are fused. Derived from the presence of the bias
    // tensors, and hashed below so a biased and a bias-free dispatch of the same shape cannot share
    // a cached program -- the kernels select the bias adds with a compile-time FUSE_BIAS define.
    bool fuse_bias = false;
    // Gate and up read from ONE [emb, 2 * hidden] tensor per expert whose tile columns interleave
    // [gate_0 | up_0 | gate_1 | up_1 ...] (the routed-expert layout of the GLM-5.3-Flash decode stack:
    // fused_experts' per-core gate|up shards). w_gates[e] and w_ups[e] are that same tensor; the
    // reader takes even tile columns, the writer odd ones. Hashed: a different kernel define.
    bool gate_up_interleaved = false;
    // Upper bound on the op's static CB bytes per core (0 = the whole unreserved worker L1). A caller whose process
    // keeps persistent L1 tensors (a decode stack's intermediates pools) passes the room under them: tt-metal refuses a
    // program whose CB region ends above the lowest L1 buffer of its cores. Hashed: it selects the geometry.
    uint32_t l1_budget_bytes = 0;
    // Gathered x read (x_row_index): dispatch row d of an expert's region reads its activation stick from
    // input row x_row_index[d] instead of row d, so a caller can pass the UNDISPATCHED token rows and skip the
    // dispatch gather. Derived from the presence of x_row_index and hashed, like fuse_bias: the reader selects
    // the gathered address with a compile-time MOE_GATHER_X define and carries one extra scratch CB.
    bool gather_x = false;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::BFLOAT8_B;
    tt::tt_metal::MemoryConfig output_memory_config{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "experts_per_chip",
        "m_tiles",
        "grid_x",
        "grid_y",
        "read_x_at_offset",
        "min_active_tokens",
        "max_active_tokens",
        "activation",
        "fuse_bias",
        "gate_up_interleaved",
        "l1_budget_bytes",
        "gather_x",
        "output_dtype",
        "output_memory_config",
        "compute_kernel_config");

    auto attribute_values() const {
        return std::forward_as_tuple(
            experts_per_chip,
            m_tiles,
            grid_x,
            grid_y,
            read_x_at_offset,
            min_active_tokens,
            max_active_tokens,
            activation,
            fuse_bias,
            gate_up_interleaved,
            l1_budget_bytes,
            gather_x,
            output_dtype,
            output_memory_config,
            compute_kernel_config);
    }
};

struct TensorArguments {
    Tensor activations;
    // One weight tensor per local expert. Expert 0 is the layout representative:
    // the program is built once and the kernels reuse a single accessor layout
    // descriptor per role, varying only the per-expert base address.
    std::vector<Tensor> w_gates;
    std::vector<Tensor> w_ups;
    std::vector<Tensor> w_downs;
    // Optional per-local-expert projection biases (gpt-oss). Either all three lists are populated
    // -- one bias per local expert, same length and order as the weight lists -- or all three are
    // empty: a partially biased dispatch would bias some projections and not others, which is wrong
    // numbers with no error. gate/up bias is (1, hidden), down is (1, emb).
    std::vector<Tensor> gate_biases;
    std::vector<Tensor> up_biases;
    std::vector<Tensor> down_biases;
    Tensor counts;
    Tensor global_expert_idx_table;
    std::optional<Tensor> optional_output;
    std::optional<Tensor> expert_region_offsets;
    // Optional UINT32 ROW_MAJOR [.., N] dispatch-row -> input-row table (one DRAM page): with read_x_at_offset
    // and bf16 ROW_MAJOR activations, dispatch row d of an expert's region reads activations row
    // x_row_index[d]. N (not the activation rows) is then the region space that expert_region_offsets index
    // and the shared output mirrors. Last member, so its buffer enumerates after every existing one.
    std::optional<Tensor> x_row_index;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu
