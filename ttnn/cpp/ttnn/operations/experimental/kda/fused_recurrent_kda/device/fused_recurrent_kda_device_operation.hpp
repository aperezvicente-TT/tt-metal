// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include "fused_recurrent_kda_device_operation_types.hpp"
#include "fused_recurrent_kda_program_factory.hpp"

namespace ttnn::experimental::prim {

// Returns {o [B,T,H,V]} or {o, state}; state is [B,T,H,K,V] (per token) or [B,H,K,V] (final).
struct FusedRecurrentKdaDeviceOperation {
    using operation_attributes_t = FusedRecurrentKdaParams;
    using tensor_args_t = FusedRecurrentKdaInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<FusedRecurrentKdaProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> fused_recurrent_kda(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& initial_state,
    const FusedRecurrentKdaParams& params);

}  // namespace ttnn::experimental::prim
