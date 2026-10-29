// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "fused_recurrent_kda_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct FusedRecurrentKdaProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const FusedRecurrentKdaParams& attrs, const FusedRecurrentKdaInputs& in, std::vector<Tensor>& outputs);
};

}  // namespace ttnn::experimental::prim
