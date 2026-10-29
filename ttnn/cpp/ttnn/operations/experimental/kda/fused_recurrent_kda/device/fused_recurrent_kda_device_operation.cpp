// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fused_recurrent_kda_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {

constexpr const char* kOp = "fused_recurrent_kda";

void check_tensor(const Tensor& t, const char* name, const Tensor& ref) {
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr,
        "{}: {} must be allocated on device",
        kOp,
        name);
    TT_FATAL(t.device() == ref.device(), "{}: {} must be on the same device as q", kOp, name);
    TT_FATAL(!t.is_sharded(), "{}: {} must be interleaved, not sharded", kOp, name);
    TT_FATAL(t.layout() == Layout::TILE, "{}: {} must be TILE layout", kOp, name);
    TT_FATAL(t.dtype() == DataType::FLOAT32, "{}: {} must be fp32, got {}", kOp, name, t.dtype());
}

// The reader addresses one (b, t, h) row of `d` contiguous values; they must sit in one tensor row.
void check_rows(const Tensor& t, const char* name, uint64_t volume, uint32_t d) {
    const auto& s = t.logical_shape();
    TT_FATAL(s.rank() >= 2, "{}: {} must have rank >= 2, got {}", kOp, name, s.rank());
    TT_FATAL(
        static_cast<uint64_t>(s.volume()) == volume,
        "{}: {} must hold B*T*H*{} = {} elements, got shape {}",
        kOp,
        name,
        d,
        volume,
        s);
    TT_FATAL(s[-1] % d == 0, "{}: {} last dim ({}) must be a multiple of {}", kOp, name, s[-1], d);
}

}  // namespace

FusedRecurrentKdaDeviceOperation::program_factory_t FusedRecurrentKdaDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return FusedRecurrentKdaProgramFactory{};
}

void FusedRecurrentKdaDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& in) {
    using tt::constants::TILE_WIDTH;
    TT_FATAL(a.batch >= 1 && a.tokens >= 1 && a.heads >= 1, "{}: B, T, H must be >= 1", kOp);
    TT_FATAL(
        a.key_dim % TILE_WIDTH == 0 && a.key_dim > 0 && a.key_dim <= 4 * TILE_WIDTH,
        "{}: K must be a multiple of {} in [32, 128], got {}",
        kOp,
        TILE_WIDTH,
        a.key_dim);
    TT_FATAL(
        a.value_dim % TILE_WIDTH == 0 && a.value_dim > 0 && a.value_dim <= 4 * TILE_WIDTH,
        "{}: V must be a multiple of {} in [32, 128], got {}",
        kOp,
        TILE_WIDTH,
        a.value_dim);
    const bool fp32_dest_acc_en =
        std::get<2>(get_compute_kernel_config_args(in.q.device()->arch(), a.compute_kernel_config));
    TT_FATAL(fp32_dest_acc_en, "{}: requires fp32_dest_acc_en (the step runs in fp32 DST)", kOp);

    check_tensor(in.q, "q", in.q);
    check_tensor(in.k, "k", in.q);
    check_tensor(in.v, "v", in.q);
    check_tensor(in.g, "g", in.q);
    check_tensor(in.beta, "beta", in.q);
    check_tensor(in.initial_state, "initial_state", in.q);

    const uint64_t rows = static_cast<uint64_t>(a.batch) * a.tokens * a.heads;
    check_rows(in.q, "q", rows * a.key_dim, a.key_dim);
    check_rows(in.k, "k", rows * a.key_dim, a.key_dim);
    check_rows(in.g, "g", rows * a.key_dim, a.key_dim);
    check_rows(in.v, "v", rows * a.value_dim, a.value_dim);
    check_rows(in.beta, "beta", rows, 1);

    const auto& s = in.initial_state.logical_shape();
    TT_FATAL(
        s.rank() == 4 && s[0] == a.batch && s[1] == a.heads && s[2] == a.key_dim && s[3] == a.value_dim,
        "{}: initial_state must be [B, H, K, V] = [{}, {}, {}, {}], got {}",
        kOp,
        a.batch,
        a.heads,
        a.key_dim,
        a.value_dim,
        s);
}

FusedRecurrentKdaDeviceOperation::spec_return_value_t FusedRecurrentKdaDeviceOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    const auto layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), a.output_mem_config);
    spec_return_value_t specs;
    specs.emplace_back(ttnn::Shape({a.batch, a.tokens, a.heads, a.value_dim}), layout);
    if (a.output_state) {
        specs.emplace_back(
            a.output_per_token_state ? ttnn::Shape({a.batch, a.tokens, a.heads, a.key_dim, a.value_dim})
                                     : ttnn::Shape({a.batch, a.heads, a.key_dim, a.value_dim}),
            layout);
    }
    return specs;
}

FusedRecurrentKdaDeviceOperation::tensor_return_value_t FusedRecurrentKdaDeviceOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    tensor_return_value_t outs;
    const auto specs = compute_output_specs(a, in);
    outs.push_back(create_device_tensor(specs[0], in.q.device()));
    if (specs.size() > 1) {
        // in place: the final state goes back into the caller's initial_state buffer (same spec, validated)
        outs.push_back(a.state_in_place ? in.initial_state : create_device_tensor(specs[1], in.q.device()));
    }
    return outs;
}

std::vector<Tensor> fused_recurrent_kda(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& initial_state,
    const FusedRecurrentKdaParams& params) {
    using OperationType = FusedRecurrentKdaDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params,
        OperationType::tensor_args_t{.q = q, .k = k, .v = v, .g = g, .beta = beta, .initial_state = initial_state});
}

}  // namespace ttnn::experimental::prim
