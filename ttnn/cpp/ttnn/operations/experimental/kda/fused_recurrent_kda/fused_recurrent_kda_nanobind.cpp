// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fused_recurrent_kda_nanobind.hpp"
#include "fused_recurrent_kda.hpp"

#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

namespace ttnn::operations::experimental::kda::fused_recurrent_kda::detail {

void bind_fused_recurrent_kda(nb::module_& mod) {
    ttnn::bind_function<"fused_recurrent_kda", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Token-by-token Kimi Delta Attention recurrence (GLM-5.3-Flash ``recurrent_kda``) for decode (T = 1) and
        speculative verify (T = K + 1). One work item per (b, h); the fp32 state stays in L1 for the item's tokens
        and the whole step runs in fp32 on the SFPU. Per token:

            q, k  = q / sqrt(sum q^2 + 1e-6), k / sqrt(sum k^2 + 1e-6)   (use_qk_l2norm)
            q     = q * scale
            S     = S * exp(g)[:, None]
            delta = (v - k^T S) * beta
            S     = S + k delta^T
            o     = q^T S

        Args:
            q (ttnn.Tensor): ``[B, T, H, K]`` or ``[B, T, H*K]``.
            k (ttnn.Tensor): ``[B, T, H, K]`` or ``[B, T, H*K]``.
            v (ttnn.Tensor): ``[B, T, H, V]`` or ``[B, T, H*V]``.
            g (ttnn.Tensor): ``[B, T, H, K]`` or ``[B, T, H*K]`` per-key-channel log decay (<= 0); exp(g) is
                applied in-kernel.
            beta (ttnn.Tensor): ``[B, T, H]`` write strength (already sigmoid'd). Decode layout: q/k/v/g
                ``[1, 1, B, H*D]`` with beta ``[1, 1, B, H]`` = B users, one token each.

        Keyword Args:
            scale (float, optional): query scale, defaults to ``K**-0.5``.
            initial_state (ttnn.Tensor, optional): ``[B, H, K, V]`` fp32; zeros when absent.
            output_final_state (bool): return the state after the last token, ``[B, H, K, V]``.
            output_per_token_state (bool): return the state after every token, ``[B, T, H, K, V]``.
            use_qk_l2norm (bool): L2-normalise q and k in-kernel. Defaults to True.
            memory_config (ttnn.MemoryConfig, optional): interleaved output memory config, DRAM by default.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): needs fp32_dest_acc_en.
            inplace_state (bool): write the final state into ``initial_state`` (fp32 TILE) and return it as the
                state output (no new state buffer, no copy-back). Needs output_final_state.

        Returns:
            tuple[ttnn.Tensor, Optional[ttnn.Tensor]]: o ``[B, T, H, V]`` fp32 and the requested state (or None).

        Note:
            Inputs are read in their flat (b, t, h, d) order, so any of the listed shapes is a view of the same data.
            K and V must be multiples of 32, at most 128.
        )doc",
        &ttnn::experimental::kda::fused_recurrent_kda,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("beta").noconvert(),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("output_final_state") = false,
        nb::arg("output_per_token_state") = false,
        nb::arg("use_qk_l2norm") = true,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("inplace_state") = false);
}

}  // namespace ttnn::operations::experimental::kda::fused_recurrent_kda::detail
