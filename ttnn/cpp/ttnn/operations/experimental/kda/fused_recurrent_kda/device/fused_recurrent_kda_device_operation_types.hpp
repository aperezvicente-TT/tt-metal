// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Types for ttnn.experimental.kda.fused_recurrent_kda: the token-by-token Kimi Delta Attention recurrence
// (`recurrent_kda` in the GLM-5.3-Flash reference, modeling:462-513). One work item per (b, h); each core walks
// its items, and for each item the T tokens sequentially with the fp32 state S [K,V] resident in L1.
//
// Per token t (all fp32):
//   q, k  <- l2norm(q), l2norm(k)            (x / sqrt(sum x^2 + 1e-6), when use_qk_l2norm)
//   q     <- q * scale                        (scale = K^-0.5 by default)
//   S     <- S * exp(g_t)[:, None]            (per-key-channel decay: row r of S scales by exp(g_t[r]))
//   kv    =  sum_r k[r] S[r, :]
//   delta =  (v - kv) * beta_t
//   S     <- S + k[:, None] * delta[None, :]
//   o     =  sum_r q[r] S[r, :]

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct FusedRecurrentKdaParams {
    uint32_t batch;       // B
    uint32_t tokens;      // T (1 = decode, K+1 = speculative verify)
    uint32_t heads;       // H
    uint32_t key_dim;     // K (multiple of 32)
    uint32_t value_dim;   // V (multiple of 32)
    uint32_t scale_bits;  // fp32 bit pattern of the q scale
    bool use_qk_l2norm;
    bool output_state;            // write any state output
    bool output_per_token_state;  // state after every token ([B,T,H,K,V]) instead of the final state ([B,H,K,V])
    bool state_in_place;          // final state written into initial_state's buffer (returned as the state output)
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

// Inputs are fp32 TILE interleaved tensors addressed by their flat row-major element order:
//   q, k, g: (b, t, h, k) order, last dim a multiple of K  (e.g. [B,T,H,K] or [B,T,H*K])
//   v:       (b, t, h, v) order, last dim a multiple of V
//   beta:    (b, t, h) order
//   initial_state: [B, H, K, V]
// g is the log-space decay (<= 0); exp(g) is taken in-kernel.
struct FusedRecurrentKdaInputs {
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor g;
    Tensor beta;
    Tensor initial_state;
};

}  // namespace ttnn::experimental::prim
