// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_tensor.h"
#include <hip/hip_runtime.h>

namespace aiter {
void mhc_pre_gemm_sqrsum(aiter_tensor_t& out,    // (split_k, m, hc_mult3) / (m, hc_mult3)
                         aiter_tensor_t& sqrsum, // (split_k, m) / (m)
                         aiter_tensor_t& x,      // (m, hc_hidden_size)
                         aiter_tensor_t& fn,     // (hc_mult3, hc_hidden_size)
                         int tile_k = 128,
                         int is_fn_pack_bf16 = 0);
void mhc_pre_big_fuse(aiter_tensor_t& post_mix,        // (m, hc_mult)
                      aiter_tensor_t& comb_mix,        // (m, hc_mult * hc_mult)
                      aiter_tensor_t& layer_input,     // (m, hidden_size)
                      aiter_tensor_t& gemm_out_mul,    // (split_k, m, hc_mult3)
                      aiter_tensor_t& gemm_out_sqrsum, // (split_k, m)
                      aiter_tensor_t& hc_scale,        // (3)
                      aiter_tensor_t& hc_base,         // (hc_mult3)
                      aiter_tensor_t& residual,        // (m, hc_mult, hidden_size)
                      float rms_eps            = 1e-6,
                      float hc_pre_eps         = 1e-6,
                      float hc_sinkhorn_eps    = 1e-6,
                      float hc_post_mult_value = 1.0,
                      int sinkhorn_repeat      = 20);
void mhc_pre_big_fuse_rmsnorm(aiter_tensor_t& post_mix,        // (m, hc_mult)
                              aiter_tensor_t& comb_mix,        // (m, hc_mult * hc_mult)
                              aiter_tensor_t& out,             // (m, hidden_size)
                              aiter_tensor_t& gemm_out_mul,    // (split_k, m, hc_mult3)
                              aiter_tensor_t& gemm_out_sqrsum, // (split_k, m)
                              aiter_tensor_t& hc_scale,        // (3)
                              aiter_tensor_t& hc_base,         // (hc_mult3)
                              aiter_tensor_t& residual,        // (m, hc_mult, hidden_size)
                              aiter_tensor_t& norm_weight,     // (hidden_size)
                              float rms_eps            = 1e-6,
                              float hc_pre_eps         = 1e-6,
                              float hc_sinkhorn_eps    = 1e-6,
                              float norm_eps           = 1e-6,
                              float hc_post_mult_value = 1.0,
                              int sinkhorn_repeat      = 20);
void mhc_post(aiter_tensor_t& out,            // (m, hc_mult, hidden_size)
              aiter_tensor_t& x,              // (m, hidden_size)
              aiter_tensor_t& residual,       // (m, hc_mult, hidden_size)
              aiter_tensor_t& post_layer_mix, // (m, hc_mult)
              aiter_tensor_t& comb_res_mix,   // (m, hc_mult, hc_mult)
              int store_nt                    = -1);
// Optimized mhc_post launch on raw device pointers (used by fused AR+MHC split epilogue).
void launch_mhc_post_raw(hipStream_t stream,
                         AiterDtype dtype,
                         void* out,
                         void* x,
                         void* residual,
                         void* post_layer_mix,
                         void* comb_res_mix,
                         int m,
                         int hidden_size,
                         int x_stride,
                         int residual_stride,
                         int store_nt = -1);
void mhc_fused_post_pre_gemm_sqrsum(
    aiter_tensor_t& gemm_out_mul,    // (split_k, m, hc_mult3)
    aiter_tensor_t& gemm_out_sqrsum, // (split_k, m)
    aiter_tensor_t& next_residual,   // (m, hc_mult, hidden_size)
    aiter_tensor_t& layer_input,     // (m, hidden_size)
    aiter_tensor_t& residual_in,     // (m, hc_mult, hidden_size)
    aiter_tensor_t& post_layer_mix,  // (m, hc_mult)
    aiter_tensor_t& comb_res_mix,    // (m, hc_mult, hc_mult)
    aiter_tensor_t& fn,              // (hc_mult3, hc_mult * hidden_size)
    int tile_m                       = 16,
    int tile_n                       = 32,
    int tile_k                       = 32,
    int is_fn_pack_bf16              = 0);
} // namespace aiter
