// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>
#include "custom_all_reduce.h"

namespace aiter {
void mhc_pre_gemm_sqrsum(torch::Tensor& out,    // (split_k, m, hc_mult3) / (m, hc_mult3)
                         torch::Tensor& sqrsum, // (split_k, m) / (m)
                         torch::Tensor& x,      // (m, hc_hidden_size)
                         torch::Tensor& fn,     // (hc_mult3, hc_hidden_size)
                         int tile_k = 128);
void mhc_pre_big_fuse(torch::Tensor& post_mix,        // (m, hc_mult)
                      torch::Tensor& comb_mix,        // (m, hc_mult * hc_mult)
                      torch::Tensor& layer_input,     // (m, hidden_size)
                      torch::Tensor& gemm_out_mul,    // (split_k, m, hc_mult3)
                      torch::Tensor& gemm_out_sqrsum, // (split_k, m)
                      torch::Tensor& hc_scale,        // (3)
                      torch::Tensor& hc_base,         // (hc_mult3)
                      torch::Tensor& residual,        // (m, hc_mult, hidden_size)
                      float rms_eps            = 1e-6,
                      float hc_pre_eps         = 1e-6,
                      float hc_sinkhorn_eps    = 1e-6,
                      float hc_post_mult_value = 1.0,
                      int sinkhorn_repeat      = 20);
void mhc_pre_big_fuse_rmsnorm(torch::Tensor& post_mix,        // (m, hc_mult)
                              torch::Tensor& comb_mix,        // (m, hc_mult * hc_mult)
                              torch::Tensor& out,             // (m, hidden_size)
                              torch::Tensor& gemm_out_mul,    // (split_k, m, hc_mult3)
                              torch::Tensor& gemm_out_sqrsum, // (split_k, m)
                              torch::Tensor& hc_scale,        // (3)
                              torch::Tensor& hc_base,         // (hc_mult3)
                              torch::Tensor& residual,        // (m, hc_mult, hidden_size)
                              torch::Tensor& norm_weight,     // (hidden_size)
                              float rms_eps            = 1e-6,
                              float hc_pre_eps         = 1e-6,
                              float hc_sinkhorn_eps    = 1e-6,
                              float norm_eps           = 1e-6,
                              float hc_post_mult_value = 1.0,
                              int sinkhorn_repeat      = 20);
void mhc_post(torch::Tensor& out,            // (m, hc_mult, hidden_size)
              torch::Tensor& x,              // (m, hidden_size)
              torch::Tensor& residual,       // (m, hc_mult, hidden_size)
              torch::Tensor& post_layer_mix, // (m, hc_mult)
              torch::Tensor& comb_res_mix,   // (m, hc_mult, hc_mult)
              int store_nt                   = -1);
void launch_fused_ar_mhc_gemm_sqrsum_unified(
    fptr_t _fa,
    torch::Tensor& inp,
    torch::Tensor& gemm_out_mul,
    torch::Tensor& gemm_out_sqrsum,
    torch::Tensor& next_residual,
    torch::Tensor& residual_in,
    torch::Tensor& post_layer_mix,
    torch::Tensor& comb_res_mix,
    torch::Tensor& fn,
    torch::Tensor& post_mix,
    torch::Tensor& comb_mix,
    torch::Tensor& layer_input_out,
    torch::Tensor& hc_scale,
    torch::Tensor& hc_base,
    torch::Tensor& norm_weight,
    float rms_eps,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float norm_eps,
    float hc_post_mult_value,
    int sinkhorn_repeat,
    int tile_m,
    int tile_n,
    int tile_k,
    int64_t reg_ptr = 0);
void mhc_fused_post_pre_gemm_sqrsum(
    torch::Tensor& gemm_out_mul,    // (split_k * hc_mult, m, hc_mult3)
    torch::Tensor& gemm_out_sqrsum, // (split_k * hc_mult, m)
    torch::Tensor& next_residual,   // (m, hc_mult, hidden_size)
    torch::Tensor& layer_input,     // (m, hidden_size)
    torch::Tensor& residual_in,     // (m, hc_mult, hidden_size)
    torch::Tensor& post_layer_mix,  // (m, hc_mult)
    torch::Tensor& comb_res_mix,    // (m, hc_mult, hc_mult)
    torch::Tensor& fn,              // (hc_mult3, hc_mult * hidden_size)
    int tile_m                       = 16,
    int tile_n                       = 32,
    int tile_k                       = 32);
} // namespace aiter
