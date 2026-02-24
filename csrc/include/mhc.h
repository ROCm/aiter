// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

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
} // namespace aiter
