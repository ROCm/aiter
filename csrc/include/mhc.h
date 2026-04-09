// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_tensor.h"

namespace aiter {
void mhc_pre_gemm_sqrsum(const aiter_tensor_t& out,    // (split_k, m, hc_mult3) / (m, hc_mult3)
                         const aiter_tensor_t& sqrsum, // (split_k, m) / (m)
                         const aiter_tensor_t& x,      // (m, hc_hidden_size)
                         const aiter_tensor_t& fn,     // (hc_mult3, hc_hidden_size)
                         int tile_k = 128);
void mhc_pre_big_fuse(const aiter_tensor_t& post_mix,        // (m, hc_mult)
                      const aiter_tensor_t& comb_mix,        // (m, hc_mult * hc_mult)
                      const aiter_tensor_t& layer_input,     // (m, hidden_size)
                      const aiter_tensor_t& gemm_out_mul,    // (split_k, m, hc_mult3)
                      const aiter_tensor_t& gemm_out_sqrsum, // (split_k, m)
                      const aiter_tensor_t& hc_scale,        // (3)
                      const aiter_tensor_t& hc_base,         // (hc_mult3)
                      const aiter_tensor_t& residual,        // (m, hc_mult, hidden_size)
                      float rms_eps            = 1e-6,
                      float hc_pre_eps         = 1e-6,
                      float hc_sinkhorn_eps    = 1e-6,
                      float hc_post_mult_value = 1.0,
                      int sinkhorn_repeat      = 20);
void mhc_post(const aiter_tensor_t& out,            // (m, hc_mult, hidden_size)
              const aiter_tensor_t& x,              // (m, hidden_size)
              const aiter_tensor_t& residual,       // (m, hc_mult, hidden_size)
              const aiter_tensor_t& post_layer_mix, // (m, hc_mult)
              const aiter_tensor_t& comb_res_mix    // (m, hc_mult, hc_mult)
);
} // namespace aiter
