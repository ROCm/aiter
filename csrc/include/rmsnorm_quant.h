// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

namespace aiter {

void add_rmsnorm_quant(aiter_tensor_t& out,
                       aiter_tensor_t& input,
                       aiter_tensor_t& residual_in,
                       aiter_tensor_t& residual_out,
                       aiter_tensor_t& scale,
                       aiter_tensor_t& weight,
                       double epsilon,
                       int group_size          = 0,
                       bool shuffle_scale      = false,
                       bool gemma_norm         = false,
                       bool scale_layout_m32k4 = false);

void add_rmsnorm(aiter_tensor_t& out,
                 aiter_tensor_t& input,
                 aiter_tensor_t& residual_in,
                 aiter_tensor_t& residual_out,
                 aiter_tensor_t& weight,
                 double epsilon,
                 bool gemma_norm = false);

void rmsnorm_quant(aiter_tensor_t& out,
                   aiter_tensor_t& input,
                   aiter_tensor_t& scale,
                   aiter_tensor_t& weight,
                   double epsilon,
                   int group_size          = 0,
                   bool shuffle_scale      = false,
                   bool gemma_norm         = false,
                   bool scale_layout_m32k4 = false); // fp8 + e8m0 + group 32: ASM GEMM A-scale layout

void rmsnorm(aiter_tensor_t& out,
             aiter_tensor_t& input,
             aiter_tensor_t& weight,
             double epsilon,
             bool gemma_norm = false);

// DeepSeek MLA q_a (MXFP4 out, m32k4 e8m0 scale) + kv_a (bf16 out) RMSNorm in one launch.
void qkv_a_rmsnorm_fp4(aiter_tensor_t& q_out,
                       aiter_tensor_t& q_scale,
                       aiter_tensor_t& kv_out,
                       aiter_tensor_t& q_in,
                       aiter_tensor_t& kv_in,
                       aiter_tensor_t& q_weight,
                       aiter_tensor_t& kv_weight,
                       double q_eps,
                       double kv_eps,
                       bool gemma_norm = false);

} // namespace aiter
