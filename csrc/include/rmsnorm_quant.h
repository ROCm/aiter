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
                       bool scale_layout_m32k4 = false,
                       int scale_layout        = -1);

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
                   bool scale_layout_m32k4 = false,
                   int scale_layout        = -1);

void rmsnorm(aiter_tensor_t& out,
             aiter_tensor_t& input,
             aiter_tensor_t& weight,
             double epsilon,
             bool gemma_norm = false);

} // namespace aiter
