// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

namespace aiter {

void add_rmsnorm_quant(const aiter_tensor_t& out,
                       const aiter_tensor_t& input,
                       const aiter_tensor_t& residual_in,
                       const aiter_tensor_t& residual_out,
                       const aiter_tensor_t& scale,
                       const aiter_tensor_t& weight,
                       double epsilon,
                       int group_size     = 0,
                       bool shuffle_scale = false);

void add_rmsnorm(const aiter_tensor_t& out,
                 const aiter_tensor_t& input,
                 const aiter_tensor_t& residual_in,
                 const aiter_tensor_t& residual_out,
                 const aiter_tensor_t& weight,
                 double epsilon);

void rmsnorm_quant(const aiter_tensor_t& out,
                   const aiter_tensor_t& input,
                   const aiter_tensor_t& scale,
                   const aiter_tensor_t& weight,
                   double epsilon,
                   int group_size     = 0,
                   bool shuffle_scale = false);

void rmsnorm(const aiter_tensor_t& out, const aiter_tensor_t& input, const aiter_tensor_t& weight, double epsilon);

} // namespace aiter
