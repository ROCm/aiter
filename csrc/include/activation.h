#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

namespace aiter {

void silu_and_mul(const aiter_tensor_t& out, const aiter_tensor_t& input);
void swiglu_and_mul(const aiter_tensor_t& out, const aiter_tensor_t& input);
void silu_and_mul_bias(const aiter_tensor_t& out,
                       const aiter_tensor_t& input,
                       const aiter_tensor_t& expert_ids,
                       const aiter_tensor_t& bias);
void swiglu_and_mul_bias(const aiter_tensor_t& out,
                         const aiter_tensor_t& input,
                         const aiter_tensor_t& expert_ids,
                         const aiter_tensor_t& bias);
void scaled_silu_and_mul(const aiter_tensor_t& out, const aiter_tensor_t& input, const aiter_tensor_t& scale);
void gelu_and_mul(const aiter_tensor_t& out, const aiter_tensor_t& input);
void gelu_tanh_and_mul(const aiter_tensor_t& out, const aiter_tensor_t& input);
void gelu_fast(const aiter_tensor_t& out, const aiter_tensor_t& input);

} // namespace aiter
