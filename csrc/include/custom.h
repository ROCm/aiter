#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

namespace aiter {

void wvSpltK(aiter_tensor_t& in_a,
             aiter_tensor_t& in_b,
             aiter_tensor_t& out_c,
             const int64_t N_in,
             const int64_t CuCount);

void wv_splitk_small_fp16_bf16_wrapper(aiter_tensor_t& in_a,
                                       aiter_tensor_t& in_b,
                                       aiter_tensor_t& out_c,
                                       const int64_t N_in,
                                       const int64_t CuCount);

void LLMM1(aiter_tensor_t& in_a, aiter_tensor_t& in_b, aiter_tensor_t& out_c, const int64_t rows_per_block);

void wvSplitKQ(aiter_tensor_t& in_a,
               aiter_tensor_t& in_b,
               aiter_tensor_t& out_c,
               aiter_tensor_t& scale_a,
               aiter_tensor_t& scale_b,
               const int64_t CuCount);
} // namespace aiter
