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
}
