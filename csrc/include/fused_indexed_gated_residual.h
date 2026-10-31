// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

// out[m] = residual[m] + gate[indices[m]] * x[m] for [M, N] bf16 rows with unit
// column stride (row strides free), a [G, N] gate table and int32/int64 [M] indices.
// All operands share one dtype; out may alias residual or x.
void fused_indexed_gated_residual(const aiter_tensor_t& residual,
                                  const aiter_tensor_t& x,
                                  const aiter_tensor_t& gate,
                                  const aiter_tensor_t& indices,
                                  aiter_tensor_t& out,
                                  bool round_intermediate);
