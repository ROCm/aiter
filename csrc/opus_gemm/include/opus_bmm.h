// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <optional>

// Opus BMM public C++ API. These frontends use BMM/grouped layouts (for example
// DSV4 wo_a) while reusing the shared opus GEMM backend kernels.

// fp8 e8m0 mxscale (block-scale) BMM (zero-copy DSV4 wo_a): O/Y are [M, batch,
// *], wo_a/w_scale batch-major. Y dtype in {fp32, bf16}. dim0=M, dim1=batch (K
// contiguous); the batch axis memory position is otherwise free (see host
// stride checks). The global kid is exact: no fallback or redirect is allowed.
void opus_gemm_a8w8_mxscale_bmm_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& Y,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& w_scale,
    std::optional<aiter_tensor_t> workspace,
    int kid,
    int split_k);
