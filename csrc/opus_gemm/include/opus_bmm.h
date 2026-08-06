// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

// Opus BMM public C++ API. These frontends use BMM/grouped layouts (for example
// DSV4 wo_a) while reusing the shared opus GEMM backend kernels.

// fp8 e8m0 mxscale (block-scale) BMM (zero-copy DSV4 wo_a): O/Y are [M, batch,
// *], wo_a/w_scale batch-major. Y dtype in {fp32, bf16}. dim0=M, dim1=batch (K
// contiguous); the batch axis memory position is otherwise free (see host
// stride checks). kid-dispatched; driven by bmm_a8w8_mxscale_opus (Python).
void opus_bmm_a8w8_mxscale(aiter_tensor_t& O,
                           aiter_tensor_t& wo_a,
                           aiter_tensor_t& Y,
                           aiter_tensor_t& x_scale,
                           aiter_tensor_t& w_scale,
                           int splitK,
                           int kernelId);
