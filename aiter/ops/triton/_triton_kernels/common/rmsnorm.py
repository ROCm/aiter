# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared Triton RMSNorm operation for fused normalization kernels."""

import triton
import triton.language as tl


@triton.jit
def _rmsnorm_op(row, weight, n_cols, epsilon, USE_RECIPROCAL: tl.constexpr = False):
    row_norm = tl.sum(row * row, axis=-1)
    if USE_RECIPROCAL:
        variance = row_norm * (1.0 / n_cols)
    else:
        variance = row_norm / n_cols
    norm_factor = tl.math.rsqrt(variance + epsilon)
    return row * norm_factor * weight
