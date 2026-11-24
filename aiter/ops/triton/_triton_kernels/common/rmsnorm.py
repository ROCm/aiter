# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared Triton RMSNorm operation for fused normalization kernels."""

import triton
import triton.language as tl


@triton.jit
def _rmsnorm_op(row, weight, n_cols, epsilon, USE_RECIPROCAL: tl.constexpr = False):
    norm_factor = _inverse_rms(row, n_cols, epsilon, USE_RECIPROCAL)
    return row * norm_factor * weight


@triton.jit
def _inverse_rms(row, n_cols, epsilon, USE_RECIPROCAL: tl.constexpr):
    """1 / sqrt(mean(row**2) + epsilon) on the last axis."""
    sum_of_squares = tl.sum(row * row, axis=-1)
    if USE_RECIPROCAL:
        variance = sum_of_squares * (1.0 / n_cols)
    else:
        variance = sum_of_squares / n_cols
    return tl.math.rsqrt(variance + epsilon)
