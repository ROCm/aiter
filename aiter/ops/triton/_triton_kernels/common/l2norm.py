# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared device helpers for row-wise L2 normalization."""

import triton
import triton.language as tl


@triton.jit
def l2norm_bwd_rows(x, dy, eps):
    """``dx`` for ``y = x / sqrt(sum(x^2) + eps)`` per row of fp32 ``[rows, D]`` tiles."""
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) + eps)[:, None]
    y = x * rstd
    return dy * rstd - tl.sum(dy * y, axis=1)[:, None] * y * rstd
