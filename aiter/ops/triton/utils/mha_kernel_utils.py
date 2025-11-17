# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    # compute fp8 scaling and descaling factor for a block
    x_amax = tl.max(tl.abs(x))  # NOTE: abs deals with negative values
    x_amax = tl.where(x_amax <= 1e-9, 1e-9, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x
