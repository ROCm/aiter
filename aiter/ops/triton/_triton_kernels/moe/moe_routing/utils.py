# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_keyed_add_repr = make_kernel_repr("moe_routing_keyed_add", [])


@triton.jit(repr=_keyed_add_repr)
def keyed_add(x, y):
    """Add packed ``[16-bit key | 16-bit count]`` values by equal-key runs."""
    key_mask: tl.constexpr = 0xFFFF0000
    kx = x & key_mask
    ky = y & key_mask
    return tl.where(kx == ky, x + y - kx, y)
