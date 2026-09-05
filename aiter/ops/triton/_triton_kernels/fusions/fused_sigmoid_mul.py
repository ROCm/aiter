# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.activation import _sigmoid
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_fused_sigmoid_mul_repr = make_kernel_repr(
    "_fused_sigmoid_mul_kernel",
    ["BLOCK_SIZE_N", "NEED_MASK"],
)


@triton.jit(repr=_fused_sigmoid_mul_repr)
def _fused_sigmoid_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    N,
    BLOCK_SIZE_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """
    out[i] = x[i] * sigmoid(gate[i])
    """
    offs = tl.program_id(0).to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = None
    if NEED_MASK:
        mask = offs < N

    gate = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + offs, x * _sigmoid(gate), mask=mask)
