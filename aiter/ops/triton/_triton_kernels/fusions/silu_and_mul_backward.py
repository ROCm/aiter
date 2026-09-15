# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.activation import _sigmoid_exp2
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_silu_and_mul_backward_kernel_repr = make_kernel_repr(
    "_silu_and_mul_backward_kernel",
    ["BLOCK_M", "BLOCK_N", "num_warps", "num_stages"],
)


@triton.jit(repr=_silu_and_mul_backward_kernel_repr)
def _silu_and_mul_backward_kernel(
    grad_output_ptr,
    input_ptr,
    grad_input_ptr,
    n_rows,
    n_cols,
    grad_output_row_stride,
    grad_output_col_stride,
    input_row_stride,
    input_col_stride,
    grad_input_row_stride,
    grad_input_col_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Backward for ``silu(gate) * up`` with concatenated gate/up inputs."""
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)

    input_offsets = rows[:, None] * input_row_stride + cols[None, :] * input_col_stride
    grad_output_offsets = (
        rows[:, None] * grad_output_row_stride + cols[None, :] * grad_output_col_stride
    )
    grad_input_offsets = (
        rows[:, None] * grad_input_row_stride + cols[None, :] * grad_input_col_stride
    )
    n_cols_i64 = n_cols.to(tl.int64)

    gate = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(
        input_ptr + input_offsets + n_cols_i64 * input_col_stride,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    grad_output = tl.load(
        grad_output_ptr + grad_output_offsets, mask=mask, other=0.0
    ).to(tl.float32)

    sigmoid_gate = _sigmoid_exp2(gate)
    silu_gate = gate * sigmoid_gate
    grad_gate = grad_output * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    grad_up = grad_output * silu_gate

    tl.store(
        grad_input_ptr + grad_input_offsets,
        grad_gate.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        grad_input_ptr + grad_input_offsets + n_cols_i64 * grad_input_col_stride,
        grad_up.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )
