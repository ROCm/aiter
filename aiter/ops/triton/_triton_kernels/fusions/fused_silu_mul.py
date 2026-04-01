# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.activation import _silu_exp2


@triton.jit
def fused_silu_mul_kernel(
    inp_ptr,
    out_ptr,
    n_rows,
    n_cols,
    row_stride_in,
    col_stride_in,
    row_stride_out,
    col_stride_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    SiLU on the first half of the last dimension, multiply by the second half.
    Each row has 2 * n_cols input elements; writes n_cols outputs.
    2D grid: axis 0 tiles rows (BLOCK_M), axis 1 tiles columns (BLOCK_N).
    """
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)
    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    row_idx = m_pid * BLOCK_M + m_offs
    col_idx = n_pid * BLOCK_N + n_offs

    row_in = row_idx * row_stride_in
    row_out = row_idx * row_stride_out

    first_half_ptrs = inp_ptr + row_in[:, None] + col_idx[None, :] * col_stride_in
    second_half_ptrs = inp_ptr + row_in[:, None] + (n_cols + col_idx)[None, :] * col_stride_in
    out_ptrs = out_ptr + row_out[:, None] + col_idx[None, :] * col_stride_out

    mask = (row_idx < n_rows)[:, None] & (col_idx < n_cols)[None, :]
    a = tl.load(first_half_ptrs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(second_half_ptrs, mask=mask, other=0.0).to(tl.float32)
    silu_a = _silu_exp2(a)
    o = (silu_a * b).to(out_ptr.dtype.element_ty)
    tl.store(out_ptrs, o, mask=mask)
