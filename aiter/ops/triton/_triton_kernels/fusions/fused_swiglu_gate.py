# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused clamp + sigmoid(alpha*gate) + mul + (up+1) on separated gate|up rows.

Each program tile loads gate and up from ``[M, 2 * N]`` with the same layout as
``torch.chunk(2, dim=-1)`` on gate-up GEMM output (MiniMax-M3 / GPT-OSS swigluoai).
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_fused_swiglu_gate_repr = make_kernel_repr(
    "_fused_swiglu_gate_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "HAVE_SWIGLU_CLAMP",
        "ADD_RESIDUAL",
    ],
)


@triton.jit(repr=_fused_swiglu_gate_repr)
def _fused_swiglu_gate_kernel(
    inp_ptr,
    out_ptr,
    n_rows,
    n_cols,
    row_stride_in,
    col_stride_in,
    row_stride_out,
    col_stride_out,
    swiglu_alpha,
    swiglu_limit,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAVE_SWIGLU_CLAMP: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
):
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)
    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    row_idx = m_pid * BLOCK_M + m_offs
    col_idx = n_pid * BLOCK_N + n_offs

    row_in = row_idx * row_stride_in
    row_out = row_idx * row_stride_out

    gate_ptrs = inp_ptr + row_in[:, None] + col_idx[None, :] * col_stride_in
    up_ptrs = inp_ptr + row_in[:, None] + (n_cols + col_idx)[None, :] * col_stride_in
    out_ptrs = out_ptr + row_out[:, None] + col_idx[None, :] * col_stride_out

    mask = (row_idx < n_rows)[:, None] & (col_idx < n_cols)[None, :]
    gate = tl.load(gate_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)

    if HAVE_SWIGLU_CLAMP:
        gate = tl.minimum(gate, swiglu_limit)
        up = tl.clamp(up, -swiglu_limit, swiglu_limit)

    s = gate / (1 + tl.exp2(-1.44269504089 * swiglu_alpha * gate))
    if ADD_RESIDUAL:
        out = tl.fma(s, up, s)
    else:
        out = s * up

    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=mask)
