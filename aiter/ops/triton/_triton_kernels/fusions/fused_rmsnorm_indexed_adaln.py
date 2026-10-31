# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.fusions.indexed_table import (
    _gather_indexed_rows,
    _load_block_indices,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_fused_rmsnorm_indexed_adaln_repr = make_kernel_repr(
    "_fused_rmsnorm_indexed_adaln_kernel",
    ["BLOCK_M", "BLOCK_N", "ROUND_INTERMEDIATE"],
)


@triton.jit(repr=_fused_rmsnorm_indexed_adaln_repr)
def _fused_rmsnorm_indexed_adaln_kernel(
    out_ptr,
    x_ptr,
    w_ptr,
    shift_ptr,
    scale_ptr,
    idx_ptr,
    M,
    N,
    stride_x_row,
    stride_out_row,
    stride_shift_row,
    stride_scale_row,
    eps,
    ROUND_INTERMEDIATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """out[m] = rmsnorm(x[m]) * (1 + scale[idx[m]]) + shift[idx[m]]

    One program owns BLOCK_M rows and walks the row in BLOCK_N column tiles:
    once to accumulate the sum of squares, once to normalise and modulate. The
    row is small enough to still be in cache on the second pass, so x is read
    from memory once and written once -- against read/write/read/write for a
    separate norm followed by a separate modulation.
    """
    row_block = tl.program_id(0)
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    idx, first, uniform = _load_block_indices(
        idx_ptr, row_block, rows, row_mask, BLOCK_M
    )

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start in tl.range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        x = tl.load(
            x_ptr + rows[:, None] * stride_x_row + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x * x, axis=1)

    # torch.nn.RMSNorm reduces in fp32 for reduced-precision input; match that
    # rather than the storage dtype, or the norm drifts on long rows.
    rstd = tl.rsqrt(acc / N + eps)

    for start in tl.range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        mask2d = row_mask[:, None] & col_mask[None, :]

        x = tl.load(
            x_ptr + rows[:, None] * stride_x_row + cols[None, :],
            mask=mask2d,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        normed = x * rstd[:, None] * w[None, :]
        if ROUND_INTERMEDIATE:
            # The eager path stores the normalised activation before modulating
            # it, so it rounds here. Reproduce that when bit-comparability with
            # an unfused reference matters more than the extra precision.
            normed = normed.to(out_ptr.dtype.element_ty).to(tl.float32)

        scale = _gather_indexed_rows(
            scale_ptr,
            stride_scale_row,
            idx,
            first,
            uniform,
            cols,
            row_mask,
            col_mask,
            BLOCK_M,
            BLOCK_N,
        )
        shift = _gather_indexed_rows(
            shift_ptr,
            stride_shift_row,
            idx,
            first,
            uniform,
            cols,
            row_mask,
            col_mask,
            BLOCK_M,
            BLOCK_N,
        )

        modulated = normed * (1.0 + scale)
        if ROUND_INTERMEDIATE:
            modulated = modulated.to(out_ptr.dtype.element_ty).to(tl.float32)
        out = modulated + shift

        tl.store(
            out_ptr + rows[:, None] * stride_out_row + cols[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask2d,
        )
