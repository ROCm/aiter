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
    [
        "BLOCK_M",
        "BLOCK_N",
        "ROUND_INTERMEDIATE",
        "IS_LAYERNORM",
        "HAS_WEIGHT",
        "HAS_BIAS",
    ],
)


@triton.jit(repr=_fused_rmsnorm_indexed_adaln_repr)
def _fused_rmsnorm_indexed_adaln_kernel(
    out_ptr,
    x_ptr,
    w_ptr,
    b_ptr,
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
    IS_LAYERNORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """out[m] = norm(x[m]) * (1 + scale[idx[m]]) + shift[idx[m]]

    norm is RMSNorm, or LayerNorm when IS_LAYERNORM. The weight and the bias
    of the norm are optional (HAS_WEIGHT, HAS_BIAS).

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

    if IS_LAYERNORM:
        # Mean and variance per column tile, merged with Chan's formula. This
        # keeps two passes over the row and avoids E[x^2] - E[x]^2. The
        # statistics are of x - x[:, 0], so a large mean does not cost precision.
        ref = tl.load(x_ptr + rows * stride_x_row, mask=row_mask, other=0.0).to(
            tl.float32
        )
        mean = tl.zeros((BLOCK_M,), dtype=tl.float32)
        m2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        count = 0.0
        for start in tl.range(0, N, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            x = tl.load(
                x_ptr + rows[:, None] * stride_x_row + cols[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            x = tl.where(col_mask[None, :], x - ref[:, None], 0.0)
            n_tile = tl.minimum(N - start, BLOCK_N).to(tl.float32)
            tile_mean = tl.sum(x, axis=1) / n_tile
            centered = tl.where(col_mask[None, :], x - tile_mean[:, None], 0.0)
            tile_m2 = tl.sum(centered * centered, axis=1)
            total = count + n_tile
            delta = tile_mean - mean
            mean += delta * (n_tile / total)
            m2 += tile_m2 + delta * delta * (count * n_tile / total)
            count = total
        rstd = tl.rsqrt(m2 / N + eps)
    else:
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
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        if IS_LAYERNORM:
            normed = ((x - ref[:, None]) - mean[:, None]) * rstd[:, None]
            if HAS_WEIGHT:
                normed = normed * w[None, :]
        elif HAS_WEIGHT:
            normed = x * rstd[:, None] * w[None, :]
        else:
            normed = x * rstd[:, None]
        if HAS_BIAS:
            b = tl.load(b_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            normed = normed + b[None, :]
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
