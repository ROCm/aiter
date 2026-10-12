# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_fused_indexed_gated_residual_repr = make_kernel_repr(
    "_fused_indexed_gated_residual_kernel",
    ["BLOCK_M", "BLOCK_N", "ROUND_INTERMEDIATE"],
)


@triton.jit(repr=_fused_indexed_gated_residual_repr)
def _fused_indexed_gated_residual_kernel(
    out_ptr,
    residual_ptr,
    x_ptr,
    gate_ptr,
    idx_ptr,
    M,
    N,
    stride_residual_row,
    stride_x_row,
    stride_out_row,
    stride_gate_row,
    ROUND_INTERMEDIATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """out[m] = residual[m] + gate[idx[m]] * x[m]

    One program owns a [BLOCK_M, BLOCK_N] tile. Every element is read once and
    written once; the gate table row is gathered per token rather than being
    materialised at [M, N] first.
    """
    row_block = tl.program_id(0)
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    # M * N can exceed the int32 range for long sequences.
    rows64 = rows.to(tl.int64)

    # The block's first row always exists, so its index is a safe fill for the
    # tail lanes -- they must not make a uniform block look mixed.
    first = tl.load(idx_ptr + row_block * BLOCK_M)
    idx = tl.load(idx_ptr + rows, mask=row_mask, other=first)
    if tl.min(idx) == tl.max(idx):
        # Consecutive tokens usually share one gate row (runs of one modality),
        # so load it once and broadcast it across the block.
        gate = tl.broadcast_to(
            tl.load(
                gate_ptr + first.to(tl.int64) * stride_gate_row + cols,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)[None, :],
            (BLOCK_M, BLOCK_N),
        )
    else:
        gate = tl.load(
            gate_ptr + idx.to(tl.int64)[:, None] * stride_gate_row + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

    x = tl.load(
        x_ptr + rows64[:, None] * stride_x_row + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + rows64[:, None] * stride_residual_row + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    gated = gate * x
    if ROUND_INTERMEDIATE:
        # The eager `residual + gate[idx] * x` stores the product before adding
        # it. Reproduce that when bit-comparability matters more than precision.
        gated = gated.to(out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(
        out_ptr + rows64[:, None] * stride_out_row + cols[None, :],
        (residual + gated).to(out_ptr.dtype.element_ty),
        mask=mask,
    )
