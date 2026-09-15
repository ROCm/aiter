# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# AdaLN-Zero: LayerNorm (no affine) fused with adaptive scale/shift modulation.
import triton
import triton.language as tl


@triton.jit
def _adaln_zero_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    shift_ptr,
    eps,
    seq,
    N,
    x_stride_m,
    out_stride_m,
    scale_stride_b,
    shift_stride_b,
    BLOCK_SIZE_N: tl.constexpr,
):
    """One program per token row (grid = B*seq). ``scale``/``shift`` are indexed by
    the batch ``b = row // seq`` and broadcast over the sequence dimension."""
    pid_m = tl.program_id(0)
    tl.assume(pid_m >= 0)
    b = pid_m // seq

    n_offs = tl.arange(0, BLOCK_SIZE_N)
    mask = n_offs < N

    x = tl.load(
        x_ptr + pid_m * x_stride_m + n_offs,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)

    mean = tl.sum(x, axis=-1) / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=-1) / N
    x_norm = x_centered * tl.math.rsqrt(var + eps)

    scale = tl.load(scale_ptr + b * scale_stride_b + n_offs, mask=mask, other=0.0).to(
        tl.float32
    )
    shift = tl.load(shift_ptr + b * shift_stride_b + n_offs, mask=mask, other=0.0).to(
        tl.float32
    )

    out = x_norm * (1.0 + scale) + shift
    tl.store(
        out_ptr + pid_m * out_stride_m + n_offs,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
