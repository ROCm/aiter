# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# RMSNorm + optional residual add. Non-Gluon fallback for archs without TDM.
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_op(row, weight, n_cols, epsilon):
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)
    return row * norm_factor * weight


@triton.jit
def _triton_fused_rms_kernel(
    x_ptr,
    w_ptr,
    res_ptr,
    out_ptr,
    out_res_ptr,
    eps,
    M,
    N,
    x_stride_m,
    res_stride_m,
    out_stride_m,
    out_res_stride_m,
    BLOCK_SIZE_N: tl.constexpr,
    FIRST_INPUT_RES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    tl.assume(pid_m >= 0)

    n_offs = tl.arange(0, BLOCK_SIZE_N)
    mask = n_offs < N

    x = tl.load(
        x_ptr + pid_m * x_stride_m + n_offs,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)

    if FIRST_INPUT_RES:
        res = tl.load(
            res_ptr + pid_m * res_stride_m + n_offs,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        x = x + res
        tl.store(
            out_res_ptr + pid_m * out_res_stride_m + n_offs,
            x.to(out_res_ptr.dtype.element_ty),
            mask=mask,
        )

    w = tl.load(w_ptr + n_offs, mask=mask, other=0.0).to(tl.float32)
    out = _rmsnorm_op(x, w, N, eps).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + pid_m * out_stride_m + n_offs, out, mask=mask)
