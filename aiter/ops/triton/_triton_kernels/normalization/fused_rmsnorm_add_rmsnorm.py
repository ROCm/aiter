# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""RMSNorm, residual add, and RMSNorm with FP32 intermediates."""

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


_fused_rmsnorm_add_rmsnorm_repr = make_kernel_repr(
    "_fused_rmsnorm_add_rmsnorm_kernel", ["BLOCK_SIZE_N"]
)


@triton.jit(repr=_fused_rmsnorm_add_rmsnorm_repr)
def _fused_rmsnorm_add_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    post_weight_ptr,
    pre_weight_ptr,
    residual_out_ptr,
    pre_norm_ptr,
    width: tl.constexpr,
    post_eps,
    pre_eps,
    BLOCK_SIZE_N: tl.constexpr,
):
    row = tl.program_id(0)
    inv_width = 1.0 / width
    col = tl.arange(0, BLOCK_SIZE_N)
    valid = col < width
    offset = row * width + col

    x = tl.load(x_ptr + offset, valid, other=0).to(tl.float32)
    residual = tl.load(residual_ptr + offset, valid, other=0).to(tl.float32)
    post_weight = tl.load(post_weight_ptr + col, valid, other=0).to(tl.float32)
    pre_weight = tl.load(pre_weight_ptr + col, valid, other=0).to(tl.float32)

    post_rsqrt = tl.rsqrt(tl.sum(x * x, 0) * inv_width + post_eps)
    post_norm = x * post_rsqrt * (post_weight + 1.0)
    added = post_norm + residual
    pre_rsqrt = tl.rsqrt(tl.sum(added * added, 0) * inv_width + pre_eps)
    pre_norm = added * pre_rsqrt * (pre_weight + 1.0)

    tl.store(residual_out_ptr + offset, added.to(tl.bfloat16), valid)
    tl.store(pre_norm_ptr + offset, pre_norm.to(tl.bfloat16), valid)


@gluon.jit
def _fused_rmsnorm_add_rmsnorm_large_m_kernel(
    x_ptr,
    residual_ptr,
    post_weight_ptr,
    pre_weight_ptr,
    residual_out_ptr,
    pre_norm_ptr,
    M,
    width: gl.constexpr,
    post_eps,
    pre_eps,
    BLOCK_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
):
    # Each of the four waves owns a row, and visits four rows per tile.
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    wave_rows = gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, layout))
    valid_cols = cols < width
    inv_width = 1.0 / width

    for row_group in range(BLOCK_M // 4):
        rows = gl.program_id(0) * BLOCK_M + row_group * 4 + wave_rows
        offsets = rows[:, None] * width + cols[None, :]
        valid = (rows[:, None] < M) & valid_cols[None, :]

        x = gl.load(x_ptr + offsets, valid, other=0).to(gl.float32)
        post_rsqrt = gl.rsqrt(gl.sum(x * x, 1) * inv_width + post_eps)
        post_weight = gl.load(post_weight_ptr + cols, valid_cols, other=0).to(
            gl.float32
        )
        post_norm = x * post_rsqrt[:, None] * (post_weight[None, :] + 1.0)
        residual = gl.load(residual_ptr + offsets, valid, other=0).to(gl.float32)
        added = post_norm + residual
        pre_rsqrt = gl.rsqrt(gl.sum(added * added, 1) * inv_width + pre_eps)
        pre_weight = gl.load(pre_weight_ptr + cols, valid_cols, other=0).to(
            gl.float32
        )
        pre_norm = added * pre_rsqrt[:, None] * (pre_weight[None, :] + 1.0)

        gl.store(residual_out_ptr + offsets, added.to(gl.bfloat16), valid)
        gl.store(pre_norm_ptr + offsets, pre_norm.to(gl.bfloat16), valid)
