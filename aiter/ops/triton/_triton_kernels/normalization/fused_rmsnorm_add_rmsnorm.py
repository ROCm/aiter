# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""RMSNorm, residual add, and RMSNorm with FP32 intermediates."""

import triton
import triton.language as tl

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
    width,
    inv_width,
    post_eps,
    pre_eps,
    BLOCK_SIZE_N: tl.constexpr,
):
    row = tl.program_id(0)
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

    tl.store(residual_out_ptr + offset, added, valid)
    # The BF16 destination rounds only the final norm result.
    tl.store(pre_norm_ptr + offset, pre_norm, valid)
