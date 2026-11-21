# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_channel_rmsnorm_repr = make_kernel_repr(
    "_channel_rmsnorm_kernel",
    ["BLOCK_C", "BLOCK_S", "ONE_PASS", "HAS_BIAS", "USE_ACTIVATION", "num_warps"],
)


@triton.jit
def _channel_rmsnorm_epilogue(
    x,
    denom,
    gamma,
    bias,
    scale,
    out_dtype: tl.constexpr,
    activation: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Same steps and rounding points as the eager reference: the normalize is
    # in fp32, then each step rounds to the output type.
    y = (x / denom[None, :]).to(out_dtype).to(tl.float32)
    y = (y * scale).to(out_dtype).to(tl.float32)
    y = (y * gamma[:, None]).to(out_dtype).to(tl.float32)
    if HAS_BIAS:
        y = (y + bias[:, None]).to(out_dtype).to(tl.float32)
    if USE_ACTIVATION:
        y = activation(y)
    return y.to(out_dtype)


@triton.jit(repr=_channel_rmsnorm_repr)
def _channel_rmsnorm_kernel(
    x_ptr,
    gamma_ptr,
    bias_ptr,
    out_ptr,
    C,
    S,
    stride_n,
    stride_c,
    scale,
    eps,
    activation: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ONE_PASS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Normalize x[n, :, s] over C; one program does BLOCK_S positions of one n."""
    n = tl.program_id(1).to(tl.int64)
    offs_s = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    x_base = x_ptr + n * stride_n
    out_base = out_ptr + n * stride_n
    out_dtype = out_ptr.dtype.element_ty

    if ONE_PASS:
        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        mask = mask_c[:, None] & mask_s[None, :]
        offs = offs_c[:, None].to(tl.int64) * stride_c + offs_s[None, :]
        x = tl.load(x_base + offs, mask=mask, other=0.0).to(tl.float32)
        denom = tl.maximum(tl.sqrt(tl.sum(x * x, axis=0)), eps)
        gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        # Without a bias, gamma fills the unused argument.
        bias = gamma
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        y = _channel_rmsnorm_epilogue(
            x,
            denom,
            gamma,
            bias,
            scale,
            out_dtype,
            activation,
            USE_ACTIVATION,
            HAS_BIAS,
        )
        tl.store(out_base + offs, y, mask=mask)
    else:
        sumsq = tl.zeros((BLOCK_S,), dtype=tl.float32)
        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask = (offs_c[:, None] < C) & mask_s[None, :]
            offs = offs_c[:, None].to(tl.int64) * stride_c + offs_s[None, :]
            x = tl.load(x_base + offs, mask=mask, other=0.0).to(tl.float32)
            sumsq += tl.sum(x * x, axis=0)
        denom = tl.maximum(tl.sqrt(sumsq), eps)
        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask_c = offs_c < C
            mask = mask_c[:, None] & mask_s[None, :]
            offs = offs_c[:, None].to(tl.int64) * stride_c + offs_s[None, :]
            x = tl.load(x_base + offs, mask=mask, other=0.0).to(tl.float32)
            gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
            bias = gamma
            if HAS_BIAS:
                bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
            y = _channel_rmsnorm_epilogue(
                x,
                denom,
                gamma,
                bias,
                scale,
                out_dtype,
                activation,
                USE_ACTIVATION,
                HAS_BIAS,
            )
            tl.store(out_base + offs, y, mask=mask)
