# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops
import torch.nn.functional as F
import functools
from .enum import QuantType, ActivationType
from . import triton
from ..utility import dtypes, fp4_utils


@compile_ops("module_smoothquant")
def smoothquant_fwd(input: Tensor, out: Tensor, x_scale: Tensor, y_scale: Tensor): ...


@compile_ops("module_smoothquant")
def moe_smoothquant_fwd(
    input: Tensor, out: Tensor, x_scale: Tensor, topk_ids: Tensor, y_scale: Tensor
): ...


# following are pure torch implement
@functools.lru_cache()
def get_dtype_max(dtype):
    try:
        dtypeMax = torch.finfo(dtype).max
    except:
        dtypeMax = torch.iinfo(dtype).max
    return dtypeMax


def pertoken_quant(
    x,
    scale=None,
    x_scale=None,  # smooth_scale
    scale_dtype=dtypes.fp32,
    quant_dtype=dtypes.i8,
    dtypeMax=None,
):
    x = x.to(dtypes.fp32)
    if x_scale is None:
        hidden_states = x
    else:
        # smooth quant
        hidden_states = x * x_scale

    if dtypeMax is None:
        dtypeMax = get_dtype_max(quant_dtype)

    per_token_scale = scale
    if scale is None:
        # [m, 1]
        per_token_amax, _ = torch.max(
            input=torch.abs(hidden_states), dim=-1, keepdim=True
        )
        per_token_scale = per_token_amax / dtypeMax
        per_token_scale[per_token_scale == 0] = 1

    # quant hidden_states
    y = (hidden_states / per_token_scale).to(dtype=quant_dtype)
    y_scale = per_token_scale.to(scale_dtype)
    return y, y_scale


def per_1x32_f4_quant(x, scale=None, quant_dtype=dtypes.fp4x2, shuffle=False):
    assert quant_dtype == dtypes.fp4x2
    block_size = 32
    F8E8M0_EXP_BIAS = 127
    F4E2M1_MAX = 6.0
    MAX_POW2 = int(torch.log2(torch.tensor(F4E2M1_MAX, dtype=torch.float32)).item())
    # dtypeMax = F4E2M1_MAX
    dtypeMax = 2.0**MAX_POW2

    shape_original = x.shape
    x = x.view(-1, shape_original[-1])

    m, n = x.shape
    x = x.view(-1, block_size)
    max_abs = torch.amax(torch.abs(x.float()), 1)
    # max_abs = max_abs.view(torch.int32)
    # max_abs = ((max_abs + 0x200000) & 0xFF800000).view(torch.float32)

    # fp8e8m0fnu_from_fp32_value
    scale_e8m0_biased = fp4_utils.f32_to_e8m0(max_abs / dtypeMax)

    # Float8_e8m0fnu to float
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0_biased)

    y = x.float() / scale_f32.view(-1, 1)
    y = fp4_utils.f32_to_mxfp4(y)
    y = y.view(*shape_original[:-1], -1)
    scale = scale_e8m0_biased.view(m, -1).view(torch.uint8)
    if shuffle:
        scale = fp4_utils.e8m0_shuffle(scale)
    return y, scale.view(dtypes.fp8_e8m0)


def per_tensor_quant(
    x, scale=None, scale_dtype=dtypes.fp32, quant_dtype=dtypes.i8, dtypeMax=None
):
    x = x.to(dtypes.fp32)
    if scale is None:
        if dtypeMax is None:
            dtypeMax = get_dtype_max(quant_dtype)
        scale = torch.abs(x).max() / dtypeMax
    y = x / scale

    return y.to(quant_dtype), scale.view(1).to(scale_dtype)


def per_block_quant_wrapper(block_shape=(1, 128)):
    def decorator(per_token_quant_func):
        def wrapper(x, scale=None, quant_dtype=dtypes.i8):
            blk_m, blk_n = block_shape
            assert (
                x.shape[-1] % blk_n == 0
            ), f"block size {blk_n} not match {x.shape[-1]}"
            assert blk_m == 1, "only support 1xN block, TODO: support MxN"
            m, n = x.shape
            x = x.view(-1, blk_n)
            y, scale = per_token_quant_func(x, scale=scale, quant_dtype=quant_dtype)
            return y.view(m, -1), scale.view(m, -1)

        return wrapper

    return decorator


@functools.lru_cache()
def get_torch_quant(qType):
    tmp = {
        QuantType.No: lambda *a, **k: (a[0], None),
        QuantType.per_Tensor: per_tensor_quant,
        QuantType.per_Token: pertoken_quant,
        QuantType.per_1x32: per_1x32_f4_quant,
        QuantType.per_1x128: per_block_quant_wrapper((1, 128))(pertoken_quant),
    }

    def raise_NotImplementedError(*a, **k):
        raise NotImplementedError(f"unsupported quant type {qType=}")

    return tmp.get(qType, raise_NotImplementedError)


@functools.lru_cache()
def get_hip_quant(qType):
    tmp = {
        QuantType.No: lambda *a, **k: (a[0], None),
        QuantType.per_Tensor: per_tensor_quant_hip,
        QuantType.per_Token: per_token_quant_hip,
        QuantType.per_1x32: per_1x32_f4_quant_hip,
        QuantType.per_1x128: per_block_quant_wrapper((1, 128))(per_token_quant_hip),
    }

    def raise_NotImplementedError(*a, **k):
        raise NotImplementedError(f"unsupported quant type {qType=}")

    return tmp.get(qType, raise_NotImplementedError)


@functools.lru_cache()
def get_triton_quant(qType):
    tmp = {
        QuantType.No: lambda *a, **k: (a[0], None),
        QuantType.per_Tensor: per_tensor_quant_triton,
        QuantType.per_Token: per_token_quant_triton,
        QuantType.per_1x32: per_1x32_f4_quant_triton,
        QuantType.per_1x128: per_block_quant_wrapper((1, 128))(per_token_quant_triton),
    }

    def raise_NotImplementedError(*a, **k):
        raise NotImplementedError(f"unsupported quant type {qType=}")

    return tmp.get(qType, raise_NotImplementedError)


def per_token_quant_hip(x, scale=None, quant_dtype=dtypes.i8):
    shape = x.shape
    device = x.device
    if scale is None:
        scale = torch.empty((*shape[:-1], 1), dtype=dtypes.fp32, device=device)
    else:
        raise ValueError("unsupported: static per token quant")

    if 1:
        y = torch.empty(shape, dtype=quant_dtype, device=device)
        dynamic_per_token_scaled_quant(y, x, scale)
    elif quant_dtype == dtypes.i8:
        M, N = x.view(-1, shape[-1]).shape
        y = torch.empty((M, N), dtype=dtypes.i8, device=device)
        scale = torch.empty(M, dtype=dtypes.fp32, device=device)
        smooth_scale = torch.ones(N, dtype=dtypes.fp32, device=device)
        smoothquant_fwd(y, x, smooth_scale, scale)
        y = y.view(shape)
    else:
        raise ValueError(f"unsupported: {quant_dtype=}")
    return y, scale


def per_1x32_f4_quant_hip(x, scale=None, quant_dtype=dtypes.fp4x2, shuffle=False):
    m, n = x.shape
    assert quant_dtype == dtypes.fp4x2
    assert n % 2 == 0
    device = x.device
    if scale is None:
        if shuffle:
            scale = (
                torch.empty(
                    (
                        (m + 255) // 256 * 256,
                        (n // 32 + 7) // 8 * 8,
                    ),
                    dtype=torch.uint8,
                    device=device,
                )
                # .fill_(0x7F)
                .view(dtypes.fp8_e8m0)
            )
        else:
            scale = (
                torch.empty(
                    (m, n // 32),
                    dtype=torch.uint8,
                    device=device,
                )
                # .fill_(0x7F)
                .view(dtypes.fp8_e8m0)
            )
    else:
        raise ValueError("unsupported: static per token quant")
    y = torch.empty(m, n // 2, dtype=quant_dtype, device=device)
    dynamic_per_group_scaled_quant_fp4(y, x, scale, 32, shuffle_scale=shuffle)
    return y.view(torch.uint8), scale


def per_tensor_quant_hip(x, scale=None, quant_dtype=dtypes.i8):
    y = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
    if quant_dtype in [dtypes.fp8, dtypes.i8]:
        if scale is None:
            scale = torch.empty(1, dtype=dtypes.fp32, device=x.device)
            dynamic_per_tensor_quant(y, x, scale)
        else:
            static_per_tensor_quant(y, x, scale)
    else:
        raise ValueError(f"unsupported: {quant_dtype=}")
    return y, scale.view(1)


def per_token_quant_triton(x, scale=None, quant_dtype=dtypes.i8):
    shape = x.shape
    device = x.device
    dtypeMax = get_dtype_max(quant_dtype)
    y = torch.empty(shape, dtype=quant_dtype, device=device)
    if scale is None:
        scale = torch.empty((*shape[:-1], 1), dtype=dtypes.fp32, device=device)
        triton.quant.dynamic_per_token_fp8_quant(
            y,
            x.view(-1, x.shape[-1]),
            scale,
            quant_dtype=quant_dtype,
            dtypeMax=dtypeMax,
        )
    else:
        raise ValueError("unsupported: static per token quant")

    return y, scale


def per_1x32_f4_quant_triton(x, scale=None, quant_dtype=dtypes.fp4x2, shuffle=False):
    assert quant_dtype == dtypes.fp4x2
    # y, scale = triton.quant.dynamic_mxfp4_quant(x)
    y, scale = fp4_utils.dynamic_mxfp4_quant(x, shuffle=shuffle)
    return y, scale


def per_tensor_quant_triton(x, scale=None, quant_dtype=dtypes.i8):
    y = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
    x = x.view(-1, x.shape[-1])
    if scale is None:
        scale = torch.zeros(1, dtype=dtypes.fp32, device=x.device)
        triton.quant.dynamic_per_tensor_fp8_quant(y, x, scale)
    else:
        triton.quant.static_per_tensor_fp8_quant(y, x, scale)
    return y, scale


@functools.lru_cache()
def get_torch_act(aType):
    tmp = {
        ActivationType.No: lambda *a, **k: a[0],
        ActivationType.Silu: F.silu,
        ActivationType.Gelu: F.gelu,
    }
    return tmp.get(aType, NotImplementedError)


@compile_ops("module_quant")
def static_per_tensor_quant(out: Tensor, input: Tensor, scale: Tensor): ...


@compile_ops("module_quant")
def dynamic_per_tensor_quant(out: Tensor, input: Tensor, scale: Tensor): ...


@compile_ops("module_quant")
def dynamic_per_token_scaled_quant(
    out: Tensor,
    input: Tensor,
    scales: Tensor,
    scale_ub: Optional[Tensor] = None,
    shuffle_scale=True,
): ...


@compile_ops("module_quant")
def dynamic_per_group_scaled_quant_fp4(
    out: Tensor,
    input: Tensor,
    scales: Tensor,
    group_size: Optional[int] = 32,
    shuffle_scale=True,
):
    """
    Only support group_size in [32, 64, 128]
    """
    ...
