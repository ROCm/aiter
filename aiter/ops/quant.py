# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import List, Optional
from enum import Enum
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR
import torch.nn.functional as F
import functools
from .enum import *
from . import triton


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
    scale_dtype=torch.float,
    quant_dtype=torch.int8,
    dtypeMax=None,
):
    x = x.to(torch.float)
    if x_scale is None:
        hidden_states = x
    else:
        # smooth quant
        hidden_states = x * x_scale
    # [m, 1]
    per_token_amax, _ = torch.max(input=torch.abs(hidden_states), dim=-1, keepdim=True)

    if dtypeMax is None:
        dtypeMax = get_dtype_max(quant_dtype)

    per_token_scale = scale
    if scale is None:
        per_token_scale = per_token_amax / dtypeMax
        per_token_scale[per_token_scale == 0] = 1

    # quant hidden_states
    y = (hidden_states / per_token_scale).to(dtype=quant_dtype)
    y_scale = per_token_scale.to(scale_dtype)
    return y, y_scale


def per_tensor_quant(
    x, scale=None, scale_dtype=torch.float, quant_dtype=torch.int8, dtypeMax=None
):
    x = x.to(torch.float)
    if scale is None:
        if dtypeMax is None:
            dtypeMax = get_dtype_max(quant_dtype)
        scale = torch.abs(x).max() / dtypeMax
    y = x / scale

    return y.to(quant_dtype), scale.view(1).to(scale_dtype)


@functools.lru_cache()
def get_torch_quant(qType):
    tmp = {
        QuantType.No: lambda *a, **k: (a[0], None),
        QuantType.per_Tensor: per_tensor_quant,
        QuantType.per_Token: pertoken_quant,
    }
    return tmp.get(qType, NotImplementedError)


@functools.lru_cache()
def get_hip_quant(qType):
    tmp = {
        QuantType.No: lambda *a, **k: (a[0], None),
        QuantType.per_Tensor: per_tensor_quant_hip,
        QuantType.per_Token: per_token_quant_hip,
        QuantType.per_1x128: lambda a, **k: per_token_quant_hip(a.view(-1, 128), **k),
    }
    return tmp.get(qType, NotImplementedError)


@functools.lru_cache()
def get_triton_quant(qType):
    tmp = {
        QuantType.No: lambda *a, **k: (a[0], None),
        QuantType.per_Tensor: per_tensor_quant_fp8_triton,
        QuantType.per_Token: per_token_quant_triton,
        QuantType.per_1x128: lambda a, **k: per_token_quant_triton(
            a.view(-1, 128), **k
        ),
    }
    return tmp.get(qType, NotImplementedError)


def per_token_quant_hip(x, scale=None, quant_dtype=torch.int8):
    shape = x.shape
    device = x.device
    if scale is None:
        scale = torch.empty(shape[:-1], dtype=torch.float, device=device)
    else:
        raise ValueError(f"unsupported: static per token quant")

    if quant_dtype == torch.float8_e4m3fnuz:
        y = torch.empty(shape, dtype=quant_dtype, device=device)
        dynamic_per_token_scaled_fp8_quant(y, x, scale)
    elif quant_dtype == torch.int8:
        M, N = x.view(-1, shape[-1]).shape
        y = torch.empty((M, N), dtype=torch.int8, device=device)
        scale = torch.empty(M, dtype=torch.float, device=device)
        smooth_scale = torch.ones(N, dtype=torch.float, device=device)
        smoothquant_fwd(y, x, smooth_scale, scale)
        y = y.view(shape)
    else:
        raise ValueError(f"unsupported: {quant_dtype=}")
    return y, scale


def per_tensor_quant_hip(x, scale=None, quant_dtype=torch.float8_e4m3fnuz):
    y = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
    if quant_dtype == torch.float8_e4m3fnuz:
        if scale is None:
            scale = torch.empty(1, dtype=torch.float, device=x.device)
            dynamic_scaled_fp8_quant(y, x, scale)
        else:
            static_scaled_fp8_quant(y, x, scale)
    else:
        raise ValueError(f"unsupported: {quant_dtype=}")
    return y, scale


def per_token_quant_triton(x, scale=None, quant_dtype=torch.int8):
    shape = x.shape
    device = x.device
    dtypeMax = get_dtype_max(quant_dtype)
    y = torch.empty(shape, dtype=quant_dtype, device=device)
    if scale is None:
        scale = torch.empty(shape[:-1], dtype=torch.float, device=device)
        triton.quant.dynamic_per_token_fp8_quant(
            y, x, scale, quant_dtype=quant_dtype, dtypeMax=dtypeMax
        )
    else:
        raise ValueError(f"unsupported: static per token quant")

    return y, scale


def per_tensor_quant_fp8_triton(x, scale=None):
    y = torch.empty(x.shape, dtype=torch.float8_e4m3fnuz, device=x.device)
    if scale is None:
        scale = torch.empty(1, dtype=torch.float, device=x.device)
        triton.dynamic_scaled_fp8_quant(y, x, scale)
    else:
        triton.static_scaled_fp8_quant(y, x, scale)
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
def static_scaled_fp8_quant(out: Tensor, input: Tensor, scale: Tensor): ...


@compile_ops("module_quant")
def dynamic_scaled_fp8_quant(out: Tensor, input: Tensor, scale: Tensor): ...


@compile_ops("module_quant")
def dynamic_per_token_scaled_fp8_quant(
    out: Tensor, input: Tensor, scales: Tensor, scale_ub: Optional[Tensor] = None
): ...
