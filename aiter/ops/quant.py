# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR


@compile_ops("module_smoothquant")
def smoothquant_fwd(input: Tensor, out: Tensor,
                    x_scale: Tensor, y_scale: Tensor): ...


@compile_ops("module_smoothquant")
def moe_smoothquant_fwd(
    input: Tensor, out: Tensor, x_scale: Tensor, topk_ids: Tensor, y_scale: Tensor
): ...


# following are pure torch implement
def get_dtype_max(dtype):
    try:
        dtypeMax = torch.finfo(dtype).max
    except:
        dtypeMax = torch.iinfo(dtype).max
    return dtypeMax


def pertoken_quant(x, y_scale_dtype=torch.float, x_scale=None, quant_dtype=torch.int8, dtypeMax=None):
    if x_scale is None:
        hidden_states = x
    else:
        # smooth quant
        hidden_states = x * x_scale
    # [m, 1]
    per_token_amax, _ = torch.max(
        input=torch.abs(hidden_states),
        dim=-1,
        keepdim=True
    )

    if not dtypeMax:
        dtypeMax = get_dtype_max(quant_dtype)

    per_token_scale = per_token_amax / dtypeMax
    per_token_scale[per_token_scale == 0] = 1

    # quant hidden_states
    y = (hidden_states / per_token_scale).to(dtype=quant_dtype)
    y_scale = per_token_scale.to(y_scale_dtype)
    return y, y_scale


def per_tensor_quant(x, scale=None, scale_dtype=torch.float, quant_dtype=torch.int8):
    x = x.to(torch.float)
    if scale is None:
        dtypeMax = get_dtype_max(quant_dtype)
        scale = torch.abs(x).max() / dtypeMax
    y = x/scale

    return y.to(quant_dtype), scale.to(scale_dtype)


def per_tensor_quant_fp8_hip(x, scale=None):
    y = torch.empty(x.shape, dtype=torch.float8_e4m3fnuz, device=x.device)
    if scale is None:
        scale = torch.empty(1, dtype=torch.float, device=x.device)
        dynamic_scaled_fp8_quant(y, x, scale)
    else:
        static_scaled_fp8_quant(y, x, scale)
    return y, scale


@compile_ops("module_quant")
def static_scaled_fp8_quant(
    out: Tensor, input: Tensor, scale: Tensor
): ...


@compile_ops("module_quant")
def dynamic_scaled_fp8_quant(
    out: Tensor, input: Tensor, scale: Tensor
):...


@compile_ops("module_quant")
def dynamic_per_token_scaled_fp8_quant(
    out: Tensor, input: Tensor, scales: Tensor, scale_ub: Optional[Tensor] = None
): ...
