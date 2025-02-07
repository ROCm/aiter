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


def pertoken_quant(x, y_scale_dtype=torch.float, x_scale=None, quant_dtype=torch.int8):
    if x_scale is None:
        hidden_states = x
    else:
        # smooth quant
        hidden_states = x.to(x_scale) * x_scale
    # [m, 1]
    per_token_amax, _ = torch.max(
        input=torch.abs(hidden_states),
        dim=-1,
        keepdim=True
    )

    dtypeMax = get_dtype_max(quant_dtype)

    per_token_scale = per_token_amax.to(dtype=torch.float32) / dtypeMax
    per_token_scale[per_token_scale == 0] = 1

    # quant hidden_states
    y = (hidden_states / per_token_scale).to(dtype=quant_dtype)
    y_scale = per_token_scale.to(y_scale_dtype)
    return y, y_scale


def per_tensor_quant(x, scale=None, scale_dtype=torch.float, quant_dtype=torch.int8):
    if scale is None:
        dtypeMax = get_dtype_max(quant_dtype)
        scale = torch.abs(x.to(torch.float)).max() / dtypeMax
    y = x/scale

    return y.to(quant_dtype), scale.to(scale_dtype)


@compile_ops("module_quant")
def static_scaled_fp8_quant(
    out: Tensor, input: Tensor, scale: Tensor
): ...


@compile_ops("module_quant")
def dynamic_scaled_fp8_quant(
    out: Tensor, input: Tensor, scale: Tensor
):
    '''
    must init out as zeroes
    '''
    ...


@compile_ops("module_quant")
def dynamic_per_token_scaled_fp8_quant(
    out: Tensor, input: Tensor, scales: Tensor, scale_ub: Optional[Tensor] = None
): ...
