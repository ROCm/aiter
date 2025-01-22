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
def pertoken_quant(hidden_states_input, y_scale_dtype=torch.float, x_scale=None, quant_dtype=torch.int8):
    if x_scale is None:
        hidden_states = hidden_states_input
    else:
        # smooth quant
        hidden_states = hidden_states_input.to(x_scale) * x_scale
    # [m, 1]
    per_token_amax, _ = torch.max(
        input=torch.abs(hidden_states),
        dim=-1,
        keepdim=True
    )

    try:
        dtypeMax = torch.finfo(quant_dtype).max
    except:
        dtypeMax = torch.iinfo(quant_dtype).max

    per_token_scale = per_token_amax.to(dtype=torch.float32) / dtypeMax
    per_token_scale[per_token_scale == 0] = 1

    # quant hidden_states
    hidden_states = (hidden_states / per_token_scale).to(dtype=quant_dtype)

    return hidden_states, per_token_scale.to(y_scale_dtype)


@compile_ops("module_quant")
def static_scaled_fp8_quant(
    out: Tensor, input: Tensor, scale: Tensor
):...


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
):...
