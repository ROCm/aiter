# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

import torch
from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_norm"


def gen_layer_norm_fake_tensors(
    input: Tensor,
    # normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
    x_bias: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(
        input,
        dtype=input.dtype,
        device=input.device,
    )


@compile_ops(
    "module_norm", fc_name="layernorm2d_fwd", gen_fake=gen_layer_norm_fake_tensors
)
def layer_norm(
    input: Tensor,
    # normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    epsilon: float = 1e-5,
    x_bias: Optional[Tensor] = None,
) -> Tensor: ...


@compile_ops(
    "module_norm", fc_name="layernorm2d_fwd", gen_fake=gen_layer_norm_fake_tensors
)
def layernorm2d_fwd(
    input: Tensor,
    # normalized_shape: List[int],
    weight: Tensor,
    bias: Tensor,
    epsilon: float = 1e-5,
    x_bias: Optional[Tensor] = None,
) -> Tensor: ...


@compile_ops("module_norm")
def layernorm2d_fwd_with_add(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
    x_bias: Optional[Tensor] = None,
) -> None: ...


@compile_ops("module_norm")
def layernorm2d_fwd_with_smoothquant(
    out: Tensor,
    input: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
    x_bias: Optional[Tensor] = None,
) -> None: ...


@compile_ops("module_norm")
def layernorm2d_fwd_with_add_smoothquant(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
    x_bias: Optional[Tensor] = None,
) -> None: ...


# @compile_ops("module_norm")
# def layernorm2d_fwd_with_dynamicquant(
#     out: Tensor,
#     input: Tensor,
#     yscale: Tensor,
#     weight: Tensor,
#     bias: Tensor,
#     epsilon: float,
#     x_bias: Optional[Tensor] = None,):...


# @compile_ops("module_norm")
# def layernorm2d_fwd_with_add_dynamicquant(
#     out: Tensor,
#     input: Tensor,
#     residual_in: Tensor,
#     residual_out: Tensor,
#     yscale: Tensor,
#     weight: Tensor,
#     bias: Tensor,
#     epsilon: float,
#     x_bias: Optional[Tensor] = None,):...
@compile_ops("module_norm")
def layernorm2d_with_add_asm(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
    x_bias: Optional[Tensor] = None,
) -> None: ...


@compile_ops("module_norm")
def layernorm2d_with_add_smoothquant_asm(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
    x_bias: Optional[Tensor] = None,
) -> None: ...
