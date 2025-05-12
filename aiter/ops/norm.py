# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops

MD_NAME = "module_norm"


@compile_ops("module_norm", fc_name="layernorm2d_fwd")
def layer_norm(
    input: Tensor,
    # normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
    x_bias: Optional[Tensor] = None,
) -> Tensor: ...


@compile_ops("module_norm", fc_name="layernorm2d_fwd")
def layernorm2d_fwd(
    input: Tensor,
    # normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
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
): ...


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
): ...


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
): ...


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
): ...
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
): ...
