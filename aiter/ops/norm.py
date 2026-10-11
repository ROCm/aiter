# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
from torch import Tensor

from ..jit.core import ENABLE_CK, compile_ops

MD_NAME = "module_norm"


def gen_layer_norm_fake_tensors(
    input: Tensor,
    # normalized_shape: List[int],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
    x_bias: Tensor | None = None,
) -> Tensor:
    return torch.empty_like(
        input,
        dtype=input.dtype,
        device=input.device,
    )


@compile_ops(
    "module_norm", fc_name="layernorm2d_fwd", gen_fake=gen_layer_norm_fake_tensors
)
def layer_norm_ck(
    input: Tensor,
    # normalized_shape: List[int],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    epsilon: float = 1e-5,
    x_bias: Tensor | None = None,
) -> Tensor: ...


@compile_ops(
    "module_norm", fc_name="layernorm2d_fwd", gen_fake=gen_layer_norm_fake_tensors
)
def layernorm2d_fwd_ck(
    input: Tensor,
    # normalized_shape: List[int],
    weight: Tensor,
    bias: Tensor,
    epsilon: float = 1e-5,
    x_bias: Tensor | None = None,
) -> Tensor: ...


@compile_ops("module_norm", fc_name="layernorm2d_fwd_with_add")
def layernorm2d_fwd_with_add_ck(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
    x_bias: Tensor | None = None,
) -> None: ...


@compile_ops("module_norm", fc_name="layernorm2d_fwd_with_smoothquant")
def layernorm2d_fwd_with_smoothquant_ck(
    out: Tensor,
    input: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
    x_bias: Tensor | None = None,
) -> None: ...


@compile_ops("module_norm", fc_name="layernorm2d_fwd_with_add_smoothquant")
def layernorm2d_fwd_with_add_smoothquant_ck(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
    x_bias: Tensor | None = None,
) -> None: ...


# The Triton kernels take 2-D rows and have no x_bias input; the CK kernels add
# x_bias to the input and treat any shape as (numel / N, N).
def _triton_rows(input: Tensor, x_bias: Tensor | None) -> Tensor:
    if x_bias is not None:
        input = input + x_bias
    return input.reshape(-1, input.shape[-1])


# The Triton fused-add kernels index residual_in and residual_out with the input
# row stride, while CK takes one stride per tensor, so all three are passed as
# contiguous rows; residual_out goes through a temporary when it is not.
def _triton_add_rows(
    input: Tensor, residual_in: Tensor, residual_out: Tensor, x_bias: Tensor | None
) -> tuple[Tensor, Tensor, Tensor]:
    n = input.shape[-1]
    res_out = residual_out.view(-1, n)
    if not res_out.is_contiguous():
        res_out = torch.empty(res_out.shape, dtype=res_out.dtype, device=res_out.device)
    return (
        _triton_rows(input, x_bias).contiguous(),
        residual_in.reshape(-1, n).contiguous(),
        res_out,
    )


def _copy_back(residual_out: Tensor, res_out: Tensor) -> None:
    dst = residual_out.view(-1, res_out.shape[-1])
    if res_out.data_ptr() != dst.data_ptr():
        dst.copy_(res_out)


# The CK binding rejects None for weight / bias; reject it the same way here.
def _check_affine(weight: Tensor | None, bias: Tensor | None) -> None:
    if weight is None or bias is None:
        raise TypeError("LayerNorm requires weight and bias tensors")


def layer_norm(
    input: Tensor,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    epsilon: float = 1e-5,
    x_bias: Tensor | None = None,
) -> Tensor:
    if not ENABLE_CK:
        _check_affine(weight, bias)
        from .triton.normalization.norm import layer_norm as layer_norm_triton

        out = layer_norm_triton(_triton_rows(input, x_bias), weight, bias, epsilon)
        return out.view(input.shape)
    return layer_norm_ck(input, weight, bias, epsilon, x_bias)


def layernorm2d_fwd(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float = 1e-5,
    x_bias: Tensor | None = None,
) -> Tensor:
    if not ENABLE_CK:
        return layer_norm(input, weight, bias, epsilon, x_bias)
    return layernorm2d_fwd_ck(input, weight, bias, epsilon, x_bias)


def layernorm2d_fwd_with_add(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
    x_bias: Tensor | None = None,
) -> None:
    if not ENABLE_CK:
        _check_affine(weight, bias)
        from .triton.normalization.norm import (
            layernorm2d_fwd_with_add as layernorm2d_fwd_with_add_triton,
        )

        x, res_in, res_out = _triton_add_rows(input, residual_in, residual_out, x_bias)
        layernorm2d_fwd_with_add_triton(
            out.view(-1, input.shape[-1]),
            x,
            res_in,
            res_out,
            weight,
            bias,
            epsilon,
        )
        _copy_back(residual_out, res_out)
        return
    layernorm2d_fwd_with_add_ck(
        out, input, residual_in, residual_out, weight, bias, epsilon, x_bias
    )


def layernorm2d_fwd_with_smoothquant(
    out: Tensor,
    input: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
    x_bias: Tensor | None = None,
) -> None:
    if not ENABLE_CK:
        _check_affine(weight, bias)
        from .triton.normalization.norm import (
            layernorm2d_fwd_with_smoothquant as layernorm2d_fwd_with_smoothquant_triton,
        )

        n = input.shape[-1]
        layernorm2d_fwd_with_smoothquant_triton(
            out.view(-1, n),
            _triton_rows(input, x_bias),
            xscale,
            yscale,
            weight,
            bias,
            epsilon,
        )
        return
    layernorm2d_fwd_with_smoothquant_ck(
        out, input, xscale, yscale, weight, bias, epsilon, x_bias
    )


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
    x_bias: Tensor | None = None,
) -> None:
    if not ENABLE_CK:
        _check_affine(weight, bias)
        from .triton.normalization.norm import (
            layernorm2d_fwd_with_add_smoothquant as layernorm2d_fwd_with_add_smoothquant_triton,
        )

        x, res_in, res_out = _triton_add_rows(input, residual_in, residual_out, x_bias)
        layernorm2d_fwd_with_add_smoothquant_triton(
            out.view(-1, input.shape[-1]),
            x,
            res_in,
            res_out,
            xscale,
            yscale,
            weight,
            bias,
            epsilon,
        )
        _copy_back(residual_out, res_out)
        return
    layernorm2d_fwd_with_add_smoothquant_ck(
        out,
        input,
        residual_in,
        residual_out,
        xscale,
        yscale,
        weight,
        bias,
        epsilon,
        x_bias,
    )


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
