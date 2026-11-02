# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
from torch import Tensor

from ..jit.core import ENABLE_CK, compile_ops
from ..jit.utils.chip_info import get_gfx_runtime as get_gfx
from ..jit.utils.torch_guard import torch_compile_guard

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


# The Triton kernels take 2-D rows with unit column stride and have no x_bias
# input; the CK kernels add x_bias to the input and treat any shape as
# (numel / N, N).
def _triton_rows(input: Tensor, x_bias: Tensor | None) -> Tensor:
    if x_bias is not None:
        input = input + x_bias
    rows = input.reshape(-1, input.shape[-1])
    return rows if rows.stride(-1) == 1 else rows.contiguous()


# A caller-provided output that has no such row view (for example a permuted
# 3-D tensor), or is not contiguous when contiguous=True, is written through a
# contiguous temporary; _copy_back then copies it into the caller's tensor.
def _out_rows(out: Tensor, n: int, contiguous: bool = False) -> Tensor:
    if out.is_contiguous() or (
        not contiguous and out.dim() == 2 and out.stride(-1) == 1
    ):
        return out.view(-1, n)
    return torch.empty((out.numel() // n, n), dtype=out.dtype, device=out.device)


def _copy_back(out: Tensor, rows: Tensor) -> None:
    if rows.data_ptr() != out.data_ptr():
        out.copy_(rows.view(out.shape))


# The CK binding and the registered fake allocate the result with
# empty_like(input), which keeps a dense permuted layout; the Triton kernel
# returns contiguous rows.
def _like_input(input: Tensor, rows: Tensor) -> Tensor:
    y = rows.view(input.shape)
    if input.is_contiguous():
        return y
    out = torch.empty_like(input)
    if out.stride() == y.stride():
        return y
    return out.copy_(y)


# The Triton fused-add kernels index residual_in and residual_out with the input
# row stride, while CK takes one stride per tensor, so all three are passed as
# contiguous rows.
def _triton_add_rows(
    input: Tensor, residual_in: Tensor, residual_out: Tensor, x_bias: Tensor | None
) -> tuple[Tensor, Tensor, Tensor]:
    n = input.shape[-1]
    return (
        _triton_rows(input, x_bias).contiguous(),
        residual_in.reshape(-1, n).contiguous(),
        _out_rows(residual_out, n, contiguous=True),
    )


def _use_triton() -> bool:
    return not ENABLE_CK or get_gfx() == "gfx1250"


# The CK binding rejects None for weight / bias; reject it the same way here.
def _check_affine(weight: Tensor | None, bias: Tensor | None) -> None:
    if weight is None or bias is None:
        raise TypeError("LayerNorm requires weight and bias tensors")


@torch_compile_guard(gen_fake=gen_layer_norm_fake_tensors)
def layer_norm(
    input: Tensor,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    epsilon: float = 1e-5,
    x_bias: Tensor | None = None,
) -> Tensor:
    if _use_triton():
        _check_affine(weight, bias)
        from .triton.normalization.norm import layer_norm as layer_norm_triton

        out = layer_norm_triton(_triton_rows(input, x_bias), weight, bias, epsilon)
        return _like_input(input, out)
    return layer_norm_ck(input, weight, bias, epsilon, x_bias)


@torch_compile_guard(gen_fake=gen_layer_norm_fake_tensors)
def layernorm2d_fwd(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float = 1e-5,
    x_bias: Tensor | None = None,
) -> Tensor:
    if _use_triton():
        _check_affine(weight, bias)
        from .triton.normalization.norm import layer_norm as layer_norm_triton

        out = layer_norm_triton(_triton_rows(input, x_bias), weight, bias, epsilon)
        return _like_input(input, out)
    return layernorm2d_fwd_ck(input, weight, bias, epsilon, x_bias)


@torch_compile_guard(gen_fake=lambda *args, **kwargs: None)
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
    if _use_triton():
        _check_affine(weight, bias)
        from .triton.normalization.norm import (
            layernorm2d_fwd_with_add as layernorm2d_fwd_with_add_triton,
        )

        x, res_in, res_out = _triton_add_rows(input, residual_in, residual_out, x_bias)
        y = _out_rows(out, input.shape[-1])
        layernorm2d_fwd_with_add_triton(y, x, res_in, res_out, weight, bias, epsilon)
        _copy_back(out, y)
        _copy_back(residual_out, res_out)
        return
    layernorm2d_fwd_with_add_ck(
        out, input, residual_in, residual_out, weight, bias, epsilon, x_bias
    )


@torch_compile_guard(gen_fake=lambda *args, **kwargs: None)
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
    if _use_triton():
        _check_affine(weight, bias)
        from .triton.normalization.norm import (
            layernorm2d_fwd_with_smoothquant as layernorm2d_fwd_with_smoothquant_triton,
        )

        y = _out_rows(out, input.shape[-1])
        layernorm2d_fwd_with_smoothquant_triton(
            y,
            _triton_rows(input, x_bias),
            xscale,
            yscale,
            weight,
            bias,
            epsilon,
        )
        _copy_back(out, y)
        return
    layernorm2d_fwd_with_smoothquant_ck(
        out, input, xscale, yscale, weight, bias, epsilon, x_bias
    )


@torch_compile_guard(gen_fake=lambda *args, **kwargs: None)
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
    if _use_triton():
        _check_affine(weight, bias)
        from .triton.normalization.norm import (
            layernorm2d_fwd_with_add_smoothquant as layernorm2d_fwd_with_add_smoothquant_triton,
        )

        x, res_in, res_out = _triton_add_rows(input, residual_in, residual_out, x_bias)
        y = _out_rows(out, input.shape[-1])
        layernorm2d_fwd_with_add_smoothquant_triton(
            y,
            x,
            res_in,
            res_out,
            xscale,
            yscale,
            weight,
            bias,
            epsilon,
        )
        _copy_back(out, y)
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
