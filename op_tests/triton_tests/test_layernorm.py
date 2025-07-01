# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import torch
import torch.nn.functional as F
import pytest
import aiter
from aiter.ops.triton.norm import (
    layer_norm,
    layernorm2d_fwd_with_add,
    layernorm2d_fwd_with_dynamicquant,
    layernorm2d_fwd_with_smoothquant,
    layernorm2d_fwd_with_add_dynamicquant,
    layernorm2d_fwd_with_add_smoothquant,
)


def run_torch(
    input, weight, bias, eps, residual=None, x_scale=None, y_scale_dtype=None
):
    if residual is None:
        residual_out = None
        output = F.layer_norm(
            input=input,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps,
        )
    else:
        residual_out = input + residual
        output = F.layer_norm(
            input=residual_out,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps,
        )
    if y_scale_dtype is None:
        y_scale = None
    else:
        output, y_scale = aiter.pertoken_quant(
            output, x_scale=x_scale, quant_dtype=torch.int8
        )
    return output, residual_out, y_scale


def run_triton(
    input,
    weight,
    bias,
    eps,
    residual=None,
    x_bias=None,
    x_scale=None,
    y_scale_dtype=None,
):
    if y_scale_dtype is None:
        y_scale = None
        if residual is None:
            residual_out = None
            output = layer_norm(input, weight, bias, eps, x_bias)
        else:
            residual_out = torch.empty_like(input)
            output = torch.empty_like(input)
            layernorm2d_fwd_with_add(
                output, input, residual, residual_out, weight, bias, eps, x_bias
            )
    elif x_scale is None:
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")
        output = torch.empty(input.shape, dtype=torch.int8, device="cuda")
        if residual is None:
            residual_out = None
            layernorm2d_fwd_with_dynamicquant(output, input, y_scale, weight, bias, eps)
        elif residual is not None:
            residual_out = torch.empty_like(input)
            layernorm2d_fwd_with_add_dynamicquant(
                output, input, residual, residual_out, y_scale, weight, bias, eps
            )
    else:
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")
        output = torch.empty(input.shape, dtype=torch.int8, device="cuda")
        if residual is None:
            residual_out = None
            layernorm2d_fwd_with_smoothquant(
                output, input, x_scale, y_scale, weight, bias, eps
            )
        elif residual is not None:
            residual_out = torch.empty_like(input)
            layernorm2d_fwd_with_add_smoothquant(
                output,
                input,
                residual,
                residual_out,
                x_scale,
                y_scale,
                weight,
                bias,
                eps,
            )

    return output, residual_out, y_scale


# TODO: Enable the commented shapes once the bug
# discussed in this issue is solved:
# https://github.com/ROCm/triton-internal/issues/843
def get_vals():

    vals = [
        # (1823, 781),
        (2, 128),
        (1, 4),
        (128, 2),
        (1, 128),
        # (8192, 8192),
        # (4096, 8192),
        (359, 1),
        (1, 359),
        (1, 131072),
        (1, 89999),
    ]

    return vals


# pytest
@pytest.mark.parametrize("dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N",
    [(shape) for shape in get_vals()],
)
def test_layernorm(M, N, dtype_str, eps=1e-5):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = arg_to_torch_dtype[dtype_str]
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w_shape = (N,)
    b = torch.rand(w_shape, device="cuda", dtype=dtype)
    w = torch.rand(w_shape, device="cuda", dtype=dtype)

    # forward pass
    y_torch, *_ = run_torch(x, w, b, eps)
    y_triton, *_ = run_triton(x, w, b, eps)

    if dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-5, 1e-5

    triton.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)


# pytest
@pytest.mark.parametrize("dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N",
    [(shape) for shape in get_vals()],
)
def test_fused_add_layernorm(M, N, dtype_str, eps=1e-5):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = arg_to_torch_dtype[dtype_str]
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    res = torch.randn(M, N, device="cuda", dtype=dtype)
    w_shape = (N,)
    w = torch.rand(w_shape, device="cuda", dtype=dtype)
    b = torch.rand(w_shape, device="cuda", dtype=dtype)

    # forward pass
    y_torch, res_torch, *_ = run_torch(x, w, b, eps, residual=res)
    y_triton, res_triton, *_ = run_triton(x, w, b, eps, residual=res)

    if dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-5, 1e-5

    triton.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)
    triton.testing.assert_close(res_triton, res_torch, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("scale_dtype_str", ["fp32"])
@pytest.mark.parametrize(
    "M, N",
    [(shape) for shape in get_vals()],
)
def test_layernorm_smoothquant(M, N, dtype_str, scale_dtype_str, eps=1e-5):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = arg_to_torch_dtype[dtype_str]
    scale_dtype = arg_to_torch_dtype[scale_dtype_str]
    torch.manual_seed(0)

    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w_shape = (N,)
    b = torch.rand(w_shape, device="cuda", dtype=dtype)
    w = torch.rand(w_shape, device="cuda", dtype=dtype)
    x_scale = torch.rand(w_shape, device="cuda", dtype=scale_dtype)

    y_torch, _, y_scale_torch = run_torch(
        x, w, b, eps, x_scale=x_scale, y_scale_dtype=scale_dtype
    )
    y_triton, _, y_scale_triton = run_triton(
        x, w, b, eps, x_scale=x_scale, y_scale_dtype=scale_dtype
    )

    xq_dequant = y_triton.to(torch.int32) * y_scale_triton
    xq_dequant = xq_dequant.to(dtype)
    ref_xq_dequant = y_torch.to(torch.int32) * y_scale_torch
    ref_xq_dequant = xq_dequant.to(dtype)

    if dtype == torch.float32:
        atol = 1e-5
        rtol = 1e-5
    else:
        atol = 1e-2
        rtol = 1e-2

    triton.testing.assert_close(xq_dequant, ref_xq_dequant, atol=atol, rtol=rtol)
    triton.testing.assert_close(y_triton, y_torch, atol=1, rtol=0)
    triton.testing.assert_close(y_scale_triton, y_scale_torch, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("scale_dtype_str", ["fp32"])
@pytest.mark.parametrize(
    "M, N",
    [(shape) for shape in get_vals()],
)
def test_layernorm_dynamicquant(M, N, dtype_str, scale_dtype_str, eps=1e-3):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = arg_to_torch_dtype[dtype_str]
    scale_dtype = arg_to_torch_dtype[scale_dtype_str]
    torch.manual_seed(0)

    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w_shape = (N,)
    b = torch.rand(w_shape, device="cuda", dtype=dtype)
    w = torch.rand(w_shape, device="cuda", dtype=dtype)

    # forward pass
    y_torch, _, y_scale_torch = run_torch(x, w, b, eps, y_scale_dtype=scale_dtype)
    y_triton, _, y_scale_triton = run_triton(x, w, b, eps, y_scale_dtype=scale_dtype)

    xq_dequant = y_triton.to(torch.int32) * y_scale_triton
    xq_dequant = xq_dequant.to(dtype)
    ref_xq_dequant = y_torch.to(torch.int32) * y_scale_torch
    ref_xq_dequant = xq_dequant.to(dtype)

    if dtype == torch.float32:
        atol = 1e-5
        rtol = 1e-5
    else:
        atol = 1e-2
        rtol = 1e-2

    triton.testing.assert_close(xq_dequant, ref_xq_dequant, atol=atol, rtol=rtol)
    triton.testing.assert_close(y_triton, y_torch, atol=1, rtol=0)
    triton.testing.assert_close(y_scale_triton, y_scale_torch, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("scale_dtype_str", ["fp32"])
@pytest.mark.parametrize(
    "M, N",
    [(shape) for shape in get_vals()],
)
def test_layernorm_fused_add_smoothquant(M, N, dtype_str, scale_dtype_str, eps=1e-5):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = arg_to_torch_dtype[dtype_str]
    scale_dtype = arg_to_torch_dtype[scale_dtype_str]
    torch.manual_seed(0)

    x = torch.randn(M, N, device="cuda", dtype=dtype)
    res = torch.randn(M, N, device="cuda", dtype=dtype)
    w_shape = (N,)
    b = torch.rand(w_shape, device="cuda", dtype=dtype)
    w = torch.rand(w_shape, device="cuda", dtype=dtype)
    x_scale = torch.rand(w_shape, device="cuda", dtype=scale_dtype)

    y_torch, res_torch, y_scale_torch = run_torch(
        x, w, b, eps, residual=res, x_scale=x_scale, y_scale_dtype=scale_dtype
    )
    y_triton, res_triton, y_scale_triton = run_triton(
        x, w, b, eps, residual=res, x_scale=x_scale, y_scale_dtype=scale_dtype
    )

    xq_dequant = y_triton.to(torch.int32) * y_scale_triton
    xq_dequant = xq_dequant.to(dtype)
    ref_xq_dequant = y_torch.to(torch.int32) * y_scale_torch
    ref_xq_dequant = xq_dequant.to(dtype)

    if dtype == torch.float32:
        atol = 1e-5
        rtol = 1e-5
    else:
        atol = 1e-2
        rtol = 1e-2

    triton.testing.assert_close(xq_dequant, ref_xq_dequant, atol=atol, rtol=rtol)
    triton.testing.assert_close(y_triton, y_torch, atol=1, rtol=0)
    triton.testing.assert_close(res_triton, res_torch, atol=atol, rtol=rtol)
    triton.testing.assert_close(y_scale_triton, y_scale_torch, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("scale_dtype_str", ["fp32"])
@pytest.mark.parametrize(
    "M, N",
    [(shape) for shape in get_vals()],
)
def test_layernorm_fused_add_dynamicquant(M, N, dtype_str, scale_dtype_str, eps=1e-3):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = arg_to_torch_dtype[dtype_str]
    scale_dtype = arg_to_torch_dtype[scale_dtype_str]
    torch.manual_seed(0)

    x = torch.randn(M, N, device="cuda", dtype=dtype)
    res = torch.randn(M, N, device="cuda", dtype=dtype)
    w_shape = (N,)
    b = torch.rand(w_shape, device="cuda", dtype=dtype)
    w = torch.rand(w_shape, device="cuda", dtype=dtype)

    # forward pass
    y_torch, res_torch, y_scale_torch = run_torch(
        x, w, b, eps, residual=res, y_scale_dtype=scale_dtype
    )
    y_triton, res_triton, y_scale_triton = run_triton(
        x, w, b, eps, residual=res, y_scale_dtype=scale_dtype
    )

    xq_dequant = y_triton.to(torch.int32) * y_scale_triton
    xq_dequant = xq_dequant.to(dtype)
    ref_xq_dequant = y_torch.to(torch.int32) * y_scale_torch
    ref_xq_dequant = xq_dequant.to(dtype)

    if dtype == torch.float32:
        atol = 1e-5
        rtol = 1e-5
    else:
        atol = 1e-2
        rtol = 1e-2

    triton.testing.assert_close(xq_dequant, ref_xq_dequant, atol=atol, rtol=rtol)
    triton.testing.assert_close(y_triton, y_torch, atol=1, rtol=0)
    triton.testing.assert_close(res_triton, res_torch, atol=atol, rtol=rtol)
    triton.testing.assert_close(y_scale_triton, y_scale_torch, atol=1e-3, rtol=1e-3)
