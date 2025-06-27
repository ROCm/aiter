# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton
from aiter.ops.triton.rmsnorm import (
    rms_norm,
    rmsnorm2d_fwd_with_add,
)


def generate_rmsnorm_inputs(M, N, dtype):
    x = torch.randn((M, N), dtype=dtype).cuda()
    weight = torch.randn(N, dtype=dtype).cuda()

    return x, weight


def torch_rmsnorm(x, g, out_dtype=torch.float16, epsilon=1e-6):
    M, N = x.shape
    # cast to float32 as the triton kernel
    x_f32 = x.float()
    g_f32 = g.float()
    rms = torch.sqrt(torch.sum(x_f32 * x_f32, dim=-1) * 1 / N)
    rsigma = 1.0 / rms
    rms_norm_f32 = x_f32 * rsigma.unsqueeze(1) * g_f32
    rms_norm = rms_norm_f32.to(out_dtype)
    return rms_norm


def run_torch(x, weight, eps, residual=None):
    if residual is None:
        residual_out = None
        output = torch_rmsnorm(x, weight, x.dtype, eps)
    else:
        residual_out = x + residual
        output = torch_rmsnorm(residual_out, weight, residual_out.dtype, eps)
    return output, residual_out


def run_triton(x, weight, eps, residual=None):
    if residual is None:
        residual_out = None
        output = rms_norm(x, weight, eps)
    else:
        residual_out = torch.empty_like(x)
        output = torch.empty_like(x)
        output = rmsnorm2d_fwd_with_add(output, x, residual, residual_out, weight, eps)
    return output, residual_out


def get_vals():

    vals = [
        (1, 4),
        (2, 10),
        (256, 4096),
        (4096, 8192),
        (1, 31744),
        (8192, 65536),
        (873, 1245),
        (4096, 5120),
        (8192, 8192),
        (2048, 4096),
        (768, 2048),
        (256, 1024),
        (128, 768),
        (64, 512),
        (173, 409),
        (71, 3571),
        (29, 17389),
    ]

    return vals


@pytest.mark.parametrize("in_dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N",
    [(shape) for shape in get_vals()],
)
def test_rmsnorm(M, N, in_dtype_str):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    in_dtype = arg_to_torch_dtype[in_dtype_str]
    out_dtype = in_dtype
    torch.manual_seed(0)

    x, weight = generate_rmsnorm_inputs(M, N, in_dtype)

    dy = torch.randn_like(x)
    x.requires_grad_(True)
    weight.requires_grad_(True)

    # forward pass
    y_torch, *_ = run_torch(x, weight, 1e-5)
    y_triton, *_ = run_triton(x, weight, 1e-5)

    # backward pass (triton)
    y_triton.backward(dy, retain_graph=True)
    dx_triton, dg_triton = [_.grad.clone() for _ in [x, weight]]
    x.grad, weight.grad = None, None

    # backward pass (torch)
    y_torch.backward(dy, retain_graph=True)
    dx_torch, dg_torch = [_.grad.clone() for _ in [x, weight]]

    if out_dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-4, 1e-4

    assert (
        y_triton.dtype == out_dtype
    ), f"y_triton has dtype={y_triton.dtype}, expected {out_dtype}"
    assert (
        y_torch.dtype == out_dtype
    ), f"y_torch has dtype={y_torch.dtype}, expected {out_dtype}"

    triton.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)
    triton.testing.assert_close(dx_triton, dx_torch, rtol=rtol, atol=atol)
    triton.testing.assert_close(dg_triton, dg_torch, rtol=rtol, atol=atol)


@pytest.mark.parametrize("in_dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N",
    [(shape) for shape in get_vals()],
)
def test_fused_add_rmsnorm(M, N, in_dtype_str):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    in_dtype = arg_to_torch_dtype[in_dtype_str]
    out_dtype = in_dtype
    torch.manual_seed(0)

    x = torch.randn(M, N, device="cuda", dtype=in_dtype)
    weight = torch.randn(N, device="cuda", dtype=in_dtype)
    res = torch.randn(M, N, device="cuda", dtype=in_dtype)

    dy = torch.randn_like(x)
    x.requires_grad_(True)
    weight.requires_grad_(True)

    # forward pass
    y_torch, res_torch, *_ = run_torch(x, weight, 1e-5, residual=res)
    y_triton, res_triton, *_ = run_triton(x, weight, 1e-5, residual=res)

    # backward pass (triton)
    y_triton.backward(dy, retain_graph=True)
    dx_triton, dg_triton = [_.grad.clone() for _ in [x, weight]]
    x.grad, weight.grad = None, None

    # backward pass (torch)
    y_torch.backward(dy, retain_graph=True)
    dx_torch, dg_torch = [_.grad.clone() for _ in [x, weight]]

    if out_dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-4, 1e-4

    assert (
        y_triton.dtype == out_dtype
    ), f"y_triton has dtype={y_triton.dtype}, expected {out_dtype}"
    assert (
        y_torch.dtype == out_dtype
    ), f"y_torch has dtype={y_torch.dtype}, expected {out_dtype}"

    triton.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)
    triton.testing.assert_close(res_triton, res_torch, atol=atol, rtol=rtol)
    triton.testing.assert_close(dx_triton, dx_torch, rtol=rtol, atol=atol)
    triton.testing.assert_close(dg_triton, dg_torch, rtol=rtol, atol=atol)
