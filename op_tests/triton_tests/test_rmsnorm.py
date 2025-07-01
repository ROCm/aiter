# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton
import aiter
from aiter.ops.triton.rmsnorm import (
    rms_norm,
    rmsnorm2d_fwd_with_add,
    rmsnorm2d_fwd_with_smoothquant,
    rmsnorm2d_fwd_with_dynamicquant,
    rmsnorm2d_fwd_with_add_smoothquant,
    rmsnorm2d_fwd_with_add_dynamicquant,
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


def run_torch(input, weight, eps, residual=None, x_scale=None, y_scale_dtype=None):
    if residual is None:
        residual_out = None
        output = torch_rmsnorm(input, weight, input.dtype, eps)
    else:
        residual_out = input + residual
        output = torch_rmsnorm(residual_out, weight, residual_out.dtype, eps)
    if y_scale_dtype is None:
        y_scale = None
        output_q = output
    else:
        output_q, y_scale = aiter.pertoken_quant(output, x_scale=x_scale)
    return output_q, residual_out, y_scale, output


def run_triton(input, weight, eps, residual=None, x_scale=None, y_scale_dtype=None):
    # out_before_quant = None
    if y_scale_dtype is None:
        y_scale = None
        if residual is None:
            residual_out = None
            output = rms_norm(input, weight, eps)
        else:
            residual_out = torch.empty_like(input)
            output = torch.empty_like(input)
            output = rmsnorm2d_fwd_with_add(
                output, input, residual, residual_out, weight, eps
            )
    elif x_scale is None:
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")
        output = torch.empty(input.shape, dtype=torch.int8, device="cuda")
        if residual is None:
            residual_out = None
            rmsnorm2d_fwd_with_dynamicquant(output, input, y_scale, weight, eps)
        elif residual is not None:
            residual_out = torch.empty_like(input)
            rmsnorm2d_fwd_with_add_dynamicquant(
                output, input, residual, residual_out, y_scale, weight, eps
            )
    else:
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")
        output = torch.empty(input.shape, dtype=torch.int8, device="cuda")
        if residual is None:
            residual_out = None
            rmsnorm2d_fwd_with_smoothquant(output, input, x_scale, y_scale, weight, eps)
        else:
            residual_out = torch.empty_like(input)
            # out_before_quant = torch.empty_like(input)
            rmsnorm2d_fwd_with_add_smoothquant(
                output,
                input,
                residual,
                residual_out,
                x_scale,
                y_scale,
                weight,
                eps,
                # out_before_quant=out_before_quant,
            )
    return output, residual_out, y_scale  # , out_before_quant


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


@pytest.mark.parametrize("in_dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("scale_dtype_str", ["fp32"])
@pytest.mark.parametrize(
    "M, N",
    [(shape) for shape in get_vals()],
)
def test_rmsnorm_smoothquant(M, N, in_dtype_str, scale_dtype_str):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    in_dtype = arg_to_torch_dtype[in_dtype_str]
    scale_dtype = arg_to_torch_dtype[scale_dtype_str]

    torch.manual_seed(0)

    x = torch.randn(M, N, device="cuda", dtype=in_dtype)
    weight = torch.randn(N, device="cuda", dtype=in_dtype)
    x_scale = torch.randn(N, device="cuda", dtype=scale_dtype)

    (y_torch, _, yscale_torch, *_) = run_torch(
        x, weight, 1e-5, x_scale=x_scale, y_scale_dtype=scale_dtype
    )
    (y_triton, _, yscale_triton, *_) = run_triton(
        x, weight, 1e-5, x_scale=x_scale, y_scale_dtype=scale_dtype
    )

    triton.testing.assert_close(y_triton, y_torch, atol=1, rtol=0)
    triton.testing.assert_close(yscale_triton, yscale_torch, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("in_dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("scale_dtype_str", ["fp32"])
@pytest.mark.parametrize(
    "M, N",
    [(shape) for shape in get_vals()],
)
def test_rmsnorm_dynamicquant(M, N, in_dtype_str, scale_dtype_str):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    in_dtype = arg_to_torch_dtype[in_dtype_str]
    scale_dtype = arg_to_torch_dtype[scale_dtype_str]

    torch.manual_seed(0)

    x = torch.randn(M, N, device="cuda", dtype=in_dtype)
    weight = torch.randn(N, device="cuda", dtype=in_dtype)

    (y_torch, _, yscale_torch, *_) = run_torch(
        x, weight, 1e-5, y_scale_dtype=scale_dtype
    )
    (y_triton, _, yscale_triton, *_) = run_triton(
        x, weight, 1e-5, y_scale_dtype=scale_dtype
    )

    triton.testing.assert_close(y_triton, y_torch, atol=1, rtol=0)
    triton.testing.assert_close(yscale_triton, yscale_torch, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("in_dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("scale_dtype_str", ["fp32"])
@pytest.mark.parametrize(
    "M, N",
    [(shape) for shape in get_vals()],
)
def test_rmsnorm_fused_add_smoothquant(M, N, in_dtype_str, scale_dtype_str):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    in_dtype = arg_to_torch_dtype[in_dtype_str]
    scale_dtype = arg_to_torch_dtype[scale_dtype_str]

    torch.manual_seed(0)

    x = torch.randn(M, N, device="cuda", dtype=in_dtype)
    weight = torch.randn(N, device="cuda", dtype=in_dtype)
    res = torch.randn(M, N, device="cuda", dtype=in_dtype)
    x_scale = torch.randn(N, device="cuda", dtype=scale_dtype)

    (y_torch, res_torch, yscale_torch, *_) = run_torch(
        x, weight, 1e-5, residual=res, x_scale=x_scale, y_scale_dtype=scale_dtype
    )
    (y_triton, res_triton, yscale_triton, *_) = run_triton(
        x, weight, 1e-5, residual=res, x_scale=x_scale, y_scale_dtype=scale_dtype
    )

    triton.testing.assert_close(y_triton, y_torch, atol=1, rtol=0)
    triton.testing.assert_close(res_triton, res_torch, atol=1e-3, rtol=1e-3)
    triton.testing.assert_close(yscale_triton, yscale_torch, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("in_dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("scale_dtype_str", ["fp32"])
@pytest.mark.parametrize(
    "M, N",
    [(shape) for shape in get_vals()],
)
def test_rmsnorm_fused_add_dynamicquant(M, N, in_dtype_str, scale_dtype_str):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    in_dtype = arg_to_torch_dtype[in_dtype_str]
    scale_dtype = arg_to_torch_dtype[scale_dtype_str]

    torch.manual_seed(0)

    x = torch.randn(M, N, device="cuda", dtype=in_dtype)
    weight = torch.randn(N, device="cuda", dtype=in_dtype)
    res = torch.randn(M, N, device="cuda", dtype=in_dtype)

    (y_torch, res_torch, yscale_torch, *_) = run_torch(
        x, weight, 1e-5, residual=res, y_scale_dtype=scale_dtype
    )
    (y_triton, res_triton, yscale_triton, *_) = run_triton(
        x, weight, 1e-5, residual=res, y_scale_dtype=scale_dtype
    )

    triton.testing.assert_close(y_triton, y_torch, atol=1, rtol=0)
    triton.testing.assert_close(res_triton, res_torch, atol=1e-3, rtol=1e-3)
    triton.testing.assert_close(yscale_triton, yscale_torch, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("B", [1, 4, 8])
@pytest.mark.parametrize("T", [128, 512, 2048])
@pytest.mark.parametrize("D", [64, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_dynamic_per_token_fp8_quant(
    B: int, T: int, D: int, dtype: torch.dtype
) -> None:
    B_T = B * T
    # Use integers to ensure consistent results across layouts,
    # avoiding discrepancies in floating-point reductions with varying data layouts
    x = torch.floor(torch.distributions.Uniform(-3, 3).sample((B_T, D))).to(
        dtype=dtype, device="cuda"
    )
    w = torch.floor(torch.distributions.Uniform(-3, 3).sample((D,))).to(
        dtype=dtype, device="cuda"
    )

    EPS = 1e-6
    quant_dtype = torch.float8_e4m3fnuz

    xq_fused_triton = torch.empty(x.shape, dtype=quant_dtype, device="cuda")
    x_scale_fused = torch.empty(x.shape[0], 1, dtype=torch.float32, device="cuda")

    x_normed = rmsnorm2d_fwd_with_dynamicquant(
        xq_fused_triton, x, x_scale_fused, w, EPS, dump_rms_norm=True
    )

    ref_x_normed = torch_rmsnorm(x, w, dtype, EPS)
    ref_xq, ref_x_scale = aiter.pertoken_quant(ref_x_normed, quant_dtype=quant_dtype)

    xq_dequant = xq_fused_triton.to(torch.float32) * x_scale_fused
    xq_dequant = xq_dequant.to(dtype)
    ref_xq_dequant = ref_xq.to(torch.float32) * ref_x_scale
    ref_xq_dequant = xq_dequant.to(dtype)

    if dtype == torch.float32:
        atol = 1e-5
        rtol = 1e-5
    else:
        atol = 1e-2
        rtol = 1e-2

    torch.testing.assert_close(xq_dequant, ref_xq_dequant, atol=atol, rtol=rtol)
    torch.testing.assert_close(x_normed, ref_x_normed, atol=atol, rtol=rtol)
