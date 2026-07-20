# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
from aiter.ops.triton.utils.types import str_to_torch_dtype
from aiter.ops.triton.normalization.fused_rmsnorm_add import fused_rmsnorm_add


def generate_fused_rmsnorm_add_inputs(M, N, dtype, has_res):
    x = torch.randn((M, N), dtype=dtype, device="cuda")
    weight = torch.randn(N, dtype=dtype, device="cuda")
    res1 = torch.randn((M, N), dtype=dtype, device="cuda") if has_res else None
    return x, weight, res1


def torch_rmsnorm(x, weight, epsilon):
    # compute in float32 like the triton/gluon kernel
    x_f32 = x.float()
    w_f32 = weight.float()
    N = x.shape[-1]
    var = torch.sum(x_f32 * x_f32, dim=-1, keepdim=True) / N
    rsigma = torch.rsqrt(var + epsilon)
    out = x_f32 * rsigma * w_f32
    return out.to(x.dtype)


def run_torch(x, weight, epsilon, res1=None):
    if res1 is None:
        return torch_rmsnorm(x, weight, epsilon), None
    out_res1 = x + res1
    out = torch_rmsnorm(out_res1, weight, epsilon)
    return out, out_res1


def run_triton(x, weight, epsilon, res1=None):
    if res1 is None:
        out = fused_rmsnorm_add(x, weight, epsilon, res1=None)
        return out, None
    out, out_res1 = fused_rmsnorm_add(x, weight, epsilon, res1=res1)
    return out, out_res1


def get_vals():
    vals = [
        (1, 4),
        (2, 10),
        (256, 4096),
        (4096, 8192),
        (1, 31744),
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
        (16380, 1536),
    ]
    return vals


# @pytest.mark.parametrize("in_dtype_str", ["fp16", "bf16"])
@pytest.mark.parametrize("in_dtype_str", ["bf16"])
@pytest.mark.parametrize("has_res", [True, False])
@pytest.mark.parametrize(
    "M, N",
    [(shape) for shape in get_vals()],
)
def test_fused_rmsnorm_add(M, N, has_res, in_dtype_str):
    in_dtype = str_to_torch_dtype[in_dtype_str]
    out_dtype = in_dtype
    torch.manual_seed(0)

    x, weight, res1 = generate_fused_rmsnorm_add_inputs(M, N, in_dtype, has_res)
    epsilon = 1e-5

    y_torch, res_torch = run_torch(x, weight, epsilon, res1=res1)
    y_triton, res_triton = run_triton(x, weight, epsilon, res1=res1)

    atol, rtol = 1e-2, 1e-2

    assert (
        y_triton.dtype == out_dtype
    ), f"y_triton has dtype={y_triton.dtype}, expected {out_dtype}"

    torch.testing.assert_close(y_triton, y_torch, atol=atol, rtol=rtol)
    if has_res:
        torch.testing.assert_close(res_triton, res_torch, atol=atol, rtol=rtol)
