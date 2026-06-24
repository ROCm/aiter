# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Backend tests for explicit FlyDSL LayerNorm APIs."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flydsl")
from aiter.ops.flydsl import (  # noqa: E402
    flydsl_add_layernorm,
    flydsl_add_layernorm_dynamicquant,
    flydsl_add_layernorm_smoothquant,
    flydsl_layernorm,
    flydsl_layernorm_dynamicquant,
    flydsl_layernorm_smoothquant,
    is_flydsl_available,
)

if not is_flydsl_available():
    pytest.skip("flydsl is not available", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm is not available", allow_module_level=True)


def _make_input(m: int, n: int, dtype: torch.dtype, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn((m, n), device="cuda", dtype=torch.float32).to(dtype).contiguous()
    weight = torch.rand((n,), device="cuda", dtype=torch.float32).to(dtype).contiguous()
    bias = torch.rand((n,), device="cuda", dtype=torch.float32).to(dtype).contiguous()
    return x, weight, bias


def _reference_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
):
    x_f32 = x.float()
    weight_f32 = weight.float()
    bias_f32 = bias.float()
    mean = x_f32.mean(dim=1, keepdim=True)
    var = x_f32.var(dim=1, keepdim=True, unbiased=False)
    return (x_f32 - mean) * torch.rsqrt(var + eps) * weight_f32 + bias_f32


def _reference_quant(y: torch.Tensor):
    yscale = y.abs().amax(dim=1) / 127.0
    yscale = torch.where(yscale == 0, torch.ones_like(yscale), yscale)
    q = torch.clamp(torch.trunc(y / yscale.unsqueeze(1)), -127, 127).to(torch.int8)
    return q, yscale


def _assert_norm_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype):
    if dtype is torch.float32:
        atol, rtol = 1e-4, 1e-4
    elif dtype is torch.float16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 2e-2, 2e-2
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "m,n,dtype,eps",
    [
        (4, 2000, torch.float32, 1e-6),  # generic path + non-default epsilon
        (4, 2000, torch.float16, 1e-5),
        (2, 8192, torch.bfloat16, 1e-5),  # vectorized fast path
    ],
)
def test_flydsl_layernorm_matches_torch(m: int, n: int, dtype: torch.dtype, eps: float):
    x, weight, bias = _make_input(m, n, dtype)
    out = flydsl_layernorm(x, weight, bias, epsilon=eps)
    ref = _reference_layernorm(x, weight, bias, eps)
    assert out.shape == x.shape
    assert out.dtype == dtype
    _assert_norm_close(out, ref, dtype)


def test_flydsl_add_layernorm_matches_torch():
    m, n, dtype, eps = 4, 2000, torch.bfloat16, 1e-6
    x, weight, bias = _make_input(m, n, dtype, seed=1)
    residual = torch.randn_like(x).contiguous()

    out, residual_out = flydsl_add_layernorm(x, residual, weight, bias, epsilon=eps)
    residual_ref = (x + residual).float()
    ref = _reference_layernorm(x + residual, weight, bias, eps)

    assert out.shape == residual_out.shape == x.shape
    _assert_norm_close(residual_out, residual_ref, dtype)
    _assert_norm_close(out, ref, dtype)


@pytest.mark.parametrize("smooth", [False, True])
def test_flydsl_layernorm_quant_matches_torch(smooth: bool):
    m, n, dtype, eps = 4, 2000, torch.float16, 1e-6
    x, weight, bias = _make_input(m, n, dtype, seed=2)
    xscale = (torch.rand((n,), device="cuda", dtype=dtype) + 0.5).contiguous()

    if smooth:
        out, yscale = flydsl_layernorm_smoothquant(x, weight, bias, xscale, epsilon=eps)
        ref_y = _reference_layernorm(x, weight, bias, eps) * xscale.float()
    else:
        out, yscale = flydsl_layernorm_dynamicquant(x, weight, bias, epsilon=eps)
        ref_y = _reference_layernorm(x, weight, bias, eps)
    q_ref, yscale_ref = _reference_quant(ref_y)

    assert out.dtype == torch.int8
    assert yscale.shape == (m,)
    assert (out.to(torch.int16) - q_ref.to(torch.int16)).abs().max().item() <= 1
    torch.testing.assert_close(yscale.cpu(), yscale_ref.cpu(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("smooth", [False, True])
def test_flydsl_add_layernorm_quant_matches_torch(smooth: bool):
    m, n, dtype, eps = 4, 2000, torch.float16, 1e-6
    x, weight, bias = _make_input(m, n, dtype, seed=3)
    residual = torch.randn_like(x).contiguous()
    xscale = (torch.rand((n,), device="cuda", dtype=dtype) + 0.5).contiguous()

    if smooth:
        out, residual_out, yscale = flydsl_add_layernorm_smoothquant(
            x,
            residual,
            weight,
            bias,
            xscale,
            epsilon=eps,
        )
        ref_y = _reference_layernorm(x + residual, weight, bias, eps) * xscale.float()
    else:
        out, residual_out, yscale = flydsl_add_layernorm_dynamicquant(
            x,
            residual,
            weight,
            bias,
            epsilon=eps,
        )
        ref_y = _reference_layernorm(x + residual, weight, bias, eps)
    q_ref, yscale_ref = _reference_quant(ref_y)

    _assert_norm_close(residual_out, (x + residual).float(), dtype)
    assert out.dtype == torch.int8
    assert yscale.shape == (m,)
    assert (out.to(torch.int16) - q_ref.to(torch.int16)).abs().max().item() <= 1
    torch.testing.assert_close(yscale.cpu(), yscale_ref.cpu(), atol=1e-3, rtol=1e-3)
