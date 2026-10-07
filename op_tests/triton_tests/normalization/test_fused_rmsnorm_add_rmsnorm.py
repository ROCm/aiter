# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.normalization.fused_rmsnorm_add_rmsnorm import (
    fused_rmsnorm_add_rmsnorm,
)


POST_EPS = 1e-8
PRE_EPS = 1e-5


def reference(x, residual, post_weight, pre_weight):
    x_f32 = x.float()
    post = (
        x_f32
        * torch.rsqrt(x_f32.square().mean(-1, keepdim=True) + POST_EPS)
        * (post_weight.float() + 1.0)
    )
    added = post + residual.float()
    pre = (
        added
        * torch.rsqrt(added.square().mean(-1, keepdim=True) + PRE_EPS)
        * (pre_weight.float() + 1.0)
    ).to(torch.bfloat16)
    return added, pre


@pytest.mark.parametrize("rows, width", [(1, 6656), (64, 6656), (4096, 6656), (7, 257)])
@pytest.mark.parametrize("residual_dtype", [torch.bfloat16, torch.float32])
def test_fused_rmsnorm_add_rmsnorm(rows, width, residual_dtype):
    torch.manual_seed(1064)
    x = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((rows, width), device="cuda", dtype=residual_dtype)
    post_weight = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1
    pre_weight = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1

    expected_residual, expected_norm = reference(
        x, residual, post_weight, pre_weight
    )
    residual_out, pre_norm = fused_rmsnorm_add_rmsnorm(
        x, residual, post_weight, pre_weight, POST_EPS, PRE_EPS
    )

    assert residual_out.dtype == torch.float32
    assert pre_norm.dtype == torch.bfloat16
    torch.testing.assert_close(residual_out, expected_residual, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(pre_norm, expected_norm, atol=0.02, rtol=0.02)


def test_fused_rmsnorm_add_rmsnorm_torch_compile():
    x = torch.randn((64, 6656), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.zeros(6656, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(fused_rmsnorm_add_rmsnorm, fullgraph=True)
    actual = compiled(x, residual, weight, weight, POST_EPS, PRE_EPS)
    expected = fused_rmsnorm_add_rmsnorm(
        x, residual, weight, weight, POST_EPS, PRE_EPS
    )
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
