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
    return added.to(torch.bfloat16), pre


@pytest.mark.parametrize("rows, width", [(1, 6656), (64, 6656), (4096, 6656), (7, 257)])
@pytest.mark.parametrize("provide_out", [False, True])
def test_fused_rmsnorm_add_rmsnorm(rows, width, provide_out):
    torch.manual_seed(1064)
    x = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
    post_weight = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1
    pre_weight = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1

    expected_residual, expected_norm = reference(
        x, residual, post_weight, pre_weight
    )
    residual_out = torch.empty_like(x)
    out = torch.empty_like(x) if provide_out else None
    pre_norm = fused_rmsnorm_add_rmsnorm(
        x, residual, post_weight, pre_weight, POST_EPS, PRE_EPS, residual_out, out
    )

    if provide_out:
        assert pre_norm is out
    assert residual_out.dtype == torch.bfloat16
    assert pre_norm.dtype == torch.bfloat16
    torch.testing.assert_close(residual_out, expected_residual, atol=0.02, rtol=0.02)
    torch.testing.assert_close(pre_norm, expected_norm, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("provide_out", [False, True])
def test_fused_rmsnorm_add_rmsnorm_torch_compile(provide_out):
    x = torch.randn((64, 6656), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.zeros(6656, device="cuda", dtype=torch.bfloat16)
    compiled_residual = torch.empty_like(x)
    eager_residual = torch.empty_like(x)
    compiled_out = torch.empty_like(x) if provide_out else None
    eager_out = torch.empty_like(x) if provide_out else None
    compiled = torch.compile(fused_rmsnorm_add_rmsnorm, fullgraph=True)
    actual = compiled(
        x, residual, weight, weight, POST_EPS, PRE_EPS, compiled_residual,
        compiled_out,
    )
    expected = fused_rmsnorm_add_rmsnorm(
        x, residual, weight, weight, POST_EPS, PRE_EPS, eager_residual, eager_out
    )
    if provide_out:
        assert actual is compiled_out
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(compiled_residual, eager_residual, atol=0, rtol=0)
