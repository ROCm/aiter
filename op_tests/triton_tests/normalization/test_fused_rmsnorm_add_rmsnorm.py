# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.normalization.fused_rmsnorm_add_rmsnorm import (
    fused_rmsnorm_add_rmsnorm,
)

POST_EPS = 1e-8
PRE_EPS = 1e-5


def generate_fused_rmsnorm_add_rmsnorm_inputs(rows, width):
    torch.manual_seed(1064)
    x = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
    post_weight = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1
    pre_weight = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1
    return x, residual, post_weight, pre_weight


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
def test_fused_rmsnorm_add_rmsnorm(rows, width):
    x, residual, post_weight, pre_weight = generate_fused_rmsnorm_add_rmsnorm_inputs(
        rows, width
    )

    expected_residual, expected_norm = reference(x, residual, post_weight, pre_weight)
    residual_out = torch.empty_like(x)
    pre_norm = fused_rmsnorm_add_rmsnorm(
        x, residual, post_weight, pre_weight, POST_EPS, PRE_EPS, residual_out
    )

    assert residual_out.dtype == torch.bfloat16
    assert pre_norm.dtype == torch.bfloat16
    torch.testing.assert_close(residual_out, expected_residual, atol=0.02, rtol=0.02)
    torch.testing.assert_close(pre_norm, expected_norm, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("rows, width", [(64, 6656), (256, 257)])
def test_fused_rmsnorm_add_rmsnorm_torch_compile(rows, width):
    x = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.zeros(width, device="cuda", dtype=torch.bfloat16)
    compiled_residual = torch.empty_like(x)
    eager_residual = torch.empty_like(x)
    compiled = torch.compile(fused_rmsnorm_add_rmsnorm, fullgraph=True)
    actual = compiled(x, residual, weight, weight, POST_EPS, PRE_EPS, compiled_residual)
    expected = fused_rmsnorm_add_rmsnorm(
        x, residual, weight, weight, POST_EPS, PRE_EPS, eager_residual
    )
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(compiled_residual, eager_residual, atol=0.02, rtol=0.02)


@pytest.mark.parametrize(
    "case,name",
    [
        ("fp32_output", "residual_out"),
        ("strided_output", "residual_out"),
        ("short_post_weight", "post_weight"),
        ("short_pre_weight", "pre_weight"),
        ("short_residual", "residual"),
        ("cpu_weight", "pre_weight"),
    ],
)
def test_fused_rmsnorm_add_rmsnorm_rejects_invalid_inputs(case, name):
    x, residual, post_weight, pre_weight = generate_fused_rmsnorm_add_rmsnorm_inputs(
        2, 16
    )
    args = {
        "x": x,
        "residual": residual,
        "post_weight": post_weight,
        "pre_weight": pre_weight,
        "residual_out": torch.empty_like(x),
    }
    if case == "fp32_output":
        args["residual_out"] = torch.empty_like(x, dtype=torch.float32)
    elif case == "strided_output":
        args["residual_out"] = torch.empty((2, 32), device="cuda", dtype=x.dtype)[
            :, ::2
        ]
    elif case == "short_post_weight":
        args["post_weight"] = post_weight[:-1]
    elif case == "short_pre_weight":
        args["pre_weight"] = pre_weight[:-1]
    elif case == "short_residual":
        args["residual"] = residual[:1]
    else:
        args["pre_weight"] = pre_weight.cpu()

    with pytest.raises(ValueError, match=name):
        fused_rmsnorm_add_rmsnorm(
            args["x"],
            args["residual"],
            args["post_weight"],
            args["pre_weight"],
            POST_EPS,
            PRE_EPS,
            args["residual_out"],
        )
