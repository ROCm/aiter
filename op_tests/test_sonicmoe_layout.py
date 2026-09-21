# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.sonicmoe import (
    SonicMoEActivationType,
    moe_general_routing_inputs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


@pytest.mark.parametrize("with_bias", [False, True])
def test_general_routing_grouped_weights_match_legacy_layout(with_bias):
    torch.manual_seed(43)
    device = torch.device("cuda")
    tokens, hidden, intermediate, experts = 64, 64, 64, 2
    token_indices = torch.arange(tokens, dtype=torch.int32, device=device)
    expert_indices = torch.repeat_interleave(
        torch.arange(experts, dtype=torch.int32, device=device),
        torch.tensor([32, 32], dtype=torch.int32, device=device),
    )

    x_grouped = torch.randn(
        tokens, hidden, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    x_legacy = x_grouped.detach().clone().requires_grad_(True)
    scores_grouped = torch.rand(
        tokens, dtype=torch.float32, device=device, requires_grad=True
    )
    scores_legacy = scores_grouped.detach().clone().requires_grad_(True)
    w1_grouped = torch.randn(
        experts,
        hidden,
        2 * intermediate,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w2_grouped = torch.randn(
        experts,
        intermediate,
        hidden,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w1_legacy = w1_grouped.detach().permute(2, 1, 0).contiguous().requires_grad_(True)
    w2_legacy = w2_grouped.detach().permute(2, 1, 0).contiguous().requires_grad_(True)
    b1_grouped = (
        torch.randn(
            experts, 2 * intermediate, dtype=torch.bfloat16, device=device
        ).requires_grad_(True)
        if with_bias
        else None
    )
    b2_grouped = (
        torch.randn(
            experts, hidden, dtype=torch.bfloat16, device=device
        ).requires_grad_(True)
        if with_bias
        else None
    )
    b1_legacy = b1_grouped.detach().clone().requires_grad_(True) if with_bias else None
    b2_legacy = b2_grouped.detach().clone().requires_grad_(True) if with_bias else None

    common = (
        token_indices,
        expert_indices,
        experts,
        torch.cuda.current_stream().cuda_stream,
        SonicMoEActivationType.SWIGLU,
        False,
        True,
    )
    output_grouped, _ = moe_general_routing_inputs(
        x_grouped,
        scores_grouped,
        common[0],
        common[1],
        w1_grouped,
        b1_grouped,
        w2_grouped,
        b2_grouped,
        *common[2:],
        grouped_weight_layout=True,
    )
    output_legacy, _ = moe_general_routing_inputs(
        x_legacy,
        scores_legacy,
        common[0],
        common[1],
        w1_legacy,
        b1_legacy,
        w2_legacy,
        b2_legacy,
        *common[2:],
    )

    torch.testing.assert_close(output_grouped, output_legacy)
    grad = torch.randn_like(output_grouped)
    output_grouped.backward(grad)
    output_legacy.backward(grad)
    torch.testing.assert_close(x_grouped.grad, x_legacy.grad)
    torch.testing.assert_close(scores_grouped.grad, scores_legacy.grad)
    torch.testing.assert_close(w1_grouped.grad, w1_legacy.grad.permute(2, 1, 0))
    torch.testing.assert_close(w2_grouped.grad, w2_legacy.grad.permute(2, 1, 0))
    if with_bias:
        torch.testing.assert_close(b1_grouped.grad, b1_legacy.grad)
        torch.testing.assert_close(b2_grouped.grad, b2_legacy.grad)
