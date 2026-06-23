# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter


@pytest.mark.parametrize("tokens", [1, 4, 8])
@pytest.mark.parametrize("num_experts", [256, 384])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("num_shared", [1, 2])
def test_biased_grouped_topk_fused_shared_append(tokens, num_experts, topk, num_shared):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("AITER grouped topk shared append test requires ROCm")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_expert_group = 8
    topk_group = 4
    scale = 1.0

    gating_output = torch.randn(tokens, num_experts, device=device, dtype=dtype)
    correction_bias = torch.randn(num_experts, device=device, dtype=dtype) * 0.01

    routed_weights = torch.empty(tokens, topk, device=device, dtype=torch.float32)
    routed_ids = torch.empty(tokens, topk, device=device, dtype=torch.int32)
    aiter.biased_grouped_topk(
        gating_output,
        correction_bias,
        routed_weights,
        routed_ids,
        num_expert_group,
        topk_group,
        True,
        scale,
    )

    expected_ids = torch.empty(tokens, topk + num_shared, device=device, dtype=torch.int32)
    expected_weights = torch.empty(
        tokens, topk + num_shared, device=device, dtype=torch.float32
    )
    expected_ids[:, :topk] = routed_ids
    expected_weights[:, :topk] = routed_weights
    for i in range(num_shared):
        expected_ids[:, topk + i] = num_experts + i
        expected_weights[:, topk + i] = 1.0

    fused_ids = torch.empty_like(expected_ids)
    fused_weights = torch.empty_like(expected_weights)
    aiter.biased_grouped_topk(
        gating_output,
        correction_bias,
        fused_weights,
        fused_ids,
        num_expert_group,
        topk_group,
        True,
        scale,
        num_shared,
        1.0,
        num_experts,
    )

    torch.testing.assert_close(fused_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(fused_weights, expected_weights, rtol=1e-6, atol=1e-6)
