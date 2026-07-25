# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import dtypes


def _run_biased_grouped_topk(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
):
    num_tokens = logits.shape[0]
    weights = torch.empty(num_tokens, topk, device="cuda", dtype=dtypes.fp32)
    ids = torch.empty(num_tokens, topk, device="cuda", dtype=dtypes.i32)
    aiter.biased_grouped_topk_hip(
        logits,
        correction_bias,
        weights,
        ids,
        num_expert_group,
        topk_group,
        True,
        1.0,
    )
    order = ids.argsort(dim=-1)
    return weights.gather(1, order), ids.gather(1, order)


@pytest.mark.skipif(torch.version.hip is None, reason="ROCm/HIP is required")
@pytest.mark.parametrize("num_tokens", [2, 4, 32])
def test_kimi_k3_biased_grouped_topk_row_strided(num_tokens: int):
    """K3 fused MoE-front router slice matches its dense copy."""
    torch.manual_seed(0)
    gate_up_width = 1536
    num_experts = 896
    routed_width = 3584
    fused_front_width = gate_up_width + num_experts + routed_width
    topk = 16

    backing = torch.randn(
        num_tokens,
        fused_front_width,
        device="cuda",
        dtype=torch.bfloat16,
    )
    logits = backing[:, gate_up_width : gate_up_width + num_experts]
    logits_dense = logits.contiguous()
    correction_bias = torch.randn(
        num_experts, device="cuda", dtype=torch.bfloat16
    )

    assert logits.stride() == (fused_front_width, 1)
    assert not logits.is_contiguous()

    strided_weights, strided_ids = _run_biased_grouped_topk(
        logits, correction_bias, topk, 1, 1
    )
    dense_weights, dense_ids = _run_biased_grouped_topk(
        logits_dense, correction_bias, topk, 1, 1
    )

    torch.testing.assert_close(strided_ids, dense_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        strided_weights, dense_weights, rtol=1e-4, atol=1e-5
    )


@pytest.mark.skipif(torch.version.hip is None, reason="ROCm/HIP is required")
def test_opt_sort_accepts_row_strided_input():
    """The opt-sort kernel accepts row-strided inputs."""
    torch.manual_seed(1)
    num_tokens = 4
    num_experts = 256
    topk = 8
    backing_width = 320

    backing = torch.randn(
        num_tokens, backing_width, device="cuda", dtype=torch.bfloat16
    )
    logits = backing[:, 32 : 32 + num_experts]
    logits_dense = logits.contiguous()
    correction_bias = torch.randn(
        num_experts, device="cuda", dtype=torch.bfloat16
    )

    assert logits.stride() == (backing_width, 1)
    assert logits_dense.stride() == (num_experts, 1)

    strided_weights, strided_ids = _run_biased_grouped_topk(
        logits, correction_bias, topk, 8, 4
    )
    dense_weights, dense_ids = _run_biased_grouped_topk(
        logits_dense, correction_bias, topk, 8, 4
    )

    torch.testing.assert_close(strided_ids, dense_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        strided_weights, dense_weights, rtol=1e-4, atol=1e-5
    )
