# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for moe_wgrad.

Uses constant inputs so the expected result is analytically known:
  grad = c * ones(T, N), inp = d * ones(T, K), all T tokens -> expert 0
  → dW[0] = T * c * d * ones(N, K)
"""

import pytest
import torch
import triton

from aiter.ops.triton.moe.moe_wgrad import moe_wgrad


def _call_moe_wgrad(num_tokens, E, N, K, grad, inp, dtype, block_size=64):
    """Build sorted_token_ids / expert_ids for a uniform-routing scenario
    (all tokens -> experts in round-robin order, top_k=1) and call moe_wgrad.
    """
    tpe = num_tokens // E  # tokens per expert
    assert tpe * E == num_tokens
    top_k = 1
    blocks_per_expert = triton.cdiv(tpe, block_size)
    padded_per_expert = blocks_per_expert * block_size
    total_padded = padded_per_expert * E

    sorted_token_ids = torch.full(
        (total_padded,), num_tokens, dtype=torch.int32, device=grad.device
    )
    expert_ids = torch.full(
        (blocks_per_expert * E,), -1, dtype=torch.int32, device=grad.device
    )
    for e in range(E):
        start = e * tpe
        out_start = e * padded_per_expert
        sorted_token_ids[out_start : out_start + tpe] = torch.arange(
            start, start + tpe, dtype=torch.int32, device=grad.device
        )
        for b in range(blocks_per_expert):
            expert_ids[e * blocks_per_expert + b] = e

    ntpp = torch.tensor([total_padded], dtype=torch.int32, device=grad.device)

    return moe_wgrad(
        grad,
        inp,
        sorted_token_ids,
        expert_ids,
        ntpp,
        num_experts=E,
        top_k=top_k,
        weight_shape=(E, N, K),
        block_size_m=block_size,
    )


@pytest.mark.parametrize("num_tokens, E, N, K", [(128, 1, 64, 64), (256, 4, 128, 64)])
@pytest.mark.parametrize("c, d", [(2.0, 3.0), (0.1, -0.5)])
def test_moe_wgrad_constant_inputs(num_tokens, E, N, K, c, d):
    """With constant inputs the result is analytically known.

    dW[e, n, k] = (num_tokens // E) * c * d  for every (e, n, k).
    """
    device, dtype = "cuda", torch.float32
    tpe = num_tokens // E

    grad = torch.full((num_tokens, N), c, dtype=dtype, device=device)
    inp = torch.full((num_tokens, K), d, dtype=dtype, device=device)

    dW = _call_moe_wgrad(num_tokens, E, N, K, grad, inp, dtype)

    expected = tpe * c * d
    assert dW.shape == (E, N, K)
    assert dW.dtype == dtype
    # Allow 1% relative tolerance for fp32 accumulation over many tokens
    assert torch.allclose(
        dW,
        torch.full_like(dW, expected),
        rtol=0.01,
        atol=abs(expected) * 0.01 + 1e-4,
    ), f"max err {(dW - expected).abs().max():.4f}, expected {expected}"


@pytest.mark.parametrize("num_tokens, E, N, K", [(128, 2, 64, 64), (256, 4, 128, 128)])
def test_moe_wgrad_linearity(num_tokens, E, N, K):
    """moe_wgrad is linear: wgrad(a) + wgrad(-a) = 0 over the same input."""
    device, dtype = "cuda", torch.float32
    torch.manual_seed(0)
    grad = torch.randn(num_tokens, N, dtype=dtype, device=device) * 0.1
    inp = torch.randn(num_tokens, K, dtype=dtype, device=device) * 0.1

    dW_pos = _call_moe_wgrad(num_tokens, E, N, K, grad, inp, dtype)
    dW_neg = _call_moe_wgrad(num_tokens, E, N, K, -grad, inp, dtype)

    assert torch.allclose(
        dW_pos + dW_neg,
        torch.zeros_like(dW_pos),
        atol=1e-4,
    ), f"linearity violated: max {(dW_pos + dW_neg).abs().max():.6f}"


def test_moe_wgrad_zero_padded_tokens():
    """num_tokens_post_padded=0 returns zero dW without error."""
    E, N, K, block_size = 4, 64, 64, 64
    grad = torch.randn(64, N, dtype=torch.float32, device="cuda") * 0.1
    inp = torch.randn(64, K, dtype=torch.float32, device="cuda") * 0.1
    sorted_ids = torch.full((block_size,), 64, dtype=torch.int32, device="cuda")
    expert_ids = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    ntpp = torch.tensor([0], dtype=torch.int32, device="cuda")
    dW = moe_wgrad(grad, inp, sorted_ids, expert_ids, ntpp, E, 1, (E, N, K), block_size)
    assert dW.shape == (E, N, K)
    assert dW.abs().max() == 0
