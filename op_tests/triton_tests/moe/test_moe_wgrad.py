# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for moe_wgrad Triton kernel.

The kernel computes dW[e] = sum_{tokens t -> expert e} grad[t/top_k].T @ input[t/top_k].
sorted_token_ids and expert_ids are produced here by a pure-Python reference
that mimics moe_align_block_size, so the test has no dependency on the ASM JIT
module and runs under the standard pytest suite.
"""

import pytest
import torch

from aiter.ops.triton.moe.moe_wgrad import moe_wgrad

# ── helpers ──────────────────────────────────────────────────────────


def _build_sorted_metadata(topk_ids: torch.Tensor, num_experts: int, block_size: int):
    """Pure-Python stand-in for moe_align_block_size.

    topk_ids: [T, top_k]  int32, values in [0, E)
    Returns: sorted_token_ids, expert_ids, num_tokens_post_padded, num_valid

    The kernel masks invalid slots with ``offs_token < num_valid_tokens`` where
    ``num_valid_tokens = T * top_k``.  Padding entries must therefore be set to
    a value >= T * top_k so that they are excluded; using 0 (as max(-1, 0)
    would give) incorrectly includes padding in the gradient accumulation.
    """
    T, top_k = topk_ids.shape
    num_valid = T * top_k  # sentinel: any slot index >= this is padding

    # flat list of (expert, global_token_slot) pairs
    pairs = []
    for t in range(T):
        for k in range(top_k):
            e = int(topk_ids[t, k])
            pairs.append((e, t * top_k + k))
    pairs.sort(key=lambda p: p[0])

    # pad each expert's block to a multiple of block_size
    padded = []
    expert_ids_list = []
    for e in range(num_experts):
        slots = [slot for ex, slot in pairs if ex == e]
        n = len(slots)
        if n == 0:
            continue  # no tokens for this expert — skip (matches moe_align_block_size)
        pad_to = ((n + block_size - 1) // block_size) * block_size
        slots += [-1] * (pad_to - n)
        for i in range(0, len(slots), block_size):
            blk = slots[i : i + block_size]
            padded.extend(blk)
            expert_ids_list.append(e if any(s >= 0 for s in blk) else -1)

    # Use num_valid as the padding sentinel so token_mask = (slot < num_valid) is False.
    sorted_token_ids = torch.tensor(
        [s if s >= 0 else num_valid for s in padded], dtype=torch.int32, device="cuda"
    )
    expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32, device="cuda")
    num_tokens_post_padded = torch.tensor(
        [len(padded)], dtype=torch.int32, device="cuda"
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded, num_valid


def _ref_moe_wgrad(grad, input, topk_ids, num_experts, top_k):
    """Reference: dense scatter-add per expert."""
    T, N = grad.shape
    K = input.shape[1]
    dW = torch.zeros(num_experts, N, K, dtype=grad.dtype, device=grad.device)
    for t in range(T):
        for k in range(top_k):
            e = int(topk_ids[t, k])
            dW[e] += grad[t].float().unsqueeze(1) @ input[t].float().unsqueeze(0)
    return dW.to(grad.dtype)


# ── tests ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "T, N, K, E, top_k",
    [
        (64, 128, 64, 4, 2),
        (128, 256, 128, 8, 2),
        (32, 64, 32, 4, 1),
        (64, 128, 128, 8, 4),
    ],
)
def test_moe_wgrad_correctness(T, N, K, E, top_k):
    torch.manual_seed(42)
    block_size = 64

    # Use float32 so that atomic_add accumulation stays numerically clean.
    # With bfloat16 each atomic_add has ~0.8% error; after O(T) accumulations
    # cosine similarity can drop to ~0.98 even with a correct kernel, making
    # the test unreliable as a regression guard.
    grad = torch.randn(T, N, dtype=torch.float32, device="cuda")
    inp = torch.randn(T, K, dtype=torch.float32, device="cuda")
    # random expert assignments, each token picks top_k distinct experts
    topk_ids = torch.stack(
        [torch.randperm(E, device="cuda")[:top_k] for _ in range(T)]
    ).to(torch.int32)

    sorted_token_ids, expert_ids, num_tokens_post_padded, _ = _build_sorted_metadata(
        topk_ids, E, block_size
    )

    dW = moe_wgrad(
        grad,
        inp,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_experts=E,
        top_k=top_k,
        weight_shape=(E, N, K),
        block_size_m=block_size,
    )

    ref = _ref_moe_wgrad(grad, inp, topk_ids, E, top_k)

    cos = torch.nn.functional.cosine_similarity(
        dW.float().reshape(-1), ref.float().reshape(-1), dim=0
    ).item()
    assert (
        cos > 0.999
    ), f"moe_wgrad cosine similarity {cos:.6f} < 0.999 for T={T} N={N} K={K} E={E} top_k={top_k}"
