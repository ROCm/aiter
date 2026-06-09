# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from op_tests.op_benchmarks.sonic_moe_a2_layout import (
    pack_a2_sorted,
    sorted_stage2_triton,
    unpack_a2_sorted,
)


def _manual_sorted_metadata(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_num, topk = topk_ids.shape
    rows: list[int] = []
    weights: list[float] = []
    expert_ids: list[int] = []
    for expert in range(num_experts):
        expert_rows: list[tuple[int, float]] = []
        for token in range(token_num):
            for slot in range(topk):
                if int(topk_ids[token, slot].item()) == expert:
                    packed = token | (slot << 24)
                    expert_rows.append((packed, float(topk_weights[token, slot].item())))
        pad = (-len(expert_rows)) % block_m
        expert_rows.extend([(token_num, 0.0)] * pad)
        for block_start in range(0, len(expert_rows), block_m):
            block = expert_rows[block_start : block_start + block_m]
            if not block:
                continue
            rows.extend(packed for packed, _ in block)
            weights.extend(weight for _, weight in block)
            expert_ids.append(expert)
    device = topk_ids.device
    return (
        torch.tensor(rows, dtype=torch.int32, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
        torch.tensor(expert_ids, dtype=torch.int32, device=device),
        torch.tensor([len(rows)], dtype=torch.int32, device=device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
def test_a2_sorted_pack_roundtrip():
    token_num, topk, inter_dim = 8, 3, 5
    a2 = torch.arange(
        token_num * topk * inter_dim,
        dtype=torch.float32,
        device="cuda",
    ).view(token_num, topk, inter_dim)
    order = torch.randperm(token_num * topk, device="cuda")
    token_ids = order // topk
    slot_ids = order % topk
    sorted_ids = (token_ids | (slot_ids << 24)).to(torch.int32)
    num_valid_ids = torch.tensor([sorted_ids.numel()], dtype=torch.int32, device="cuda")

    packed = pack_a2_sorted(a2, sorted_ids, num_valid_ids, topk)
    unpacked = unpack_a2_sorted(packed, sorted_ids, num_valid_ids, token_num, topk)

    assert_close(unpacked, a2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
def test_sorted_stage2_triton_matches_torch_reference():
    torch.manual_seed(0)
    token_num, model_dim, inter_dim, experts, topk = 8, 16, 32, 4, 2
    block_m = 4
    dtype = torch.bfloat16
    a2 = torch.randn(token_num, topk, inter_dim, dtype=dtype, device="cuda") * 0.1
    w2 = torch.randn(experts, model_dim, inter_dim, dtype=dtype, device="cuda") * 0.1
    token_idx = torch.arange(token_num, device="cuda").view(-1, 1)
    slot_idx = torch.arange(topk, device="cuda").view(1, -1)
    topk_ids = ((token_idx + slot_idx) % experts).to(torch.int64)
    topk_weights = torch.full(
        (token_num, topk), 1.0 / topk, dtype=torch.float32, device="cuda"
    )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = (
        _manual_sorted_metadata(topk_ids, topk_weights, experts, block_m)
    )
    a2_sorted = pack_a2_sorted(a2, sorted_ids, num_valid_ids, topk)

    actual = sorted_stage2_triton(
        a2_sorted,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        token_num,
        topk,
        block_m=block_m,
        block_n=16,
        block_k=32,
    )

    expected = torch.zeros(token_num, model_dim, dtype=torch.float32, device="cuda")
    for token in range(token_num):
        for slot in range(topk):
            expert = int(topk_ids[token, slot].item())
            expected[token] += (
                topk_weights[token, slot]
                * (a2[token, slot].float() @ w2[expert].float().t())
            )

    assert_close(actual, expected, atol=2e-2, rtol=2e-2)
