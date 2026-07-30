# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression tests for the fused route-sort + MXFP4 input quant kernel."""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.ops.quant import fused_dynamic_mxfp4_quant_moe_sort, per_1x32_f4_quant_hip

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm/CUDA device is required"
)

HIDDEN = 7168
BLOCK_M = 32
# The last expert of the (385, 9) route is a fused shared expert every token
# is routed to; the other routes dispatch shared experts separately.
FUSED_SHARED_ROUTE = (385, 9)
ROUTE_VARIANTS = (
    pytest.param(385, 9, id="fused_shared_e385_topk9"),
    pytest.param(384, 8, id="separate_shared_e384_topk8"),
    pytest.param(256, 8, id="separate_shared_e256_topk8"),
)
# Includes non-power-of-2 counts: the kernel's contract is M in [1, 128], not
# a whitelist, and the tail handling differs for partial 32-row blocks.
TOKEN_COUNTS = [1, 2, 3, 4, 8, 16, 17, 32, 64, 100, 127, 128]


def _make_inputs(tokens: int, experts: int, topk: int):
    torch.manual_seed(1000 + tokens)
    hidden = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    if (experts, topk) != FUSED_SHARED_ROUTE:
        topk_ids = torch.stack(
            [
                torch.randperm(experts, dtype=torch.int32, device="cuda")[:topk]
                for _ in range(tokens)
            ]
        )
        topk_weights = torch.rand((tokens, topk), dtype=torch.float32, device="cuda")
    else:
        shared = experts - 1
        routed_ids = torch.stack(
            [
                torch.randperm(shared, dtype=torch.int32, device="cuda")[: topk - 1]
                for _ in range(tokens)
            ]
        )
        topk_ids = torch.cat(
            (
                routed_ids,
                torch.full((tokens, 1), shared, dtype=torch.int32, device="cuda"),
            ),
            dim=1,
        )
        topk_weights = torch.cat(
            (
                torch.rand((tokens, topk - 1), dtype=torch.float32, device="cuda"),
                torch.ones((tokens, 1), dtype=torch.float32, device="cuda"),
            ),
            dim=1,
        )
    return hidden, topk_ids, topk_weights


def _canonical_entries(sorted_ids, sorted_weights, sorted_experts, num_valid_ids):
    valid, tokens = [int(v) for v in num_valid_ids.cpu().tolist()]
    entries = []
    for block, expert in enumerate(sorted_experts[: valid // BLOCK_M].cpu().tolist()):
        for row in range(block * BLOCK_M, block * BLOCK_M + BLOCK_M):
            packed = int(sorted_ids[row].item())
            token = packed & 0x00FFFFFF
            if token != tokens:
                entries.append(
                    (
                        expert,
                        token,
                        (packed >> 24) & 0xFF,
                        float(sorted_weights[row].item()),
                    )
                )
    return sorted(entries)


def _run_fused(hidden, topk_ids, topk_weights, experts):
    tokens = hidden.shape[0]
    route_count = topk_ids.numel()
    active_experts = min(experts, route_count)
    max_sorted = (
        (route_count + active_experts * (BLOCK_M - 1) + BLOCK_M - 1)
        // BLOCK_M
        * BLOCK_M
    )
    sorted_ids = torch.empty(max_sorted, dtype=torch.int32, device="cuda")
    sorted_weights = torch.empty(max_sorted, dtype=torch.float32, device="cuda")
    sorted_experts = torch.empty(
        (max_sorted + BLOCK_M - 1) // BLOCK_M, dtype=torch.int32, device="cuda"
    )
    num_valid_ids = torch.empty(2, dtype=torch.int32, device="cuda")
    moe_buf = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    activation_quant = torch.empty(
        (tokens, HIDDEN // 2), dtype=dtypes.fp4x2, device="cuda"
    )
    activation_scale_token = torch.empty(
        (tokens, HIDDEN // 32), dtype=dtypes.fp8_e8m0, device="cuda"
    )
    aiter.mxfp4_moe_sort_quant_fwd(
        hidden,
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_experts,
        num_valid_ids,
        moe_buf,
        activation_quant,
        activation_scale_token,
        experts,
    )
    return (
        sorted_ids,
        sorted_weights,
        sorted_experts,
        num_valid_ids,
        moe_buf,
        activation_quant,
        activation_scale_token,
    )


@pytest.mark.parametrize("experts, topk", ROUTE_VARIANTS)
@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_fused_sort_quant_matches_opus_and_generic_quant(tokens, experts, topk):
    hidden, topk_ids, topk_weights = _make_inputs(tokens, experts, topk)

    ref_ids, ref_weights, ref_experts, ref_valid, _ = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        HIDDEN,
        dtypes.bf16,
        BLOCK_M,
    )
    ref_quant, _ = fused_dynamic_mxfp4_quant_moe_sort(
        hidden,
        sorted_ids=ref_ids,
        num_valid_ids=ref_valid,
        token_num=tokens,
        topk=topk,
        block_size=BLOCK_M,
        sorted_weights=ref_weights,
    )
    compact_quant, compact_scale = per_1x32_f4_quant_hip(hidden, shuffle=False)
    fused = _run_fused(hidden, topk_ids, topk_weights, experts)
    torch.cuda.synchronize()

    (
        fused_ids,
        fused_weights,
        fused_experts,
        fused_valid,
        fused_moe_buf,
        fused_quant,
        fused_scale_token,
    ) = fused
    assert _canonical_entries(
        ref_ids, ref_weights, ref_experts, ref_valid
    ) == _canonical_entries(fused_ids, fused_weights, fused_experts, fused_valid)
    torch.testing.assert_close(fused_valid, ref_valid, rtol=0, atol=0)
    torch.testing.assert_close(fused_moe_buf, torch.zeros_like(fused_moe_buf))
    torch.testing.assert_close(
        fused_quant.view(torch.uint8), ref_quant.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        fused_quant.view(torch.uint8), compact_quant.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        fused_scale_token.view(torch.uint8),
        compact_scale.view(torch.uint8),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("experts, topk", ROUTE_VARIANTS)
def test_fused_sort_quant_is_cuda_graph_replay_safe(experts, topk):
    hidden, topk_ids, topk_weights = _make_inputs(4, experts, topk)
    # Compile/warm up before capture. The graph itself only records the
    # fixed-address operator launch; all buffers are preallocated.
    _run_fused(hidden, topk_ids, topk_weights, experts)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused = _run_fused(hidden, topk_ids, topk_weights, experts)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()

    assert int(fused[3][1].item()) == 4
    torch.testing.assert_close(fused[4], torch.zeros_like(fused[4]))
