# SPDX-License-Identifier: MIT
"""Async stage-1 LDS loads must finish before another wave consumes the tile."""

import pytest
import torch

from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 LDS-DMA regression")
@pytest.mark.parametrize("tile_m,tile_n", [(64, 128), (64, 256), (128, 128)])
def test_stage1_async_loads_match_synchronous_and_replay(tile_m, tile_n):
    pytest.importorskip("flydsl")
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

    torch.manual_seed(42)
    tokens, model_dim, inter_dim, experts, topk = 128, 5120, 512, 4, 2
    activation = torch.randn(tokens, model_dim, device="cuda", dtype=torch.bfloat16) / 4
    weight = (
        torch.randn(
            experts, 2 * inter_dim, model_dim, device="cuda", dtype=torch.bfloat16
        )
        / 4
    )
    expert_ids = torch.stack(
        [torch.randperm(experts, device="cuda")[:topk] for _ in range(tokens)]
    ).to(torch.int32)
    routing_weights = torch.full((tokens, topk), 1 / topk, device="cuda")
    sorted_ids, _, sorted_experts, valid_ids, _ = moe_sorting(
        expert_ids, routing_weights, experts, model_dim, torch.bfloat16, tile_m
    )
    activation_q, activation_scale = per_1x32_f8_scale_f8_quant(
        activation, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    weight_q, weight_scale = per_1x32_f4_quant(weight, quant_dtype=dtypes.fp4x2)
    weight_q = weight_q.view(experts, 2 * inter_dim, model_dim // 2)
    weight_q = shuffle_weight_a16w4(weight_q, 16, True)
    weight_scale = shuffle_scale_a16w4(weight_scale, experts, True)
    activation_scale = mxfp4_moe_sort_fwd(
        activation_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=valid_ids,
        token_num=tokens,
        cols=model_dim,
    )
    output = torch.empty(tokens, topk, inter_dim, device="cuda", dtype=torch.bfloat16)

    def launch(async_copy):
        return flydsl_moe_stage1(
            activation_q,
            weight_q,
            sorted_ids,
            sorted_experts,
            valid_ids,
            out=output,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=256,
            a_dtype="fp8",
            b_dtype="fp4",
            out_dtype="bf16",
            act="silu",
            gate_mode="interleave",
            w1_scale=weight_scale,
            a1_scale=activation_scale,
            use_async_copy=async_copy,
            waves_per_eu=4,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        synchronous = launch(False).clone()
        expected = launch(True).clone()
        torch.testing.assert_close(expected, synchronous, rtol=0.02, atol=0.02)
        for _ in range(50):
            output.fill_(float("nan"))
            launch(True)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch(True)
        for _ in range(20):
            output.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.cuda.current_stream().wait_stream(stream)
