# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import (
    fused_dynamic_mxfp8_quant_moe_route,
    fused_dynamic_mxfp8_quant_moe_sort,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm device"),
    pytest.mark.skipif(get_gfx() != "gfx950", reason="requires gfx950 MXFP8"),
]


def _scale_indices(rows, valid_cols, padded_cols):
    row = rows.to(torch.int64)[:, None]
    col = torch.arange(valid_cols, device=rows.device, dtype=torch.int64)[None, :]
    return (
        (row // 32 * padded_cols) * 32
        + (col // 8) * 256
        + (col % 4) * 64
        + (row % 16) * 4
        + (col % 8 // 4) * 2
        + (row % 32 // 16)
    ).reshape(-1)


@pytest.mark.parametrize("tokens", [1, 8, 32])
def test_route_quant_matches_sorted_row_quant(tokens):
    torch.manual_seed(tokens)
    hidden_dim, experts, topk = 6144, 129, 5
    hidden = torch.randn((tokens, hidden_dim), dtype=dtypes.bf16, device="cuda")
    topk_ids = torch.topk(
        torch.rand((tokens, experts), device="cuda"), topk, dim=-1
    ).indices.to(dtypes.i32)
    topk_weights = torch.rand((tokens, topk), dtype=torch.float32, device="cuda")
    (
        sorted_ids,
        sorted_weights,
        _,
        num_valid_ids,
        _,
        reverse_sorted,
    ) = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        hidden_dim,
        dtypes.bf16,
        block_size=32,
        return_reverse_sorted=True,
    )

    expected_out, expected_scale = fused_dynamic_mxfp8_quant_moe_sort(
        hidden,
        sorted_ids,
        num_valid_ids,
        tokens,
        topk,
        32,
        sorted_weights=sorted_weights,
        num_experts_upper_bound=experts,
    )
    actual_out, actual_scale = fused_dynamic_mxfp8_quant_moe_route(
        hidden, sorted_ids, reverse_sorted, tokens, topk
    )
    torch.cuda.synchronize()

    assert torch.equal(expected_out.view(torch.uint8), actual_out.view(torch.uint8))
    indices = _scale_indices(reverse_sorted, hidden_dim // 32, expected_scale.shape[1])
    assert torch.equal(
        expected_scale.view(torch.uint8).reshape(-1)[indices],
        actual_scale.view(torch.uint8).reshape(-1)[indices],
    )

    if tokens == 8:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out, graph_scale = fused_dynamic_mxfp8_quant_moe_route(
                hidden, sorted_ids, reverse_sorted, tokens, topk
            )
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(expected_out.view(torch.uint8), graph_out.view(torch.uint8))
        assert torch.equal(
            expected_scale.view(torch.uint8).reshape(-1)[indices],
            graph_scale.view(torch.uint8).reshape(-1)[indices],
        )
