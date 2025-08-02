# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import partial

import pytest
import torch
from aiter.ops.triton.moe_routing_sigmoid_top1_fused import routing_sigmoid_top1


def torch_routing_sigmoid_top1(
    x, w, topk, fused_shared_experts=False, dummy_ids=None, dummy_weights=None
):
    scores = torch.matmul(x, w)  # [M, N]

    scores = torch.sigmoid(scores.to(torch.float32))  # [M, N]

    assert topk == 1

    topk_weights, topk_ids = torch.topk(scores, topk, dim=1)  # [M, topk]

    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)

    if fused_shared_experts:
        topk_ids = torch.cat(
            [
                topk_ids,
                dummy_ids,
            ],
            dim=1,
        )
        topk_weights = torch.cat(
            [topk_weights, dummy_weights],
            dim=1,
        )

    return topk_ids, topk_weights


@pytest.mark.parametrize("M", [128, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("N", [16, 128])
@pytest.mark.parametrize("K", [16, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_routing_sigmoid_top1(M, N, K, dtype):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    TOPK = 1

    torch.manual_seed(7)

    device = "cuda"

    x = torch.randint(-2, 3, (M, K), device=device).to(dtype)
    w = torch.randint(-2, 3, (K, N), device=device).to(dtype)

    dummy_ids = torch.ones((M, 1), dtype=torch.int32, device="cuda") * N
    dummy_weights = torch.ones((M, 1), dtype=torch.float32, device="cuda")
    _eager = partial(
        torch_routing_sigmoid_top1, dummy_ids=dummy_ids, dummy_weights=dummy_weights
    )

    topk_ids, topk_weights = routing_sigmoid_top1(x, w, TOPK, fused_shared_experts=True)
    _eager_topk_ids, _eager_topk_weights = _eager(x, w, TOPK, fused_shared_experts=True)

    torch.testing.assert_close(_eager_topk_ids, topk_ids, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(_eager_topk_weights, topk_weights, atol=1e-5, rtol=1e-5)
