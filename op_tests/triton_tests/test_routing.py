# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import partial

import pytest
import torch
from aiter.ops.triton.routing import routing_sigmoid_top1, torch_routing_sigmoid_top1


@pytest.mark.parametrize("M", [128, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("N", [16, 128])
@pytest.mark.parametrize("K", [16, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_routing_sigmoid_top1(M, N, K, dtype):
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
