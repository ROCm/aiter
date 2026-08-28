# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from . import _get_compiled


def torch_routing_sigmoid_top1(x, w, topk=1):
    logits = x @ w
    weights = torch.sigmoid(logits)
    topk_weights, topk_ids = torch.topk(weights, topk, dim=-1)
    return topk_ids.to(torch.int32), topk_weights.to(torch.float32)


@pytest.mark.parametrize("M, K, N", [(64, 128, 8), (128, 256, 16), (256, 512, 32)])
def test_compile_moe_routing(M, K, N):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    topk = 1
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(K, N, device="cuda", dtype=torch.float16)

    ids_eager, weights_eager = torch_routing_sigmoid_top1(x, w, topk=topk)

    def fn(x, w):
        return torch_routing_sigmoid_top1(x, w, topk=topk)

    compiled_fn = _get_compiled(fn)
    ids_compiled, weights_compiled = compiled_fn(x, w)
    torch.cuda.synchronize()

    assert not torch.isnan(weights_compiled).any(), "torch.compile produced NaN"
    torch.testing.assert_close(ids_compiled, ids_eager)
    torch.testing.assert_close(weights_compiled, weights_eager, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
