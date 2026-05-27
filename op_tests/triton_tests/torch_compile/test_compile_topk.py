# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from . import _get_compiled


def torch_topk(x, k):
    return torch.topk(x, k, dim=-1)


@pytest.mark.parametrize("k", [8, 32])
@pytest.mark.parametrize("M, N", [(64, 256), (128, 512), (256, 1024)])
def test_compile_topk(M, N, k):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)

    values_eager, indices_eager = torch_topk(x, k)

    def fn(x):
        return torch_topk(x, k)

    compiled_fn = _get_compiled(fn)
    values_compiled, indices_compiled = compiled_fn(x)
    torch.cuda.synchronize()

    assert not torch.isnan(values_compiled).any(), "torch.compile produced NaN"
    torch.testing.assert_close(values_compiled, values_eager, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(indices_compiled, indices_eager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
