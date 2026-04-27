# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F


def _get_compiled(fn):
    return torch.compile(fn, backend="inductor", options={"max_autotune": True})


def torch_softmax(x, dim=-1):
    return F.softmax(x, dim=dim)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M, N", [(128, 64), (256, 512), (1024, 1024)])
def test_compile_softmax(M, N, dtype):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    x = torch.randn(M, N, device="cuda", dtype=dtype)

    out_eager = torch_softmax(x)

    compiled_fn = _get_compiled(torch_softmax)
    out_compiled = compiled_fn(x)
    torch.cuda.synchronize()

    assert not torch.isnan(out_compiled).any(), "torch.compile produced NaN"
    torch.testing.assert_close(out_compiled, out_eager, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
