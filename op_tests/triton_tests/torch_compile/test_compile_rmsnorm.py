# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch


def _get_compiled(fn):
    return torch.compile(fn, backend="inductor", options={"max_autotune": True})


def torch_rmsnorm(x, weight, eps):
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M, N", [(128, 256), (256, 512), (512, 1024)])
def test_compile_rmsnorm(M, N, dtype):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    eps = 1e-6
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    weight = torch.ones(N, device="cuda", dtype=dtype)

    out_eager = torch_rmsnorm(x, weight, eps)

    def fn(x, weight):
        return torch_rmsnorm(x, weight, eps)

    compiled_fn = _get_compiled(fn)
    out_compiled = compiled_fn(x, weight)
    torch.cuda.synchronize()

    assert not torch.isnan(out_compiled).any(), "torch.compile produced NaN"
    torch.testing.assert_close(out_compiled, out_eager, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
