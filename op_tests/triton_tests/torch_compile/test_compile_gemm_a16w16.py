# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch


def _get_compiled(fn):
    return torch.compile(fn, backend="inductor", options={"max_autotune": True})


def torch_gemm(x, w, bias=None):
    out = torch.mm(x, w.t())
    if bias is not None:
        out = out + bias
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M, N, K", [(128, 256, 64), (256, 512, 128), (512, 1024, 256)]
)
def test_compile_gemm(M, N, K, dtype):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    w = torch.randn(N, K, device="cuda", dtype=dtype)

    out_eager = torch_gemm(x, w)

    def fn(x, w):
        return torch_gemm(x, w)

    compiled_fn = _get_compiled(fn)
    out_compiled = compiled_fn(x, w)
    torch.cuda.synchronize()

    assert not torch.isnan(out_compiled).any(), "torch.compile produced NaN"
    torch.testing.assert_close(out_compiled, out_eager, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M, N, K", [(128, 256, 64), (256, 512, 128)])
def test_compile_gemm_with_bias(M, N, K, dtype):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    w = torch.randn(N, K, device="cuda", dtype=dtype)
    bias = torch.randn(N, device="cuda", dtype=dtype)

    out_eager = torch_gemm(x, w, bias=bias)

    def fn(x, w, bias):
        return torch_gemm(x, w, bias=bias)

    compiled_fn = _get_compiled(fn)
    out_compiled = compiled_fn(x, w, bias)
    torch.cuda.synchronize()

    assert not torch.isnan(out_compiled).any(), "torch.compile produced NaN"
    torch.testing.assert_close(out_compiled, out_eager, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
