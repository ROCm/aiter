# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch


def _get_compiled(fn):
    return torch.compile(
        fn, backend="inductor", fullgraph=True, options={"max_autotune": True}
    )


def torch_fused_mul_add(x, a, b):
    return x * a + b


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M, N", [(128, 256), (256, 512), (512, 1024)])
@pytest.mark.parametrize("scalar_ab", [False, True])
def test_compile_fused_mul_add(M, N, dtype, scalar_ab):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    x = torch.randn(M, N, device="cuda", dtype=dtype)

    if scalar_ab:
        a, b = 2.0, 0.5
    else:
        a = torch.randn(M, N, device="cuda", dtype=dtype)
        b = torch.randn(M, N, device="cuda", dtype=dtype)

    out_eager = torch_fused_mul_add(x, a, b)

    def fn(x, a, b):
        return torch_fused_mul_add(x, a, b)

    compiled_fn = _get_compiled(fn)
    out_compiled = compiled_fn(x, a, b)
    torch.cuda.synchronize()

    assert not torch.isnan(out_compiled).any(), "torch.compile produced NaN"
    tol = (0.1, 0.1) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(out_compiled, out_eager, atol=tol[0], rtol=tol[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
