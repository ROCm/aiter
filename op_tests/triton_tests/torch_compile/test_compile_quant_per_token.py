# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch


def _get_compiled(fn):
    return torch.compile(
        fn, backend="inductor", fullgraph=True, options={"max_autotune": True}
    )


FP8_MAX = torch.finfo(torch.float8_e4m3fnuz).max


def torch_dynamic_per_token_quant_fp8(x):
    x_float = x.to(torch.float32)
    amax = x_float.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    qx = (x_float / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fnuz)
    return qx, scale


@pytest.mark.parametrize("M, N", [(64, 128), (128, 256), (256, 512)])
def test_compile_quant_per_token(M, N):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)

    qx_eager, scale_eager = torch_dynamic_per_token_quant_fp8(x)

    compiled_fn = _get_compiled(torch_dynamic_per_token_quant_fp8)
    qx_compiled, scale_compiled = compiled_fn(x)
    torch.cuda.synchronize()

    torch.testing.assert_close(scale_compiled, scale_eager, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        qx_compiled.to(torch.float32),
        qx_eager.to(torch.float32),
        atol=1.0,
        rtol=1e-1,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
