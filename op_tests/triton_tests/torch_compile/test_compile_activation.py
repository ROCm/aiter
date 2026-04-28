# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F


def _get_compiled(fn):
    return torch.compile(fn, backend="inductor", fullgraph=True, options={"max_autotune": True})


def torch_silu_mul(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return F.silu(x1) * x2


def torch_gelu_mul(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return F.gelu(x1) * x2


@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M, N", [(64, 256), (128, 512), (256, 1024)])
def test_compile_activation(M, N, dtype, activation):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    x = torch.randn(M, N, device="cuda", dtype=dtype)

    act_fn = torch_silu_mul if activation == "silu" else torch_gelu_mul
    out_eager = act_fn(x)

    compiled_fn = _get_compiled(act_fn)
    out_compiled = compiled_fn(x)
    torch.cuda.synchronize()

    assert not torch.isnan(out_compiled).any(), "torch.compile produced NaN"
    torch.testing.assert_close(out_compiled, out_eager, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
