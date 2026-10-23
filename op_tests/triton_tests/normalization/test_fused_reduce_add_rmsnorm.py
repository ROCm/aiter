# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (
    _get_config as _get_gemm_config,
)
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.normalization.fused_reduce_add_rmsnorm import (
    fused_reduce_add_rmsnorm,
)


def run_torch(x, residual, weight, eps, gemma_norm):
    h = (x.float().sum(dim=0) if x.dim() == 3 else x.float()) + residual.float()
    w = weight.float() + 1.0 if gemma_norm else weight.float()
    out = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps) * w
    return out.to(residual.dtype), h.to(residual.dtype)


@pytest.mark.parametrize("M", [1, 4, 64, 255, 2048])
@pytest.mark.parametrize("N", [128, 1000, 5120])
@pytest.mark.parametrize("num_ksplit", [0, 1, 3, 4])
@pytest.mark.parametrize("gemma_norm", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_reduce_add_rmsnorm(M, N, num_ksplit, gemma_norm, dtype):
    # num_ksplit == 0: a reduced [M, N] GEMM output in dtype instead of fp32 partials.
    torch.manual_seed(0)
    if num_ksplit == 0:
        x = torch.randn((M, N), dtype=dtype, device="cuda")
    else:
        x = torch.randn((num_ksplit, M, N), dtype=torch.float32, device="cuda")
    residual = torch.randn((M, N), dtype=dtype, device="cuda")
    weight = torch.randn((N,), dtype=dtype, device="cuda")

    out, residual_out = fused_reduce_add_rmsnorm(
        x, residual, weight, 1e-6, gemma_norm=gemma_norm
    )
    out_ref, residual_ref = run_torch(x, residual, weight, 1e-6, gemma_norm)

    torch.testing.assert_close(residual_out, residual_ref)
    torch.testing.assert_close(out, out_ref)


@pytest.mark.parametrize("M", [16, 64])
@pytest.mark.parametrize("N, K", [(5120, 6144), (5120, 17408)])
@pytest.mark.parametrize("gemma_norm", [False, True])
def test_fused_reduce_add_rmsnorm_gemm_partials(M, N, K, gemma_norm):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    x = torch.randn((M, K), dtype=dtype, device="cuda") / K**0.5
    w = torch.randn((N, K), dtype=dtype, device="cuda")
    residual = torch.randn((M, N), dtype=dtype, device="cuda")
    weight = torch.randn((N,), dtype=dtype, device="cuda")

    config, _ = _get_gemm_config(M, N, K)
    partials = gemm_a16w16(x, w, dtype=dtype, config=config, skip_reduce=True)
    assert partials.shape == (config["NUM_KSPLIT"], M, N)
    assert partials.dtype == torch.float32

    out, residual_out = fused_reduce_add_rmsnorm(
        partials, residual, weight, 1e-6, gemma_norm=gemma_norm
    )
    y = gemm_a16w16(x, w, dtype=dtype, config=config)
    out_ref, residual_ref = run_torch(y, residual, weight, 1e-6, gemma_norm)

    # The reference rounds the GEMM output to bf16 before the add; the fused path does not.
    torch.testing.assert_close(residual_out, residual_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out, out_ref, atol=2e-2, rtol=2e-2)
