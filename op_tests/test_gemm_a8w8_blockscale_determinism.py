# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch


@pytest.mark.parametrize("backend", ["ck", "cktile"])
@pytest.mark.parametrize("split_k", [2, 3])
def test_gemm_a8w8_blockscale_splitk_is_deterministic(backend, split_k):
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU is required")

    from aiter import dtypes
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
    from aiter.ops.gemm_op_a8w8 import (
        gemm_a8w8_blockscale_ck,
        gemm_a8w8_blockscale_cktile,
    )

    if get_gfx_runtime() != "gfx950" or get_cu_num() != 256:
        pytest.skip("the affected tuned split-K rows target 256-CU gfx950")
    gemm = gemm_a8w8_blockscale_ck if backend == "ck" else gemm_a8w8_blockscale_cktile

    # DeepSeek-V3.2 o_proj at decode batch 4. The gfx950 tuned config selected
    # splitK=3 for this shape, which used eight atomic writers per output tile.
    m, n, k = 4, 7168, 4096
    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda").clamp_(-3, 3)
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda").clamp_(-3, 3)
    x = x.to(dtypes.fp8)
    weight = weight.to(dtypes.fp8)
    x_scale = torch.ones((m, k // 128), dtype=torch.float32, device="cuda")
    w_scale = torch.ones((n // 128, k // 128), dtype=torch.float32, device="cuda")

    def run(requested_split_k):
        out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        if gemm is gemm_a8w8_blockscale_cktile:
            return gemm(
                x,
                weight,
                x_scale,
                w_scale,
                out,
                isBpreshuffled=False,
                splitK=requested_split_k,
            )
        return gemm(x, weight, x_scale, w_scale, out, splitK=requested_split_k)

    reference = run(1)
    outputs = [run(split_k) for _ in range(20)]
    torch.cuda.synchronize()

    max_diff = max(
        (reference.float() - output.float()).abs().max().item() for output in outputs
    )
    assert all(torch.equal(reference, output) for output in outputs), (
        f"splitK={split_k} was not clamped to deterministic splitK=1, "
        f"max absolute difference={max_diff}"
    )
