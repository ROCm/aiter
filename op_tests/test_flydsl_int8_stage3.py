# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8
from aiter.ops.shuffle import shuffle_weight
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8 import (
    generate_gemm_a8w8_inputs,
    run_torch,
)


@pytest.mark.skipif(get_gfx_runtime() != "gfx950", reason="gfx950 FlyDSL kernel")
@torch.no_grad()
def test_flydsl_int8_stage3_matches_reference():
    m, n, k = 513, 512, 512
    x, weight, _, x_scale, w_scale, _, _ = generate_gemm_a8w8_inputs(
        m,
        n,
        k,
        torch.int8,
        torch.bfloat16,
        output=False,
        shuffle=False,
    )
    weight_shuffled = shuffle_weight(weight, layout=(16, 16))
    actual = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    flydsl_preshuffle_gemm_a8(
        x,
        weight_shuffled,
        x_scale,
        w_scale,
        actual,
        tile_m=128,
        tile_n=128,
        tile_k=128,
        use_async_copy=1,
        waves_per_eu=1,
        xcd_swizzle=8,
        lds_stage=3,
        enable_scheduler=True,
        split_k=1,
    )
    expected = run_torch(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=1e-2)
