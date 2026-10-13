# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_8wave import (
    flydsl_8wave_gemm_a8,
    parse_8wave_kernel_name,
    run_gemm_a8w8_bpreshuffle_8wave,
)
from aiter.ops.shuffle import shuffle_weight


def test_int8_8wave_kernel_name_contract():
    name = "flydsl_bpreshuffle_8w_128x256x128_I8_I8_B16_1x8"
    assert parse_8wave_kernel_name(name) == {
        "block_m": 128,
        "block_n": 256,
        "block_k": 128,
        "q_dtype_a": "I8",
        "q_dtype_w": "I8",
        "dtype": "B16",
        "waves_per_eu": 1,
        "xcd_swizzle": 8,
    }


def test_8wave_kernel_name_rejects_dtype_mismatch():
    x = torch.empty((1, 256), dtype=torch.int8)
    weight = torch.empty((256, 256), dtype=torch.int8)
    scale_a = torch.ones((1, 1), dtype=torch.float32)
    scale_b = torch.ones((1, 256), dtype=torch.float32)
    out = torch.empty((1, 256), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requests fp8"):
        run_gemm_a8w8_bpreshuffle_8wave(
            x,
            weight,
            scale_a,
            scale_b,
            out,
            "flydsl_bpreshuffle_8w_128x256x128_F8_F8_B16_1x0",
        )


@pytest.mark.skipif(get_gfx_runtime() != "gfx950", reason="gfx950 FlyDSL kernel")
@torch.no_grad()
def test_flydsl_int8_8wave_bit_exact_runtime_tails():
    # K=384 exercises the steady-state body as well as both peeled pipeline
    # steps. M/N deliberately tail the 128x256 workgroup tile.
    m, n, k = 131, 272, 384
    generator = torch.Generator(device="cuda").manual_seed(20261013)
    x = torch.randint(
        -31, 32, (m, k), dtype=torch.int8, device="cuda", generator=generator
    )
    weight = torch.randint(
        -31, 32, (n, k), dtype=torch.int8, device="cuda", generator=generator
    )
    x_scale = torch.rand(
        (m, 1), dtype=torch.float32, device="cuda", generator=generator
    )
    w_scale = torch.rand(
        (1, n), dtype=torch.float32, device="cuda", generator=generator
    )
    weight_shuffled = shuffle_weight(weight, layout=(16, 16))
    actual = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    flydsl_8wave_gemm_a8(
        x,
        weight_shuffled,
        x_scale,
        w_scale,
        actual,
        block_m=128,
        block_n=256,
        waves_per_eu=1,
        xcd_swizzle=0,
    )

    # Values are intentionally small enough that the FP32 linear reference
    # represents every integer product and sum exactly. Match the kernel's
    # single post-reduction scale product and require BF16 bit identity.
    accum = torch.nn.functional.linear(x.float(), weight.float())
    expected = (accum * (x_scale * w_scale)).to(torch.bfloat16)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(get_gfx_runtime() != "gfx950", reason="gfx950 FlyDSL kernel")
@torch.no_grad()
def test_flydsl_8wave_fp8_regression():
    m, n, k = 17, 256, 256
    generator = torch.Generator(device="cuda").manual_seed(20261013)
    x_float = torch.randn(
        (m, k), dtype=torch.float32, device="cuda", generator=generator
    )
    w_float = torch.randn(
        (n, k), dtype=torch.float32, device="cuda", generator=generator
    )
    x_scale = x_float.abs().amax(dim=1, keepdim=True).clamp_min_(1e-12) / 448.0
    w_scale = (
        w_float.abs().amax(dim=1, keepdim=True).T.contiguous().clamp_min_(1e-12) / 448.0
    )
    x = (x_float / x_scale).to(torch.float8_e4m3fn)
    weight = (w_float / w_scale.T).to(torch.float8_e4m3fn)
    weight_shuffled = shuffle_weight(weight, layout=(16, 16))
    actual = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    flydsl_8wave_gemm_a8(
        x,
        weight_shuffled,
        x_scale,
        w_scale,
        actual,
        block_m=128,
        block_n=256,
        waves_per_eu=1,
        xcd_swizzle=0,
    )

    expected = torch.nn.functional.linear(x.float(), weight.float())
    expected = (expected * (x_scale * w_scale)).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=1e-2)
