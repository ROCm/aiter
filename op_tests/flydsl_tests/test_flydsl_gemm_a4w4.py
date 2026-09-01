# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available

_SKIP = pytest.mark.skipif(
    get_gfx() != "gfx950" or not is_flydsl_available(),
    reason="gfx950 and FlyDSL are required",
)


@pytest.mark.parametrize(
    "m,n,k,expected",
    [
        (127, 8448, 7168, 16),
        (128, 8448, 7168, 64),
        (511, 7168, 4224, 16),
        (512, 7168, 4224, 64),
        (2047, 1536, 7168, 16),
        (2048, 1536, 7168, 64),
        (511, 7168, 768, 16),
        (512, 7168, 768, 64),
        (4096, 256, 7168, 16),
    ],
)
def test_flydsl_dense_a4w4_bm_selection(m, n, k, expected):
    from aiter.ops.flydsl.gemm_a4w4 import _select_bm

    assert _select_bm(m, n, k) == expected


@_SKIP
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 1536, 7168),
        (17, 1536, 7168),
        (1, 1536, 4224),
        (17, 1536, 4224),
    ],
)
def test_flydsl_dense_a4w4_matches_triton(m, n, k):
    from aiter.ops.flydsl.gemm_a4w4 import (
        flydsl_gemm_a4w4,
        prepare_gemm_a4w4_weight,
    )
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    torch.manual_seed(1)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    w, w_scale = dynamic_mxfp4_quant(w_bf16)
    a_quant, a_scale = dynamic_mxfp4_quant(a)

    expected = gemm_afp4wfp4(a_quant, w, a_scale, w_scale, dtype=torch.bfloat16)
    prepared = prepare_gemm_a4w4_weight(w, w_scale)
    actual = flydsl_gemm_a4w4(a, prepared, _bm=16)

    actual_f32, expected_f32 = actual.float(), expected.float()
    assert torch.isfinite(actual_f32).all()
    assert torch.isfinite(expected_f32).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    )
    relative_rmse = (actual_f32 - expected_f32).square().mean().sqrt()
    relative_rmse /= expected_f32.square().mean().sqrt().clamp_min(1e-6)
    assert cosine > 0.998
    assert relative_rmse < 0.06


@_SKIP
@pytest.mark.parametrize("k", [7168, 4224])
@pytest.mark.parametrize("m", [1, 15, 16, 17, 63, 64, 65, 127, 128, 129])
def test_flydsl_dense_a4w4_bm64_boundaries(m, k):
    from aiter.ops.flydsl.gemm_a4w4 import (
        flydsl_gemm_a4w4,
        prepare_gemm_a4w4_weight,
    )
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    n = 256
    torch.manual_seed(2)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w, w_scale = dynamic_mxfp4_quant(
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    )
    a_quant, a_scale = dynamic_mxfp4_quant(a)
    expected = gemm_afp4wfp4(a_quant, w, a_scale, w_scale, dtype=torch.bfloat16)

    actual = flydsl_gemm_a4w4(a, prepare_gemm_a4w4_weight(w, w_scale), _bm=64)
    actual_f32, expected_f32 = actual.float(), expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    )
    relative_rmse = (actual_f32 - expected_f32).square().mean().sqrt()
    relative_rmse /= expected_f32.square().mean().sqrt().clamp_min(1e-6)
    assert torch.isfinite(actual_f32).all()
    assert cosine > 0.998
    assert relative_rmse < 0.06


@_SKIP
def test_flydsl_dense_a4w4_bm64_scale_boundaries_and_row_groups():
    from aiter.ops.flydsl.gemm_a4w4 import (
        flydsl_gemm_a4w4,
        prepare_gemm_a4w4_weight,
    )
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    m, n, k = 64, 256, 768
    base = torch.linspace(-1.0625, 1.0625, 32, device="cuda")
    rows = []
    for row in range(m):
        exponent = (row % 12) - 6
        block = base * (2.0**exponent)
        rows.append(block.repeat(k // 32))
    a = torch.stack(rows).to(torch.bfloat16)
    torch.manual_seed(3)
    w, w_scale = dynamic_mxfp4_quant(
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    )
    a_quant, a_scale = dynamic_mxfp4_quant(a)
    expected = gemm_afp4wfp4(a_quant, w, a_scale, w_scale, dtype=torch.bfloat16)
    actual = flydsl_gemm_a4w4(a, prepare_gemm_a4w4_weight(w, w_scale), _bm=64)

    actual_f32, expected_f32 = actual.float(), expected.float()
    assert torch.isfinite(actual_f32).all()
    assert torch.isfinite(expected_f32).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    )
    relative_rmse = (actual_f32 - expected_f32).square().mean().sqrt()
    relative_rmse /= expected_f32.square().mean().sqrt().clamp_min(1e-6)
    assert cosine > 0.998
    assert relative_rmse < 0.06


@_SKIP
@pytest.mark.parametrize(
    "n,k",
    [(8448, 7168), (7168, 4224), (1536, 7168), (7168, 768), (1536, 4224)],
)
def test_flydsl_dense_a4w4_required_shapes_compile_and_run(n, k):
    from aiter.ops.flydsl.gemm_a4w4 import flydsl_gemm_a4w4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    a = torch.zeros((1, k), device="cuda", dtype=torch.bfloat16)
    w, w_scale = dynamic_mxfp4_quant(
        torch.zeros((n, k), device="cuda", dtype=torch.bfloat16)
    )
    out = flydsl_gemm_a4w4(a, w, w_scale)
    assert out.shape == (1, n)
    assert not out.isnan().any()


@_SKIP
def test_flydsl_dense_a4w4_k4224_scale_padding():
    from aiter.ops.flydsl.gemm_a4w4 import prepare_gemm_a4w4_weight

    weight = torch.zeros((256, 2112), device="cuda", dtype=torch.uint8)
    weight_scale = torch.zeros((256, 132), device="cuda", dtype=torch.uint8)
    prepared = prepare_gemm_a4w4_weight(weight, weight_scale)
    assert prepared.weight.shape == (256, 2112)
    assert prepared.scale.shape == (256, 136)


@_SKIP
def test_flydsl_dense_a4w4_rejects_non_128_aligned_k():
    from aiter.ops.flydsl.gemm_a4w4 import prepare_gemm_a4w4_weight

    weight = torch.zeros((256, 96), device="cuda", dtype=torch.uint8)
    weight_scale = torch.zeros((256, 6), device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="positive multiple of 128"):
        prepare_gemm_a4w4_weight(weight, weight_scale)
