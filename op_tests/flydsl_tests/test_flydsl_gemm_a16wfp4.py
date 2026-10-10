# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness coverage for dense gfx950 FlyDSL BF16 x MXFP4 GEMM."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("flydsl")

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_a16wfp4 import (
    DenseGemmConfig,
    _select_gemm_config,
    a16wfp4_config_legal,
    a16wfp4_shape_supported,
    flydsl_gemm_a16wfp4,
    prepare_gemm_a16wfp4_weight,
)
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.test_common import run_perftest
from aiter.utility import fp4_utils

_SKIP = pytest.mark.skipif(
    get_gfx() != "gfx950" or not is_flydsl_available(),
    reason="gfx950 and FlyDSL are required",
)

_E8M0_EDGE_CODES = (0x00, 0x01, 0x7E, 0x7F, 0x80, 0xFE, 0xFF)
_K3_SHAPES = ((8448, 7168), (1536, 7168), (7168, 768))


def _oracle(
    a: torch.Tensor, packed_weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    n, packed_k = packed_weight.shape
    k = packed_k * 2
    decoded = fp4_utils.mxfp4_to_f32(packed_weight)
    scale_f32 = fp4_utils.e8m0_to_f32(scale).float().unsqueeze(-1)
    weight_bf16 = (
        (decoded.float().view(n, k // 32, 32) * scale_f32)
        .reshape(n, k)
        .to(torch.bfloat16)
    )
    return F.linear(a, weight_bf16)


def _make_inputs(m: int, n: int, k: int, seed: int = 0):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    packed = torch.randint(
        0,
        256,
        (n, k // 2),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    scale = torch.full((n, k // 32), 0x7F, device="cuda", dtype=torch.uint8)
    return a, packed, scale


def _assert_close_to_oracle(m: int, n: int, k: int, seed: int):
    a, packed, scale = _make_inputs(m, n, k, seed=seed)
    a_before = a.clone()
    prepared = prepare_gemm_a16wfp4_weight(packed, scale)

    actual = flydsl_gemm_a16wfp4(a, prepared)
    expected = _oracle(a, packed, scale)

    torch.testing.assert_close(a, a_before, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=1.0)


@_SKIP
@pytest.mark.parametrize("n,k", [(1024, 3584), (3584, 512)])
@pytest.mark.parametrize("m", [1, 16, 128, 512, 4096])
def test_kimi_tp2_shapes(m: int, n: int, k: int):
    _assert_close_to_oracle(m, n, k, seed=m + n + k)


@_SKIP
@pytest.mark.parametrize("n,k", list(_K3_SHAPES))
@pytest.mark.parametrize("m", [1, 128])
def test_kimi_k3_shapes(m: int, n: int, k: int):
    _assert_close_to_oracle(m, n, k, seed=m + n + k)


@_SKIP
def test_k3_qkvo_fallback_shape_is_rejected():
    assert a16wfp4_shape_supported(7168, 4224) is False
    n, k = 7168, 4224
    packed = torch.zeros((n, k // 2), device="cuda", dtype=torch.uint8)
    scale = torch.zeros((n, k // 32), device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="K must be divisible by 256"):
        prepare_gemm_a16wfp4_weight(packed, scale)


@_SKIP
@pytest.mark.parametrize("n,k", list(_K3_SHAPES))
@pytest.mark.parametrize("m", [1, 4096])
def test_kimi_k3_microbench(m: int, n: int, k: int):
    a, packed, scale = _make_inputs(m, n, k, seed=m + n + k)
    prepared = prepare_gemm_a16wfp4_weight(packed, scale)
    flydsl_gemm_a16wfp4(a, prepared)
    torch.cuda.synchronize()

    _, us = run_perftest(
        flydsl_gemm_a16wfp4,
        a,
        prepared,
        num_iters=21,
        num_warmup=2,
        num_rotate_args=1,
        use_cuda_event=True,
    )
    flops = 2.0 * m * n * k
    tflops = flops / (us * 1e-6) / 1e12 if us > 0 else 0.0
    bytes_moved = m * k * 2 + n * (k / 2) + n * (k / 32) + m * n * 2
    tb_s = bytes_moved / (us * 1e-6) / 1e12 if us > 0 else 0.0
    cfg = _select_gemm_config(m, n, k)
    print(
        f"a16wfp4 M={m} N={n} K={k} BM={cfg.block_m} TN={cfg.tile_n} "
        f"k_wave={cfg.k_wave} wpe={cfg.waves_per_eu}: "
        f"{us:.2f} us, {tflops:.2f} TFLOPS, {tb_s:.2f} TB/s"
    )
    assert us > 0


@_SKIP
def test_exact_e8m0_edge_classification_against_canonical_oracle():
    m, n, k = 1, 1024, 512
    a = torch.zeros((m, k), device="cuda", dtype=torch.bfloat16)
    a[0, 0] = 1
    packed = torch.zeros((n, k // 2), device="cuda", dtype=torch.uint8)
    scale = torch.full((n, k // 32), 0x7F, device="cuda", dtype=torch.uint8)

    row_codes = torch.tensor(_E8M0_EDGE_CODES, device="cuda", dtype=torch.uint8)
    repeats = (n + len(_E8M0_EDGE_CODES) - 1) // len(_E8M0_EDGE_CODES)
    row_codes = row_codes.repeat(repeats)[:n]
    scale[:, 0] = row_codes

    # Low nibble is W[:, 0]. Use E2M1 1.0 normally and 6.0 with scale 0xfe,
    # which overflows BF16 to inf. Scale 0xff produces canonical NaNs.
    weight_codes = torch.full((n,), 0x2, device="cuda", dtype=torch.uint8)
    weight_codes[row_codes == 0xFE] = 0x7
    packed[:, 0] = weight_codes

    prepared = prepare_gemm_a16wfp4_weight(packed, scale)
    actual = flydsl_gemm_a16wfp4(a, prepared)
    expected = _oracle(a, packed, scale)

    torch.testing.assert_close(torch.isnan(actual), torch.isnan(expected))
    torch.testing.assert_close(torch.isinf(actual), torch.isinf(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], rtol=0, atol=0)

    for code in _E8M0_EDGE_CODES:
        rows = row_codes == code
        assert rows.any()
        torch.testing.assert_close(
            torch.isnan(actual[0, rows]), torch.isnan(expected[0, rows])
        )
        torch.testing.assert_close(
            torch.isinf(actual[0, rows]), torch.isinf(expected[0, rows])
        )


@_SKIP
def test_prepared_storage_and_warmed_launch_do_not_materialize_bf16_weight():
    m, n, k = 16, 1024, 3584
    a, packed, scale = _make_inputs(m, n, k, seed=7)
    prepared = prepare_gemm_a16wfp4_weight(packed, scale)

    assert prepared.weight.dtype == torch.uint8
    assert prepared.scale.dtype == torch.uint8
    assert prepared.weight.element_size() == packed.element_size() == 1
    assert prepared.scale.element_size() == scale.element_size() == 1
    assert prepared.weight.numel() == packed.numel()
    assert prepared.scale.numel() == scale.numel()
    assert prepared.weight.untyped_storage().nbytes() == packed.numel()
    assert prepared.scale.untyped_storage().nbytes() == scale.numel()
    assert prepared.weight.numel() + prepared.scale.numel() == (
        packed.numel() + scale.numel()
    )

    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    returned = flydsl_gemm_a16wfp4(a, prepared, out=out)
    torch.cuda.synchronize()
    assert returned.data_ptr() == out.data_ptr()

    torch.cuda.reset_peak_memory_stats(a.device)
    baseline = torch.cuda.memory_allocated(a.device)
    returned = flydsl_gemm_a16wfp4(a, prepared, out=out)
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated(a.device) - baseline

    assert returned.data_ptr() == out.data_ptr()
    assert peak_delta < n * k * torch.bfloat16.itemsize, (
        f"warmed launch allocated {peak_delta} bytes, enough for a full "
        f"BF16 [{n}, {k}] weight"
    )


@_SKIP
def test_shape_and_dtype_validation():
    assert a16wfp4_shape_supported(1024, 3584)
    assert a16wfp4_shape_supported(8448, 7168)
    assert a16wfp4_shape_supported(1536, 7168)
    assert a16wfp4_shape_supported(7168, 768)
    assert not a16wfp4_shape_supported(7168, 4224)

    assert _select_gemm_config(4095, 1024, 3584) == (16, 128, 256, 1, 2)
    assert _select_gemm_config(4096, 1024, 3584) == (32, 256, 256, 1, 1)
    assert _select_gemm_config(2047, 1024, 512) == (16, 128, 256, 1, 2)
    assert _select_gemm_config(2048, 1024, 512) == (32, 256, 256, 1, 1)
    assert _select_gemm_config(4095, 1024, 768) == (16, 128, 256, 1, 2)
    assert _select_gemm_config(1, 1024, 3584) == (16, 32, 256, 2, 1)
    assert _select_gemm_config(1, 8448, 7168) == (16, 32, 256, 4, 4)
    assert _select_gemm_config(32, 8448, 7168) == (16, 64, 256, 4, 4)
    assert _select_gemm_config(1, 1536, 7168) == (16, 16, 256, 4, 1)
    assert _select_gemm_config(1, 7168, 768) == (16, 128, 256, 1, 2)
    assert _select_gemm_config(128, 8448, 7168) == (32, 128, 256, 1, 1)
    assert _select_gemm_config(512, 8448, 7168) == (48, 192, 256, 1, 2)
    assert _select_gemm_config(1024, 8448, 7168) == (64, 192, 128, 1, 2)
    assert _select_gemm_config(4096, 8448, 7168) == (64, 192, 128, 1, 2)
    assert _select_gemm_config(128, 1536, 7168) == (16, 96, 128, 4, 2)
    assert _select_gemm_config(767, 1536, 7168) == (16, 96, 128, 4, 2)
    assert _select_gemm_config(768, 1536, 7168) == (32, 128, 256, 1, 2)
    assert _select_gemm_config(2048, 1536, 7168) == (32, 256, 128, 1, 1)
    assert _select_gemm_config(4096, 1536, 7168) == (32, 256, 128, 1, 1)
    assert _select_gemm_config(128, 7168, 768) == (32, 128, 256, 1, 2)
    assert _select_gemm_config(512, 7168, 768) == (32, 128, 256, 1, 2)
    assert _select_gemm_config(1024, 7168, 768) == (64, 256, 128, 1, 2)
    assert _select_gemm_config(4096, 7168, 768) == (64, 256, 128, 1, 2)
    assert _select_gemm_config(512, 3584, 512) == (32, 128, 256, 1, 2)
    assert _select_gemm_config(1024, 3584, 512) == (32, 256, 128, 1, 1)

    _, packed, scale = _make_inputs(1, 1024, 512)
    prepared = prepare_gemm_a16wfp4_weight(packed, scale)

    with pytest.raises(TypeError, match="BF16"):
        flydsl_gemm_a16wfp4(torch.randn(1, 512, device="cuda"), prepared)
    with pytest.raises(ValueError, match="does not match"):
        flydsl_gemm_a16wfp4(
            torch.randn(1, 256, device="cuda", dtype=torch.bfloat16), prepared
        )
    with pytest.raises(ValueError, match="scale must have shape"):
        prepare_gemm_a16wfp4_weight(packed, scale[:, :-1])


@_SKIP
def test_decode_configs_fill_the_grid():
    """Decode is grid-bound: never idle a CU while a narrower legal tile exists."""
    for n, k in ((8448, 7168), (1536, 7168), (7168, 768), (1024, 3584), (3584, 512)):
        for m in (1, 8, 16, 17, 32, 63):
            cfg = _select_gemm_config(m, n, k)
            assert a16wfp4_config_legal(n, k, cfg)
            blocks = ((m + cfg.block_m - 1) // cfg.block_m) * (n // cfg.tile_n)
            if cfg.k_wave == 1:
                continue  # short K cannot hide the dispatch of a narrower tile
            if blocks >= 256:
                continue
            narrower = cfg._replace(tile_n=cfg.tile_n // 2)
            assert not a16wfp4_config_legal(n, k, narrower), (
                f"({n},{k}) M={m} leaves {blocks} tiles for 256 CUs while "
                f"TILE_N={narrower.tile_n} is legal"
            )


@_SKIP
def test_explicit_config_override_matches_oracle():
    m, n, k = 1, 1024, 512
    a, packed, scale = _make_inputs(m, n, k, seed=3)
    prepared = prepare_gemm_a16wfp4_weight(packed, scale)
    cfg = DenseGemmConfig(16, 128, 256, 1, 1)
    actual = flydsl_gemm_a16wfp4(a, prepared, config=cfg)
    expected = _oracle(a, packed, scale)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=1.0)
