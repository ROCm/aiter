# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness coverage for the gfx950 FlyDSL A16W16 GEMM."""

from __future__ import annotations

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL HGEMM tests.",
        allow_module_level=True,
    )
if get_gfx() != "gfx950":
    pytest.skip("The FlyDSL A16W16 kernel requires gfx950.", allow_module_level=True)

from aiter.ops.flydsl.gemm_kernels import (
    flydsl_hgemm,
    flydsl_hgemm_kernel_name,
    get_flydsl_hgemm_kernel_params,
)
from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
    gemm_a16w16,
)


def _layout_matrix(rows, cols, dtype, transposed):
    if transposed:
        return torch.empty((cols, rows), dtype=dtype, device="cuda").t()
    return torch.empty((rows, cols), dtype=dtype, device="cuda")


def _tolerance(k, config):
    return float(k) / 2048 * 6e-1 * config.get("split_k", 1) * config.get("k_waves", 1)


@pytest.mark.parametrize("layout", ("nn", "nt", "tn", "tt"))
@pytest.mark.parametrize("out_dtype", (torch.bfloat16, torch.float32))
def test_gemm_a16w16_layouts(layout, out_dtype):
    m, n, k = 128, 128, 256
    config = {
        "block_m": 128,
        "block_n": 128,
        "block_k": 64,
        "stages": 2,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 2,
        "k_waves": 1,
        "group_m": 0,
        "use_half_tile_interleaved": False,
    }
    a = _layout_matrix(m, k, torch.bfloat16, layout[0] == "t").uniform_(-1, 1)
    b = _layout_matrix(k, n, torch.bfloat16, layout[1] == "t").uniform_(-1, 1)
    bias = torch.empty(n, dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)

    actual = gemm_a16w16(
        a,
        b,
        bias=bias,
        user_kwargs=config,
        layout=layout,
        out_dtype=out_dtype,
    )
    expected = torch.addmm(bias.float(), a.float(), b.float()).to(out_dtype)

    tolerance = _tolerance(k, config)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "m,n,k,config",
    (
        (
            32,
            384,
            7168,
            {
                "block_m": 32,
                "block_n": 64,
                "block_k": 64,
                "stages": 5,
                "split_k": 8,
                "m_waves": 2,
                "n_waves": 2,
                "k_waves": 1,
                "group_m": 0,
                "use_half_tile_interleaved": False,
            },
        ),
        (
            256,
            256,
            256,
            {
                "block_m": 256,
                "block_n": 256,
                "block_k": 64,
                "stages": 2,
                "split_k": 1,
                "m_waves": 2,
                "n_waves": 4,
                "k_waves": 1,
                "group_m": 0,
                "use_half_tile_interleaved": True,
            },
        ),
    ),
)
@pytest.mark.parametrize("with_bias", (False, True))
def test_flydsl_hgemm_policies(m, n, k, config, with_bias):
    a = torch.empty((m, k), dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)
    weight = torch.empty((n, k), dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)
    bias = (
        torch.empty(n, dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)
        if with_bias
        else None
    )

    actual = flydsl_hgemm(
        a,
        weight,
        bias=bias,
        tile_m=config["block_m"],
        tile_n=config["block_n"],
        tile_k=config["block_k"],
        stages=config["stages"],
        split_k=config["split_k"],
        block_m_warps=config["m_waves"],
        block_n_warps=config["n_waves"],
        block_k_warps=config["k_waves"],
        group_m=config["group_m"],
        policy="ht" if config["use_half_tile_interleaved"] else "ft",
    )
    expected = torch.nn.functional.linear(a.float(), weight.float())
    if bias is not None:
        expected += bias.float()
    expected = expected.to(actual.dtype)

    tolerance = _tolerance(k, config)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_flydsl_hgemm_kernel_name_round_trip():
    config = {
        "block_m": 256,
        "block_n": 256,
        "block_k": 64,
        "stages": 2,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 4,
        "k_waves": 1,
        "group_m": 4,
        "use_half_tile_interleaved": True,
    }
    name = flydsl_hgemm_kernel_name(
        dtype=torch.bfloat16,
        out_dtype=torch.float32,
        config=config,
        has_bias=False,
    )
    assert name == (
        "flydsl_hgemm_abf16_wbf16_fp32_t256x256x64x2_ks1_"
        "w2x4x1_bias0_ktail0_gm4_pht_gfx950"
    )
    parsed = get_flydsl_hgemm_kernel_params(name)

    assert parsed is not None
    for key, value in config.items():
        assert parsed[key] == value
    assert parsed["dtype"] == "bf16"
    assert parsed["out_dtype"] == "fp32"
    assert parsed["has_bias"] is False
