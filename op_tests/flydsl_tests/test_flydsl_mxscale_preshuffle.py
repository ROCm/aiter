# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU correctness coverage for FlyDSL MX-scale preshuffle GEMM."""

import pytest
import torch
import torch.nn.functional as F

from aiter import dtypes
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.quant import per_1x32_f4_quant, per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight
from aiter.test_common import checkAllclose
from aiter.utility import fp4_utils

if not torch.cuda.is_available():
    pytest.skip("ROCm is unavailable", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip("FlyDSL is unavailable", allow_module_level=True)

from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
    flydsl_mxscale_preshuffle_gemm,
    gemm_mxscale_preshuffle,
    get_mxscale_preshuffle_config,
)

_SHAPES = [
    (64, 8192, 8192, 64, 128, 128),
    (32, 8192, 8192, 32, 128, 256),
]
_SPLITK_SHAPES = [
    (8, 2048, 7168, 32, 128, 256, 2),
    (8, 2048, 7168, 32, 128, 256, 4),
    (1, 2048, 7168, 32, 128, 128, 2),
    (16, 4096, 8192, 32, 128, 256, 8),
]


def _require_gfx950():
    architecture = str(get_rocm_arch())
    if architecture != "gfx950":
        pytest.skip(f"MX-scale preshuffle GEMM requires gfx950, got {architecture}")


def _random_inputs(M, N, K):
    m_aligned = (M + 31) // 32 * 32
    n_aligned = (N + 31) // 32 * 32
    a = torch.zeros(m_aligned, K, device="cuda")
    b = torch.zeros(n_aligned, K, device="cuda")
    a[:M] = torch.randn(M, K, device="cuda")
    b[:N] = torch.randn(N, K, device="cuda")
    return a, b


def _assert_close(reference, actual, message):
    error = checkAllclose(
        reference,
        actual,
        rtol=1e-2,
        atol=1e-2,
        msg=message,
        catastrophic_check=True,
    )
    assert error < 0.01, f"{message} mismatch ratio={error:.4f}"


@pytest.mark.parametrize("M,N,K,tile_m,tile_n,tile_k", _SHAPES)
def test_a4w4(M, N, K, tile_m, tile_n, tile_k):
    _require_gfx950()
    a_float, b_float = _random_inputs(M, N, K)
    a_quant, a_scale_raw = per_1x32_f4_quant(
        a_float, quant_dtype=dtypes.fp4x2
    )
    b_quant, b_scale_raw = per_1x32_f4_quant(
        b_float, quant_dtype=dtypes.fp4x2
    )
    a_codes, b_codes = a_quant[:M], b_quant[:N]
    b_shuffled = shuffle_weight(b_codes, layout=(16, 16))
    a_scale = shuffle_scale_a16w4(a_scale_raw, 1, False)
    b_scale = shuffle_scale_a16w4(b_scale_raw, 1, False)

    a_dequant = fp4_utils.mxfp4_to_f32(a_codes) * fp4_utils.e8m0_to_f32(
        a_scale_raw[:M].repeat_interleave(32, dim=1)
    )
    b_dequant = fp4_utils.mxfp4_to_f32(b_codes) * fp4_utils.e8m0_to_f32(
        b_scale_raw[:N].repeat_interleave(32, dim=1)
    )
    reference = F.linear(a_dequant.float(), b_dequant.float()).to(torch.bfloat16)
    out = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    flydsl_mxscale_preshuffle_gemm(
        a_codes,
        b_shuffled,
        a_scale,
        b_scale,
        out,
        a_dtype="fp4",
        b_dtype="fp4",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )
    torch.cuda.synchronize()
    _assert_close(reference, out, "a4w4")


def _prepare_a8w8(M, N, K):
    a_float, b_float = _random_inputs(M, N, K)
    a_quant, a_scale_raw = per_1x32_f8_scale_f8_quant(
        a_float, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    b_quant, b_scale_raw = per_1x32_f8_scale_f8_quant(
        b_float, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    a_codes, b_codes = a_quant[:M], b_quant[:N]
    b_shuffled = shuffle_weight(b_codes, layout=(16, 16))
    a_scale = shuffle_scale_a16w4(a_scale_raw, 1, False)
    b_scale = shuffle_scale_a16w4(b_scale_raw, 1, False)
    a_dequant = a_codes.float() * fp4_utils.e8m0_to_f32(
        a_scale_raw[:M].repeat_interleave(32, dim=1)
    )
    b_dequant = b_codes.float() * fp4_utils.e8m0_to_f32(
        b_scale_raw[:N].repeat_interleave(32, dim=1)
    )
    reference = F.linear(a_dequant.float(), b_dequant.float()).to(torch.bfloat16)
    return a_codes, b_shuffled, a_scale, b_scale, reference


@pytest.mark.parametrize("M,N,K,tile_m,tile_n,tile_k", _SHAPES)
def test_a8w8(M, N, K, tile_m, tile_n, tile_k):
    _require_gfx950()
    a_codes, b_shuffled, a_scale, b_scale, reference = _prepare_a8w8(M, N, K)
    out = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    flydsl_mxscale_preshuffle_gemm(
        a_codes,
        b_shuffled,
        a_scale,
        b_scale,
        out,
        a_dtype="fp8",
        b_dtype="fp8",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )
    torch.cuda.synchronize()
    _assert_close(reference, out, "a8w8")


def test_minimax_exact_tuned_dispatch():
    _require_gfx950()
    M, N, K = 4, 1536, 6144
    a_codes, b_shuffled, a_scale, b_scale, reference = _prepare_a8w8(M, N, K)
    config = get_mxscale_preshuffle_config(
        M, N, K, a_dtype="fp8", b_dtype="fp8"
    )
    assert config is not None
    assert (
        config["kernelName"]
        == "flydsl_mxpsh_32x32x256_F8_F8_B16_w2_x0_sk3"
    )
    out = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    gemm_mxscale_preshuffle(
        a_codes,
        b_shuffled,
        a_scale,
        b_scale,
        out,
        a_dtype="fp8",
        b_dtype="fp8",
        require_tuned=True,
    )
    torch.cuda.synchronize()
    _assert_close(reference, out, "MiniMax exact tuned dispatch")


@pytest.mark.parametrize("M,N,K,tile_m,tile_n,tile_k,split_k", _SPLITK_SHAPES)
def test_a8w8_splitk(M, N, K, tile_m, tile_n, tile_k, split_k):
    _require_gfx950()
    a_codes, b_shuffled, a_scale, b_scale, reference = _prepare_a8w8(M, N, K)

    def run(selected_split_k, workspace=None):
        out = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
        flydsl_mxscale_preshuffle_gemm(
            a_codes,
            b_shuffled,
            a_scale,
            b_scale,
            out,
            a_dtype="fp8",
            b_dtype="fp8",
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            split_k=selected_split_k,
            splitk_workspace=workspace,
        )
        torch.cuda.synchronize()
        return out

    split_out = run(split_k)
    caller_workspace = torch.empty(
        (split_k, M, N), device="cuda", dtype=torch.float32
    )
    workspace_out = run(split_k, caller_workspace)
    single_out = run(1)
    _assert_close(reference, split_out, f"a8w8 split_k={split_k}")
    _assert_close(single_out, split_out, f"split_k={split_k} versus split_k=1")
    assert torch.equal(split_out, workspace_out)
