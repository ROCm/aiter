# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the FlyDSL MXFP4/MXFP8 preshuffle GEMM (gfx950 MFMA).

Covers A4W4 (MXFP4 A x MXFP4 B) and A8W8 (MXFP8 A x MXFP8 B), per-1x32 E8M0
microscale folded into the scaled 16x16x128 MFMA. All quant/shuffle/dequant
reuse aiter's own helpers (aiter.ops.quant / aiter.ops.shuffle /
aiter.utility.fp4_utils) — no kernel-specific reference port is needed.

Usage:
    python aiter/ops/flydsl/test_flydsl_mxfp4_preshuffle.py
    pytest -q aiter/ops/flydsl/test_flydsl_mxfp4_preshuffle.py
"""

from __future__ import annotations

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip("flydsl is not installed. Skipping FlyDSL tests.", allow_module_level=True)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

from aiter import dtypes  # noqa: E402
from aiter.ops.quant import per_1x32_f4_quant, per_1x32_f8_scale_f8_quant  # noqa: E402
from aiter.ops.shuffle import shuffle_weight, shuffle_scale_a16w4  # noqa: E402
from aiter.utility import fp4_utils  # noqa: E402
from aiter.ops.flydsl.mxfp4_preshuffle_kernels import flydsl_mxfp4_preshuffle_gemm  # noqa: E402

torch.set_default_device("cuda")

_SHAPES = [
    (64, 8192, 8192, 64, 128, 128),
    (32, 8192, 8192, 32, 128, 256),
]


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _skip_if_not_gfx950():
    arch = str(get_rocm_arch())
    if arch != "gfx950":
        pytest.skip(f"MXFP preshuffle GEMM requires gfx950, got {arch}")


def _rand_ab(M, N, K, dev):
    Ma, Na = (M + 31) // 32 * 32, (N + 31) // 32 * 32
    a_f = torch.zeros(Ma, K, device=dev)
    b_f = torch.zeros(Na, K, device=dev)
    a_f[:M] = torch.randn(M, K, device=dev)
    b_f[:N] = torch.randn(N, K, device=dev)
    return a_f, b_f


@pytest.mark.parametrize("M, N, K, tile_m, tile_n, tile_k", _SHAPES)
def test_a4w4(M, N, K, tile_m, tile_n, tile_k):
    """MXFP4 A x MXFP4 B."""
    _skip_if_not_gfx950()
    dev = torch.device("cuda")
    a_f, b_f = _rand_ab(M, N, K, dev)

    a_q, sa = per_1x32_f4_quant(a_f, quant_dtype=dtypes.fp4x2)
    b_q, sb = per_1x32_f4_quant(b_f, quant_dtype=dtypes.fp4x2)
    a_codes, b_codes = a_q[:M], b_q[:N]
    b_shuf = shuffle_weight(b_codes, layout=(16, 16))
    scale_a = shuffle_scale_a16w4(sa, 1, False)
    scale_b = shuffle_scale_a16w4(sb, 1, False)

    a_deq = fp4_utils.mxfp4_to_f32(a_codes) * fp4_utils.e8m0_to_f32(sa[:M].repeat_interleave(32, dim=1))
    b_deq = fp4_utils.mxfp4_to_f32(b_codes) * fp4_utils.e8m0_to_f32(sb[:N].repeat_interleave(32, dim=1))
    c_ref = (a_deq @ b_deq.T).float()

    out = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    flydsl_mxfp4_preshuffle_gemm(
        a_codes, b_shuf, scale_a, scale_b, out,
        a_dtype="fp4", b_dtype="fp4", tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
    )
    torch.cuda.synchronize()
    cs = _cos(out, c_ref)
    assert cs > 0.99, f"a4w4 cos={cs:.5f}"


@pytest.mark.parametrize("M, N, K, tile_m, tile_n, tile_k", _SHAPES)
def test_a8w8(M, N, K, tile_m, tile_n, tile_k):
    """MXFP8 (E4M3) A x MXFP8 (E4M3) B."""
    _skip_if_not_gfx950()
    dev = torch.device("cuda")
    a_f, b_f = _rand_ab(M, N, K, dev)

    a_q, sa = per_1x32_f8_scale_f8_quant(a_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)
    b_q, sb = per_1x32_f8_scale_f8_quant(b_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)
    a_codes, b_codes = a_q[:M], b_q[:N]
    b_shuf = shuffle_weight(b_codes, layout=(16, 16))
    scale_a = shuffle_scale_a16w4(sa, 1, False)
    scale_b = shuffle_scale_a16w4(sb, 1, False)

    a_deq = a_codes.float() * fp4_utils.e8m0_to_f32(sa[:M].repeat_interleave(32, dim=1))
    b_deq = b_codes.float() * fp4_utils.e8m0_to_f32(sb[:N].repeat_interleave(32, dim=1))
    c_ref = (a_deq @ b_deq.T).float()

    out = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    flydsl_mxfp4_preshuffle_gemm(
        a_codes, b_shuf, scale_a, scale_b, out,
        a_dtype="fp8", b_dtype="fp8", tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
    )
    torch.cuda.synchronize()
    cs = _cos(out, c_ref)
    assert cs > 0.99, f"a8w8 cos={cs:.5f}"


if __name__ == "__main__":
    for shp in _SHAPES:
        test_a4w4(*shp)
        test_a8w8(*shp)
        print(f"OK {shp}")
