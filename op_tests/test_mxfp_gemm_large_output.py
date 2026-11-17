# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP GEMM outputs larger than the kernels' 2 GiB buffer range."""

import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.gemm_op_a4w6 import gemm_a4w6
from aiter.ops.gemm_op_a6w4 import gemm_a6w4
from aiter.ops.gemm_op_a6w6 import gemm_a6w6

# A 90112 x 12288 bf16 output is 2.06 GiB; row 87381 crosses the 2 GiB mark.
M, N, K = 90112, 12288, 256
ROWS = 256
CHECKED_TILES = (0, 87296, M - ROWS)

FAMILIES = {
    "a6w6": (aiter.quant_mxfp6_gemm, aiter.quant_mxfp6_gemm, gemm_a6w6),
    "a6w4": (aiter.quant_mxfp6_gemm, aiter.quant_mxfp4_gemm, gemm_a6w4),
    "a4w6": (aiter.quant_mxfp4_gemm, aiter.quant_mxfp6_gemm, gemm_a4w6),
}


def _is_gfx950():
    try:
        return torch.cuda.is_available() and get_gfx_runtime() == "gfx950"
    except (KeyError, RuntimeError):
        return False


@pytest.mark.skipif(not _is_gfx950(), reason="MXFP assembly GEMMs require gfx950")
@pytest.mark.parametrize("family", sorted(FAMILIES))
@torch.no_grad()
def test_output_past_2gib_matches_row_slices(family):
    quant_a, quant_b, gemm = FAMILIES[family]
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    A, A_scale = quant_a(x)
    B, B_scale = quant_b(w)
    out = gemm(A, B, A_scale, B_scale, M, N, K)
    for start in CHECKED_TILES:
        a, a_scale = quant_a(x[start : start + ROWS].contiguous())
        expected = gemm(a, B, a_scale, B_scale, ROWS, N, K)
        assert torch.equal(out[start : start + ROWS], expected), f"rows from {start}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
