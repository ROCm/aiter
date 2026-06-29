# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16, _is_gluon_available
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config


def _require_gfx1250_gluon():
    if not torch.cuda.is_available():
        pytest.skip("FFM a16w16 smoke test requires a CUDA/HIP GPU")
    if not _is_gluon_available():
        pytest.skip("FFM a16w16 smoke test requires gfx1250/Gluon backend")


@pytest.mark.parametrize(
    "M,N,K",
    [
        (16, 128, 2880),
        (32, 5120, 2880),
        (64, 2880, 4096),
    ],
)
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_gemm_a16w16_ffm_smoke(M: int, N: int, K: int, kernel_type: str):
    _require_gfx1250_gluon()

    cfg, _ = get_gemm_config("GEMM-A16W16", M, N, K)
    assert K % cfg["BLOCK_K"] == 0

    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")

    expected = F.linear(x, w)
    actual = gemm_a16w16(x, w, backend="gluon", kernel_type=kernel_type)

    torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-2)
