# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_dual_splitk import (
    fused_gemm_a16w16_dual_splitk,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)


def run_torch(x, w, dtype=torch.bfloat16):
    return F.linear(x.float(), w.float()).to(dtype)


def run_triton(
    x, w, n2, dtype=torch.bfloat16, y=None, skip_reduce=False, config=None, **kwargs
):
    return fused_gemm_a16w16_dual_splitk(
        x, w, n2, dtype=dtype, y=y, skip_reduce=skip_reduce, config=config, **kwargs
    )


def _config(ks1, ks2, bm=16, bn1=64, bk1=128, bn2=32, bk2=128):
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N1": bn1,
        "BLOCK_SIZE_K1": bk1,
        "NUM_KSPLIT1": ks1,
        "BLOCK_SIZE_N2": bn2,
        "BLOCK_SIZE_K2": bk2,
        "NUM_KSPLIT2": ks2,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg",
    }


def get_x_vals():
    x_vals = [(1, 1, 1, 128)]  # minimal case
    x_vals += [
        (m, n1, n2, k)
        for k in [1024, 5120]
        for n1, n2 in [(256, 32), (4096, 96)]
        for m in [1, 8, 64, 128]
    ]
    x_vals += [(m, 16384, 96, 5120) for m in [1, 4, 16, 64, 128]]  # tuned shape
    return x_vals


# Partial tiles and a K tail, and a second part with more split-K pieces than the first
# part has tiles (programs that run only part 2).
_ODD_SHAPES = [(33, 1000, 40, 1000), (5, 256, 24, 512), (16, 64, 96, 4096)]


def _assert_close(y, ref, n1):
    triton.testing.assert_close(y[:, :n1], ref[:, :n1], atol=0.1, rtol=1e-1)
    triton.testing.assert_close(y[:, n1:], ref[:, n1:], atol=0.1, rtol=1e-1)


@pytest.mark.parametrize("M, N1, N2, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_gemm(M, N1, N2, K, dtype, output):
    x, w, _, _, y = generate_gemm_a16w16_inputs(M, N1 + N2, K, dtype, output=output)
    y_triton = run_triton(x, w, N2, dtype, y)
    if output:
        assert y_triton.data_ptr() == y.data_ptr()
    _assert_close(y_triton, run_torch(x, w, dtype), N1)


@pytest.mark.parametrize("M, N1, N2, K", get_x_vals()[1:5] + _ODD_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_gemm_skip_reduce(M, N1, N2, K, dtype, output):
    x, w, _, _, y = generate_gemm_a16w16_inputs(M, N1 + N2, K, dtype, output=output)
    y_triton, y2_pp = run_triton(
        x, w, N2, dtype, y, skip_reduce=True, config=_config(1, 4)
    )
    assert y2_pp.shape == (4, M, N2) and y2_pp.dtype == torch.float32
    ref = run_torch(x, w, dtype)
    triton.testing.assert_close(y_triton[:, :N1], ref[:, :N1], atol=0.1, rtol=1e-1)
    triton.testing.assert_close(
        y2_pp.sum(0).to(dtype), ref[:, N1:], atol=0.1, rtol=1e-1
    )


@pytest.mark.parametrize("M, N1, N2, K", [(64, 16384, 96, 5120)] + _ODD_SHAPES)
@pytest.mark.parametrize("ks1, ks2", [(1, 1), (1, 4), (2, 1), (2, 4)])
def test_gemm_ksplit(M, N1, N2, K, ks1, ks2):
    """Explicit split counts per part, through both the first-part reduce and the
    second-part reduce."""
    x, w, _, _, _ = generate_gemm_a16w16_inputs(
        M, N1 + N2, K, torch.bfloat16, output=False
    )
    y_triton = run_triton(x, w, N2, config=_config(ks1, ks2))
    _assert_close(y_triton, run_torch(x, w), N1)


@pytest.mark.parametrize("M, N1, N2, K", [(4, 16384, 96, 5120), (16, 1000, 40, 1000)])
@pytest.mark.parametrize("num_ksplit2", [3, 10, 20])
def test_gemm_exact_num_ksplit2(M, N1, N2, K, num_ksplit2):
    """num_ksplit2 is kept exactly, including splits past K (zero partials)."""
    x, w, _, _, _ = generate_gemm_a16w16_inputs(
        M, N1 + N2, K, torch.bfloat16, output=False
    )
    y_triton, y2_pp = run_triton(x, w, N2, skip_reduce=True, num_ksplit2=num_ksplit2)
    assert y2_pp.shape == (num_ksplit2, M, N2)
    ref = run_torch(x, w)
    triton.testing.assert_close(y_triton[:, :N1], ref[:, :N1], atol=0.1, rtol=1e-1)
    triton.testing.assert_close(
        y2_pp.sum(0).to(torch.bfloat16), ref[:, N1:], atol=0.1, rtol=1e-1
    )
