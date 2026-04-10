# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Quick correctness tests for interleaved, basic_pipelined, and basic_pipelined_unrolled gluon GEMM kernels."""

import torch
import torch.nn.functional as F
import pytest
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16, _is_gluon_available


requires_gluon = pytest.mark.skipif(
    not _is_gluon_available(), reason="Gluon not supported on this architecture"
)

# Shapes that exercise the kernels (BLOCK_K=64 requires K >= 64)
SHAPES = [
    (256, 256, 64),       # minimal: one K tile
    (256, 256, 128),      # two K tiles
    (256, 256, 192),      # three K tiles (needs K padding to 256)
    (1024, 2048, 2880),   # the target shape from run_kernel.py
    (512, 512, 512),      # square
    (128, 256, 640),      # smaller M
]

# Extra shapes to test even/odd K-tile counts for unrolled kernel
UNROLLED_SHAPES = SHAPES + [
    (256, 256, 256),      # 4 tiles → main_iters=2 (even)
    (256, 256, 320),      # 5 tiles → main_iters=3 (odd)
    (256, 256, 384),      # 6 tiles → main_iters=4 (even)
    (256, 256, 448),      # 7 tiles → main_iters=5 (odd)
]

CONFIG_3BUF = {
    "BLOCK_M": 256,
    "BLOCK_N": 256,
    "BLOCK_K": 64,
    "NUM_BUFFERS": 3,
    "num_warps": 4,
    "L2_PREFETCH_DISTANCE": 0,
}


# ── interleaved kernel tests ──


@requires_gluon
@pytest.mark.parametrize("M,N,K", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_interleaved_correctness(M, N, K, dtype):
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")

    ref = F.linear(x, w)
    out = gemm_a16w16(
        x, w, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="interleaved",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


@requires_gluon
def test_interleaved_with_bias():
    M, N, K = 1024, 2048, 2880
    dtype = torch.bfloat16
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    ref = F.linear(x, w, bias=bias)
    out = gemm_a16w16(
        x, w, bias=bias, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="interleaved",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


@requires_gluon
@pytest.mark.parametrize("num_buffers", [2, 3])
def test_interleaved_num_buffers(num_buffers):
    """Verify the kernel works with both double and triple buffering."""
    M, N, K = 256, 256, 256
    dtype = torch.bfloat16
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")

    config = {**CONFIG_3BUF, "NUM_BUFFERS": num_buffers}
    ref = F.linear(x, w)
    out = gemm_a16w16(
        x, w, dtype=dtype, config=config,
        backend="gluon", kernel_type="interleaved",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


# ── basic_pipelined kernel tests ──


@requires_gluon
@pytest.mark.parametrize("M,N,K", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_basic_pipelined_correctness(M, N, K, dtype):
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")

    ref = F.linear(x, w)
    out = gemm_a16w16(
        x, w, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="basic_pipelined",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


@requires_gluon
def test_basic_pipelined_with_bias():
    M, N, K = 1024, 2048, 2880
    dtype = torch.bfloat16
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    ref = F.linear(x, w, bias=bias)
    out = gemm_a16w16(
        x, w, bias=bias, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="basic_pipelined",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


# ── basic_pipelined_unrolled kernel tests ──


@requires_gluon
@pytest.mark.parametrize("M,N,K", UNROLLED_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_basic_pipelined_unrolled_correctness(M, N, K, dtype):
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")

    ref = F.linear(x, w)
    out = gemm_a16w16(
        x, w, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="basic_pipelined_unrolled",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


@requires_gluon
def test_basic_pipelined_unrolled_with_bias():
    M, N, K = 1024, 2048, 2880
    dtype = torch.bfloat16
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    ref = F.linear(x, w, bias=bias)
    out = gemm_a16w16(
        x, w, bias=bias, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="basic_pipelined_unrolled",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


# ── interleaved_pipelined kernel tests ──


@requires_gluon
@pytest.mark.parametrize("M,N,K", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_interleaved_pipelined_correctness(M, N, K, dtype):
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")

    ref = F.linear(x, w)
    out = gemm_a16w16(
        x, w, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="interleaved_pipelined",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


@requires_gluon
def test_interleaved_pipelined_with_bias():
    M, N, K = 1024, 2048, 2880
    dtype = torch.bfloat16
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    ref = F.linear(x, w, bias=bias)
    out = gemm_a16w16(
        x, w, bias=bias, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="interleaved_pipelined",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


# ── interleaved_pipelined_unrolled kernel tests ──


@requires_gluon
@pytest.mark.parametrize("M,N,K", UNROLLED_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_interleaved_pipelined_unrolled_correctness(M, N, K, dtype):
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")

    ref = F.linear(x, w)
    out = gemm_a16w16(
        x, w, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="interleaved_pipelined_unrolled",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


@requires_gluon
def test_interleaved_pipelined_unrolled_with_bias():
    M, N, K = 1024, 2048, 2880
    dtype = torch.bfloat16
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    ref = F.linear(x, w, bias=bias)
    out = gemm_a16w16(
        x, w, bias=bias, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="interleaved_pipelined_unrolled",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


# ── finer_interleaved_pipelined kernel tests ──


@requires_gluon
@pytest.mark.parametrize("M,N,K", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_finer_interleaved_pipelined_correctness(M, N, K, dtype):
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")

    ref = F.linear(x, w)
    out = gemm_a16w16(
        x, w, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="finer_interleaved_pipelined",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


@requires_gluon
def test_finer_interleaved_pipelined_with_bias():
    M, N, K = 1024, 2048, 2880
    dtype = torch.bfloat16
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    ref = F.linear(x, w, bias=bias)
    out = gemm_a16w16(
        x, w, bias=bias, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="finer_interleaved_pipelined",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


# ── basic_pipelined_v2 kernel tests ──


@requires_gluon
@pytest.mark.parametrize("M,N,K", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_buffers", [2, 3])
def test_basic_pipelined_v2_correctness(M, N, K, dtype, num_buffers):
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")

    config = {**CONFIG_3BUF, "NUM_BUFFERS": num_buffers}
    ref = F.linear(x, w)
    out = gemm_a16w16(
        x, w, dtype=dtype, config=config,
        backend="gluon", kernel_type="basic_pipelined_v2",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


@requires_gluon
def test_basic_pipelined_v2_with_bias():
    M, N, K = 1024, 2048, 2880
    dtype = torch.bfloat16
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    ref = F.linear(x, w, bias=bias)
    out = gemm_a16w16(
        x, w, bias=bias, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="basic_pipelined_v2",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


# ── finer_interleaved_pipelined_v2 kernel tests ──


@requires_gluon
@pytest.mark.parametrize("M,N,K", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_finer_interleaved_pipelined_v2_correctness(M, N, K, dtype):
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")

    ref = F.linear(x, w)
    out = gemm_a16w16(
        x, w, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="finer_interleaved_pipelined_v2",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


@requires_gluon
def test_finer_interleaved_pipelined_v2_with_bias():
    M, N, K = 1024, 2048, 2880
    dtype = torch.bfloat16
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    ref = F.linear(x, w, bias=bias)
    out = gemm_a16w16(
        x, w, bias=bias, dtype=dtype, config=CONFIG_3BUF,
        backend="gluon", kernel_type="finer_interleaved_pipelined_v2",
    )

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
