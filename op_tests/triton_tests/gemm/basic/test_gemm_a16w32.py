# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter.ops.triton.gemm.basic.gemm_a16w32 as a16w32_op
from aiter.ops.triton.gemm.basic.gemm_a16w32 import gemm_a16w32, split_bf16x3
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.utility.graph_alloc import ROUTES_INSIDE_CAPTURE, capture_scoped_cache

pytestmark = pytest.mark.skipif(
    "gfx950" not in get_arch(), reason="gemm_a16w32 has configs for gfx950 only"
)


def generate_gemm_a16w32_inputs(M, N, K, output=True, seed=0):
    """(x, w, w_bf16x3, y): bf16 x (M, K), the fp32 weight w (N, K), its bf16 terms
    (3, N, K) and an fp32 output (M, N), or None."""
    torch.manual_seed(seed)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.float32, device="cuda") * 0.02
    w_bf16x3 = split_bf16x3(w)
    y = torch.empty((M, N), dtype=torch.float32, device="cuda") if output else None
    return x, w, w_bf16x3, y


def _check(out, x, w):
    # An fp64 reference, and a bound scaled by sum |x| |w| for the fp32 accumulation.
    ref = x.double() @ w.double().T
    scale = (x.double().abs() @ w.double().abs().T).max().item()
    torch.testing.assert_close(out.double(), ref, rtol=0, atol=4e-6 * scale)


SHAPES_NK = [(128, 6144), (256, 3072), (128, 7168)]


def test_split_bf16x3_is_exact():
    _, w, w_bf16x3, _ = generate_gemm_a16w32_inputs(4, 128, 6144, output=False)
    total = w_bf16x3[0].float() + w_bf16x3[1].float() + w_bf16x3[2].float()
    assert torch.equal(total, w)


def test_split_bf16x3_limits():
    # Exact from 2**-110 to below 2**128 - 2**119, and lossy only under 2**-133.
    ulp = 2.0**-23
    exact = [0.0, 2.0**-110 * (1 + ulp), -(2.0**-110) * (2 - ulp), 2.0**128 - 2.0**120]
    tiny = [2.0**-126 + 2.0**-149, -(2.0**-120) * (1 + ulp)]
    w = torch.tensor([exact + tiny], dtype=torch.float32, device="cuda")
    # fp64 on the CPU keeps the subnormal terms.
    total = split_bf16x3(w).cpu().double().sum(dim=0)
    w64 = w.cpu().double()
    assert torch.equal(total[:, : len(exact)], w64[:, : len(exact)])
    assert (total - w64).abs().max().item() <= 2.0**-134
    for bad in (2.0**128 - 2.0**119, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            split_bf16x3(torch.full((1, 4), bad, dtype=torch.float32, device="cuda"))


@pytest.mark.parametrize("M", [1, 3, 16, 32, 33, 64, 127, 256, 1000, 2048, 4096])
@pytest.mark.parametrize("N, K", SHAPES_NK)
def test_gemm_a16w32(M, N, K):
    x, w, w_bf16x3, y = generate_gemm_a16w32_inputs(M, N, K)
    out = gemm_a16w32(x, w, w_bf16x3, y=y)
    assert out is y
    _check(out, x, w)


@pytest.mark.parametrize("M", range(1, 34))
def test_gemm_a16w32_small_m(M):
    # Every M of the fp32 FMA buckets, and the first M of the bf16x3 kernel.
    x, w, w_bf16x3, y = generate_gemm_a16w32_inputs(M, 128, 6144)
    _check(gemm_a16w32(x, w, w_bf16x3, y=y), x, w)


@pytest.mark.parametrize("M", [1, 32, 64])
@pytest.mark.parametrize("K", [96, 128])
def test_gemm_a16w32_small_k(M, K):
    # K = 96 is not a multiple of BLOCK_SIZE_K. At M = 64, NUM_KSPLIT becomes 1.
    x, w, w_bf16x3, y = generate_gemm_a16w32_inputs(M, 128, K)
    _check(gemm_a16w32(x, w, w_bf16x3, y=y), x, w)


@pytest.mark.parametrize("M", [1, 31, 512])
def test_gemm_a16w32_allocates_y_and_takes_strided_x(M):
    N, K = 128, 6144
    x, w, w_bf16x3, _ = generate_gemm_a16w32_inputs(M, N, K, output=False)
    wide = torch.zeros((M, K + 64), dtype=torch.bfloat16, device="cuda")
    wide[:, :K] = x
    out = gemm_a16w32(wide[:, :K], w, w_bf16x3)
    assert out.dtype == torch.float32 and out.shape == (M, N)
    _check(out, x, w)


@pytest.mark.parametrize("M", [1, 8, 32, 33, 64, 128])
def test_gemm_a16w32_repeats_bitwise(M):
    # New inputs each call: a stale split-K partial or a counter that was not reset
    # gives a wrong result. The same inputs must give the same bits.
    N, K = 128, 6144
    x0, w0, w0_bf16x3, y0 = generate_gemm_a16w32_inputs(M, N, K)
    first = gemm_a16w32(x0, w0, w0_bf16x3, y=y0).clone()
    for seed in range(1, 20):
        x, w, w_bf16x3, y = generate_gemm_a16w32_inputs(M, N, K, seed=seed)
        _check(gemm_a16w32(x, w, w_bf16x3, y=y), x, w)
    again = gemm_a16w32(x0, w0, w0_bf16x3, y=torch.empty_like(y0))
    assert torch.equal(again, first)


@pytest.mark.skipif(
    not ROUTES_INSIDE_CAPTURE, reason="needs torch >= 2.10 to allocate in a capture"
)
@pytest.mark.parametrize("M", [1, 32, 33, 64])
def test_gemm_a16w32_in_graph_capture(M, monkeypatch):
    # A fresh counter cache makes the first split-K call happen inside the capture.
    torch.cuda.synchronize()
    fresh = capture_scoped_cache(a16w32_op._split_counters.__wrapped__)
    monkeypatch.setattr(a16w32_op, "_split_counters", fresh)
    N, K = 128, 6144
    x, w, w_bf16x3, y = generate_gemm_a16w32_inputs(M, N, K)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm_a16w32(x, w, w_bf16x3, y=y)
    for seed in (1, 2):
        x_new, w_new, w_bf16x3_new, _ = generate_gemm_a16w32_inputs(
            M, N, K, output=False, seed=seed
        )
        x.copy_(x_new)
        w.copy_(w_new)
        w_bf16x3.copy_(w_bf16x3_new)
        graph.replay()
        torch.cuda.synchronize()
        _check(y, x, w)
    # An eager call after the capture must not use the counters of the graph.
    _check(gemm_a16w32(x, w, w_bf16x3), x, w)


@pytest.mark.parametrize("M", [8, 64])
def test_gemm_a16w32_more_accurate_than_bf16_weight(M):
    N, K = 128, 6144
    x, w, w_bf16x3, y = generate_gemm_a16w32_inputs(M, N, K)
    ref = x.double() @ w.double().T
    err = (gemm_a16w32(x, w, w_bf16x3, y=y).double() - ref).abs().max()
    err_bf16 = ((x @ w.to(torch.bfloat16).T).double() - ref).abs().max()
    assert err * 100 < err_bf16
