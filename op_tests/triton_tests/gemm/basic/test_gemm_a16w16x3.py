# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter.ops.triton.gemm.basic.gemm_a16w16x3 as x3_op
from aiter.ops.triton.gemm.basic.gemm_a16w16x3 import gemm_a16w16x3, split_bf16x3
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.utility.graph_alloc import ROUTES_INSIDE_CAPTURE, capture_scoped_cache

pytestmark = pytest.mark.skipif(
    "gfx950" not in get_arch(), reason="gemm_a16w16x3 has configs for gfx950 only"
)


def generate_gemm_a16w16x3_inputs(M, N, K, output=True, seed=0):
    """(x, weight, w, y): bf16 x (M, K), the fp32 weight (N, K), its bf16x3 terms
    (3, N, K) and an fp32 output (M, N), or None."""
    torch.manual_seed(seed)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((N, K), dtype=torch.float32, device="cuda") * 0.02
    w = split_bf16x3(weight)
    y = torch.empty((M, N), dtype=torch.float32, device="cuda") if output else None
    return x, weight, w, y


def _check(out, x, weight):
    # An fp64 reference, and a bound scaled by sum |x| |w| for the fp32 accumulation.
    ref = x.double() @ weight.double().T
    scale = (x.double().abs() @ weight.double().abs().T).max().item()
    torch.testing.assert_close(out.double(), ref, rtol=0, atol=4e-6 * scale)


SHAPES_NK = [(128, 6144), (256, 3072), (128, 7168)]


def test_split_bf16x3_is_exact():
    _, weight, w, _ = generate_gemm_a16w16x3_inputs(4, 128, 6144, output=False)
    total = w[0].float() + w[1].float() + w[2].float()
    assert torch.equal(total, weight)


@pytest.mark.parametrize("M", [1, 3, 16, 31, 32, 64, 127, 256, 1000, 2048, 4096])
@pytest.mark.parametrize("N, K", SHAPES_NK)
def test_gemm_a16w16x3(M, N, K):
    x, weight, w, y = generate_gemm_a16w16x3_inputs(M, N, K)
    out = gemm_a16w16x3(x, w, y=y)
    assert out is y
    _check(out, x, weight)


@pytest.mark.parametrize("M", [1, 64])
@pytest.mark.parametrize("K", [96, 128])
def test_gemm_a16w16x3_small_k_without_split_k(M, K):
    # For a K this small, compute_splitk_params sets NUM_KSPLIT to 1.
    x, weight, w, y = generate_gemm_a16w16x3_inputs(M, 128, K)
    _check(gemm_a16w16x3(x, w, y=y), x, weight)


@pytest.mark.parametrize("M", [1, 31, 512])
def test_gemm_a16w16x3_allocates_y_and_takes_strided_x(M):
    N, K = 128, 6144
    x, weight, w, _ = generate_gemm_a16w16x3_inputs(M, N, K, output=False)
    wide = torch.zeros((M, K + 64), dtype=torch.bfloat16, device="cuda")
    wide[:, :K] = x
    out = gemm_a16w16x3(wide[:, :K], w)
    assert out.dtype == torch.float32 and out.shape == (M, N)
    _check(out, x, weight)


@pytest.mark.parametrize("M", [1, 8, 32, 64, 128])
def test_gemm_a16w16x3_repeats_bitwise(M):
    # New inputs each call: a stale split-K partial or a counter that was not reset
    # gives a wrong result. The same inputs must give the same bits.
    N, K = 128, 6144
    x0, _, w0, y0 = generate_gemm_a16w16x3_inputs(M, N, K)
    first = gemm_a16w16x3(x0, w0, y=y0).clone()
    for seed in range(1, 20):
        x, weight, w, y = generate_gemm_a16w16x3_inputs(M, N, K, seed=seed)
        _check(gemm_a16w16x3(x, w, y=y), x, weight)
    again = gemm_a16w16x3(x0, w0, y=torch.empty_like(y0))
    assert torch.equal(again, first)


@pytest.mark.skipif(
    not ROUTES_INSIDE_CAPTURE, reason="needs torch >= 2.10 to allocate in a capture"
)
@pytest.mark.parametrize("M", [1, 64])
def test_gemm_a16w16x3_first_split_k_use_inside_capture(M, monkeypatch):
    # A fresh counter cache makes the first split-K call happen inside the capture.
    torch.cuda.synchronize()
    fresh = capture_scoped_cache(x3_op._split_counters.__wrapped__)
    monkeypatch.setattr(x3_op, "_split_counters", fresh)
    N, K = 128, 6144
    x, _, w, y = generate_gemm_a16w16x3_inputs(M, N, K)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm_a16w16x3(x, w, y=y)
    for seed in (1, 2):
        x_new, weight_new, w_new, _ = generate_gemm_a16w16x3_inputs(
            M, N, K, output=False, seed=seed
        )
        x.copy_(x_new)
        w.copy_(w_new)
        graph.replay()
        torch.cuda.synchronize()
        _check(y, x, weight_new)
    # An eager call after the capture must not use the counters of the graph.
    _check(gemm_a16w16x3(x, w), x, weight_new)


def test_gemm_a16w16x3_more_accurate_than_bf16_weight():
    M, N, K = 64, 128, 6144
    x, weight, w, y = generate_gemm_a16w16x3_inputs(M, N, K)
    ref = x.double() @ weight.double().T
    err_x3 = (gemm_a16w16x3(x, w, y=y).double() - ref).abs().max()
    err_bf16 = ((x @ weight.to(torch.bfloat16).T).double() - ref).abs().max()
    assert err_x3 * 100 < err_bf16
