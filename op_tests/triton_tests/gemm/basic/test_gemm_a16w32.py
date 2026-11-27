# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_a16w32 import gemm_a16w32, split_bf16x3
from aiter.ops.triton.utils import fused_splitk
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
    # An fp64 reference. The bound of each output is 2**-21 sum_k |x w|. The fp32
    # sums of both kernels stay below 2**-21.9 of it (K = 96) and 2**-23.6 (K >=
    # 3072). A lost third bf16 term gives 2**-22.7 or more, which the exact test
    # below catches at every K.
    ref = x.double() @ w.double().T
    bound = 2.0**-21 * (x.double().abs() @ w.double().abs().T)
    excess = (out.double() - ref).abs() - bound
    worst = excess.argmax().item()
    m, n = divmod(worst, out.shape[1])
    assert excess.max().item() <= 0, (
        f"out[{m}, {n}] = {out[m, n].item():.9e}, fp64 reference "
        f"{ref[m, n].item():.9e}, bound {bound[m, n].item():.3e}"
    )


def _assert_equal(actual, expected, what):
    diff = (actual.double() - expected.double()).abs()
    worst = diff.argmax().item()
    m, n = divmod(worst, actual.shape[1])
    assert torch.equal(actual, expected), (
        f"{what}: {(diff > 0).sum().item()} values differ, out[{m}, {n}] = "
        f"{actual[m, n].item():.9e}, not {expected[m, n].item():.9e}"
    )


SHAPES_NK = [(128, 6144), (256, 3072), (128, 7168)]


def test_split_bf16x3_is_exact():
    _, w, w_bf16x3, _ = generate_gemm_a16w32_inputs(4, 128, 6144, output=False)
    total = w_bf16x3[0].float() + w_bf16x3[1].float() + w_bf16x3[2].float()
    _assert_equal(total, w, "sum of the terms")


def test_split_bf16x3_limits():
    # Exact from 2**-110 to below 2**128 - 2**119, and lossy only under 2**-133.
    ulp = 2.0**-23
    exact = [0.0, 2.0**-110 * (1 + ulp), -(2.0**-110) * (2 - ulp), 2.0**128 - 2.0**120]
    tiny = [2.0**-126 + 2.0**-149, -(2.0**-120) * (1 + ulp)]
    w = torch.tensor([exact + tiny], dtype=torch.float32, device="cuda")
    # fp64 on the CPU keeps the subnormal terms.
    total = split_bf16x3(w).cpu().double().sum(dim=0)
    w64 = w.cpu().double()
    _assert_equal(total[:, : len(exact)], w64[:, : len(exact)], "exact range")
    lost = (total - w64).abs().max().item()
    assert lost <= 2.0**-134, f"tiny values lose {lost:.3e}, more than 2**-134"
    for bad in (2.0**128 - 2.0**119, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            split_bf16x3(torch.full((1, 4), bad, dtype=torch.float32, device="cuda"))


@pytest.mark.parametrize("M", [1, 3, 16, 32, 33, 64, 127, 256, 1000, 2048, 4096])
@pytest.mark.parametrize("N, K", SHAPES_NK)
def test_gemm_a16w32(M, N, K):
    x, w, _, y = generate_gemm_a16w32_inputs(M, N, K)
    out = gemm_a16w32(x, w, y=y)
    assert out is y, "the result is not in y"
    _check(out, x, w)


@pytest.mark.parametrize("M", [5, 6, 7, 8, 9, 10, 11, 12])
def test_gemm_a16w32_small_m(M):
    # The fp32 FMA kernel uses BLOCK_SIZE_M 4 for M 5 to 32: each remainder of
    # M % 4 in the M_LEQ_8 and M_LEQ_16 buckets. test_gemm_a16w32 has 16 and 32.
    x, w, _, y = generate_gemm_a16w32_inputs(M, 128, 6144)
    _check(gemm_a16w32(x, w, y=y), x, w)


@pytest.mark.parametrize("M", [1, 32, 64])
@pytest.mark.parametrize("K", [96, 128])
def test_gemm_a16w32_small_k(M, K):
    # K = 96 is not a multiple of BLOCK_SIZE_K. At M = 64, NUM_KSPLIT becomes 1.
    x, w, _, y = generate_gemm_a16w32_inputs(M, 128, K)
    _check(gemm_a16w32(x, w, y=y), x, w)


@pytest.mark.parametrize("M", [33, 256, 1000, 5000])
def test_gemm_a16w32_split_k_with_k_tail(M):
    # K = 6112 is not a multiple of BLOCK_SIZE_K: the last split has a partial K step.
    x, w, _, y = generate_gemm_a16w32_inputs(M, 128, 6112)
    _check(gemm_a16w32(x, w, y=y), x, w)


@pytest.mark.parametrize("M", [1, 16, 33, 256, 1000])
@pytest.mark.parametrize("N, K", [(128, 6144), (128, 6112)])
def test_gemm_a16w32_one_hot_rows_are_exact(M, N, K):
    # Each row of x has one value, a power of two, so each output is that value
    # times one weight, exact in fp32. A kernel that loses a bf16 term fails here.
    _, w, _, y = generate_gemm_a16w32_inputs(M, N, K)
    gen = torch.Generator(device="cuda").manual_seed(M)
    cols = torch.randint(0, K, (M,), device="cuda", generator=gen)
    exps = torch.randint(-2, 3, (M,), device="cuda", generator=gen)
    signs = 1 - 2 * torch.randint(0, 2, (M,), device="cuda", generator=gen)
    scale = torch.exp2(exps.to(torch.float32)) * signs.to(torch.float32)
    x = torch.zeros((M, K), dtype=torch.bfloat16, device="cuda")
    x[torch.arange(M, device="cuda"), cols] = scale.to(torch.bfloat16)
    expected = w[:, cols].T * scale[:, None]
    _assert_equal(gemm_a16w32(x, w, y=y), expected, "one-hot rows")


@pytest.mark.parametrize("M", [1, 33, 256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("N, K", [(128, 6144), (256, 3072)])
def test_gemm_a16w32_takes_w_bf16x3(M, N, K):
    # vLLM gives the terms as the third argument. The result has the same bits. The
    # terms use more LDS than the fp32 weight, so each M bucket of the tuned file
    # (128, 6144) and of DEFAULT.json (256, 3072) must fit with them.
    x, w, w_bf16x3, _ = generate_gemm_a16w32_inputs(M, N, K, output=False)
    _assert_equal(gemm_a16w32(x, w, w_bf16x3), gemm_a16w32(x, w), "with w_bf16x3")


@pytest.mark.parametrize("M", [1, 512])
def test_gemm_a16w32_allocates_y_and_takes_strided_x(M):
    N, K = 128, 6144
    x, w, _, _ = generate_gemm_a16w32_inputs(M, N, K, output=False)
    wide = torch.zeros((M, K + 64), dtype=torch.bfloat16, device="cuda")
    wide[:, :K] = x
    out = gemm_a16w32(wide[:, :K], w)
    assert out.dtype == torch.float32 and out.shape == (
        M,
        N,
    ), f"y is {out.dtype} {tuple(out.shape)}"
    _check(out, x, w)


@pytest.mark.parametrize("M", [33, 128])
def test_gemm_a16w32_repeats_bitwise(M):
    # New inputs each call: a stale split-K partial or a counter that was not reset
    # gives a wrong result. The same inputs must give the same bits.
    N, K = 128, 6144
    x0, w0, _, y0 = generate_gemm_a16w32_inputs(M, N, K)
    first = gemm_a16w32(x0, w0, y=y0).clone()
    for seed in range(1, 20):
        x, w, _, y = generate_gemm_a16w32_inputs(M, N, K, seed=seed)
        _check(gemm_a16w32(x, w, y=y), x, w)
    _assert_equal(gemm_a16w32(x0, w0), first, "second call")


def _capture_and_replay(M, monkeypatch, routes_inside_capture):
    # A fresh counter cache makes the first split-K call happen inside the capture.
    torch.cuda.synchronize()
    fresh = capture_scoped_cache(fused_splitk._split_counters_keyed.__wrapped__)
    monkeypatch.setattr(fused_splitk, "_split_counters_keyed", fresh)
    monkeypatch.setattr(fused_splitk, "ROUTES_INSIDE_CAPTURE", routes_inside_capture)
    N, K = 128, 6144
    x, w, _, y = generate_gemm_a16w32_inputs(M, N, K)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm_a16w32(x, w, y=y)
    for seed in (1, 2):
        x_new, w_new, _, _ = generate_gemm_a16w32_inputs(
            M, N, K, output=False, seed=seed
        )
        x.copy_(x_new)
        w.copy_(w_new)
        graph.replay()
        torch.cuda.synchronize()
        _check(y, x, w)
    # An eager call after the capture must not use the counters of the graph.
    _check(gemm_a16w32(x, w), x, w)


@pytest.mark.skipif(
    not ROUTES_INSIDE_CAPTURE, reason="needs torch >= 2.10 to allocate in a capture"
)
@pytest.mark.parametrize("M", [1, 33])
def test_gemm_a16w32_in_graph_capture(M, monkeypatch):
    _capture_and_replay(M, monkeypatch, routes_inside_capture=True)


@pytest.mark.parametrize("M", [33, 256])
def test_gemm_a16w32_in_graph_capture_without_counters(M, monkeypatch):
    # Before torch 2.10 a capture cannot make the counters: the partials go to the
    # reduce kernel.
    _capture_and_replay(M, monkeypatch, routes_inside_capture=False)


@pytest.mark.parametrize("M", [8, 64])
def test_gemm_a16w32_more_accurate_than_bf16_weight(M):
    N, K = 128, 6144
    x, w, _, y = generate_gemm_a16w32_inputs(M, N, K)
    ref = x.double() @ w.double().T
    err = (gemm_a16w32(x, w, y=y).double() - ref).abs().max().item()
    err_bf16 = ((x @ w.to(torch.bfloat16).T).double() - ref).abs().max().item()
    assert err * 100 < err_bf16, f"error {err:.3e}, bf16 weight error {err_bf16:.3e}"
