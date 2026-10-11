# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the delayed mHC gates from split-K partials (``mhc_delayed_gates``)."""

import pytest
import torch

from aiter.ops.triton.fusions.mhc_delayed_gates import mhc_delayed_gates

H = 5120
K = 4 * H
# rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, sinkhorn_repeat (DeepSeek-V4.1-Flash)
ARGS = (1e-20, 1e-6, 1e-6, 2.0, 20)


def ref_gates_fp64(mixes, sq, hc_scale, hc_base, rms_eps, pre_eps, sk_eps, mult, rep):
    """fp64 reference: vLLM's mhc_pre_delayed_torch gates on exact mixes."""
    mixes = mixes * torch.rsqrt(sq / K + rms_eps)[:, None]
    s, b = hc_scale.double(), hc_base.double()
    pre = torch.sigmoid(mixes[:, :4] * s[0] + b[:4]) + pre_eps
    post = torch.sigmoid(mixes[:, 4:8] * s[1] + b[4:8]) * mult
    comb = mixes[:, 8:].view(-1, 4, 4) * s[2] + b[8:].view(1, 4, 4)
    comb = torch.softmax(comb, dim=-1) + sk_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + sk_eps)
    for _ in range(rep - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + sk_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sk_eps)
    return post.unsqueeze(-1).float(), comb.float(), pre.float()


def make_partials(T, S, seed=0, device="cuda"):
    """Split-K partials of a realistic seam: R' ~ N(0, 1) in bf16, fn ~ N(0, 2e-3)."""
    g = torch.Generator(device=device).manual_seed(seed)
    r = torch.randn(T, K, device=device, generator=g).bfloat16().double()
    fn = torch.randn(24, K, device=device, generator=g).double() * 2e-3
    hc_scale = 0.5 + torch.rand(3, device=device, generator=g)
    hc_base = torch.randn(24, device=device, generator=g) * 0.5
    bounds = [K * s // S for s in range(S + 1)]
    gemm = torch.zeros(S, T, 32, device=device, dtype=torch.float32)
    sqrsum = torch.empty(S, T, device=device, dtype=torch.float32)
    for s in range(S):
        a, b = bounds[s], bounds[s + 1]
        gemm[s, :, :24] = (r[:, a:b] @ fn[:, a:b].t()).float()
        sqrsum[s] = r[:, a:b].square().sum(-1).float()
    mixes = r @ fn.t()
    sq = r.square().sum(-1)
    return gemm, sqrsum, hc_scale, hc_base, mixes, sq


# S = 10: the gfx942 seam kernel's chunk partials; 1 / 4 / 160: AITER split-K choices
@pytest.mark.parametrize("S", [1, 4, 10, 160])
@pytest.mark.parametrize("T", [1, 7, 100, 4097])
def test_mhc_delayed_gates(T, S):
    gemm, sqrsum, hc_scale, hc_base, mixes, sq = make_partials(T, S, seed=T + S)
    post, comb, pre = mhc_delayed_gates(
        gemm[:, :, :24], sqrsum, hc_scale, hc_base, *ARGS, K
    )
    ref = ref_gates_fp64(mixes, sq, hc_scale, hc_base, *ARGS)
    for got, want in zip((post, comb, pre), ref):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-4)


def test_mhc_delayed_gates_strided_sqrsum():
    """The seam kernel's layout: (S, T, 32) rows with the sum of squares in column 24."""
    T, S = 333, 10
    gemm, sqrsum, hc_scale, hc_base, _, _ = make_partials(T, S)
    want = mhc_delayed_gates(gemm, sqrsum, hc_scale, hc_base, *ARGS, K)
    gemm[:, :, 24] = sqrsum
    got = mhc_delayed_gates(gemm, gemm[:, :, 24], hc_scale, hc_base, *ARGS, K)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_mhc_delayed_gates_separate_eps():
    """hc_pre_eps only shifts pre; hc_sinkhorn_eps only enters comb."""
    T, S = 64, 10
    gemm, sqrsum, hc_scale, hc_base, mixes, sq = make_partials(T, S)
    args = (1e-20, 1e-3, 1e-4, 2.0, 20)
    post, comb, pre = mhc_delayed_gates(gemm, sqrsum, hc_scale, hc_base, *args, K)
    ref = ref_gates_fp64(mixes, sq, hc_scale, hc_base, *args)
    for got, want in zip((post, comb, pre), ref):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-4)


def test_mhc_delayed_gates_empty():
    gemm = torch.empty(10, 0, 32, device="cuda")
    hc_scale = torch.ones(3, device="cuda")
    hc_base = torch.zeros(24, device="cuda")
    post, comb, pre = mhc_delayed_gates(
        gemm, gemm[:, :, 24], hc_scale, hc_base, *ARGS, K
    )
    assert post.shape == (0, 4, 1) and comb.shape == (0, 4, 4) and pre.shape == (0, 4)
