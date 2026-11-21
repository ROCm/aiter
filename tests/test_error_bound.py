# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the per-element GEMM error bound."""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Import the module directly, without importing GPU-dependent aiter ops.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aiter" / "utility"))
import error_bound as eb


def _fp8_operands(m, n, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g).to(torch.float8_e4m3fn)
    w = torch.randn(n, k, generator=g).to(torch.float8_e4m3fn)
    xs = torch.rand(m, 1, generator=g) + 0.5
    ws = torch.rand(n, 1, generator=g) + 0.5
    return x, w, xs, ws


def _reference(x, w, xs, ws, dtype):
    xd, wd = x.float() * xs, w.float() * ws
    return xd, wd, F.linear(xd, wd).to(dtype)


def _blocked_kernel(x, w, xs, ws, dtype, block=32):
    # fp32 accumulation of exact code products in reverse block order, scales after.
    acc = torch.zeros(x.shape[0], w.shape[0])
    for lo in reversed(range(0, x.shape[1], block)):
        acc += x[:, lo : lo + block].float() @ w[:, lo : lo + block].float().t()
    return (acc * xs * ws.t()).to(dtype)


def test_unit_roundoff():
    assert eb.unit_roundoff(torch.float32) == 2.0**-24
    assert eb.unit_roundoff(torch.bfloat16) == 2.0**-8
    assert eb.unit_roundoff(torch.float16) == 2.0**-11


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m, n, k", [(1, 64, 8192), (16, 32, 1024), (8, 8, 32)])
def test_reordered_accumulation_stays_within_bound(dtype, m, n, k):
    x, w, xs, ws = _fp8_operands(m, n, k)
    xd, wd, ref = _reference(x, w, xs, ws, dtype)
    out = _blocked_kernel(x, w, xs, ws, dtype)
    assert eb.check_error_bound(ref, out, eb.gemm_error_bound(xd, wd, ref)) <= 1


def test_cancelling_row_bound_matches_closed_form():
    # K = 4096, exact result 0, mass 4096: only the accumulation, scale and
    # subnormal terms remain. lam = 10, so lam * sqrt(K) = 640.
    u = 2.0**-24
    x = torch.tensor([[1.0, -1.0] * 2048])
    w = torch.ones(1, 4096)
    ref = torch.zeros(1, 1, dtype=torch.bfloat16)
    g = math.expm1(640 * u + 4096 * u * u / (1 - u))
    expected = (1 + 2.0**-8) * 2 * (g + 2 * u / (1 - 2 * u)) * 4096 + 2.0**-133
    assert eb.gemm_error_bound(x, w, ref).total.item() == pytest.approx(
        expected, rel=1e-6
    )


def test_output_rounding_term_matches_closed_form():
    # K = 1: the |ref| term dominates; fp16 has u = 2**-11.
    x = torch.tensor([[3.0]])
    w = torch.tensor([[2.0]])
    ref = torch.tensor([[6.0]], dtype=torch.float16)
    u, u16 = 2.0**-24, 2.0**-11
    g = math.expm1(10 * u + u * u / (1 - u))
    expected = (
        (1 + u16) * 2 * (g + 2 * u / (1 - 2 * u)) * 6
        + 2 * u16 / (1 - u16) * 6
        + 2.0**-24
    )
    assert eb.gemm_error_bound(x, w, ref).total.item() == pytest.approx(
        expected, rel=1e-6
    )


def test_shape_mismatch_and_empty():
    ref = torch.zeros(4, 4)
    with pytest.raises(AssertionError, match="shapes differ"):
        eb.check_error_bound(ref, torch.zeros(4, 1), torch.ones(4, 4))
    empty = torch.zeros(0, 4)
    assert eb.check_error_bound(empty, empty, empty) == 0


def test_one_ulp_rounding_flip_passes_two_ulps_fail():
    one = torch.ones(1, 1)
    ref = one.to(torch.bfloat16)
    bound = eb.gemm_error_bound(one, one, ref)
    eb.check_error_bound(ref, ref + 2.0**-7, bound)
    with pytest.raises(AssertionError, match="1 of 1 elements over"):
        eb.check_error_bound(ref, ref + 2.0**-6, bound)


def test_cancelling_sum_keeps_a_budget_for_its_terms():
    x = torch.tensor([[1000.0, -1000.0]])
    w = torch.tensor([[1.0, 1.0]])
    ref = torch.zeros(1, 1, dtype=torch.bfloat16)
    bound = eb.gemm_error_bound(x, w, ref)
    assert bound.total.item() > 2000 * 2.0**-24
    eb.check_error_bound(ref, ref + bound.total.item() / 2, bound)


def test_nan_fails_and_matching_inf_passes():
    one = torch.ones(1, 1)
    ref = torch.full((1, 1), float("inf"), dtype=torch.float16)
    bound = eb.gemm_error_bound(one, one, ref)
    assert eb.check_error_bound(ref, ref.clone(), bound) == 0
    with pytest.raises(AssertionError):
        eb.check_error_bound(ref, torch.full_like(ref, float("nan")), bound)


def test_scale_error_is_caught():
    x, w, xs, ws = _fp8_operands(64, 64, 1024)
    xd, wd, ref = _reference(x, w, xs, ws, torch.bfloat16)
    out = _blocked_kernel(x, w, xs, ws * (1 + 2.0**-6), torch.bfloat16)
    with pytest.raises(AssertionError, match="max err/bound"):
        eb.check_error_bound(ref, out, eb.gemm_error_bound(xd, wd, ref))


def test_scale_bias_under_one_ulp_fails_in_aggregate():
    # 2**-9 moves no element by more than one bf16 ulp, but every element the same way.
    x, w, xs, ws = _fp8_operands(128, 128, 1024)
    xd, wd, ref = _reference(x, w, xs, ws, torch.bfloat16)
    bound = eb.gemm_error_bound(xd, wd, ref)
    eb.check_error_bound(ref, _blocked_kernel(x, w, xs, ws, torch.bfloat16), bound)
    out = _blocked_kernel(x, w, xs, ws * (1 + 2.0**-9), torch.bfloat16)
    with pytest.raises(AssertionError, match=r"0 of 16384 elements over; bias"):
        eb.check_error_bound(ref, out, bound)


def test_independent_one_ulp_flips_pass_in_aggregate():
    g = torch.Generator().manual_seed(0)
    x = (1 + torch.rand(65536, 1, generator=g)).to(torch.bfloat16).float()
    ref = x.to(torch.bfloat16)
    flip = torch.randint(0, 2, x.shape, generator=g).float() * 2 - 1
    out = (x + flip * 2.0**-7).to(torch.bfloat16)
    eb.check_error_bound(ref, out, eb.gemm_error_bound(x, torch.ones(1, 1), ref))


def test_shared_scale_rounding_is_allowed_in_aggregate():
    # fp32 output, K = 1: one fp32 rounding of a shared scale moves every element
    # the same way by far more than independent roundings would.
    x, w, xs, ws = _fp8_operands(256, 256, 1)
    xd, wd, ref = _reference(x, w, xs, ws, torch.float32)
    bound = eb.gemm_error_bound(xd, wd, ref)
    out = _blocked_kernel(x, w, xs, ws * (1 + 2.0**-23), torch.float32)
    eb.check_error_bound(ref, out, bound)
    with pytest.raises(AssertionError, match="bias"):
        eb.check_error_bound(ref, out, bound.total)


def test_round_toward_zero_output_fails_in_aggregate():
    x, w, xs, ws = _fp8_operands(128, 128, 1024)
    xd, wd, ref = _reference(x, w, xs, ws, torch.bfloat16)
    acc = _blocked_kernel(x, w, xs, ws, torch.float32)
    out = (acc.view(torch.int32) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)
    with pytest.raises(AssertionError, match="bias"):
        eb.check_error_bound(ref, out, eb.gemm_error_bound(xd, wd, ref))


def test_failure_reports_the_output_rounding_share():
    one = torch.ones(1, 1)
    ref = one.to(torch.bfloat16)
    bound = eb.gemm_error_bound(one, one, ref)
    with pytest.raises(AssertionError, match=r"\(output rounding 0\.0078"):
        eb.check_error_bound(ref, ref + 2.0**-6, bound)
