# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter.ops.triton.fusions.fused_mixed_sample import fused_mixed_sample

EPS = 1e-10
# Two race keys closer than this (relative) are a near tie. fp32 rounding can
# order a near tie either way, in this kernel and in the HIP kernel.
NEAR_TIE = 1e-5


def _keys64(logits_row, temp, noise_row):
    p = torch.softmax(logits_row.double() / max(float(temp), 1e-5), dim=0)
    return p / (noise_row.double() + EPS)


def _assert_race(out, logits, temps, noise, greedy_eps=0.0):
    """Check each row against the float64 race, or the lowest-id argmax."""
    for r in range(logits.shape[0]):
        tok = int(out[r])
        assert 0 <= tok < logits.shape[1], f"row {r}: id {tok} out of range"
        if float(temps[r]) <= greedy_eps:
            row = logits[r].float()
            expected = int(torch.nonzero(row == row.max())[0])
            assert tok == expected, f"greedy row {r}: {tok} != {expected}"
            continue
        keys = _keys64(logits[r], temps[r], noise[r])
        best = int(keys.argmax())
        gap = float((keys[best] - keys[tok]) / keys[best])
        assert tok == best or gap < NEAR_TIE, f"row {r}: {tok} != {best}, gap {gap}"


def _inputs(rows, vocab, dtype, seed):
    torch.manual_seed(seed)
    logits = torch.randn((rows, vocab), dtype=torch.float32, device="cuda") * 4
    logits = logits.to(dtype)
    noise = torch.empty((rows, vocab), dtype=torch.float32, device="cuda")
    noise.exponential_(1)
    temps = torch.rand(rows, dtype=torch.float32, device="cuda") + 0.3
    temps[::3] = 0.0
    return logits, temps, noise


@pytest.mark.parametrize("vocab", [1000, 4097, 50000, 163840])
@pytest.mark.parametrize("rows", [1, 7, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_mixed_sample_matches_the_race(rows, vocab, dtype):
    logits, temps, noise = _inputs(rows, vocab, dtype, seed=rows * vocab)
    # Ties at the max of a greedy row: the lowest id must win.
    top = logits[0].float().max().to(dtype)
    logits[0, vocab // 3] = top
    logits[0, vocab - 1] = top
    out = fused_mixed_sample(logits, noise, temps, eps=EPS)
    assert out.dtype == torch.int32 and out.shape == (rows,)
    _assert_race(out, logits, temps, noise)


# The HIP kernel gives ids out of range for a vocab that is not a multiple of 4
# or is below 4096 before #7373, so compare at sizes that are correct on both.
@pytest.mark.parametrize("vocab", [8192, 163840])
def test_fused_mixed_sample_agrees_with_hip_kernel(vocab):
    rows = 64
    logits, temps, noise = _inputs(rows, vocab, torch.bfloat16, seed=vocab)
    temps = temps.clamp_min(0.3)
    hip = torch.empty(rows, dtype=torch.int32, device="cuda")
    aiter.mixed_sample_outer_exponential(hip, logits, noise, temps, eps=EPS)
    out = fused_mixed_sample(logits, noise, temps, eps=EPS)
    for r in torch.nonzero(out != hip).flatten().tolist():
        keys = _keys64(logits[r], temps[r], noise[r])
        a, b = keys[int(out[r])], keys[int(hip[r])]
        gap = float((a - b).abs() / torch.maximum(a, b))
        assert gap < NEAR_TIE, f"row {r}: {int(out[r])} vs HIP {int(hip[r])}"


@pytest.mark.parametrize("forced", [0, 5, 4095, 4096, 100000, 163839])
def test_fused_mixed_sample_forced_row(forced):
    # All ids but one are -inf, as a required-tool-call mask makes them.
    vocab = 163840
    logits = torch.full((4, vocab), float("-inf"), dtype=torch.bfloat16, device="cuda")
    logits[:, forced] = 3.0
    temps = torch.tensor([1.0, 0.7, 0.0, 1e-7], dtype=torch.float32, device="cuda")
    noise = torch.empty((4, vocab), dtype=torch.float32, device="cuda")
    noise.exponential_(1)
    out = fused_mixed_sample(logits, noise, temps, eps=EPS)
    assert out.tolist() == [forced] * 4


def test_fused_mixed_sample_masked_rows():
    rows, vocab = 6, 163840
    logits, temps, noise = _inputs(rows, vocab, torch.bfloat16, seed=7)
    logits[:, : 20 * 4096] = float("-inf")
    banned = torch.randint(0, vocab, (256,), device="cuda")
    logits[:, banned] = float("-inf")
    logits[rows - 1] = float("-inf")
    out = fused_mixed_sample(logits, noise, temps, eps=EPS)
    assert ((out >= 0) & (out < vocab)).all(), out
    for r in range(rows - 1):
        assert logits[r, int(out[r])] != float("-inf"), f"row {r} drew a masked id"
    _assert_race(out[: rows - 1], logits[: rows - 1], temps, noise)


def test_fused_mixed_sample_broadcast_noise():
    rows, vocab = 8, 32000
    logits, temps, _ = _inputs(rows, vocab, torch.bfloat16, seed=11)
    row = torch.empty((1, vocab), dtype=torch.float32, device="cuda").exponential_(1)
    shared = row.expand(rows, vocab)
    out = fused_mixed_sample(logits, shared, temps, eps=EPS)
    full = fused_mixed_sample(logits, shared.contiguous(), temps, eps=EPS)
    torch.testing.assert_close(out, full, atol=0, rtol=0)


def test_fused_mixed_sample_out_and_empty():
    logits, temps, noise = _inputs(5, 4096, torch.bfloat16, seed=3)
    out = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    assert fused_mixed_sample(logits, noise, temps, eps=EPS, out=out) is out
    _assert_race(out, logits, temps, noise)
    empty = fused_mixed_sample(
        torch.empty((0, 4096), dtype=torch.bfloat16, device="cuda"),
        torch.empty((0, 4096), dtype=torch.float32, device="cuda"),
        torch.empty((0,), dtype=torch.float32, device="cuda"),
    )
    assert empty.shape == (0,) and empty.dtype == torch.int32


def test_fused_mixed_sample_rejects_bad_inputs():
    logits, temps, noise = _inputs(4, 4096, torch.bfloat16, seed=5)
    with pytest.raises(ValueError):
        fused_mixed_sample(logits, noise.to(torch.bfloat16), temps)
    with pytest.raises(ValueError):
        fused_mixed_sample(logits, noise[:, :100], temps)
    with pytest.raises(ValueError):
        fused_mixed_sample(logits.t(), noise.t(), temps)
