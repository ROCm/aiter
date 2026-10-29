# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The small-k top-p draw of the top-k-first fast path against the exact
top-k / top-p distribution (aiter's accepted set: the mass strictly above an
element is below p)."""
import pytest
import torch

import aiter.ops.sampling as sampling
from aiter.ops.triton.top_p_small_k import top_p_sampling_small_k

VOCAB = 248320


def _exact(row, k, p):
    vals, ids = row.topk(k)
    gt = (vals[None, :] * (vals[None, :] > vals[:, None])).sum(1)
    q = torch.where(gt < p, vals, torch.zeros_like(vals))
    out = torch.zeros(row.numel(), dtype=torch.float64, device=row.device)
    out[ids] = (q / q.sum()).double()
    return out


@pytest.mark.parametrize("k", [1, 8, 20, 64])
@pytest.mark.parametrize("p", [0.5, 0.95, 1.0])
@pytest.mark.parametrize("scale", [1.0, 6.0])
def test_fast_path_distribution(monkeypatch, k, p, scale):
    torch.manual_seed(k)
    monkeypatch.setattr(sampling, "_SMALL_K_TOPP", True)
    row = (torch.randn(VOCAB, device="cuda") * scale).softmax(-1)
    probs = row.expand(4096, VOCAB).contiguous()
    counts = torch.zeros(VOCAB, dtype=torch.float64, device="cuda")
    for _ in range(25):
        t = sampling.top_k_top_p_sampling_from_probs(probs, None, None, k, None, p)
        counts.index_add_(0, t.long(), torch.ones_like(t, dtype=torch.float64))
    freq = counts / counts.sum()
    ref = _exact(row, k, p)
    assert freq[ref == 0].sum().item() == 0
    assert 0.5 * (freq - ref).abs().sum().item() < 0.01


def test_degenerate_rows_stay_in_range():
    probs = torch.zeros(8, 20, device="cuda")
    top_p = torch.full((8,), 0.9, device="cuda")
    out = top_p_sampling_small_k(probs, top_p)
    assert ((out >= 0) & (out < 20)).all()


def test_nan_rows_and_top_p_stay_in_range():
    # a NaN row (the fast path then divides top_p by a NaN sum), a row with one
    # NaN, and NaN / zero / negative top_p: no element survives the top-p test
    probs = torch.rand((6, 20), dtype=torch.float32, device="cuda")
    probs[0] = float("nan")
    probs[1, 3] = float("nan")
    top_p = torch.tensor(
        [0.9, 0.9, float("nan"), 0.0, -1.0, 0.9], dtype=torch.float32, device="cuda"
    )
    out = top_p_sampling_small_k(probs, top_p)
    assert ((out >= 0) & (out < 20)).all(), out.tolist()


def test_fast_path_nan_row_gathers_in_range(monkeypatch):
    # the crash this guards: the fast path's gather from the top-k indices with
    # the small-k kernel's local index, on a row of NaN probabilities
    monkeypatch.setattr(sampling, "_SMALL_K_TOPP", True)
    probs = torch.rand((4, VOCAB), dtype=torch.float32, device="cuda").softmax(-1)
    probs[2] = float("nan")
    out = sampling.top_k_top_p_sampling_from_probs(probs, None, None, 20, None, 0.95)
    torch.cuda.synchronize()
    assert out.shape == (4,), out.shape
