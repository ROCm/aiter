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
