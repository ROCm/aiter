# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import os
import sys

import pytest
import torch

from op_tests.triton_tests.attention.test_fav3_sage import fp8_max_diff_percentage

"""
Unit coverage for bench_sage's `peaked` and `tiebreak` input distributions.

Not in test_benchmark_smoke.py: that harness asserts only returncode == 0, which cannot see a
wrong head mapping, and its shape leaves --hk defaulted to --hq, where the GQA mapping is the
identity either way.
"""


@pytest.fixture(scope="module")
def bench():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "op_benchmarks",
            "triton",
            "bench_sage.py",
        )
    )
    spec = importlib.util.spec_from_file_location("bench_sage", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: dataclasses resolves annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tensors(bench, hq, hk, sq, sk, distribution, dim=128):
    torch.manual_seed(7)
    return bench.generate_test_tensors(
        1, hq, hk, sq, sk, dim, dim, torch.float32, "cpu", distribution
    )


def test_tiebreak_rejects_too_few_keys(bench):
    """Each gap needs its own key pair, so sk below 2*len(gaps) cannot be built."""
    minimum = 2 * len(bench._TIEBREAK_GAPS)
    with pytest.raises(ValueError, match="tiebreak needs at least"):
        _tensors(bench, 2, 2, 4, minimum - 1, "tiebreak")
    q, k, v = _tensors(bench, 2, 2, 4, minimum, "tiebreak")
    assert torch.isfinite(q).all() and torch.isfinite(k).all()
    assert torch.isfinite(v).all()


@pytest.mark.parametrize("gqa_ratio", [1, 2, 4])
def test_tiebreak_ties_land_on_the_head_the_query_reads(bench, gqa_ratio):
    """Every query row must tie on the KV head its own group points at.

    Checks the tie where it is observable -- the top-2 keys of the real score matrix -- rather
    than re-deriving the index arithmetic. With `arange(hq) % hk` this holds only at ratio 1;
    above it the mapping disagrees with the `(h g)` expansion and the rows land on keys they
    never read, which measures ordinary diffuse attention instead of a tie.
    """
    hk = 2
    hq = hk * gqa_ratio
    pairs = len(bench._TIEBREAK_GAPS)
    q, k, _ = _tensors(bench, hq, hk, 2 * pairs, 2 * pairs, "tiebreak")

    for head in range(hq):
        kv_head = head // gqa_ratio
        scores = q[0, head] @ k[0, kv_head].transpose(-1, -2)
        top2 = scores.topk(2, dim=-1).indices.sort(dim=-1).values
        for row in range(scores.shape[0]):
            pair = row % pairs
            assert top2[row].tolist() == [2 * pair, 2 * pair + 1], (
                f"ratio {gqa_ratio}, q head {head} -> kv head {kv_head}, row {row}: "
                f"expected the tied pair {[2 * pair, 2 * pair + 1]}, got {top2[row].tolist()}"
            )


def test_peaked_is_more_concentrated_than_diffusion(bench):
    """`peaked` exists to average fewer V rows, which is what leaves quantization noise uncancelled.

    Compared against `diffusion` rather than an absolute bound so the check tracks the generator
    rather than a calibration constant.
    """
    hq = hk = 2
    sq = sk = 256

    def effective_keys(distribution):
        q, k, _ = _tensors(bench, hq, hk, sq, sk, distribution)
        scores = q[0, 0] @ k[0, 0].transpose(-1, -2) / q.shape[-1] ** 0.5
        weights = scores.softmax(dim=-1)
        entropy = -(weights * weights.clamp_min(1e-20).log()).sum(dim=-1)
        return entropy.exp().mean().item()

    diffusion = effective_keys("diffusion")
    peaked = effective_keys("peaked")
    assert peaked < diffusion / 4, f"peaked {peaked:.1f} vs diffusion {diffusion:.1f}"


def test_tiebreak_keeps_its_own_reference_bound(bench):
    """`tiebreak` leaves BF16 by construction, so it needs a looser bound than the sweep's.

    A tie resolved differently moves a whole output row, so reusing a neighbouring distribution's
    bound here would report breakage on the property the generator is built to create.
    """
    bound = fp8_max_diff_percentage("tiebreak")
    others = [
        fp8_max_diff_percentage(name) for name in ("normal", "diffusion", "peaked")
    ]
    assert bound > max(others), f"tiebreak {bound} vs {others}"
