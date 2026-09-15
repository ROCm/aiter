# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exact stable selection across radix boundaries and candidate-buffer overflow.

Run with ``python -m pytest op_tests/test_flydsl_topk_stable_paths.py``.
These tests launch one-block directly so small test batches cannot dispatch to
the multi-block implementation instead of exercising the changed emitters.
"""

import numpy as np
import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.kernels_common import get_warp_size
from aiter.ops.flydsl.kernels.radix_topk_one_block import (
    build_radix_topk_one_block_module,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

_MODES = (
    "first_bucket",
    "second_bucket",
    "overflow",
    "low_bits",
    "partial_tie",
    "all_equal",
    "negative",
    "signed_zero",
    "small_2",
    "small_8",
    "small_32",
    "small_64",
    "small_65",
    "small_tie",
)


def _scores(mode, length, k, rng):
    scores = np.zeros(length, dtype=np.float32)
    above = k // 4
    needed = k - above
    lower = 12000 if mode == "overflow" else 64
    bits = scores.view(np.uint32)
    scores[:above] = 2.0
    # All these candidates share the first bucket for both 12- and 14-bit
    # radix layouts. The upper group fits a single second-level bucket, while
    # the lower group is separated even by the coarser 12+10 prefix.
    base = np.uint32(0x3F810000)
    bits[above:k] = base + np.arange(needed, dtype=np.uint32) % 128
    bits[k : k + lower] = base - np.uint32(2048)
    if mode == "first_bucket":
        scores[:] = 0.0
        scores[:k] = 2.0
    elif mode == "low_bits":
        bits[above:k] = base + np.uint32(128)
        bits[k : k + lower] = base + np.uint32(1)
    elif mode == "partial_tie":
        bits[above : k + lower] = base + np.uint32(128)
    elif mode == "all_equal":
        scores[:] = 68.0
    elif mode == "negative":
        scores -= np.float32(4.0)
    elif mode == "signed_zero":
        scores[:] = np.float32(-0.0)
        scores[:k] = np.float32(0.0)
    elif mode.startswith("small_"):
        count = 8 if mode == "small_tie" else int(mode.split("_")[1])
        needed = 3 if mode == "small_tie" else count // 2
        above = k - needed
        scores[:] = 0.0
        scores[:above] = 2.0
        digits = (np.arange(count, dtype=np.uint32) * 3) % 128
        if mode == "small_tie":
            digits = np.arange(count, dtype=np.uint32) // 4
        bits[above : above + count] = base + digits
        bits[above + count : above + count + 64] = base - np.uint32(2048)
    return scores[rng.permutation(length)]


def _reference(scores, start, k):
    bits = scores.view(np.uint32)
    keys = bits ^ np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
    positions = np.arange(scores.size)
    selected = np.lexsort((positions, -keys.astype(np.int64)))[:k]
    return np.sort(selected).astype(np.int32) + start


def _check(k, write_values, length, *, is_decode=False, fallback_radix=False):
    arch = get_gfx()
    if arch not in ("gfx942", "gfx950", "gfx1250"):
        pytest.skip("FlyDSL top-k requires a supported AMD GPU")
    rows = len(_MODES)
    next_n = 2
    width = length + 16
    host = np.full((rows, width), np.inf, dtype=np.float32)
    starts = np.zeros(rows, dtype=np.int32)
    ends = np.empty(rows, dtype=np.int32)
    expected = np.empty((rows, k), dtype=np.int32)
    rng = np.random.default_rng(527)
    for row, mode in enumerate(_MODES):
        start = 0 if is_decode else 1 + row % 4
        size = length - (next_n - 1 - row % next_n) if is_decode else length - row % 3
        scores = _scores(mode, size, k, rng)
        host[row, start : start + size] = scores
        starts[row], ends[row] = start, start + size
        expected[row] = _reference(scores, start, k)

    logits = torch.from_numpy(host).cuda()
    row_starts = torch.from_numpy(starts).cuda()
    row_ends = (
        torch.full((rows // next_n,), length, dtype=torch.int32, device="cuda")
        if is_decode
        else torch.from_numpy(ends).cuda()
    )
    indices = torch.full((rows, k), -123456789, dtype=torch.int32, device="cuda")
    values = torch.empty((rows, k), dtype=torch.float32, device="cuda")
    launcher = build_radix_topk_one_block_module(
        k,
        block_threads=1024,
        write_values=write_values,
        stable=True,
        is_decode=is_decode,
        wave_size=get_warp_size(arch),
        lds_budget_bytes=78 * 1024 if arch == "gfx950" else 0,
        arch="" if fallback_radix else arch,
    )
    want = torch.from_numpy(expected).cuda()
    stream = torch.cuda.current_stream()
    for _ in range(3):
        indices.fill_(-123456789)
        _run_compiled(
            launcher,
            logits,
            row_starts,
            row_ends,
            indices,
            values if write_values else logits,
            width,
            next_n if is_decode else 1,
            rows,
            stream,
        )
        torch.cuda.synchronize()
        assert torch.equal(indices, want)
        if write_values:
            gathered = logits.gather(1, want.long())
            assert torch.equal(values.view(torch.int32), gathered.view(torch.int32))


@pytest.mark.parametrize("length", [8195, 50003, 65537])
@pytest.mark.parametrize(
    "k,write_values",
    [
        (512, False),
        (1024, True),
        (2000, False),
        (2048, False),
        (2048, True),
        (4096, False),
    ],
)
def test_stable_prefill_boundary_paths(k, write_values, length):
    _check(k, write_values, length)


@pytest.mark.parametrize("length", [50003, 65537])
def test_stable_decode_boundary_paths(length):
    _check(2048, True, length, is_decode=True)


@pytest.mark.parametrize("length", [49152, 65537])
def test_stable_fallback_radix_boundary_paths(length):
    _check(2048, True, length, fallback_radix=True)


def test_stable_global_sort_crossover():
    _check(2048, False, 65536)


@pytest.mark.parametrize("write_values", [False, True])
def test_stable_compact_sort_crossover(write_values):
    _check(2048, write_values, 49152)


def test_stable_existing_small_k_sort_crossover():
    _check(1024, True, 32768)


@pytest.mark.parametrize("write_values", [False, True])
def test_stable_streaming_boundary(write_values):
    _check(2048, write_values, 4097)
