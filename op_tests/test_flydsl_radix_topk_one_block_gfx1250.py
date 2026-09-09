# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression coverage for gfx1250 radix compaction and stable index sorting.

Run with: python -m pytest -q op_tests/test_flydsl_radix_topk_one_block_gfx1250.py
"""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.topk import flydsl_radix_topk_one_block_gfx1250


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="requires gfx1250",
)
@pytest.mark.parametrize("write_values", [False, True])
@pytest.mark.parametrize(
    "width,k,stable,distribution",
    [
        (128, 512, False, "special"),
        (128, 2048, True, "normal"),
        (4096, 4096, True, "special"),
        (4096, 1, False, "normal"),
        (4096, 31, True, "normal"),
        (4096, 513, False, "ties"),
        (4096, 513, True, "normal"),
        (4096, 2048, False, "normal"),
        (4096, 2048, True, "special"),
        (4097, 2048, True, "normal"),
        (32767, 512, True, "normal"),
        (32768, 513, True, "normal"),
        (65536, 31, True, "normal"),
        (65536, 1025, True, "normal"),
        (65536, 2048, True, "normal"),
        (65536, 4096, True, "normal"),
        (65536, 2048, False, "normal"),
        (65536, 2048, True, "bucket"),
        (65536, 2048, False, "bucket"),
        (65536, 2048, True, "equal"),
        (65536, 2048, False, "equal"),
        (65536, 2048, True, "ties"),
        (65536, 2048, False, "ties"),
        (65536, 2048, True, "special"),
        (65536, 2048, False, "special"),
    ],
)
def test_radix_compaction_and_index_sort(width, k, stable, distribution, write_values):
    pytest.importorskip("flydsl")
    rows = 8
    generator = torch.Generator(device="cuda").manual_seed(20260909)
    physical_width = width if width <= 4096 else width + rows - 1
    logits = torch.randn((rows, physical_width), generator=generator, device="cuda")
    if distribution == "bucket":
        # Distinct keys in one first-pass bucket: overflow LDS, then index-sort
        # from global memory without an ambiguous threshold tie.
        bits = 0x3F800000 + torch.arange(
            logits.shape[1], device="cuda", dtype=torch.int32
        )
        logits.copy_(bits.view(torch.float32))
    elif distribution == "equal":
        logits.fill_(1)
    elif distribution == "ties":
        logits.round_()
    elif distribution == "special":
        bits = torch.tensor(
            [0, -2147483648, 0x7F800000, -8388608, 0x7FC00001, -4194303],
            device="cuda",
            dtype=torch.int32,
        )
        logits.copy_(
            bits[torch.arange(logits.shape[1], device="cuda") % len(bits)].view(
                torch.float32
            )
        )

    starts = torch.arange(rows, device="cuda", dtype=torch.int32)
    starts[-1] = 0
    lengths = [0, 1, k - 1, k, k + 1, width - 1, width, width]
    lengths = [
        min(length, physical_width - int(starts[row]))
        for row, length in enumerate(lengths)
    ]
    ends = starts + torch.tensor(lengths, device="cuda", dtype=torch.int32)
    indices = torch.empty((rows, k), device="cuda", dtype=torch.int32)
    values = torch.empty((rows, k), device="cuda") if write_values else None
    previous = None
    for _ in range(3):
        indices.fill_(-123456789)
        flydsl_radix_topk_one_block_gfx1250(
            logits,
            starts,
            ends,
            indices,
            values,
            rows,
            logits.stride(0),
            1,
            k=k,
            stable=stable,
        )
        torch.cuda.synchronize()
        if stable and previous is not None:
            assert torch.equal(indices, previous)
        previous = indices.clone()
        for row, length in enumerate(lengths):
            count = min(k, length)
            start = int(starts[row])
            actual = indices[row, :count].long() - start
            assert torch.all((actual >= 0) & (actual < length))
            assert actual.unique().numel() == count
            assert torch.all(indices[row, count:] == -1)
            source = logits[row, start : start + length]
            raw = source.view(torch.int32).long()
            keys = ((raw ^ ((raw >> 31) & 0x7FFFFFFF)) ^ 0x80000000) & 0xFFFFFFFF
            expected = torch.argsort(keys, descending=True, stable=True)[:count]
            if stable:
                assert torch.equal(actual, expected.sort().values)
            else:
                assert torch.equal(
                    keys[actual].sort().values, keys[expected].sort().values
                )
            if write_values:
                assert torch.equal(
                    values[row, :count].view(torch.int32),
                    source[actual].view(torch.int32),
                )
                assert torch.all(values[row, count:] == -float("inf"))


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="requires gfx1250",
)
@pytest.mark.parametrize("rows", [256, 512, 513])
@pytest.mark.parametrize("write_values", [False, True])
def test_direct_output_dispatch_boundary(rows, write_values):
    """Check in-kernel copy/pad across the 256-row scalar/vector cutoff."""
    pytest.importorskip("flydsl")
    width, k = 129, 512
    generator = torch.Generator(device="cuda").manual_seed(20260909)
    logits = torch.randn((rows, width), generator=generator, device="cuda")
    logits[:, ::5] = float("nan")
    logits[:, ::7] = -float("inf")
    logits[:, 3] = -0.0
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    starts = row_ids % 8
    lengths = row_ids * 17 % (width - starts + 1)
    ends = starts + lengths
    indices = torch.full((rows, k), -123456789, device="cuda", dtype=torch.int32)
    values = torch.empty((rows, k), device="cuda") if write_values else None
    flydsl_radix_topk_one_block_gfx1250(
        logits,
        starts,
        ends,
        indices,
        values,
        rows,
        logits.stride(0),
        1,
        k=k,
        stable=True,
    )
    columns = torch.arange(k, device="cuda", dtype=torch.int32)[None, :]
    valid = columns < lengths[:, None]
    expected_indices = torch.where(valid, columns + starts[:, None], -1)
    assert torch.equal(indices, expected_indices)
    if write_values:
        expected_values = logits.gather(1, expected_indices.clamp_min(0).long())
        expected_values[~valid] = -float("inf")
        assert torch.equal(values.view(torch.int32), expected_values.view(torch.int32))
