# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Focused correctness test for the reusable prefill candidate merger."""

import math
import struct

import pytest
import torch


def _ordered_i32(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if math.isnan(value):
        return -(1 << 31)
    ordered = bits ^ (0x7FFFFFFF if bits & 0x80000000 else 0)
    return ordered - (1 << 32) if ordered & 0x80000000 else ordered


def _reference(values, indices, counts, row_offsets, topk):
    result = []
    for row in range(len(row_offsets) - 1):
        candidates = []
        for cta in range(row_offsets[row], row_offsets[row + 1]):
            for slot in range(counts[cta]):
                candidates.append((float(values[cta, slot]), int(indices[cta, slot])))
        candidates.sort(key=lambda item: (-_ordered_i32(item[0]), item[1]))
        result.append(candidates[:topk])
    return result


@pytest.mark.parametrize("topk", [512, 1024])
def test_candidate_topk_merge_total_order_and_page_map(topk):
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith("gfx9"):
        pytest.skip("candidate merger currently requires wave64")

    from aiter.ops.flydsl import flydsl_candidate_topk_merge

    rows = 3
    ctas_per_row = 2
    ctas = rows * ctas_per_row
    page_size = 64
    count0 = topk // 2 + 1
    count1 = topk + 2 - count0
    counts_cpu = [count0, count1] * rows
    row_offsets_cpu = [0, 2, 4, 6]

    values = torch.full((ctas, topk), -float("inf"), dtype=torch.float32)
    indices = torch.full((ctas, topk), -1, dtype=torch.int32)
    generator = torch.Generator().manual_seed(2026)
    for row in range(rows):
        row_values = torch.randint(
            -8,
            9,
            (topk + 2,),
            generator=generator,
            dtype=torch.int32,
        ).float()
        if row == 1:
            # Cutoff: K-1 values above zero, then +0 beats -0 even though -0
            # owns the smaller index.
            row_values.fill_(-1.0)
            row_values[2 : topk + 1] = 1.0
            row_values[0] = -0.0
            row_values[1] = 0.0
        elif row == 2:
            # NaN is below -inf. Exactly one low value is needed after K-1
            # positive values, so -inf wins and both NaNs lose.
            row_values.fill_(float("nan"))
            row_values[2 : topk + 1] = 1.0
            row_values[1] = -float("inf")

        for plane in range(ctas_per_row):
            cta = row * ctas_per_row + plane
            begin = 0 if plane == 0 else count0
            end = begin + counts_cpu[cta]
            values[cta, : counts_cpu[cta]] = row_values[begin:end]
            indices[cta, : counts_cpu[cta]] = torch.arange(
                begin,
                end,
                dtype=torch.int32,
            )

    max_blocks = (topk + 2 + page_size - 1) // page_size
    block_table = (
        torch.arange(rows * max_blocks, dtype=torch.int32).reshape(rows, max_blocks)
        + 100
    )
    row_to_batch = torch.arange(rows, dtype=torch.int32)
    expected = _reference(
        values,
        indices,
        counts_cpu,
        row_offsets_cpu,
        topk,
    )

    values = values.cuda()
    indices = indices.cuda()
    counts = torch.tensor(counts_cpu, dtype=torch.int32, device="cuda")
    row_offsets = torch.tensor(row_offsets_cpu, dtype=torch.int32, device="cuda")
    row_to_batch = row_to_batch.cuda()
    block_table = block_table.cuda()
    out_values = torch.empty(rows, topk, dtype=torch.float32, device="cuda")
    out_raw = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    out_physical = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    out_counts = torch.empty(rows, dtype=torch.int32, device="cuda")

    flydsl_candidate_topk_merge(
        values,
        indices,
        counts,
        row_offsets,
        row_to_batch,
        block_table,
        out_values,
        out_raw,
        out_physical,
        out_counts,
        page_size,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out_counts.cpu(),
        torch.full((rows,), topk, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    for row, wanted in enumerate(expected):
        wanted_indices = torch.tensor(
            sorted(index for _, index in wanted),
            dtype=torch.int32,
        )
        got_indices = torch.sort(out_raw[row].cpu()).values
        torch.testing.assert_close(got_indices, wanted_indices, rtol=0, atol=0)

        # Values and physical slots must follow each emitted raw index.
        source = {
            int(indices[cta, slot].cpu()): values[cta, slot].cpu()
            for cta in range(row_offsets_cpu[row], row_offsets_cpu[row + 1])
            for slot in range(counts_cpu[cta])
        }
        for slot in range(topk):
            raw = int(out_raw[row, slot])
            assert torch.equal(out_values[row, slot].cpu(), source[raw])
            expected_physical = (
                int(block_table[row, raw // page_size]) * page_size + raw % page_size
            )
            assert int(out_physical[row, slot]) == expected_physical
