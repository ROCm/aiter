# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Focused public-path tests for FP4 paged-MQA streaming TopK."""

from __future__ import annotations

import importlib.util
import math
import struct

import pytest
import torch


def _gfx950_flydsl_available() -> bool:
    if importlib.util.find_spec("flydsl") is None:
        return False
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx950"
    except (AttributeError, RuntimeError):
        return False


requires_gfx950_flydsl = pytest.mark.skipif(
    not _gfx950_flydsl_available(),
    reason="requires a ROCm gfx950 GPU and FlyDSL",
)


def _ordered_i32(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return -(1 << 31)
    signed = bits if bits < (1 << 31) else bits - (1 << 32)
    return signed ^ ((signed >> 31) & 0x7FFFFFFF)


def test_reference_total_order_has_required_special_values() -> None:
    assert _ordered_i32(float("nan")) < _ordered_i32(float("-inf"))
    assert _ordered_i32(float("-inf")) < _ordered_i32(-0.0)
    assert _ordered_i32(-0.0) < _ordered_i32(+0.0)
    assert _ordered_i32(+0.0) < _ordered_i32(float("inf"))


@requires_gfx950_flydsl
@pytest.mark.parametrize("topk", [512, 1024])
def test_true_and_tiled_public_paths_match_full_logits(topk: int) -> None:
    from aiter.ops.flydsl import (
        allocate_fp4_bounded_prefill_topk_workspace,
        allocate_fp4_prefill_topk_workspace,
        flydsl_pa_mqa_topk_fp4_prefill,
        flydsl_pa_mqa_topk_fp4_prefill_tiled,
    )
    from op_tests.test_flydsl_pa_mqa_fp4_prefill_topk import (
        TILE_TOKENS,
        _full_logits,
        _make_case,
        _stable_reference,
    )

    case = _make_case(seed=71 + topk)
    rows = case[0].shape[0]
    weight_scale = 1.25
    full_logits = _full_logits(case, weight_scale)
    expected = _stable_reference(
        full_logits,
        case[4],
        case[6],
        case[7],
        case[8],
        topk,
    )

    parallel_unit_num = max(512, rows)
    fused_workspace = allocate_fp4_prefill_topk_workspace(
        rows,
        parallel_unit_num,
        topk,
        case[0].device,
    )
    fused = flydsl_pa_mqa_topk_fp4_prefill(
        *case[:6],
        case[6],
        case[7],
        case[8],
        case[9],
        topk=topk,
        weight_scale=weight_scale,
        parallel_unit_num=parallel_unit_num,
        workspace=fused_workspace,
    )

    tiled_workspace = allocate_fp4_bounded_prefill_topk_workspace(
        rows,
        topk,
        case[0].device,
        tile_tokens=TILE_TOKENS,
    )
    tiled = flydsl_pa_mqa_topk_fp4_prefill_tiled(
        *case[:6],
        case[6],
        case[7],
        case[8],
        case[9],
        k=topk,
        tile_tokens=TILE_TOKENS,
        weight_scale=weight_scale,
        workspace=tiled_workspace,
    )
    torch.cuda.synchronize()

    actual_results = (
        (
            fused.values,
            fused.raw_indices,
            fused.physical_indices,
            fused.counts,
        ),
        (
            tiled.values,
            tiled.raw_indices,
            tiled.kv_indices,
            tiled.valid_counts,
        ),
    )
    for actual in actual_results:
        for got, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)


@requires_gfx950_flydsl
@pytest.mark.parametrize("topk", [512, 1024])
def test_finalizer_orders_long_rows_and_keeps_short_rows_sequential(
    topk: int,
) -> None:
    from aiter.ops.flydsl.mqa_topk_finalize import order_and_map_mqa_topk

    rows = 2
    page_size = 64
    max_seq_len = topk + page_size
    generator = torch.Generator().manual_seed(99 + topk)

    raw_long = torch.randperm(topk, generator=generator, dtype=torch.int64)
    values_by_raw = torch.linspace(-3.0, 3.0, topk, dtype=torch.float32)
    values_by_raw[0] = float("nan")
    values_by_raw[1] = -float("inf")
    values_by_raw[2] = -0.0
    values_by_raw[3] = +0.0
    values_by_raw[4:12] = 1.0
    source_values = torch.full((rows, topk), -float("inf"), dtype=torch.float32)
    source_raw = torch.full((rows, topk), -1, dtype=torch.int32)
    source_raw[0] = raw_long.to(torch.int32)
    source_values[0] = values_by_raw[raw_long]

    short_raw = torch.randperm(10, generator=generator) + 100
    source_raw[1, :10] = short_raw.to(torch.int32)
    source_values[1, :10] = torch.arange(10, dtype=torch.float32)[short_raw - 100]
    counts = torch.tensor([topk, 10], dtype=torch.int32)
    starts = torch.tensor([0, 100], dtype=torch.int32)
    ends = torch.tensor([topk + 1, 110], dtype=torch.int32)
    row_to_batch = torch.tensor([0, 1], dtype=torch.int32)
    table_width = math.ceil(max_seq_len / page_size)
    block_tables = (
        torch.arange(rows * table_width, dtype=torch.int32)
        .reshape(rows, table_width)
        .flip(1)
        .contiguous()
    )

    out_values = torch.empty_like(source_values, device="cuda")
    out_raw = torch.empty_like(source_raw, device="cuda")
    out_slots = torch.empty_like(source_raw, device="cuda")
    order_and_map_mqa_topk(
        source_values.cuda(),
        source_raw.cuda(),
        counts.cuda(),
        starts.cuda(),
        ends.cuda(),
        row_to_batch.cuda(),
        block_tables.cuda(),
        out_values,
        out_raw,
        out_slots,
        max_seq_len,
        topk,
        page_size,
    )
    torch.cuda.synchronize()

    expected_long = sorted(
        range(topk),
        key=lambda raw: (-_ordered_i32(float(values_by_raw[raw])), raw),
    )
    assert out_raw[0].cpu().tolist() == expected_long
    assert out_raw[1, :10].cpu().tolist() == list(range(100, 110))
    assert torch.all(out_raw[1, 10:] == -1)
    for row, count in ((0, topk), (1, 10)):
        raw = out_raw[row, :count].cpu()
        expected_slots = (
            block_tables[row, raw // page_size] * page_size + raw % page_size
        )
        torch.testing.assert_close(
            out_slots[row, :count].cpu(),
            expected_slots,
            rtol=0,
            atol=0,
        )
