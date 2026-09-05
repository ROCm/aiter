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


def _make_empty_window_case(table_width: int):
    device = torch.device("cuda")
    rows = 2
    max_seq_len = table_width * 64
    starts = (
        torch.tensor([0, 17], dtype=torch.int32, device=device)
        if table_width
        else torch.zeros(rows, dtype=torch.int32, device=device)
    )
    return (
        torch.zeros((rows, 64, 64), dtype=torch.uint8, device=device),
        torch.zeros((rows, 1, 4, 16, 4), dtype=torch.uint8, device=device),
        torch.empty((0, 1, 4, 64, 16), dtype=torch.uint8, device=device),
        torch.empty((0, 1, 4, 64), dtype=torch.uint8, device=device),
        torch.full((1, table_width), -1, dtype=torch.int32, device=device),
        torch.zeros((rows, 64), dtype=torch.bfloat16, device=device),
        torch.zeros(rows, dtype=torch.int32, device=device),
        starts,
        starts.clone(),
        max_seq_len,
    )


@requires_gfx950_flydsl
@pytest.mark.parametrize("table_width", [0, 1])
def test_empty_windows_never_dereference_invalid_pages(table_width: int) -> None:
    from aiter.ops.flydsl import flydsl_pa_mqa_topk_fp4_prefill

    case = _make_empty_window_case(table_width)
    result = flydsl_pa_mqa_topk_fp4_prefill(
        *case,
        topk=512,
        parallel_unit_num=case[0].shape[0],
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(result.counts) == 0
    assert torch.all(result.raw_indices == -1)
    assert torch.all(result.physical_indices == -1)
    assert torch.all(torch.isneginf(result.values))


@requires_gfx950_flydsl
def test_fused_topk_allocates_on_supplied_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    module = importlib.import_module(
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill"
    )
    case = _make_empty_window_case(1)
    stream = torch.cuda.Stream(device=case[0].device)
    original_allocate = module.allocate_fp4_prefill_topk_workspace
    allocation_streams: list[int] = []

    def checked_allocate(rows, parallel_unit_num, topk, device):
        allocation_streams.append(torch.cuda.current_stream(device).cuda_stream)
        return original_allocate(rows, parallel_unit_num, topk, device)

    monkeypatch.setattr(
        module,
        "allocate_fp4_prefill_topk_workspace",
        checked_allocate,
    )
    result = module.flydsl_pa_mqa_topk_fp4_prefill(
        *case,
        topk=512,
        parallel_unit_num=case[0].shape[0],
        stream=stream,
    )
    stream.synchronize()

    assert allocation_streams == [stream.cuda_stream]
    assert torch.count_nonzero(result.counts) == 0


@requires_gfx950_flydsl
def test_fused_topk_validates_stream_before_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    module = importlib.import_module(
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill"
    )
    case = _make_empty_window_case(1)
    allocated = False

    def unexpected_allocate(*args, **kwargs):
        nonlocal allocated
        allocated = True
        raise AssertionError("workspace allocation must follow stream validation")

    class WrongDeviceStream:
        device = torch.device("cuda", (case[0].device.index or 0) + 1)

    monkeypatch.setattr(
        module,
        "allocate_fp4_prefill_topk_workspace",
        unexpected_allocate,
    )
    with pytest.raises(ValueError, match="stream must belong"):
        module.flydsl_pa_mqa_topk_fp4_prefill(
            *case,
            topk=512,
            parallel_unit_num=case[0].shape[0],
            stream=WrongDeviceStream(),
        )
    assert not allocated


@requires_gfx950_flydsl
def test_tiled_topk_validates_stream_before_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    module = importlib.import_module("aiter.ops.flydsl.fp4_prefill_topk")
    case = _make_empty_window_case(1)
    allocated = False

    def unexpected_allocate(*args, **kwargs):
        nonlocal allocated
        allocated = True
        raise AssertionError("workspace allocation must follow stream validation")

    class WrongDeviceStream:
        device = torch.device("cuda", (case[0].device.index or 0) + 1)

    monkeypatch.setattr(
        module,
        "allocate_fp4_prefill_topk_workspace",
        unexpected_allocate,
    )
    with pytest.raises(ValueError, match="stream must belong"):
        module.flydsl_pa_mqa_fp4_prefill_topk(
            *case,
            k=512,
            stream=WrongDeviceStream(),
        )
    assert not allocated


@requires_gfx950_flydsl
def test_fused_topk_rejects_capture_before_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    module = importlib.import_module(
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill"
    )
    case = _make_empty_window_case(1)
    allocated = False

    def unexpected_allocate(*args, **kwargs):
        nonlocal allocated
        allocated = True
        raise AssertionError("capture rejection must precede workspace allocation")

    monkeypatch.setattr(
        module,
        "allocate_fp4_prefill_topk_workspace",
        unexpected_allocate,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="does not support.*graph capture"):
        module.flydsl_pa_mqa_topk_fp4_prefill(
            *case,
            topk=512,
            parallel_unit_num=case[0].shape[0],
        )
    assert not allocated


@requires_gfx950_flydsl
def test_schedule_rejects_noncontiguous_row_offsets() -> None:
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        compute_prefill_schedule,
    )

    rows = 2
    device = torch.device("cuda")
    metadata = torch.zeros(rows, dtype=torch.int32, device=device)
    backing = torch.empty(2 * (rows + 1), dtype=torch.int32, device=device)
    row_offsets = backing[::2]
    assert not row_offsets.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        compute_prefill_schedule(
            metadata,
            metadata,
            metadata,
            256,
            rows,
            64,
            row_offsets_out=row_offsets,
        )
