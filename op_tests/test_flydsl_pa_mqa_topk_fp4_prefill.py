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

HEADS = 64
HEAD_DIM = 128
KV_BLOCK_SIZE = 64


def _make_case(seed: int = 7):
    from op_tests.test_flydsl_pa_mqa_logits_fp4_prefill import (
        indexer_k_fp4_paged_preshuffle,
        quant_q_fp4_preshuffle,
    )

    torch.manual_seed(seed)
    device = torch.device("cuda")
    batch = 2
    max_seq_len = 5120
    blocks_per_seq = max_seq_len // KV_BLOCK_SIZE
    num_blocks = batch * blocks_per_seq
    block_tables = torch.randperm(
        num_blocks,
        device=device,
        dtype=torch.int32,
    ).reshape(batch, blocks_per_seq)
    kv = torch.randn(
        batch,
        max_seq_len,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    logical_batch = torch.arange(batch, device=device).repeat_interleave(max_seq_len)
    logical_token = torch.arange(max_seq_len, device=device).repeat(batch)
    physical_page = block_tables[
        logical_batch,
        logical_token // KV_BLOCK_SIZE,
    ].to(torch.int64)
    slot_mapping = (physical_page * KV_BLOCK_SIZE + logical_token % KV_BLOCK_SIZE).to(
        torch.int32
    )
    kv_cache = torch.zeros(
        num_blocks,
        1,
        4,
        KV_BLOCK_SIZE,
        16,
        dtype=torch.uint8,
        device=device,
    )
    kv_scale = torch.zeros(
        num_blocks,
        1,
        4,
        KV_BLOCK_SIZE,
        dtype=torch.uint8,
        device=device,
    )
    indexer_k_fp4_paged_preshuffle(
        kv.reshape(-1, HEAD_DIM),
        slot_mapping,
        kv_cache,
        kv_scale,
        KV_BLOCK_SIZE,
    )

    row_to_batch = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int32, device=device)
    local_starts = torch.tensor(
        [37, 0, 4011, 1700, 2200, 900],
        dtype=torch.int32,
        device=device,
    )
    local_ends = torch.tensor(
        [5097, 777, 4411, 1700, 4900, 5050],
        dtype=torch.int32,
        device=device,
    )
    rows = row_to_batch.numel()
    q = torch.randn(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    weights = (torch.randn(rows, HEADS, dtype=torch.float32, device=device) * 0.1).to(
        torch.bfloat16
    )
    q_fp4, q_scale = quant_q_fp4_preshuffle(q)
    return (
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        local_starts,
        local_ends,
        max_seq_len,
    )


def _stable_reference(full_logits, block_tables, row_to_batch, starts, ends, topk):
    rows = full_logits.shape[0]
    values = torch.full(
        (rows, topk),
        -float("inf"),
        dtype=torch.float32,
        device=full_logits.device,
    )
    raw_indices = torch.full(
        (rows, topk),
        -1,
        dtype=torch.int32,
        device=full_logits.device,
    )
    physical_indices = torch.full_like(raw_indices, -1)
    counts = torch.empty(rows, dtype=torch.int32, device=full_logits.device)
    for row in range(rows):
        start, end = int(starts[row]), int(ends[row])
        count = min(topk, max(end - start, 0))
        counts[row] = count
        if count == 0:
            continue
        if end - start <= topk:
            raw = torch.arange(start, end, device=full_logits.device)
        else:
            raw = (
                torch.argsort(
                    full_logits[row, start:end],
                    descending=True,
                    stable=True,
                )[:count]
                + start
            )
        raw_indices[row, :count] = raw.to(torch.int32)
        values[row, :count] = full_logits[row, raw]
        batch = int(row_to_batch[row])
        physical_indices[row, :count] = (
            block_tables[batch, raw // KV_BLOCK_SIZE] * KV_BLOCK_SIZE
            + raw % KV_BLOCK_SIZE
        ).to(torch.int32)
    return values, raw_indices, physical_indices, counts


def _full_logits(case, weight_scale: float):
    from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4_prefill

    return flydsl_pa_mqa_logits_fp4_prefill(
        *case,
        weight_scale=weight_scale,
        block_k=256,
        kv_block_size=KV_BLOCK_SIZE,
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
def test_fused_public_path_matches_full_logits(topk: int) -> None:
    from aiter.ops.flydsl import (
        allocate_fp4_prefill_topk_workspace,
        flydsl_pa_mqa_topk_fp4_prefill,
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

    # Match serving: one schedule slot per row. The case includes an empty row,
    # so this also verifies that compact row offsets, rather than row == CTA,
    # drive the no-radix merge.
    parallel_unit_num = rows
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

    torch.cuda.synchronize()

    actual = (
        fused.values,
        fused.raw_indices,
        fused.physical_indices,
        fused.counts,
    )
    for got, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)


@requires_gfx950_flydsl
@pytest.mark.parametrize("topk", [512, 1024])
def test_fused_split_candidate_path_matches_full_logits(topk: int) -> None:
    from aiter.ops.flydsl import (
        allocate_fp4_prefill_topk_workspace,
        flydsl_pa_mqa_topk_fp4_prefill,
    )

    case = _make_case(seed=83 + topk)
    rows = case[0].shape[0]
    parallel_unit_num = 512
    weight_scale = 1.25
    expected = _stable_reference(
        _full_logits(case, weight_scale),
        case[4],
        case[6],
        case[7],
        case[8],
        topk,
    )
    workspace = allocate_fp4_prefill_topk_workspace(
        rows,
        parallel_unit_num,
        topk,
        case[0].device,
    )
    result = flydsl_pa_mqa_topk_fp4_prefill(
        *case,
        topk=topk,
        weight_scale=weight_scale,
        parallel_unit_num=parallel_unit_num,
        workspace=workspace,
    )
    torch.cuda.synchronize()

    row_offsets = workspace.row_offsets.cpu()
    assert torch.any(row_offsets[1:] - row_offsets[:-1] > 1)
    actual = (
        result.values,
        result.raw_indices,
        result.physical_indices,
        result.counts,
    )
    for got, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)


@requires_gfx950_flydsl
@pytest.mark.parametrize("topk", [512, 1024])
def test_equal_score_pool_stays_monotonic_across_compactions(topk: int) -> None:
    from aiter.ops.flydsl import (
        allocate_fp4_prefill_topk_workspace,
        flydsl_pa_mqa_topk_fp4_prefill,
    )

    case = _make_case(seed=91 + topk)
    case[5][0].zero_()
    case[5][5].zero_()
    rows = case[0].shape[0]
    score_batch_chunks = 2
    workspace = allocate_fp4_prefill_topk_workspace(
        rows,
        rows,
        topk,
        case[0].device,
    )
    result = flydsl_pa_mqa_topk_fp4_prefill(
        *case,
        topk=topk,
        weight_scale=1.25,
        parallel_unit_num=rows,
        score_batch_chunks=score_batch_chunks,
        workspace=workspace,
    )
    torch.cuda.synchronize()

    row_offsets = workspace.row_offsets.cpu().tolist()
    for row in (0, 5):
        start = int(case[7][row])
        end = int(case[8][row])
        assert end - start > topk + score_batch_chunks * 256
        cta = row_offsets[row]
        assert row_offsets[row + 1] == cta + 1
        expected_raw = torch.arange(
            start,
            start + topk,
            dtype=torch.int32,
            device=case[0].device,
        )
        assert int(workspace.candidate_counts[cta]) == topk
        torch.testing.assert_close(
            workspace.candidate_indices[cta],
            expected_raw,
            rtol=0,
            atol=0,
        )
        assert int(result.counts[row]) == topk
        torch.testing.assert_close(
            result.raw_indices[row],
            expected_raw,
            rtol=0,
            atol=0,
        )


@requires_gfx950_flydsl
@pytest.mark.parametrize("score_batch_chunks", [1, 2])
def test_short_rows_publish_complete_score_batches(score_batch_chunks: int) -> None:
    from aiter.ops.flydsl import flydsl_pa_mqa_topk_fp4_prefill

    case = list(_make_case(seed=109 + score_batch_chunks))
    case[7] = torch.tensor(
        [37, 0, 4011, 1700, 2200, 900],
        dtype=torch.int32,
        device=case[0].device,
    )
    case[8] = case[7] + torch.tensor(
        [1, 255, 256, 257, 511, 777],
        dtype=torch.int32,
        device=case[0].device,
    )
    topk = 1024
    full_logits = _full_logits(case, 1.25)
    expected = _stable_reference(
        full_logits,
        case[4],
        case[6],
        case[7],
        case[8],
        topk,
    )

    for _ in range(3):
        result = flydsl_pa_mqa_topk_fp4_prefill(
            *case,
            topk=topk,
            weight_scale=1.25,
            parallel_unit_num=case[0].shape[0],
            score_batch_chunks=score_batch_chunks,
        )
        torch.cuda.synchronize()
        actual = (
            result.values,
            result.raw_indices,
            result.physical_indices,
            result.counts,
        )
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
def test_short_rows_never_dereference_padding_pages() -> None:
    from aiter.ops.flydsl import flydsl_pa_mqa_topk_fp4_prefill

    case = list(_make_case(seed=127))
    case[4][:, 1:] = -1
    case[7] = torch.tensor(
        [0, 1, 7, 16, 31, 48],
        dtype=torch.int32,
        device=case[0].device,
    )
    case[8] = torch.tensor(
        [1, 3, 15, 32, 63, 64],
        dtype=torch.int32,
        device=case[0].device,
    )
    result = flydsl_pa_mqa_topk_fp4_prefill(
        *case,
        topk=512,
        parallel_unit_num=case[0].shape[0],
        score_batch_chunks=1,
    )
    torch.cuda.synchronize()

    for row in range(case[0].shape[0]):
        start = int(case[7][row])
        end = int(case[8][row])
        count = end - start
        expected_raw = torch.arange(
            start,
            end,
            dtype=torch.int32,
            device=case[0].device,
        )
        batch = int(case[6][row])
        expected_physical = case[4][batch, 0] * KV_BLOCK_SIZE + expected_raw
        assert int(result.counts[row]) == count
        torch.testing.assert_close(
            result.raw_indices[row, :count], expected_raw, rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.physical_indices[row, :count],
            expected_physical,
            rtol=0,
            atol=0,
        )
        assert torch.all(result.raw_indices[row, count:] == -1)


@requires_gfx950_flydsl
def test_windows_are_clamped_to_sequence_capacity() -> None:
    from aiter.ops.flydsl import flydsl_pa_mqa_topk_fp4_prefill

    case = list(_make_case(seed=131))
    max_seq_len = case[9]
    case[7] = torch.tensor(
        [-257, -1, 0, max_seq_len - 100, max_seq_len, max_seq_len + 1],
        dtype=torch.int32,
        device=case[0].device,
    )
    case[8] = torch.tensor(
        [1, 63, max_seq_len + 513, max_seq_len + 1, max_seq_len + 2, max_seq_len + 3],
        dtype=torch.int32,
        device=case[0].device,
    )
    result = flydsl_pa_mqa_topk_fp4_prefill(
        *case,
        topk=1024,
        parallel_unit_num=case[0].shape[0],
        score_batch_chunks=2,
    )
    torch.cuda.synchronize()

    for row in range(case[0].shape[0]):
        start = max(0, min(int(case[7][row]), max_seq_len))
        end = max(start, min(int(case[8][row]), max_seq_len))
        count = min(end - start, 1024)
        assert int(result.counts[row]) == count
        if end - start <= 1024:
            expected_raw = torch.arange(
                start,
                end,
                dtype=torch.int32,
                device=case[0].device,
            )
            torch.testing.assert_close(
                result.raw_indices[row, :count], expected_raw, rtol=0, atol=0
            )
        if count:
            assert torch.all(result.raw_indices[row, :count] >= start)
            assert torch.all(result.raw_indices[row, :count] < end)
        assert torch.all(result.raw_indices[row, count:] == -1)


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


@pytest.mark.skipif(
    importlib.util.find_spec("flydsl") is None,
    reason="requires FlyDSL",
)
def test_single_cta_row_plan_property() -> None:
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        _row_plan_torch,
    )

    generator = torch.Generator().manual_seed(20260905)
    rows = 1024
    block_k = 256
    s_max = 64
    for _ in range(32):
        starts = torch.randint(
            -block_k,
            s_max * block_k,
            (rows,),
            dtype=torch.int32,
            generator=generator,
        )
        ends = torch.randint(
            -block_k,
            s_max * block_k,
            (rows,),
            dtype=torch.int32,
            generator=generator,
        )
        plan = _row_plan_torch(
            starts,
            ends,
            block_k,
            rows,
            s_max,
            s_max * block_k,
            single_cta_per_row=True,
        )

        window_starts = torch.clamp(starts, min=0, max=s_max * block_k)
        window_ends = torch.maximum(
            torch.clamp(ends, min=0, max=s_max * block_k),
            window_starts,
        )
        first_chunks = window_starts // block_k
        end_chunks = (window_ends + block_k - 1) // block_k
        chunks = torch.where(
            window_ends > window_starts,
            torch.clamp(end_chunks - first_chunks, min=0),
            0,
        )
        expected_ctas = (chunks > 0).to(torch.int32)
        actual_ctas = plan.incl - plan.excl

        torch.testing.assert_close(actual_ctas, expected_ctas, rtol=0, atol=0)
        assert int(plan.safe) == max(int(chunks.max()), 1)
        assert int(plan.total_splits) == int(expected_ctas.sum())


@pytest.mark.skipif(
    importlib.util.find_spec("flydsl") is None,
    reason="requires FlyDSL",
)
def test_score_batch_lds_budget() -> None:
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        DEFAULT_SCORE_BATCH_CHUNKS,
        SUPPORTED_SCORE_BATCH_CHUNKS,
        _streaming_topk_lds_bytes,
    )

    assert DEFAULT_SCORE_BATCH_CHUNKS == 20
    assert SUPPORTED_SCORE_BATCH_CHUNKS == (1, 2, 4, 8, 16, 20, 24)
    assert _streaming_topk_lds_bytes(512, 16) == 37_936
    assert _streaming_topk_lds_bytes(1024, 16) == 42_032
    assert _streaming_topk_lds_bytes(1024, 24) == 58_416
    assert _streaming_topk_lds_bytes(1024, 16) < 64 * 1024


@pytest.mark.skipif(
    importlib.util.find_spec("flydsl") is None,
    reason="requires FlyDSL",
)
def test_legacy_compile_cache_does_not_evict_live_modules() -> None:
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        compile_pa_mqa_logits_fp4_prefill,
    )

    assert compile_pa_mqa_logits_fp4_prefill.cache_info().maxsize is None


@requires_gfx950_flydsl
@pytest.mark.parametrize("topk", [512, 1024])
def test_single_cta_copy_uses_offsets_and_counts(topk: int) -> None:
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        _copy_single_cta_topk_candidates,
    )

    rows = 5
    row_offsets = torch.tensor(
        [0, 1, 1, 2, 3, 3],
        dtype=torch.int32,
        device="cuda",
    )
    counts = torch.tensor([topk, 7, 0], dtype=torch.int32, device="cuda")
    values = torch.arange(3 * topk, dtype=torch.float32, device="cuda").reshape(
        3,
        topk,
    )
    indices = torch.arange(3 * topk, dtype=torch.int32, device="cuda").reshape(
        3,
        topk,
    )
    out_values = torch.empty(rows, topk, dtype=torch.float32, device="cuda")
    out_indices = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    out_counts = torch.empty(rows, dtype=torch.int32, device="cuda")

    _copy_single_cta_topk_candidates(
        values,
        indices,
        counts,
        row_offsets,
        out_values,
        out_indices,
        out_counts,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out_counts.cpu(),
        torch.tensor([topk, 0, 7, 0, 0], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(out_values[0], values[0], rtol=0, atol=0)
    torch.testing.assert_close(out_indices[0], indices[0], rtol=0, atol=0)
    torch.testing.assert_close(out_values[2, :7], values[1, :7], rtol=0, atol=0)
    torch.testing.assert_close(out_indices[2, :7], indices[1, :7], rtol=0, atol=0)
    invalid_rows = torch.tensor([1, 3, 4], device="cuda")
    assert torch.all(torch.isneginf(out_values[invalid_rows]))
    assert torch.all(out_indices[invalid_rows] == -1)
    assert torch.all(torch.isneginf(out_values[2, 7:]))
    assert torch.all(out_indices[2, 7:] == -1)
