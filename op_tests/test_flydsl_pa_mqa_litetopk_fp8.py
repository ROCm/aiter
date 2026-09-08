# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes, indexer_k_quant_and_cache
from aiter.ops.flydsl import (
    FP8LiteTopKResult,
    allocate_fp4_litetopk_workspace,
    allocate_fp8_litetopk_workspace,
    flydsl_pa_mqa_litetopk_fp8_prefill,
    fp8_litetopk_workspace_nbytes,
    fp8_litetopk_workspace_size,
)
from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
    _flydsl_pa_mqa_litetopk_fp8_prefill_scan,
    _flydsl_pa_mqa_litetopk_fp8_seed,
    _flydsl_pa_mqa_logits_fp8_prefill,
)
from aiter.ops.flydsl.pa_mqa_litetopk_fp4 import (
    prepare_fp4_litetopk_seed,
)
from aiter.ops.flydsl.pa_mqa_litetopk_fp8 import (
    DEFAULT_MAX_SEQ_LEN,
    STATUS_CANDIDATE_OVERFLOW,
    STATUS_INVALID_INDEX,
    STATUS_NONFINITE,
)
from aiter.ops.triton.attention.pa_mqa_litetopk_seed import (
    build_pa_mqa_litetopk_scan_schedule,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() != "gfx950", reason="FP8 LiteTopK requires gfx950"
)


def _make_preshuffled_cache(
    keys: torch.Tensor, scales: torch.Tensor, page_size: int
) -> torch.Tensor:
    tokens, head_dim = keys.shape
    pages = (tokens + page_size - 1) // page_size
    padded_keys = torch.zeros(
        (pages * page_size, head_dim), dtype=keys.dtype, device=keys.device
    )
    padded_keys[:tokens] = keys
    payload = (
        padded_keys.view(pages, page_size // 16, 16, head_dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(pages, page_size * head_dim)
    )
    padded_scales = torch.zeros(
        (pages * page_size,), dtype=torch.float32, device=keys.device
    )
    padded_scales[:tokens] = scales
    cache = torch.empty(
        (pages, page_size * (head_dim + 4)),
        dtype=keys.dtype,
        device=keys.device,
    )
    cache[:, : page_size * head_dim] = payload
    cache[:, page_size * head_dim :] = (
        padded_scales.view(pages, page_size).view(torch.uint8).view(keys.dtype)
    )
    return cache.view(torch.uint8).view(pages, page_size, 1, head_dim + 4)


def _make_zero_case(rows: int, tokens: int):
    heads, head_dim, page_size = 32, 128, 64
    padded_tokens = ((tokens + page_size - 1) // page_size) * page_size
    q = torch.zeros((rows, heads, head_dim), dtype=dtypes.fp8, device="cuda")
    keys = torch.zeros((padded_tokens, head_dim), dtype=dtypes.fp8, device="cuda")
    scales = torch.ones((padded_tokens,), dtype=torch.float32, device="cuda")
    weights = torch.ones((rows, heads), dtype=torch.float32, device="cuda")
    cache = _make_preshuffled_cache(keys, scales, page_size)
    block_tables = torch.arange(cache.shape[0], dtype=torch.int32, device="cuda").view(
        1, -1
    )
    row_to_batch = torch.zeros((rows,), dtype=torch.int32, device="cuda")
    return q, cache, block_tables, weights, row_to_batch


def _run_zero_case_on_stream(
    *, device: torch.device, stream: torch.cuda.Stream, tokens: int
):
    with torch.cuda.device(device), torch.cuda.stream(stream):
        q, cache, block_tables, weights, row_to_batch = _make_zero_case(1, tokens)
        starts = torch.zeros(1, dtype=torch.int32, device=device)
        ends = torch.full((1,), tokens, dtype=torch.int32, device=device)
        workspace = allocate_fp8_litetopk_workspace(1, device, stream=stream)
        result = flydsl_pa_mqa_litetopk_fp8_prefill(
            q,
            cache,
            block_tables,
            weights,
            row_to_batch,
            starts,
            ends,
            tokens,
            workspace=workspace,
            stream=stream,
        )
    return result


def test_paged_flydsl_fp8_dense_matches_math_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        pytest.skip("requires gfx950")

    torch.manual_seed(7)
    rows, heads, head_dim, page_size, tokens = 2, 32, 128, 64, 1024
    q = (torch.randn((rows, heads, head_dim), device="cuda") * 0.25).to(dtypes.fp8)
    keys = (torch.randn((tokens, head_dim), device="cuda") * 0.25).to(dtypes.fp8)
    scales = torch.rand((tokens,), dtype=torch.float32, device="cuda") + 0.25
    weights = torch.randn((rows, heads), dtype=torch.float32, device="cuda")
    pages = tokens // page_size
    block_tables = torch.randperm(pages, device="cuda").to(torch.int32)[None]
    physical_keys = torch.empty_like(keys)
    physical_scales = torch.empty_like(scales)
    for logical_page in range(pages):
        physical_page = int(block_tables[0, logical_page])
        logical_slice = slice(logical_page * page_size, (logical_page + 1) * page_size)
        physical_slice = slice(
            physical_page * page_size, (physical_page + 1) * page_size
        )
        physical_keys[physical_slice] = keys[logical_slice]
        physical_scales[physical_slice] = scales[logical_slice]
    cache = _make_preshuffled_cache(physical_keys, physical_scales, page_size)
    row_to_batch = torch.zeros((rows,), dtype=torch.int32, device="cuda")
    starts = torch.tensor([17, 65], dtype=torch.int32, device="cuda")
    ends = torch.tensor([1003, 997], dtype=torch.int32, device="cuda")

    actual = _flydsl_pa_mqa_logits_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        tokens,
    )
    torch.cuda.synchronize()

    for row in range(rows):
        logical = torch.arange(int(starts[row]), int(ends[row]), device="cuda")
        dots = torch.einsum("hd,td->ht", q[row].float(), keys[logical].float())
        expected = (
            torch.relu(dots * scales[logical][None, :]) * weights[row, :, None]
        ).sum(dim=0)
        torch.testing.assert_close(actual[row, logical], expected, rtol=2e-3, atol=2e-3)


def test_flydsl_fp8_seed_matches_dense_oracle() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        pytest.skip("requires gfx950")

    torch.manual_seed(12)
    rows, heads, head_dim, page_size = 1, 32, 128, 64
    tokens, sample_len, topk, merge_cap = 8192, 8192, 2048, 16384
    q = (torch.randn((rows, heads, head_dim), device="cuda") * 0.25).to(dtypes.fp8)
    keys = (torch.randn((tokens, head_dim), device="cuda") * 0.25).to(dtypes.fp8)
    scales = torch.rand((tokens,), dtype=torch.float32, device="cuda") + 0.25
    weights = torch.rand((rows, heads), dtype=torch.float32, device="cuda")
    pages = tokens // page_size
    block_tables = torch.randperm(pages, device="cuda").to(torch.int32)[None]
    physical_keys = torch.empty_like(keys)
    physical_scales = torch.empty_like(scales)
    for logical_page in range(pages):
        physical_page = int(block_tables[0, logical_page])
        logical_slice = slice(logical_page * page_size, (logical_page + 1) * page_size)
        physical_slice = slice(
            physical_page * page_size, (physical_page + 1) * page_size
        )
        physical_keys[physical_slice] = keys[logical_slice]
        physical_scales[physical_slice] = scales[logical_slice]
    cache = _make_preshuffled_cache(physical_keys, physical_scales, page_size)
    row_to_batch = torch.zeros((rows,), dtype=torch.int32, device="cuda")
    row_starts = torch.zeros((rows,), dtype=torch.int32, device="cuda")
    row_ends = torch.full((rows,), tokens, dtype=torch.int32, device="cuda")
    cta_info = torch.empty((rows, 6), dtype=torch.int32, device="cuda")
    build_pa_mqa_litetopk_scan_schedule(
        row_to_batch,
        row_starts,
        row_ends,
        cta_info,
        segment_start=0,
        segment_end=tokens,
        block_k=512,
        stream=torch.cuda.current_stream(),
    )
    actual = allocate_fp8_litetopk_workspace(rows, "cuda")
    expected = allocate_fp4_litetopk_workspace(
        rows,
        "cuda",
        topk=topk,
        sample_len=sample_len,
        merge_cap=merge_cap,
    )
    actual.status.zero_()
    actual.page_errors.zero_()

    _flydsl_pa_mqa_litetopk_fp8_seed(
        q,
        cache,
        block_tables,
        weights,
        cta_info,
        actual.origin,
        actual.inv_delta,
        actual.threshold,
        actual.histogram,
        actual.candidate_values,
        actual.candidate_indices,
        actual.candidate_counts,
        actual.select_starts,
        actual.select_ends,
        actual.status,
        n_ctas=rows,
        merge_cap=merge_cap,
        topk=topk,
        page_errors=actual.page_errors,
    )
    dense_scores = _flydsl_pa_mqa_logits_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        row_starts,
        row_ends,
        tokens,
    )
    prepare_fp4_litetopk_seed(
        dense_scores,
        row_starts,
        row_ends,
        expected,
        max_seq_len=tokens,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.origin, expected.origin, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        actual.inv_delta, expected.inv_delta, rtol=1e-5, atol=1e-5
    )
    assert torch.equal(actual.histogram, expected.histogram)
    assert torch.equal(actual.threshold, expected.threshold)
    count = int(actual.candidate_counts[0])
    candidates = actual.candidate_indices[0, :count]
    dense_topk = torch.topk(dense_scores[0], topk).indices.to(torch.int32)
    assert set(dense_topk.cpu().tolist()).issubset(set(candidates.cpu().tolist()))
    torch.testing.assert_close(
        actual.candidate_values[0, :count],
        dense_scores[0, candidates.long()],
        rtol=1e-4,
        atol=1e-4,
    )
    assert actual.status.cpu().tolist() == [0]


def test_fp8_suffix_candidates_contain_dense_topk() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        pytest.skip("requires gfx950")

    torch.manual_seed(13)
    rows, heads, head_dim, page_size = 1, 32, 128, 64
    tokens, sample_len, topk, merge_cap = 1024, 256, 64, 1024
    q = (torch.randn((rows, heads, head_dim), device="cuda") * 0.25).to(dtypes.fp8)
    keys = (torch.randn((tokens, head_dim), device="cuda") * 0.25).to(dtypes.fp8)
    scales = torch.rand((tokens,), dtype=torch.float32, device="cuda") + 0.25
    weights = torch.randn((rows, heads), dtype=torch.float32, device="cuda")
    cache = _make_preshuffled_cache(keys, scales, page_size)
    block_tables = torch.randperm(cache.shape[0], dtype=torch.int64, device="cuda").to(
        torch.int32
    )[None]
    physical_keys = torch.empty_like(keys)
    physical_scales = torch.empty_like(scales)
    for logical_page in range(cache.shape[0]):
        physical_page = int(block_tables[0, logical_page])
        logical_slice = slice(logical_page * page_size, (logical_page + 1) * page_size)
        physical_slice = slice(
            physical_page * page_size, (physical_page + 1) * page_size
        )
        physical_keys[physical_slice] = keys[logical_slice]
        physical_scales[physical_slice] = scales[logical_slice]
    cache = _make_preshuffled_cache(physical_keys, physical_scales, page_size)
    row_to_batch = torch.zeros((rows,), dtype=torch.int32, device="cuda")
    row_starts = torch.tensor([17], dtype=torch.int32, device="cuda")
    row_ends = torch.tensor([tokens - 7], dtype=torch.int32, device="cuda")
    workspace = allocate_fp8_litetopk_workspace(
        rows,
        "cuda",
        topk=topk,
        sample_len=sample_len,
        merge_cap=merge_cap,
    )
    workspace.page_errors.zero_()
    workspace.score_errors.zero_()

    seed_cta_info = torch.empty((rows, 6), dtype=torch.int32, device="cuda")
    build_pa_mqa_litetopk_scan_schedule(
        row_to_batch,
        row_starts,
        torch.minimum(row_starts + sample_len, row_ends),
        seed_cta_info,
        segment_start=0,
        segment_end=tokens,
        block_k=512,
        stream=torch.cuda.current_stream(),
    )
    _flydsl_pa_mqa_litetopk_fp8_seed(
        q,
        cache,
        block_tables,
        weights,
        seed_cta_info,
        workspace.origin,
        workspace.inv_delta,
        workspace.threshold,
        workspace.histogram,
        workspace.candidate_values,
        workspace.candidate_indices,
        workspace.candidate_counts,
        workspace.select_starts,
        workspace.select_ends,
        workspace.status,
        n_ctas=rows,
        merge_cap=merge_cap,
        topk=topk,
        page_errors=workspace.page_errors,
    )
    suffix_cta_info = torch.empty((rows, 6), dtype=torch.int32, device="cuda")
    build_pa_mqa_litetopk_scan_schedule(
        row_to_batch,
        row_starts + sample_len,
        row_ends,
        suffix_cta_info,
        segment_start=0,
        segment_end=tokens,
        block_k=512,
        stream=torch.cuda.current_stream(),
    )
    _flydsl_pa_mqa_litetopk_fp8_prefill_scan(
        q,
        cache,
        block_tables,
        weights,
        suffix_cta_info,
        workspace.origin,
        workspace.inv_delta,
        workspace.threshold,
        workspace.histogram,
        workspace.candidate_values,
        workspace.candidate_indices,
        workspace.candidate_counts,
        workspace.page_errors,
        workspace.score_errors,
        n_ctas=rows,
        block_k=512,
        topk=topk,
        merge_cap=merge_cap,
        refresh_every=32,
    )
    torch.cuda.synchronize()

    dense_scores = _flydsl_pa_mqa_logits_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        row_starts,
        row_ends,
        tokens,
    )
    logical = torch.arange(int(row_starts[0]), int(row_ends[0]), device="cuda")
    dense_topk = logical[
        torch.topk(dense_scores[0, int(row_starts[0]) : int(row_ends[0])], topk).indices
    ].cpu()
    count = int(workspace.candidate_counts[0])
    candidates = workspace.candidate_indices[0, : min(count, merge_cap)].cpu()
    assert count >= topk
    missing = set(dense_topk.tolist()) - set(candidates.tolist())
    assert not missing, (
        f"missing={sorted(missing)[:16]}, count={count}, "
        f"threshold={int(workspace.threshold[0])}, "
        f"histogram_sum={int(workspace.histogram[0].sum())}"
    )
    assert workspace.score_errors.cpu().tolist() == [0]


def test_fp8_workspace_size_query_matches_allocation() -> None:
    workspace = allocate_fp8_litetopk_workspace(3, "cuda")
    assert fp8_litetopk_workspace_size(3) == fp8_litetopk_workspace_nbytes(workspace)


def test_fp8_full_operator_matches_dense_set_and_maps_pages() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        pytest.skip("requires gfx950")

    torch.manual_seed(23)
    rows, heads, head_dim, page_size = 1, 32, 128, 64
    tokens, topk = 12_288, 2048
    q = (torch.randn((rows, heads, head_dim), device="cuda") * 0.25).to(dtypes.fp8)
    keys = (torch.randn((tokens, head_dim), device="cuda") * 0.25).to(dtypes.fp8)
    scales = torch.rand((tokens,), dtype=torch.float32, device="cuda") + 0.25
    weights = torch.rand((rows, heads), dtype=torch.float32, device="cuda")
    num_pages = tokens // page_size
    block_tables = torch.randperm(num_pages, device="cuda").to(torch.int32)[None]
    physical_keys = torch.empty_like(keys)
    physical_scales = torch.empty_like(scales)
    for logical_page in range(num_pages):
        physical_page = int(block_tables[0, logical_page])
        logical_slice = slice(logical_page * page_size, (logical_page + 1) * page_size)
        physical_slice = slice(
            physical_page * page_size, (physical_page + 1) * page_size
        )
        physical_keys[physical_slice] = keys[logical_slice]
        physical_scales[physical_slice] = scales[logical_slice]
    cache = _make_preshuffled_cache(physical_keys, physical_scales, page_size)
    row_to_batch = torch.zeros((rows,), dtype=torch.int32, device="cuda")
    row_starts = torch.tensor([37], dtype=torch.int32, device="cuda")
    row_ends = torch.tensor([tokens - 11], dtype=torch.int32, device="cuda")
    workspace = allocate_fp8_litetopk_workspace(rows, "cuda")

    result = flydsl_pa_mqa_litetopk_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        row_starts,
        row_ends,
        tokens,
        workspace=workspace,
    )
    torch.cuda.synchronize()

    dense_scores = _flydsl_pa_mqa_logits_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        row_starts,
        row_ends,
        tokens,
    )
    expected = torch.topk(dense_scores[0], topk).indices.to(torch.int32)
    assert result.status.cpu().tolist() == [0]
    assert workspace.status_ok.cpu().tolist() == [1]
    assert result.counts.cpu().tolist() == [topk]
    assert torch.equal(
        torch.sort(result.raw_indices[0]).values.cpu(),
        torch.sort(expected).values.cpu(),
    )
    expected_physical = (
        block_tables[0, result.raw_indices[0].long() // page_size] * page_size
        + result.raw_indices[0] % page_size
    )
    assert torch.equal(result.physical_indices[0], expected_physical)


def test_fp8_full_operator_accepts_production_cache_writer_output() -> None:
    torch.manual_seed(29)
    rows, heads, head_dim, page_size = 1, 32, 128, 64
    tokens, topk = 8192, 2048
    q = (torch.randn((rows, heads, head_dim), device="cuda") * 0.25).to(dtypes.fp8)
    keys = torch.randn((tokens, head_dim), dtype=torch.bfloat16, device="cuda")
    weights = torch.rand((rows, heads), dtype=torch.float32, device="cuda")
    pages = tokens // page_size
    block_tables = torch.randperm(pages, device="cuda").to(torch.int32)[None]
    physical_slots = (
        block_tables[0, torch.arange(tokens, device="cuda") // page_size] * page_size
        + torch.arange(tokens, device="cuda") % page_size
    ).to(torch.int64)
    cache = torch.zeros(
        (pages, page_size, head_dim + 4), dtype=dtypes.fp8, device="cuda"
    )
    indexer_k_quant_and_cache(
        keys,
        cache,
        physical_slots,
        head_dim,
        "ue8m0",
        True,
    )
    cache = cache.view(pages, page_size, 1, head_dim + 4)
    row_to_batch = torch.zeros(rows, dtype=torch.int32, device="cuda")
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), tokens, dtype=torch.int32, device="cuda")

    result = flydsl_pa_mqa_litetopk_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        tokens,
    )
    dense = _flydsl_pa_mqa_logits_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        tokens,
    )
    torch.cuda.synchronize()

    expected = torch.topk(dense[0], topk).indices.to(torch.int32)
    assert result.status.cpu().tolist() == [0]
    assert torch.equal(
        torch.sort(result.raw_indices[0]).values.cpu(),
        torch.sort(expected).values.cpu(),
    )


def test_fp8_seed_boundaries_and_short_rows() -> None:
    lengths = (0, 2047, 2048, 8191, 8192, 8193)
    q, cache, block_tables, weights, row_to_batch = _make_zero_case(
        len(lengths), max(lengths)
    )
    starts = torch.zeros(len(lengths), dtype=torch.int32, device="cuda")
    ends = torch.tensor(lengths, dtype=torch.int32, device="cuda")

    result = flydsl_pa_mqa_litetopk_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        max(lengths),
    )
    torch.cuda.synchronize()

    assert result.status.cpu().tolist() == [0] * len(lengths)
    assert result.counts.cpu().tolist() == [min(length, 2048) for length in lengths]
    assert result.candidate_counts.cpu().tolist() == list(lengths)
    for row, length in enumerate(lengths):
        count = min(length, 2048)
        assert torch.equal(
            result.raw_indices[row, :count].cpu(),
            torch.arange(count, dtype=torch.int32),
        )
        assert torch.all(result.raw_indices[row, count:] == -1)


def test_fp8_suffix_cutoff_ties_return_valid_unique_set() -> None:
    tokens, topk = 12_288, 2048
    q, cache, block_tables, weights, row_to_batch = _make_zero_case(1, tokens)
    q[0, 0, 0] = 1.0
    weights.zero_()
    weights[0, 0] = 1.0
    keys = torch.zeros((tokens, 128), dtype=dtypes.fp8, device="cuda")
    keys[8192:, 0] = 1.0
    scales = torch.ones((tokens,), dtype=torch.float32, device="cuda")
    cache = _make_preshuffled_cache(keys, scales, 64)
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), tokens, dtype=torch.int32, device="cuda")

    first = flydsl_pa_mqa_litetopk_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        tokens,
    )
    torch.cuda.synchronize()

    selected = first.raw_indices[0]
    assert first.status.cpu().tolist() == [0]
    assert first.counts.cpu().tolist() == [topk]
    assert selected.unique().numel() == topk
    assert torch.all((selected >= 8192) & (selected < tokens))


@pytest.mark.parametrize("tokens,overflows", [(16_384, False), (16_385, True)])
def test_fp8_candidate_capacity_boundary(tokens: int, overflows: bool) -> None:
    q, cache, block_tables, weights, row_to_batch = _make_zero_case(1, tokens)
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), tokens, dtype=torch.int32, device="cuda")
    workspace = allocate_fp8_litetopk_workspace(1, "cuda")

    result = flydsl_pa_mqa_litetopk_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        tokens,
        workspace=workspace,
        enforce_status=False,
    )
    torch.cuda.synchronize()

    assert result.candidate_counts.cpu().tolist() == [tokens]
    if overflows:
        assert int(result.status[0]) & STATUS_CANDIDATE_OVERFLOW
        assert result.counts.cpu().tolist() == [0]
        assert torch.all(result.raw_indices == -1)
        assert torch.all(result.physical_indices == -1)
    else:
        assert result.status.cpu().tolist() == [0]
        assert result.counts.cpu().tolist() == [2048]
        assert result.raw_indices[0].unique().numel() == 2048


@pytest.mark.parametrize("tokens", [24_577, 40_961])
def test_fp8_threshold_refresh_matches_paged_dense(tokens: int) -> None:
    torch.manual_seed(tokens)
    rows, heads, head_dim, page_size, topk = 1, 32, 128, 64, 2048
    q = (torch.randn((rows, heads, head_dim), device="cuda") * 0.25).to(dtypes.fp8)
    padded_tokens = ((tokens + page_size - 1) // page_size) * page_size
    keys = (torch.randn((padded_tokens, head_dim), device="cuda") * 0.25).to(dtypes.fp8)
    scales = torch.rand((padded_tokens,), dtype=torch.float32, device="cuda") + 0.25
    weights = torch.rand((rows, heads), dtype=torch.float32, device="cuda")
    pages = padded_tokens // page_size
    block_tables = torch.randperm(pages, device="cuda").to(torch.int32)[None]
    physical_keys = torch.empty_like(keys)
    physical_scales = torch.empty_like(scales)
    for logical_page in range(pages):
        physical_page = int(block_tables[0, logical_page])
        logical_slice = slice(logical_page * page_size, (logical_page + 1) * page_size)
        physical_slice = slice(
            physical_page * page_size, (physical_page + 1) * page_size
        )
        physical_keys[physical_slice] = keys[logical_slice]
        physical_scales[physical_slice] = scales[logical_slice]
    cache = _make_preshuffled_cache(physical_keys, physical_scales, page_size)
    row_to_batch = torch.zeros(rows, dtype=torch.int32, device="cuda")
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), tokens, dtype=torch.int32, device="cuda")

    result = flydsl_pa_mqa_litetopk_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        padded_tokens,
    )
    dense = _flydsl_pa_mqa_logits_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        padded_tokens,
    )
    torch.cuda.synchronize()

    expected = torch.topk(dense[0, :tokens], topk).indices.to(torch.int32)
    assert result.status.cpu().tolist() == [0]
    assert torch.equal(
        torch.sort(result.raw_indices[0]).values.cpu(),
        torch.sort(expected).values.cpu(),
    )


def test_fp8_invalid_live_page_fails_closed() -> None:
    tokens = 12_288
    q, cache, block_tables, weights, row_to_batch = _make_zero_case(1, tokens)
    block_tables[0, 150] = cache.shape[0]
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), tokens, dtype=torch.int32, device="cuda")
    workspace = allocate_fp8_litetopk_workspace(1, "cuda")

    result = flydsl_pa_mqa_litetopk_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        tokens,
        workspace=workspace,
        enforce_status=False,
    )
    torch.cuda.synchronize()

    assert int(result.status[0]) & STATUS_INVALID_INDEX
    assert workspace.status_ok.cpu().tolist() == [0]
    assert result.counts.cpu().tolist() == [0]
    assert torch.all(result.raw_indices == -1)


def test_fp8_invalid_padding_page_is_ignored() -> None:
    q, cache, block_tables, weights, row_to_batch = _make_zero_case(1, 128)
    block_tables = torch.cat(
        (block_tables, torch.tensor([[999]], dtype=torch.int32, device="cuda")),
        dim=1,
    )
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), 128, dtype=torch.int32, device="cuda")

    result = flydsl_pa_mqa_litetopk_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        192,
    )
    torch.cuda.synchronize()

    assert result.status.cpu().tolist() == [0]
    assert result.counts.cpu().tolist() == [128]
    assert torch.equal(
        result.raw_indices[0, :128].cpu(), torch.arange(128, dtype=torch.int32)
    )


@pytest.mark.parametrize("bad_metadata", ["window", "batch"])
def test_fp8_invalid_row_metadata_fails_closed(bad_metadata: str) -> None:
    tokens = 8192
    q, cache, block_tables, weights, row_to_batch = _make_zero_case(1, tokens)
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), tokens, dtype=torch.int32, device="cuda")
    if bad_metadata == "window":
        ends[0] = tokens + 1
    else:
        row_to_batch[0] = block_tables.shape[0]

    result = flydsl_pa_mqa_litetopk_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        tokens,
        enforce_status=False,
    )
    torch.cuda.synchronize()

    assert int(result.status[0]) & STATUS_INVALID_INDEX
    assert result.counts.cpu().tolist() == [0]


def test_fp8_nonfinite_scores_fail_closed() -> None:
    tokens = 8192
    q, cache, block_tables, weights, row_to_batch = _make_zero_case(1, tokens)
    weights[0, 0] = float("inf")
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), tokens, dtype=torch.int32, device="cuda")

    result = flydsl_pa_mqa_litetopk_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        tokens,
        enforce_status=False,
    )
    torch.cuda.synchronize()

    assert int(result.status[0]) & STATUS_NONFINITE
    assert result.counts.cpu().tolist() == [0]


def test_fp8_workspace_reuse_on_non_default_stream() -> None:
    tokens = 8192
    q, cache, block_tables, weights, row_to_batch = _make_zero_case(1, tokens)
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), tokens, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    workspace = allocate_fp8_litetopk_workspace(1, "cuda", stream=stream)

    first = flydsl_pa_mqa_litetopk_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        tokens,
        workspace=workspace,
        stream=stream,
    )
    stream.synchronize()
    first_indices = first.raw_indices.clone()
    second = flydsl_pa_mqa_litetopk_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        tokens,
        workspace=workspace,
        stream=stream,
    )
    stream.synchronize()

    assert second.status.cpu().tolist() == [0]
    assert torch.equal(second.raw_indices, first_indices)


def test_fp8_full_operator_concurrent_streams() -> None:
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    results = [
        _run_zero_case_on_stream(
            device=torch.device("cuda", torch.cuda.current_device()),
            stream=stream,
            tokens=8192,
        )
        for stream in streams
    ]
    for stream in streams:
        stream.synchronize()
    for result in results:
        assert result.status.cpu().tolist() == [0]
        assert result.counts.cpu().tolist() == [2048]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_fp8_full_operator_explicit_non_current_device() -> None:
    current_device = torch.cuda.current_device()
    target_device = 1 if current_device == 0 else 0
    device = torch.device("cuda", target_device)
    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device=device)
    try:
        torch.cuda.set_device(current_device)
        result = _run_zero_case_on_stream(
            device=device,
            stream=stream,
            tokens=8192,
        )
        stream.synchronize()
        assert result.status.cpu().tolist() == [0]
        assert result.counts.cpu().tolist() == [2048]
    finally:
        torch.cuda.set_device(current_device)


def test_fp8_rejects_shifted_workspace_output_alias_and_oversized_context() -> None:
    rows, tokens = 2, 8192
    q, cache, block_tables, weights, row_to_batch = _make_zero_case(rows, tokens)
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), tokens, dtype=torch.int32, device="cuda")
    workspace = allocate_fp8_litetopk_workspace(3, "cuda")
    raw_indices = torch.empty((rows, 2048), dtype=torch.int32, device="cuda")
    out = FP8LiteTopKResult(
        values=workspace.selected_values[1:3],
        raw_indices=raw_indices,
        physical_indices=torch.empty_like(raw_indices),
        counts=workspace.output_counts[:rows],
        candidate_counts=workspace.candidate_counts[:rows],
        status=workspace.status[:rows],
    )
    with pytest.raises(ValueError, match="overlap workspace"):
        flydsl_pa_mqa_litetopk_fp8_prefill(
            q,
            cache,
            block_tables,
            weights,
            row_to_batch,
            starts,
            ends,
            tokens,
            workspace=workspace,
            out=out,
        )

    oversized_pages = DEFAULT_MAX_SEQ_LEN // 64 + 1
    oversized_table = torch.zeros(
        (1, oversized_pages), dtype=torch.int32, device="cuda"
    )
    with pytest.raises(ValueError, match="max_seq_len must be"):
        flydsl_pa_mqa_litetopk_fp8_prefill(
            q,
            cache,
            oversized_table,
            weights,
            row_to_batch,
            starts,
            ends,
            DEFAULT_MAX_SEQ_LEN + 64,
        )


def test_fp8_physical_page_base_above_4gib() -> None:
    page_size, head_dim, tokens = 64, 128, 128
    page_bytes = page_size * (head_dim + 4)
    high_page = (1 << 32) // page_bytes + 1
    required_bytes = (high_page + 2) * page_bytes
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < required_bytes + (1 << 30):
        pytest.skip("requires at least 5 GiB of free GPU memory")

    q, _, _, weights, row_to_batch = _make_zero_case(1, tokens)
    q[0, 0, 0] = 1.0
    weights.zero_()
    weights[0, 0] = 1.0
    cache = torch.empty(
        (high_page + 2, page_size, 1, head_dim + 4),
        dtype=torch.uint8,
        device="cuda",
    )
    cache[:4].zero_()
    small_keys = torch.zeros((tokens, head_dim), dtype=dtypes.fp8, device="cuda")
    small_keys[:, 0] = 1.0
    small_scales = torch.ones((tokens,), dtype=torch.float32, device="cuda")
    cache[high_page : high_page + 2].copy_(
        _make_preshuffled_cache(small_keys, small_scales, page_size)
    )
    block_tables = torch.tensor(
        [[high_page, high_page + 1]], dtype=torch.int32, device="cuda"
    )
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), tokens, dtype=torch.int32, device="cuda")

    result = flydsl_pa_mqa_litetopk_fp8_prefill(
        q,
        cache,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        tokens,
    )
    torch.cuda.synchronize()

    assert result.status.cpu().tolist() == [0]
    assert result.counts.cpu().tolist() == [tokens]
    torch.testing.assert_close(
        result.values[0, :tokens],
        torch.ones(tokens, dtype=torch.float32, device="cuda"),
        rtol=0,
        atol=0,
    )
    expected_physical = (
        block_tables[0, result.raw_indices[0, :tokens].long() // page_size] * page_size
        + result.raw_indices[0, :tokens] % page_size
    )
    assert torch.equal(result.physical_indices[0, :tokens], expected_physical)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
