# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() != "gfx950", reason="FP4 LiteTopK requires gfx950"
)

from aiter.ops.flydsl import (
    FP4_LITETOPK_SUPPORTED_TOPKS,
    allocate_fp4_litetopk_workspace,
    flydsl_pa_mqa_litetopk_fp4_prefill,
    flydsl_pa_mqa_logits_fp4_prefill,
    fp4_litetopk_workspace_nbytes,
    fp4_litetopk_workspace_size,
    prepare_fp4_litetopk_seed,
)
from aiter.ops.flydsl.pa_mqa_litetopk_fp4 import STATUS_CANDIDATE_OVERFLOW
from aiter.ops.topk import top_k_per_row_prefill
from aiter.ops.triton.attention.pa_mqa_litetopk_seed import (
    refresh_pa_mqa_litetopk_threshold,
)
from op_tests.test_flydsl_pa_mqa_logits_fp4_prefill import (
    indexer_k_fp4_paged_preshuffle,
    quant_q_fp4_preshuffle,
)


def _run_seed(sample_logits, row_starts, row_ends, *, topk, merge_cap):
    rows, sample_len = sample_logits.shape
    workspace = allocate_fp4_litetopk_workspace(
        rows,
        sample_logits.device,
        topk=topk,
        sample_len=sample_len,
        merge_cap=merge_cap,
    )
    prepare_fp4_litetopk_seed(
        sample_logits,
        row_starts,
        row_ends,
        workspace,
        max_seq_len=int(row_ends.max().item()),
    )
    torch.cuda.synchronize()
    return workspace


def test_fp4_supported_topk_contract():
    assert FP4_LITETOPK_SUPPORTED_TOPKS == (512, 1024)


def test_seed_is_conservative_and_ordered():
    torch.manual_seed(7)
    rows, sample_len, topk = 3, 1024, 128
    scores = torch.randn(rows, sample_len, dtype=torch.float32, device="cuda")
    starts = torch.tensor([0, 17, 31], dtype=torch.int32, device="cuda")
    ends = starts + torch.tensor([1024, 900, 700], dtype=torch.int32, device="cuda")
    workspace = _run_seed(scores, starts, ends, topk=topk, merge_cap=sample_len)

    for row in range(rows):
        count = int(workspace.candidate_counts[row].item())
        indices = workspace.candidate_indices[row, :count].cpu()
        assert count >= topk
        assert torch.equal(indices, indices.sort().values)
        selected = set(indices.tolist())
        reference = torch.topk(
            scores[row, : int(ends[row] - starts[row])], topk
        ).indices
        reference = set((reference.cpu() + starts[row].cpu()).tolist())
        assert reference.issubset(selected)
    assert torch.count_nonzero(workspace.status).item() == 0


def test_workspace_size_query_matches_allocation():
    workspace = allocate_fp4_litetopk_workspace(
        3, "cuda:0", topk=128, sample_len=1024, merge_cap=2048
    )
    assert fp4_litetopk_workspace_size(
        3, topk=128, sample_len=1024, merge_cap=2048
    ) == fp4_litetopk_workspace_nbytes(workspace)


def test_seed_flat_scores_and_overflow_are_explicit():
    sample_len, topk, merge_cap = 2048, 512, 1024
    scores = torch.ones(1, sample_len, dtype=torch.float32, device="cuda")
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), sample_len, dtype=torch.int32, device="cuda")
    workspace = _run_seed(scores, starts, ends, topk=topk, merge_cap=merge_cap)

    assert int(workspace.candidate_counts[0].item()) == sample_len
    assert int(workspace.select_ends[0].item()) == merge_cap
    assert int(workspace.status[0].item()) & STATUS_CANDIDATE_OVERFLOW
    assert torch.equal(
        workspace.candidate_indices[0].cpu(), torch.arange(merge_cap, dtype=torch.int32)
    )


def test_seed_helper_resets_stale_status():
    scores = torch.randn(1, 1024, dtype=torch.float32, device="cuda")
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), 1024, dtype=torch.int32, device="cuda")
    workspace = allocate_fp4_litetopk_workspace(
        1, "cuda:0", topk=128, sample_len=1024, merge_cap=2048
    )
    workspace.status.fill_(0x7FFFFFFF)

    prepare_fp4_litetopk_seed(
        scores,
        starts,
        ends,
        workspace,
        max_seq_len=1024,
    )
    torch.cuda.synchronize()

    assert workspace.status.cpu().tolist() == [0]


def test_seed_helper_rejects_input_alias_and_clears_no_emit_metadata():
    scores = torch.randn(1, 1024, dtype=torch.float32, device="cuda")
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), 1024, dtype=torch.int32, device="cuda")
    workspace = allocate_fp4_litetopk_workspace(
        1, "cuda:0", topk=128, sample_len=1024, merge_cap=2048
    )
    workspace.candidate_counts.fill_(99)
    workspace.select_starts.fill_(7)
    workspace.select_ends.fill_(88)
    prepare_fp4_litetopk_seed(
        scores,
        starts,
        ends,
        workspace,
        max_seq_len=1024,
        emit_candidates=False,
    )
    torch.cuda.synchronize()
    assert workspace.candidate_counts.cpu().tolist() == [0]
    assert workspace.select_starts.cpu().tolist() == [0]
    assert workspace.select_ends.cpu().tolist() == [0]

    with pytest.raises(ValueError, match="seed inputs must not overlap workspace"):
        prepare_fp4_litetopk_seed(
            scores,
            starts,
            workspace.status,
            workspace,
            max_seq_len=1024,
        )


def test_threshold_refresh_only_tightens():
    histogram = torch.zeros(2, 256, dtype=torch.int32, device="cuda")
    histogram[0, :4] = torch.tensor([100, 100, 200, 200], device="cuda")
    histogram[1, :8] = 100
    threshold = torch.tensor([20, 3], dtype=torch.int32, device="cuda")
    refresh_pa_mqa_litetopk_threshold(
        histogram,
        threshold,
        topk=512,
        num_buckets=256,
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    assert threshold.cpu().tolist() == [3, 3]


def test_fused_litetopk_matches_dense_set_and_maps_pages():
    _check_fused_litetopk(rows=1, seq_len=1024, starts_list=[0], seed=29)


def test_fused_litetopk_k1024_matches_dense_set_and_maps_pages():
    _check_fused_litetopk(
        rows=1,
        seq_len=16_384,
        starts_list=[37],
        ends_list=[16_379],
        seed=30,
        topk=1024,
    )


def test_full_operator_rejects_fp8_topk():
    with pytest.raises(ValueError, match=r"topk in \(512, 1024\)"):
        flydsl_pa_mqa_litetopk_fp4_prefill(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            topk=2048,
        )


def test_fused_seed_calibration_matches_triton_oracle():
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        flydsl_pa_mqa_litetopk_fp4_seed,
    )
    from aiter.ops.flydsl.pa_mqa_litetopk_fp4 import (
        STATUS_CANDIDATE_OVERFLOW,
        STATUS_NONFINITE,
        STATUS_UNDERFILLED,
    )
    from aiter.ops.triton.attention.pa_mqa_litetopk_seed import (
        build_pa_mqa_litetopk_scan_schedule,
    )

    rows, seq_len, heads, head_dim = 4, 8192, 64, 128
    torch.manual_seed(31)
    q = torch.randn(rows, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    q[1].zero_()
    k = torch.randn(seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    weights = (torch.randn(rows, heads, device="cuda") * 0.1).to(torch.bfloat16)
    weights[3].fill_(float("inf"))
    q_fp4, q_scale = quant_q_fp4_preshuffle(q)

    num_pages = seq_len // 64
    block_tables = torch.randperm(num_pages, device="cuda").to(torch.int32)[None]
    kv_cache = torch.zeros(num_pages, 1, 4, 64, 16, dtype=torch.uint8, device="cuda")
    kv_scale = torch.zeros(num_pages, 1, 4, 64, dtype=torch.uint8, device="cuda")
    logical = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    physical = block_tables[0, logical.long() // 64] * 64 + logical % 64
    indexer_k_fp4_paged_preshuffle(k, physical, kv_cache, kv_scale, 64)

    row_to_batch = torch.zeros(rows, dtype=torch.int32, device="cuda")
    starts = torch.tensor([17, 0, 400, 63], dtype=torch.int32, device="cuda")
    ends = torch.tensor([8192, 8192, 400, 128], dtype=torch.int32, device="cuda")
    sample_ends = torch.minimum(starts + 8192, ends)
    cta_info = torch.empty((rows, 6), dtype=torch.int32, device="cuda")
    build_pa_mqa_litetopk_scan_schedule(
        row_to_batch,
        starts,
        sample_ends,
        cta_info,
        segment_start=0,
        segment_end=seq_len,
        block_k=512,
        stream=torch.cuda.current_stream(),
    )

    fused_workspace = allocate_fp4_litetopk_workspace(rows, "cuda", merge_cap=1024)
    oracle_workspace = allocate_fp4_litetopk_workspace(rows, "cuda", merge_cap=1024)
    oracle_workspace.sample_logits.fill_(float("-inf"))
    fused_workspace.status.zero_()

    flydsl_pa_mqa_litetopk_fp4_seed(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        cta_info,
        fused_workspace.origin,
        fused_workspace.inv_delta,
        fused_workspace.threshold,
        fused_workspace.histogram,
        fused_workspace.candidate_values,
        fused_workspace.candidate_indices,
        fused_workspace.candidate_counts,
        fused_workspace.select_starts,
        fused_workspace.select_ends,
        fused_workspace.status,
        n_ctas=rows,
        merge_cap=fused_workspace.merge_cap,
        topk=fused_workspace.topk,
        status_candidate_overflow=STATUS_CANDIDATE_OVERFLOW,
        status_underfilled=STATUS_UNDERFILLED,
        status_nonfinite=STATUS_NONFINITE,
        block_k=512,
        num_warps=8,
    )
    flydsl_pa_mqa_logits_fp4_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        starts,
        sample_ends,
        seq_len,
        out=oracle_workspace.sample_logits,
        cta_info=cta_info,
        n_ctas=rows,
        parallel_unit_num=rows,
        relative_output=True,
        block_k=512,
        num_warps=8,
    )
    prepare_fp4_litetopk_seed(
        oracle_workspace.sample_logits,
        starts,
        ends,
        oracle_workspace,
        max_seq_len=seq_len,
    )
    torch.cuda.synchronize()

    assert torch.equal(fused_workspace.origin, oracle_workspace.origin)
    assert torch.equal(fused_workspace.inv_delta, oracle_workspace.inv_delta)
    assert torch.equal(fused_workspace.histogram, oracle_workspace.histogram)
    assert torch.equal(fused_workspace.threshold, oracle_workspace.threshold)
    assert torch.equal(
        fused_workspace.candidate_counts, oracle_workspace.candidate_counts
    )
    assert torch.equal(fused_workspace.select_starts, oracle_workspace.select_starts)
    assert torch.equal(fused_workspace.select_ends, oracle_workspace.select_ends)
    for row in range(rows):
        count = min(
            int(fused_workspace.candidate_counts[row]), fused_workspace.merge_cap
        )
        assert torch.equal(
            fused_workspace.candidate_values[row, :count].view(torch.int32),
            oracle_workspace.candidate_values[row, :count].view(torch.int32),
        )
        assert torch.equal(
            fused_workspace.candidate_indices[row, :count],
            oracle_workspace.candidate_indices[row, :count],
        )
    assert torch.equal(fused_workspace.status, oracle_workspace.status)
    assert int(fused_workspace.status[3]) & STATUS_NONFINITE


def test_fused_litetopk_long_nonzero_windows():
    _check_fused_litetopk(
        rows=2,
        seq_len=32768,
        starts_list=[37, 1000],
        ends_list=[32761, 30000],
        seed=37,
    )


def test_fused_litetopk_production_min_context():
    _check_fused_litetopk(
        rows=1,
        seq_len=65_536,
        starts_list=[113],
        ends_list=[65_531],
        seed=41,
    )


@pytest.mark.parametrize("topk", [512, 1024])
def test_fused_litetopk_production_max_context(topk: int):
    _check_fused_litetopk(
        rows=1,
        seq_len=196_608,
        starts_list=[211],
        ends_list=[196_601],
        seed=53,
        topk=topk,
    )


def test_full_operator_explicit_non_default_stream():
    stream = torch.cuda.Stream()
    _check_fused_litetopk(
        rows=1,
        seq_len=16_384,
        starts_list=[29],
        ends_list=[16_379],
        seed=59,
        stream=stream,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_full_operator_explicit_non_current_device():
    current_device = torch.cuda.current_device()
    target_device = 1 if current_device == 0 else 0
    with torch.cuda.device(target_device):
        stream = torch.cuda.Stream(device=target_device)
    try:
        torch.cuda.set_device(current_device)
        _check_fused_litetopk(
            rows=1,
            seq_len=1024,
            starts_list=[17],
            ends_list=[1019],
            seed=61,
            device=torch.device("cuda", target_device),
            stream=stream,
        )
    finally:
        torch.cuda.set_device(current_device)


def test_full_operator_concurrent_streams():
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    pending = []
    for seed, stream in zip((67, 71), streams):
        pending.append(
            _check_fused_litetopk(
                rows=1,
                seq_len=1024,
                starts_list=[seed % 31],
                ends_list=[1024],
                seed=seed,
                stream=stream,
                synchronize=False,
            )
        )
    for stream in streams:
        stream.synchronize()
    for validate in pending:
        validate()


def test_nonfinite_suffix_fails_closed():
    from aiter.ops.flydsl import pa_mqa_litetopk_fp4 as impl
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        flydsl_pa_mqa_litetopk_fp4_prefill_scan,
    )
    from aiter.ops.triton.attention.pa_mqa_litetopk_finalize import (
        prepare_pa_mqa_litetopk_select,
    )
    from aiter.ops.triton.attention.pa_mqa_litetopk_seed import (
        build_pa_mqa_litetopk_scan_schedule,
    )

    rows, seq_len, heads, head_dim = 1, 16_384, 64, 128
    q = torch.ones(rows, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.ones(seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.ones(rows, heads, dtype=torch.bfloat16, device="cuda")
    q_fp4, q_scale = quant_q_fp4_preshuffle(q)
    num_pages = seq_len // 64
    block_tables = torch.arange(num_pages, dtype=torch.int32, device="cuda")[None]
    kv_cache = torch.zeros(num_pages, 1, 4, 64, 16, dtype=torch.uint8, device="cuda")
    kv_scale = torch.zeros(num_pages, 1, 4, 64, dtype=torch.uint8, device="cuda")
    logical = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    indexer_k_fp4_paged_preshuffle(k, logical, kv_cache, kv_scale, 64)
    row_to_batch = torch.zeros(rows, dtype=torch.int32, device="cuda")
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), seq_len, dtype=torch.int32, device="cuda")
    workspace = allocate_fp4_litetopk_workspace(rows, "cuda:0")
    sample_logits = torch.ones(
        rows, workspace.sample_len, dtype=torch.float32, device="cuda"
    )
    prepare_fp4_litetopk_seed(
        sample_logits,
        starts,
        ends,
        workspace,
        max_seq_len=seq_len,
    )
    sample_ends = torch.full(
        (rows,), workspace.sample_len, dtype=torch.int32, device="cuda"
    )
    build_pa_mqa_litetopk_scan_schedule(
        row_to_batch,
        sample_ends,
        ends,
        workspace.scan_cta_info,
        segment_start=0,
        segment_end=seq_len,
        block_k=256,
        stream=torch.cuda.current_stream(),
    )
    flydsl_pa_mqa_litetopk_fp4_prefill_scan(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        workspace.scan_cta_info,
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
        merge_cap=workspace.merge_cap,
        weight_scale=float("inf"),
    )
    prepare_pa_mqa_litetopk_select(
        workspace.candidate_counts,
        workspace.page_errors,
        workspace.score_errors,
        workspace.histogram,
        starts,
        ends,
        workspace.threshold,
        workspace.select_ends,
        workspace.status,
        topk=workspace.topk,
        merge_cap=workspace.merge_cap,
        num_buckets=workspace.num_buckets,
        status_candidate_overflow=impl.STATUS_CANDIDATE_OVERFLOW,
        status_underfilled=impl.STATUS_UNDERFILLED,
        status_nonfinite=impl.STATUS_NONFINITE,
        status_invalid_index=impl.STATUS_INVALID_INDEX,
        status_bad_certificate=impl.STATUS_BAD_CERTIFICATE,
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    assert int(workspace.score_errors[0]) > 0
    assert int(workspace.status[0]) & impl.STATUS_NONFINITE
    assert int(workspace.select_ends[0]) == 0


@pytest.mark.parametrize("topk", [512, 1024])
def test_fused_litetopk_empty_and_short_rows(topk: int):
    _check_fused_litetopk(
        rows=2,
        seq_len=1024,
        starts_list=[0, 63],
        ends_list=[0, 128],
        seed=43,
        topk=topk,
    )


def test_suffix_affine_is_clamped_before_int32_conversion():
    rows, heads, head_dim, topk = 1, 64, 128, 512
    sample_len = 8192
    low_count = 8192
    seq_len = sample_len + low_count + topk
    q = torch.zeros(rows, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    q[:, : heads // 2, 0] = 1
    q[:, heads // 2 :, 0] = -1
    weights = torch.ones(rows, heads, dtype=torch.bfloat16, device="cuda")
    weights[:, : heads // 2] = -1
    k = torch.zeros(seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    k[sample_len : sample_len + low_count, 0] = 1
    k[-topk:, 0] = -1
    q_fp4, q_scale = quant_q_fp4_preshuffle(q)

    num_pages = (seq_len + 63) // 64
    block_tables = torch.arange(num_pages, dtype=torch.int32, device="cuda").unsqueeze(
        0
    )
    kv_cache = torch.zeros(num_pages, 1, 4, 64, 16, dtype=torch.uint8, device="cuda")
    kv_scale = torch.zeros(num_pages, 1, 4, 64, dtype=torch.uint8, device="cuda")
    logical = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    indexer_k_fp4_paged_preshuffle(k, logical, kv_cache, kv_scale, 64)
    row_to_batch = torch.zeros(rows, dtype=torch.int32, device="cuda")
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), seq_len, dtype=torch.int32, device="cuda")

    result = flydsl_pa_mqa_litetopk_fp4_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        seq_len,
    )
    torch.cuda.synchronize()

    assert result.status.cpu().tolist() == [0]
    assert result.candidate_counts.cpu().tolist() == [sample_len + topk]
    expected = torch.arange(seq_len - topk, seq_len, dtype=torch.int32)
    assert torch.equal(torch.sort(result.raw_indices[0]).values.cpu(), expected)


@pytest.mark.parametrize("topk", [512, 1024])
def test_full_operator_all_ties_returns_valid_unique_set(topk: int):
    rows, seq_len, heads, head_dim = 1, 16_384, 64, 128
    q = torch.zeros(rows, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.ones(rows, heads, dtype=torch.bfloat16, device="cuda")
    q_fp4, q_scale = quant_q_fp4_preshuffle(q)
    num_pages = seq_len // 64
    block_tables = torch.randperm(num_pages, device="cuda").to(torch.int32)[None]
    kv_cache = torch.zeros(num_pages, 1, 4, 64, 16, dtype=torch.uint8, device="cuda")
    kv_scale = torch.zeros(num_pages, 1, 4, 64, dtype=torch.uint8, device="cuda")
    logical = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    physical = block_tables[0, logical.long() // 64] * 64 + logical % 64
    indexer_k_fp4_paged_preshuffle(k, physical, kv_cache, kv_scale, 64)
    row_to_batch = torch.zeros(rows, dtype=torch.int32, device="cuda")
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), seq_len, dtype=torch.int32, device="cuda")
    workspace = allocate_fp4_litetopk_workspace(rows, "cuda", topk=topk)

    result = flydsl_pa_mqa_litetopk_fp4_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        seq_len,
        topk=topk,
        workspace=workspace,
    )
    torch.cuda.synchronize()

    selected = result.raw_indices[0]
    assert result.status.cpu().tolist() == [0]
    assert result.counts.cpu().tolist() == [topk]
    assert result.candidate_counts.cpu().tolist() == [seq_len]
    assert selected.unique().numel() == topk
    assert ((selected >= 0) & (selected < seq_len)).all()
    expected_physical = block_tables[0, selected.long() // 64] * 64 + selected % 64
    assert torch.equal(result.physical_indices[0], expected_physical)


def _check_fused_litetopk(
    *,
    rows: int,
    seq_len: int,
    starts_list: list[int],
    seed: int,
    ends_list: list[int] | None = None,
    device: torch.device | str = "cuda",
    stream: torch.cuda.Stream | None = None,
    synchronize: bool = True,
    topk: int = 512,
):
    heads, head_dim = 64, 128
    device = torch.device(device)
    torch.manual_seed(seed)
    q = torch.randn(rows, heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(seq_len, head_dim, dtype=torch.bfloat16, device=device)
    weights = (torch.randn(rows, heads, dtype=torch.float32, device=device) * 0.1).to(
        torch.bfloat16
    )
    q_fp4, q_scale = quant_q_fp4_preshuffle(q)

    num_pages = (seq_len + 63) // 64
    block_tables = torch.randperm(num_pages, device=device).to(torch.int32)[None]
    kv_cache = torch.zeros(num_pages, 1, 4, 64, 16, dtype=torch.uint8, device=device)
    kv_scale = torch.zeros(num_pages, 1, 4, 64, dtype=torch.uint8, device=device)
    logical = torch.arange(seq_len, dtype=torch.int32, device=device)
    physical = block_tables[0, logical.long() // 64] * 64 + logical % 64
    indexer_k_fp4_paged_preshuffle(k, physical, kv_cache, kv_scale, 64)

    row_to_batch = torch.zeros(rows, dtype=torch.int32, device=device)
    starts = torch.tensor(starts_list, dtype=torch.int32, device=device)
    ends = torch.tensor(
        ends_list if ends_list is not None else [seq_len] * rows,
        dtype=torch.int32,
        device=device,
    )
    dense = flydsl_pa_mqa_logits_fp4_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        seq_len,
        parallel_unit_num=rows,
    )
    reference = torch.empty((rows, topk), dtype=torch.int32, device=device)
    top_k_per_row_prefill(
        dense,
        starts,
        ends,
        reference,
        None,
        rows,
        dense.stride(0),
        dense.stride(1),
        topk,
        stable=True,
    )
    workspace = allocate_fp4_litetopk_workspace(rows, device, topk=topk, stream=stream)

    result = flydsl_pa_mqa_litetopk_fp4_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        seq_len,
        topk=topk,
        workspace=workspace,
        stream=stream,
    )
    if stream is None:
        torch.cuda.synchronize()
    elif synchronize:
        stream.synchronize()

    def validate():
        assert result.status.cpu().tolist() == [0] * rows
        assert workspace.status_ok.cpu().tolist() == [1]
        expected_counts = [
            min(topk, end - start) for start, end in zip(starts_list, ends)
        ]
        assert result.counts.cpu().tolist() == expected_counts
        for row in range(rows):
            count = expected_counts[row]
            assert torch.equal(
                torch.sort(result.raw_indices[row, :count]).values,
                torch.sort(reference[row, :count]).values,
            )
            assert result.raw_indices[row, count:].eq(-1).all()
            assert result.physical_indices[row, count:].eq(-1).all()
            assert torch.isneginf(result.values[row, count:]).all()
            if count:
                raw = result.raw_indices[row, :count]
                expected_physical = block_tables[0, raw.long() // 64] * 64 + raw % 64
                assert torch.equal(
                    result.physical_indices[row, :count], expected_physical
                )
                assert torch.equal(result.values[row, :count], dense[row, raw.long()])

    if synchronize:
        validate()
    return validate


@pytest.mark.parametrize(
    "seq_len,bad_page,negative", [(1024, 3, True), (16_384, 192, False)]
)
def test_invalid_live_page_fails_closed(seq_len, bad_page, negative):
    from aiter.ops.flydsl.pa_mqa_litetopk_fp4 import STATUS_INVALID_INDEX

    rows, heads, head_dim = 1, 64, 128
    torch.manual_seed(47)
    q = torch.randn(rows, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.ones(rows, heads, dtype=torch.bfloat16, device="cuda")
    q_fp4, q_scale = quant_q_fp4_preshuffle(q)
    num_pages = (seq_len + 63) // 64
    block_tables = torch.arange(num_pages, dtype=torch.int32, device="cuda")[None]
    kv_cache = torch.zeros(num_pages, 1, 4, 64, 16, dtype=torch.uint8, device="cuda")
    kv_scale = torch.zeros(num_pages, 1, 4, 64, dtype=torch.uint8, device="cuda")
    logical = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    indexer_k_fp4_paged_preshuffle(k, logical, kv_cache, kv_scale, 64)
    block_tables[0, bad_page] = -1 if negative else num_pages + 7
    row_to_batch = torch.zeros(rows, dtype=torch.int32, device="cuda")
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), seq_len, dtype=torch.int32, device="cuda")
    workspace = allocate_fp4_litetopk_workspace(rows, "cuda")

    result = flydsl_pa_mqa_litetopk_fp4_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        seq_len,
        workspace=workspace,
        enforce_status=False,
    )
    torch.cuda.synchronize()

    assert int(workspace.page_errors[0]) > 0
    assert workspace.status_ok.cpu().tolist() == [0]
    assert int(workspace.sample_ends[0]) == min(seq_len, workspace.sample_len)
    assert int(result.status[0]) & STATUS_INVALID_INDEX
    assert int(result.counts[0]) == 0
    assert result.raw_indices.eq(-1).all()
    assert result.physical_indices.eq(-1).all()

    block_tables[0, bad_page] = bad_page
    result = flydsl_pa_mqa_litetopk_fp4_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        seq_len,
        workspace=workspace,
    )
    torch.cuda.synchronize()
    assert result.status.cpu().tolist() == [0]
    assert result.counts.cpu().tolist() == [512]


def test_invalid_row_metadata_fails_closed():
    from aiter.ops.flydsl.pa_mqa_litetopk_fp4 import STATUS_INVALID_INDEX

    rows, seq_len, heads, head_dim = 4, 1024, 64, 128
    torch.manual_seed(59)
    q = torch.randn(rows, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.ones(rows, heads, dtype=torch.bfloat16, device="cuda")
    q_fp4, q_scale = quant_q_fp4_preshuffle(q)
    num_pages = seq_len // 64
    block_tables = torch.arange(num_pages, dtype=torch.int32, device="cuda")[None]
    kv_cache = torch.zeros(num_pages, 1, 4, 64, 16, dtype=torch.uint8, device="cuda")
    kv_scale = torch.zeros(num_pages, 1, 4, 64, dtype=torch.uint8, device="cuda")
    logical = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    indexer_k_fp4_paged_preshuffle(k, logical, kv_cache, kv_scale, 64)
    row_to_batch = torch.tensor([0, 1, 0, 0], dtype=torch.int32, device="cuda")
    starts = torch.tensor([-1, 0, 900, 0], dtype=torch.int32, device="cuda")
    ends = torch.tensor([1024, 1024, 800, 1025], dtype=torch.int32, device="cuda")

    result = flydsl_pa_mqa_litetopk_fp4_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        starts,
        ends,
        seq_len,
        enforce_status=False,
    )
    torch.cuda.synchronize()

    assert torch.all((result.status & STATUS_INVALID_INDEX) != 0)
    assert result.counts.eq(0).all()
    assert result.raw_indices.eq(-1).all()
    assert result.physical_indices.eq(-1).all()


def test_invalid_padding_page_outside_window_is_ignored():
    rows, seq_len, heads, head_dim = 1, 1024, 64, 128
    torch.manual_seed(61)
    q = torch.randn(rows, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.ones(rows, heads, dtype=torch.bfloat16, device="cuda")
    q_fp4, q_scale = quant_q_fp4_preshuffle(q)
    num_pages = seq_len // 64
    block_tables = torch.arange(num_pages, dtype=torch.int32, device="cuda")[None]
    block_tables[0, 15] = -1
    kv_cache = torch.zeros(num_pages, 1, 4, 64, 16, dtype=torch.uint8, device="cuda")
    kv_scale = torch.zeros(num_pages, 1, 4, 64, dtype=torch.uint8, device="cuda")
    logical = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    indexer_k_fp4_paged_preshuffle(k, logical, kv_cache, kv_scale, 64)

    result = flydsl_pa_mqa_litetopk_fp4_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        torch.zeros(rows, dtype=torch.int32, device="cuda"),
        torch.zeros(rows, dtype=torch.int32, device="cuda"),
        torch.full((rows,), 512, dtype=torch.int32, device="cuda"),
        seq_len,
    )
    torch.cuda.synchronize()

    assert result.status.cpu().tolist() == [0]
    assert result.counts.cpu().tolist() == [512]


def test_scan_schedule_rejects_aliased_dual_outputs():
    from aiter.ops.triton.attention.pa_mqa_litetopk_seed import (
        build_pa_mqa_litetopk_scan_schedule,
    )

    rows = 1
    row_to_batch = torch.zeros(rows, dtype=torch.int32, device="cuda")
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), 1024, dtype=torch.int32, device="cuda")
    cta_info = torch.empty((rows, 6), dtype=torch.int32, device="cuda")
    sample_ends = torch.empty(rows, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="must not overlap"):
        build_pa_mqa_litetopk_scan_schedule(
            row_to_batch,
            starts,
            ends,
            cta_info,
            segment_start=0,
            segment_end=1024,
            block_k=512,
            stream=torch.cuda.current_stream(),
            suffix_cta_info=cta_info,
            sample_len=512,
            sample_ends=sample_ends,
        )


def test_workspace_rejects_capture_before_allocation():
    with (
        patch("torch.cuda.is_current_stream_capturing", return_value=True),
        patch("torch.empty") as empty,
        pytest.raises(RuntimeError, match="not capture-safe"),
    ):
        allocate_fp4_litetopk_workspace(1, "cuda:0")
    empty.assert_not_called()


def test_explicit_non_default_stream():
    stream = torch.cuda.Stream()
    scores = torch.randn(1, 1024, dtype=torch.float32, device="cuda")
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    ends = torch.full((1,), 1024, dtype=torch.int32, device="cuda")
    workspace = allocate_fp4_litetopk_workspace(
        1,
        "cuda:0",
        topk=128,
        sample_len=1024,
        merge_cap=2048,
        stream=stream,
    )
    prepare_fp4_litetopk_seed(
        scores,
        starts,
        ends,
        workspace,
        max_seq_len=1024,
        stream=stream,
    )
    stream.synchronize()
    assert workspace.status.cpu().tolist() == [0]
    assert int(workspace.candidate_counts[0]) >= 128


def test_rejects_incompatible_caller_output():
    rows = 1
    workspace = allocate_fp4_litetopk_workspace(rows, "cuda:0")
    bad_result = type("Output", (), {})()
    bad_result.values = torch.empty((rows, 511), dtype=torch.float32, device="cuda")
    bad_result.raw_indices = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
    bad_result.physical_indices = torch.empty_like(bad_result.raw_indices)
    bad_result.counts = workspace.output_counts
    bad_result.candidate_counts = workspace.candidate_counts
    bad_result.status = workspace.status
    from aiter.ops.flydsl.pa_mqa_litetopk_fp4 import _validate_result

    with pytest.raises(ValueError, match="output tensors"):
        _validate_result(
            bad_result,
            workspace,
            rows=rows,
            topk=512,
            device=torch.device("cuda:0"),
        )


def test_rejects_aliased_workspace_and_output():
    from aiter.ops.flydsl.pa_mqa_litetopk_fp4 import (
        FP4LiteTopKResult,
        _validate_result,
        _validate_workspace,
    )

    workspace = allocate_fp4_litetopk_workspace(
        1, "cuda:0", topk=128, sample_len=1024, merge_cap=2048
    )
    stream = torch.cuda.current_stream()
    aliased_workspace = workspace._replace(status=workspace.output_counts)
    with pytest.raises(ValueError, match="workspace tensors must not overlap"):
        _validate_workspace(
            aliased_workspace,
            rows=1,
            device=torch.device("cuda:0"),
            stream=stream,
            topk=128,
            sample_len=1024,
            num_buckets=256,
            merge_cap=2048,
        )

    raw = torch.empty((1, 128), dtype=torch.int32, device="cuda")
    out = FP4LiteTopKResult(
        values=workspace.sample_logits[:, :128],
        raw_indices=raw,
        physical_indices=torch.empty_like(raw),
        counts=workspace.output_counts,
        candidate_counts=workspace.candidate_counts,
        status=workspace.status,
    )
    with pytest.raises(ValueError, match="outputs must not overlap workspace"):
        _validate_result(
            out,
            workspace,
            rows=1,
            topk=128,
            device=torch.device("cuda:0"),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
