# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import pytest
import torch

from aiter.ops.triton.attention.qsa import (
    qsa_expand_block_indices,
    qsa_paged_mqa_logits,
    qsa_select_paged_tokens,
    qsa_sparse_paged_gqa,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="QSA kernels require a CUDA/HIP device"
)


def _mqa_reference(q, cache, page_table, token_to_request, visible_groups):
    pages = page_table.index_select(0, token_to_request.long()).long()
    keys = cache[pages, :, 0, :].flatten(1, 2)
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys.float())
    logits = torch.relu(scores).sum(dim=-1) / math.sqrt(q.shape[-1])
    positions = torch.arange(keys.shape[1], device=q.device).unsqueeze(0)
    return logits.masked_fill(positions >= visible_groups.unsqueeze(1), -torch.inf)


def _expand_reference(
    block_indices,
    query_positions,
    row_context_lens,
    compress_ratio,
    token_topk,
):
    rows = block_indices.shape[0]
    output_width = token_topk + compress_ratio - 1
    offsets = torch.arange(compress_ratio, device=block_indices.device)
    expanded = block_indices.long().unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        block_indices.unsqueeze(-1) >= 0,
        expanded,
        torch.full_like(expanded, -1),
    ).reshape(rows, token_topk)
    expanded = torch.where(
        (expanded >= 0) & (expanded < row_context_lens.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=block_indices.device)
    visible_tokens = query_positions + 1
    tail_start = visible_tokens // compress_ratio * compress_ratio
    tail = tail_start.unsqueeze(1) + tail_offsets
    tail_valid = (
        tail_offsets.unsqueeze(0) < (visible_tokens - tail_start).unsqueeze(1)
    ) & (tail < row_context_lens.unsqueeze(1))
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat((expanded, tail), dim=1)
    order = torch.arange(output_width, device=result.device).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + output_width)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


def _attention_reference(
    q,
    k_cache,
    v_cache,
    logical_indices,
    block_table,
    token_to_request,
    scale,
):
    output = torch.zeros_like(q)
    repeats = q.shape[1] // k_cache.shape[2]
    page_size = k_cache.shape[1]
    for row in range(q.shape[0]):
        logical = logical_indices[row]
        logical = logical[logical >= 0].long()
        if logical.numel() == 0:
            continue
        request = token_to_request[row].long()
        pages = block_table[request, logical // page_size].long()
        offsets = logical % page_size
        keys = k_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        values = v_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float())
        probabilities = torch.softmax(scores * scale, dim=-1)
        output[row] = torch.einsum(
            "hk,khd->hd", probabilities, values.float()
        ).to(q.dtype)
    return output


def test_qsa_paged_mqa_logits_matches_reference():
    torch.manual_seed(1)
    q = torch.randn(3, 4, 16, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(5, 4, 1, 16, device="cuda", dtype=torch.bfloat16)
    page_table = torch.tensor([[3, 1], [4, 2]], device="cuda", dtype=torch.int32)
    token_to_request = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([4, 1, 6], device="cuda", dtype=torch.int32)
    context_lens = torch.tensor([8, 2], device="cuda", dtype=torch.int32)
    expected_visible = torch.tensor([5, 2, 7], device="cuda", dtype=torch.int32)

    actual, visible = qsa_paged_mqa_logits(
        q,
        cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        compress_ratio=1,
    )
    expected = _mqa_reference(
        q, cache, page_table, token_to_request, expected_visible
    )

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(visible, expected_visible)


def test_qsa_expand_block_indices_matches_reference():
    blocks = torch.tensor([[0, -1], [1, 0]], device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([5, 10], device="cuda", dtype=torch.int32)
    context_lens = torch.tensor([6, 11], device="cuda", dtype=torch.int32)
    token_to_request = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

    actual = qsa_expand_block_indices(
        blocks,
        query_positions,
        context_lens,
        token_to_request,
        compress_ratio=4,
        token_topk=8,
    )
    expected = _expand_reference(blocks, query_positions, context_lens, 4, 8)
    torch.testing.assert_close(actual, expected)


def test_qsa_sparse_paged_gqa_matches_reference():
    torch.manual_seed(2)
    q = torch.randn(3, 4, 16, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(4, 4, 2, 16, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
    token_to_request = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int32)
    logical_indices = torch.tensor(
        [[0, 2, 5, -1], [1, 4, 6, 7], [-1, -1, -1, -1]],
        device="cuda",
        dtype=torch.int32,
    )
    scale = q.shape[-1] ** -0.5

    actual = qsa_sparse_paged_gqa(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_request,
        scale,
    )
    expected = _attention_reference(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_request,
        scale,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_qsa_selection_pipeline_matches_reference():
    torch.manual_seed(3)
    q = torch.randn(2, 4, 16, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(8, 4, 1, 16, device="cuda", dtype=torch.bfloat16)
    page_table = torch.tensor(
        [[0, 2, 4, 6], [1, 3, 5, 7]], device="cuda", dtype=torch.int32
    )
    token_to_request = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([15, 13], device="cuda", dtype=torch.int32)
    context_lens = torch.tensor([16, 14], device="cuda", dtype=torch.int32)

    actual = qsa_select_paged_tokens(
        q,
        cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        token_topk=8,
        compress_ratio=2,
    )
    visible = torch.tensor([8, 7], device="cuda", dtype=torch.int32)
    logits = _mqa_reference(q, cache, page_table, token_to_request, visible)
    selected = torch.topk(logits, 4, dim=1).indices.to(torch.int32)
    row_context_lens = context_lens.index_select(0, token_to_request.long())
    expected = _expand_reference(
        selected, query_positions, row_context_lens, 2, 8
    )

    # Selection order is implementation-defined; attention is order-invariant.
    for row in range(expected.shape[0]):
        actual_valid = actual[row][actual[row] >= 0]
        expected_valid = expected[row][expected[row] >= 0]
        torch.testing.assert_close(
            torch.sort(actual_valid).values,
            torch.sort(expected_valid).values,
        )


def test_qsa_gfx950_representative_head_dim_128():
    torch.manual_seed(4)
    q_index = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    compressed_cache = torch.randn(
        16, 4, 1, 128, device="cuda", dtype=torch.bfloat16
    )
    page_table = torch.arange(16, device="cuda", dtype=torch.int32).reshape(2, 8)
    token_to_request = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([23, 27], device="cuda", dtype=torch.int32)
    context_lens = torch.tensor([24, 28], device="cuda", dtype=torch.int32)
    visible = torch.tensor([6, 7], device="cuda", dtype=torch.int32)

    actual_logits, actual_visible = qsa_paged_mqa_logits(
        q_index,
        compressed_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        compress_ratio=4,
    )
    expected_logits = _mqa_reference(
        q_index,
        compressed_cache,
        page_table,
        token_to_request,
        visible,
    )
    torch.testing.assert_close(actual_logits, expected_logits, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual_visible, visible)

    q = torch.randn(2, 5, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(16, 4, 1, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    logical_indices = torch.tensor(
        [
            [0, 3, 7, 8, 15, 16, 22, -1],
            [1, 5, 9, 13, 17, 21, 25, 27],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    scale = 128**-0.5
    actual_output = qsa_sparse_paged_gqa(
        q,
        k_cache,
        v_cache,
        logical_indices,
        page_table,
        token_to_request,
        scale,
    )
    expected_output = _attention_reference(
        q,
        k_cache,
        v_cache,
        logical_indices,
        page_table,
        token_to_request,
        scale,
    )
    torch.testing.assert_close(actual_output, expected_output, rtol=2e-2, atol=2e-2)


def test_qsa_sparse_paged_gqa_qwen_air_selection_width():
    torch.manual_seed(5)
    page_size = 64
    num_pages = 32
    q = torch.randn(1, 5, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(
        num_pages, page_size, 1, 128, device="cuda", dtype=torch.bfloat16
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(
        num_pages, device="cuda", dtype=torch.int32
    ).unsqueeze(0)
    token_to_request = torch.zeros(1, device="cuda", dtype=torch.int32)
    logical_indices = torch.cat(
        (
            torch.randperm(
                num_pages * page_size, device="cuda", dtype=torch.int32
            ),
            torch.full((3,), -1, device="cuda", dtype=torch.int32),
        )
    ).unsqueeze(0)
    scale = 128**-0.5

    actual = qsa_sparse_paged_gqa(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_request,
        scale,
    )
    expected = _attention_reference(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_request,
        scale,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_qsa_forced_gluon_mqa_matches_triton_across_pages_and_requests():
    torch.manual_seed(6)
    q = torch.randn(4, 8, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(9, 4, 1, 128, device="cuda", dtype=torch.bfloat16)
    page_table = torch.tensor(
        [
            [5, -1, -1, -1],
            [2, 6, 0, -1],
            [7, 3, 4, 8],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    token_to_request = torch.tensor([2, 0, 1, 2], device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([0, 7, 35, 24], device="cuda", dtype=torch.int32)
    context_lens = torch.tensor([7, 40, 64], device="cuda", dtype=torch.int32)
    expected_visible = torch.tensor([0, 1, 9, 6], device="cuda", dtype=torch.int32)

    triton_logits, triton_visible = qsa_paged_mqa_logits(
        q,
        cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        compress_ratio=4,
        backend="triton",
    )
    gluon_logits, gluon_visible = qsa_paged_mqa_logits(
        q,
        cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        compress_ratio=4,
        backend="gluon",
    )
    expected = _mqa_reference(
        q, cache, page_table, token_to_request, expected_visible
    )

    torch.testing.assert_close(triton_visible, expected_visible)
    torch.testing.assert_close(gluon_visible, expected_visible)
    torch.testing.assert_close(triton_logits, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(gluon_logits, triton_logits, rtol=2e-3, atol=2e-3)


def test_qsa_forced_gluon_sparse_gqa_qwen_geometry_and_width():
    torch.manual_seed(7)
    rows = 3
    page_size = 16
    pages_per_request = 129
    q = torch.randn(rows, 10, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(
        pages_per_request, page_size, 2, 128, device="cuda", dtype=torch.bfloat16
    )
    v_cache = torch.randn_like(k_cache)
    ascending = torch.arange(
        pages_per_request, device="cuda", dtype=torch.int32
    )
    block_table = torch.stack(
        (ascending, ascending.roll(17), torch.flip(ascending, dims=(0,)))
    )
    token_to_request = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    logical_indices = torch.full(
        (rows, 2051), -1, device="cuda", dtype=torch.int32
    )
    boundary_indices = torch.tensor(
        [0, 15, 16, 17, 31, 32, 2047, 2048, 2050],
        device="cuda",
        dtype=torch.int32,
    )
    logical_indices[0, : boundary_indices.numel()] = boundary_indices
    logical_indices[1, : boundary_indices.numel()] = boundary_indices.flip(0)
    logical_indices[2, :5] = torch.tensor(
        [1, 14, 1023, 1024, 2049], device="cuda", dtype=torch.int32
    )
    scale = 128**-0.5

    triton_output = qsa_sparse_paged_gqa(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_request,
        scale,
        backend="triton",
    )
    gluon_output = qsa_sparse_paged_gqa(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_request,
        scale,
        backend="gluon",
    )
    expected = _attention_reference(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_request,
        scale,
    )

    torch.testing.assert_close(triton_output, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(gluon_output, triton_output, rtol=2e-2, atol=2e-2)
