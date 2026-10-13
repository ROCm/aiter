# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for QSA scoring, selection and sparse paged GQA."""

import pytest
import torch

torch.set_default_device("cuda")


def _make_paged_scenario(
    rows,
    pages,
    block,
    ratio,
    index_heads,
    index_dim,
    query_heads,
    query_dim,
    kv_heads,
    start=0,
    dtype=torch.bfloat16,
):
    device = "cuda"
    positions = torch.arange(start, start + rows, device=device, dtype=torch.int32)
    tables = torch.randperm(pages, device=device).to(torch.int32)[None]
    requests = torch.zeros(rows, dtype=torch.int32, device=device)
    lengths = torch.tensor([start + rows], dtype=torch.int32, device=device)
    index_q = torch.randn(rows, index_heads, index_dim, dtype=dtype, device=device)
    compressed = torch.randn(
        pages,
        block // ratio,
        1,
        index_dim,
        dtype=dtype,
        device=device,
    )
    query = torch.randn(rows, query_heads, query_dim, dtype=dtype, device=device)
    keys = torch.randn(pages, block, kv_heads, query_dim, dtype=dtype, device=device)
    values = torch.randn_like(keys)
    return {
        "positions": positions,
        "tables": tables,
        "requests": requests,
        "lengths": lengths,
        "index_q": index_q,
        "compressed": compressed,
        "query": query,
        "keys": keys,
        "values": values,
    }


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("start", [15, 2047, 2063])
def test_qsa_verification_is_causal_and_matches_individual_queries(k, start):
    """Future draft keys must not enter an earlier query's selected context."""
    from aiter.ops.triton.attention.qsa import (
        qsa_select_paged_tokens,
        qsa_sparse_paged_gqa,
    )

    torch.manual_seed(93)
    block, ratio, budget = 16, 4, 2048
    rows, pages = k + 1, 132
    s = _make_paged_scenario(
        rows,
        pages,
        block,
        ratio,
        index_heads=4,
        index_dim=128,
        query_heads=24,
        query_dim=256,
        kv_heads=2,
        start=start,
    )
    selected = qsa_select_paged_tokens(
        s["index_q"],
        s["compressed"],
        s["tables"],
        s["requests"],
        s["positions"],
        s["lengths"],
        budget,
        ratio,
    )
    batched = qsa_sparse_paged_gqa(
        s["query"],
        s["keys"],
        s["values"],
        selected,
        s["tables"],
        s["requests"],
        num_decode_requests=1,
    )
    for row in range(rows):
        visible = selected[row][selected[row] >= 0]
        expected_count = (
            min((start + row + 1) // ratio * ratio, budget) + (start + row + 1) % ratio
        )
        assert visible.numel() == expected_count
        assert visible.unique().numel() == visible.numel()
        assert int(visible.max()) <= start + row

        individual = qsa_select_paged_tokens(
            s["index_q"][row : row + 1],
            s["compressed"],
            s["tables"],
            s["requests"][row : row + 1],
            s["positions"][row : row + 1],
            s["lengths"].new_tensor([start + row + 1]),
            budget,
            ratio,
        )
        torch.testing.assert_close(individual, selected[row : row + 1])

        reference = qsa_sparse_paged_gqa(
            s["query"][row : row + 1],
            s["keys"],
            s["values"],
            individual,
            s["tables"],
            s["requests"][row : row + 1],
            num_decode_requests=1,
        )
        torch.testing.assert_close(
            batched[row : row + 1],
            reference,
            rtol=0.001,
            atol=0.001,
        )


@pytest.mark.parametrize("rows", [512, 1024])
def test_qsa_mfma_scorer_matches_vector_path(rows):
    """The MFMA scorer (rows > 256) must match the vector-path scorer row for row."""
    from aiter.ops.triton.attention.qsa import qsa_paged_mqa_logits

    torch.manual_seed(7)
    block, ratio, pages = 64, 4, 128
    index_heads, index_dim = 4, 128
    compressed_page_size = block // ratio

    positions = torch.arange(rows, dtype=torch.int32)
    tables = torch.randperm(pages).to(torch.int32)[None]
    requests = torch.zeros(rows, dtype=torch.int32)
    lengths = torch.tensor([rows], dtype=torch.int32)
    index_q = torch.randn(rows, index_heads, index_dim, dtype=torch.bfloat16)
    compressed = torch.randn(
        pages, compressed_page_size, 1, index_dim, dtype=torch.bfloat16
    )

    # Vector path: score in chunks of 256 to stay on the non-MFMA kernel
    ref_logits_parts = []
    ref_visible_parts = []
    chunk = 256
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        sl = slice(start, end)
        part_lengths = torch.tensor([end], dtype=torch.int32, device="cuda")
        logits_part, visible_part = qsa_paged_mqa_logits(
            index_q[sl],
            compressed,
            tables,
            requests[sl],
            positions[sl],
            part_lengths,
            ratio,
        )
        ref_logits_parts.append(logits_part)
        ref_visible_parts.append(visible_part)
    ref_logits = torch.cat(ref_logits_parts, dim=0)
    ref_visible = torch.cat(ref_visible_parts, dim=0)

    # MFMA path: rows > 256 triggers the MFMA kernel
    mfma_logits, mfma_visible = qsa_paged_mqa_logits(
        index_q,
        compressed,
        tables,
        requests,
        positions,
        lengths,
        ratio,
    )

    torch.testing.assert_close(mfma_visible, ref_visible)
    for row in range(rows):
        vis = int(ref_visible[row])
        if vis > 0:
            torch.testing.assert_close(
                mfma_logits[row, :vis],
                ref_logits[row, :vis],
                rtol=0.01,
                atol=0.01,
            )
