# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Build-time layout of the MiniMax-M3 fused sparse-layer decode kernel (no GPU)."""

import pytest

from aiter.ops.flydsl.kernels.minimax_m3_mono import layout
from aiter.ops.flydsl.kernels.minimax_m3_mono.config import (
    BLOCKS,
    HEAD_DIM,
    HIDDEN,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    PAGE16,
    SPARSE_BLOCK,
    TOP_K,
    CacheLayout,
    IndexHeads,
)

TOKEN_COUNTS = range(1, MAX_TOKENS + 1)


def _regions(table):
    starts = sorted((off, name) for name, off in table.items() if name != "_bytes")
    ends = [off for off, _ in starts[1:]] + [table["_bytes"]]
    return [(name, off, end) for (off, name), end in zip(starts, ends)]


@pytest.mark.parametrize(
    "table", [layout.SCRATCH, layout.sym_layout(4)], ids=["scratch", "sym"]
)
def test_regions_are_disjoint_nonempty_and_aligned(table):
    for name, start, end in _regions(table):
        assert start % 256 == 0, name
        assert end > start, name


@pytest.mark.parametrize("npes", [1, 2, 4, 8])
def test_symmetric_regions_hold_every_rank_and_token(npes):
    table = {
        name: end - start for name, start, end in _regions(layout.sym_layout(npes))
    }
    row_bytes = HIDDEN * 4  # (value, tag) pairs of two bf16: 4 B payload per 8
    assert table["attn"] >= npes * MAX_TOKENS * row_bytes
    assert table["ffn"] >= npes * MAX_TOKENS * row_bytes
    assert table["a_ag"] >= MAX_TOKENS * row_bytes
    assert table["ffn_ag"] >= MAX_TOKENS * row_bytes


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_every_routed_up_gate_task_runs_once(tokens):
    total = sum(layout.ug_tasks_of(cta, tokens) for cta in range(BLOCKS))
    assert total == TOP_K * layout.UG_PER_SLOT * tokens


@pytest.mark.parametrize("row", ["XN8_ROW", "XSC_ROW", "MID_ROW", "MIDSC_ROW", "O_ROW"])
def test_padded_lds_rows_put_sixteen_rows_in_distinct_bank_groups(row):
    pitch = getattr(layout, row)
    assert len({(r * pitch) % 64 // 4 for r in range(16)}) == 16


@pytest.mark.parametrize("count, own", [(1, 0), (4, 0), (4, 1), (4, 2), (4, 3)])
def test_index_heads_locate_the_fused_projection_segments(count, own):
    sizes = [LOCAL_Q_HEADS * HEAD_DIM, HEAD_DIM, HEAD_DIM, count * HEAD_DIM, HEAD_DIM]
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    heads = IndexHeads(count, own)
    assert heads.rows == sum(sizes)
    assert heads.iq_off == starts[3] + own * HEAD_DIM
    assert heads.ik_off == starts[4]


@pytest.mark.parametrize("count, own", [(2, 0), (4, 4), (1, 1)])
def test_index_heads_reject_other_head_sets(count, own):
    with pytest.raises(AssertionError):
        IndexHeads(count, own)


@pytest.mark.parametrize(
    "block_pages", [SPARSE_BLOCK // PAGE16, 2 * SPARSE_BLOCK // PAGE16]
)
def test_cache_layout_block_pages(block_pages):
    assert CacheLayout(block_pages).block_pages == block_pages


@pytest.mark.parametrize("block_pages", [0, 4, 12, 32])
def test_cache_layout_rejects_other_strides(block_pages):
    with pytest.raises(AssertionError):
        CacheLayout(block_pages)
