# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_step5_paged_mqa_logits_kernel_repr = make_kernel_repr(
    "_step5_paged_mqa_logits_kernel",
    [
        "HEADS",
        "PAGE_SIZE",
        "BLOCK_R",
        "TILE",
        "TILES_PER_PROGRAM",
        "FNUZ",
    ],
)


@triton.jit
def _load_fp8(ptrs, mask, FNUZ: tl.constexpr):
    raw = tl.load(ptrs, mask, other=0)
    if FNUZ:
        # The same e4m3fn bits read as e4m3fnuz are exactly half the value;
        # 0x80 (-0 in fn) is the fnuz NaN, so flush it to +0.
        value = (
            tl.where(raw == 0x80, 0, raw).to(tl.uint8).to(tl.float8e4b8, bitcast=True)
        )
    else:
        value = raw.to(tl.float8e4nv, bitcast=True)
    return value


@triton.jit
def _row_pages(
    Table,
    batch,
    region,
    limit,
    TABLE_ROW,
    TABLE_COL: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    sees = region[None, :] < limit[:, None]
    pages = tl.load(
        Table + batch[:, None] * TABLE_ROW + (region // PAGE_SIZE)[None, :] * TABLE_COL,
        sees,
        other=0,
    )
    return sees, pages


@triton.jit
def _tile_scores(
    query,
    weights,
    KV,
    Table,
    Lengths,
    region,
    limit,
    batch,
    max_len,
    row_start,
    ROWS,
    KV_PAGE: tl.constexpr,
    TABLE_ROW,
    TABLE_COL: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    REGIONS,
    NEXT_N: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TILE: tl.constexpr,
    FNUZ: tl.constexpr,
):
    BLOCK_M: tl.constexpr = BLOCK_R * HEADS
    d = tl.arange(0, HEAD_DIM)
    m_local = tl.arange(0, BLOCK_M) // HEADS
    # Keys are shared across the row block when every row maps the regions it
    # can see to the same physical pages as the longest row.
    lead_batch = tl.max(tl.where(limit == max_len, batch, 0), 0)
    lead_valid = region < max_len
    lead_page = tl.load(
        Table + lead_batch * TABLE_ROW + (region // PAGE_SIZE) * TABLE_COL,
        lead_valid,
        other=0,
    )
    if TILE % PAGE_SIZE == 0:
        page = tl.min(region, 0) // PAGE_SIZE + tl.arange(0, TILE // PAGE_SIZE)
        page_sees = page[None, :] * PAGE_SIZE < limit[:, None]
        lead_pages = tl.load(
            Table + lead_batch * TABLE_ROW + page * TABLE_COL,
            page * PAGE_SIZE < max_len,
            other=0,
        )
        row_pages = tl.load(
            Table + batch[:, None] * TABLE_ROW + page[None, :] * TABLE_COL,
            page_sees,
            other=0,
        )
        mismatch = tl.max(
            tl.max((page_sees & (row_pages != lead_pages[None, :])).to(tl.int32), 1), 0
        )
    else:
        sees, pages = _row_pages(
            Table, batch, region, limit, TABLE_ROW, TABLE_COL, PAGE_SIZE
        )
        mismatch = tl.max(
            tl.max((sees & (pages != lead_page[None, :])).to(tl.int32), 1), 0
        )

    if mismatch == 0:
        key = _load_fp8(
            KV
            + lead_page.to(tl.int64)[None, :] * KV_PAGE
            + (region % PAGE_SIZE)[None, :] * HEAD_DIM
            + d[:, None],
            lead_valid[None, :],
            FNUZ,
        )
        dot = tl.dot(query, key, out_dtype=tl.float32)
    else:
        # Each row loads its own keys; the dot runs once per row of the block.
        dot = tl.zeros((BLOCK_M, TILE), dtype=tl.float32)
        for r in range(BLOCK_R):
            r_row = row_start + r
            r_batch = r_row // NEXT_N
            r_len = tl.minimum(
                tl.load(Lengths + r_batch * LENGTH_STRIDE, r_row < ROWS, other=0),
                REGIONS,
            )
            r_valid = region < r_len
            r_page = tl.load(
                Table + r_batch * TABLE_ROW + (region // PAGE_SIZE) * TABLE_COL,
                r_valid,
                other=0,
            ).to(tl.int64)
            r_key = _load_fp8(
                KV
                + r_page[None, :] * KV_PAGE
                + (region % PAGE_SIZE)[None, :] * HEAD_DIM
                + d[:, None],
                r_valid[None, :],
                FNUZ,
            )
            r_dot = tl.dot(query, r_key, out_dtype=tl.float32)
            dot = tl.where(m_local[:, None] == r, r_dot, dot)

    if FNUZ:
        dot = dot * 4.0
    dot = tl.maximum(dot, 0.0) * weights[:, None]
    summed = tl.sum(tl.reshape(dot, (BLOCK_R, HEADS, TILE)), 1)
    if mismatch == 0:
        scales = tl.load(
            (
                KV
                + lead_page.to(tl.int64) * KV_PAGE
                + PAGE_SIZE * HEAD_DIM
                + (region % PAGE_SIZE) * 4
            ).to(tl.pointer_type(tl.float32)),
            lead_valid,
            other=0.0,
        )
        result = summed * scales[None, :]
    else:
        sees, pages = _row_pages(
            Table, batch, region, limit, TABLE_ROW, TABLE_COL, PAGE_SIZE
        )
        row_scales = tl.load(
            (
                KV
                + pages.to(tl.int64) * KV_PAGE
                + PAGE_SIZE * HEAD_DIM
                + (region % PAGE_SIZE)[None, :] * 4
            ).to(tl.pointer_type(tl.float32)),
            sees,
            other=0.0,
        )
        result = summed * row_scales
    return result


@triton.jit(
    repr=_step5_paged_mqa_logits_kernel_repr,
    do_not_specialize=["ROWS", "TABLE_ROW", "REGIONS"],
)
def _step5_paged_mqa_logits_kernel(
    Q,  # uint8 view of fp8 e4m3fn [batch, next_n, heads, HEAD_DIM]
    KV,  # uint8 [pages, PAGE_SIZE * (HEAD_DIM + 4)]: keys, then fp32 scales
    W,  # fp32 [batch * next_n, heads]
    Lengths,  # int32 [batch]
    Table,  # int32 [batch, pages_per_req]
    Output,  # fp32 [batch * next_n, REGIONS]
    ROWS,
    Q_BATCH: tl.constexpr,
    Q_NEXT: tl.constexpr,
    Q_HEAD: tl.constexpr,
    W_ROW: tl.constexpr,
    W_HEAD: tl.constexpr,
    KV_PAGE: tl.constexpr,
    TABLE_ROW,
    TABLE_COL: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    REGIONS,
    NEXT_N: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TILE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    FNUZ: tl.constexpr,
):
    BLOCK_M: tl.constexpr = BLOCK_R * HEADS
    row_start = tl.program_id(0) * BLOCK_R
    first_tile = tl.program_id(1) * TILES_PER_PROGRAM

    rows = row_start + tl.arange(0, BLOCK_R)
    row_ok = rows < ROWS
    batch = rows // NEXT_N
    limit = tl.minimum(
        tl.load(Lengths + batch * LENGTH_STRIDE, row_ok, other=0), REGIONS
    )
    max_len = tl.max(limit, 0)

    d = tl.arange(0, HEAD_DIM)
    m = tl.arange(0, BLOCK_M)
    m_row = row_start + m // HEADS
    m_head = m % HEADS
    m_ok = m_row < ROWS
    q_offsets = (
        (m_row // NEXT_N).to(tl.int64) * Q_BATCH
        + (m_row % NEXT_N) * Q_NEXT
        + m_head * Q_HEAD
    )
    query = _load_fp8(Q + q_offsets[:, None] + d[None, :], m_ok[:, None], FNUZ)
    weights = tl.load(W + m_row.to(tl.int64) * W_ROW + m_head * W_HEAD, m_ok, 0.0)
    out_rows = Output + rows[:, None].to(tl.int64) * REGIONS

    for t in range(first_tile, first_tile + TILES_PER_PROGRAM):
        region = t * TILE + tl.arange(0, TILE)
        if t * TILE < max_len:
            result = _tile_scores(
                query,
                weights,
                KV,
                Table,
                Lengths,
                region,
                limit,
                batch,
                max_len,
                row_start,
                ROWS,
                KV_PAGE,
                TABLE_ROW,
                TABLE_COL,
                LENGTH_STRIDE,
                REGIONS,
                NEXT_N,
                HEADS,
                HEAD_DIM,
                PAGE_SIZE,
                BLOCK_R,
                TILE,
                FNUZ,
            )
        else:
            result = tl.zeros((BLOCK_R, TILE), dtype=tl.float32)
        tl.store(
            out_rows + region[None, :],
            tl.where(region[None, :] < limit[:, None], result, float("-inf")),
            row_ok[:, None] & (region[None, :] < REGIONS),
        )
