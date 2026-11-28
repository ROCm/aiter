# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_REPR_KEYS = [
    "NEXT_N",
    "HEADS",
    "HEAD_DIM",
    "PAGE_SIZE",
    "BLOCK_R",
    "TILE",
    "TILES_PER_PROGRAM",
    "READ_FNUZ",
    "FN_AS_FNUZ",
]
_pa_mqa_logits_shared_pages_kernel_repr = make_kernel_repr(
    "_pa_mqa_logits_shared_pages_kernel", _REPR_KEYS
)
_pa_mqa_logits_shared_pages_rows_kernel_repr = make_kernel_repr(
    "_pa_mqa_logits_shared_pages_rows_kernel", _REPR_KEYS
)


@triton.jit
def _load_fp8(ptrs, mask, READ_FNUZ: tl.constexpr, FN_AS_FNUZ: tl.constexpr):
    raw = tl.load(ptrs, mask, other=0)
    if FN_AS_FNUZ:
        # The same e4m3fn bits read as e4m3fnuz are exactly half the value;
        # 0x80 (-0 in fn) is the fnuz NaN, so flush it to +0.
        value = (
            tl.where(raw == 0x80, 0, raw).to(tl.uint8).to(tl.float8e4b8, bitcast=True)
        )
    elif READ_FNUZ:
        value = raw.to(tl.float8e4b8, bitcast=True)
    else:
        value = raw.to(tl.float8e4nv, bitcast=True)
    return value


@triton.jit
def _load_scales(KV, pages, region, mask, KV_PAGE, PAGE_SIZE, HEAD_DIM):
    ptrs = (
        KV
        + pages.to(tl.int64) * KV_PAGE
        + PAGE_SIZE * HEAD_DIM
        + (region % PAGE_SIZE) * 4
    )
    return tl.load(ptrs.to(tl.pointer_type(tl.float32)), mask, other=0.0)


@triton.jit
def _visible_keys(Lengths, batch, row, ok, LENGTH_STRIDE, REGIONS, NEXT_N):
    """Query row ``j`` of a request of length ``L`` sees ``L - NEXT_N + 1 + j`` keys."""
    length = tl.load(Lengths + batch * LENGTH_STRIDE + LENGTH_STRIDE - 1, ok, other=0)
    limit = tl.where(ok, length - NEXT_N + 1 + row % NEXT_N, 0)
    return tl.minimum(tl.maximum(limit, 0), REGIONS)


@triton.jit
def _block_rows(row_start, Lengths, ROWS, LENGTH_STRIDE, REGIONS, NEXT_N, BLOCK_R):
    rows = row_start + tl.arange(0, BLOCK_R)
    row_ok = rows < ROWS
    batch = rows // NEXT_N
    limit = _visible_keys(Lengths, batch, rows, row_ok, LENGTH_STRIDE, REGIONS, NEXT_N)
    max_len = tl.max(limit, 0)
    lead_batch = tl.max(tl.where(limit == max_len, batch, 0), 0)
    return rows, row_ok, batch, limit, max_len, lead_batch


@triton.jit(
    repr=_pa_mqa_logits_shared_pages_kernel_repr,
    do_not_specialize=["ROWS", "TABLE_ROW", "OUT_ROW", "REGIONS"],
)
def _pa_mqa_logits_shared_pages_kernel(
    Q,  # uint8 view of fp8 [batch, NEXT_N, HEADS, HEAD_DIM]
    KV,  # uint8 [pages, PAGE_SIZE * (HEAD_DIM + 4)]: keys, then fp32 scales
    W,  # fp32 [batch * NEXT_N, HEADS]
    Lengths,  # int32 [batch, LENGTH_STRIDE]
    Table,  # int32 [batch, pages_per_request]
    Output,  # fp32 [batch * NEXT_N, >= REGIONS]
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
    OUT_ROW,
    REGIONS,
    NEXT_N: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TILE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    READ_FNUZ: tl.constexpr,
    FN_AS_FNUZ: tl.constexpr,
):
    """Score each tile of a row block with the keys of its longest row.

    The scores are exact for every row that maps its visible keys to the same
    pages; the rows kernel rewrites the tiles where a row does not.
    """
    BLOCK_M: tl.constexpr = BLOCK_R * HEADS
    row_start = tl.program_id(0) * BLOCK_R
    first_tile = tl.program_id(1) * TILES_PER_PROGRAM
    rows, row_ok, _, limit, max_len, lead_batch = _block_rows(
        row_start, Lengths, ROWS, LENGTH_STRIDE, REGIONS, NEXT_N, BLOCK_R
    )

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
    query = _load_fp8(
        Q + q_offsets[:, None] + d[None, :], m_ok[:, None], READ_FNUZ, FN_AS_FNUZ
    )
    weights = tl.load(W + m_row.to(tl.int64) * W_ROW + m_head * W_HEAD, m_ok, 0.0)
    out_rows = Output + rows[:, None].to(tl.int64) * OUT_ROW

    for t in range(first_tile, first_tile + TILES_PER_PROGRAM):
        if t * TILE < max_len:
            region = t * TILE + tl.arange(0, TILE)
            valid = region < max_len
            pages = tl.load(
                Table + lead_batch * TABLE_ROW + (region // PAGE_SIZE) * TABLE_COL,
                valid,
                other=0,
            )
            key = _load_fp8(
                KV
                + pages.to(tl.int64)[None, :] * KV_PAGE
                + (region % PAGE_SIZE)[None, :] * HEAD_DIM
                + d[:, None],
                valid[None, :],
                READ_FNUZ,
                FN_AS_FNUZ,
            )
            dot = tl.dot(query, key, out_dtype=tl.float32)
            if FN_AS_FNUZ:
                dot = dot * 4.0
            dot = tl.maximum(dot, 0.0) * weights[:, None]
            summed = tl.sum(tl.reshape(dot, (BLOCK_R, HEADS, TILE)), 1)
            scales = _load_scales(
                KV, pages, region, valid, KV_PAGE, PAGE_SIZE, HEAD_DIM
            )
            tl.store(
                out_rows + region[None, :],
                summed * scales[None, :],
                row_ok[:, None] & (region[None, :] < limit[:, None]),
            )


@triton.jit(
    repr=_pa_mqa_logits_shared_pages_rows_kernel_repr,
    do_not_specialize=["ROWS", "TABLE_ROW", "OUT_ROW", "REGIONS"],
)
def _pa_mqa_logits_shared_pages_rows_kernel(
    Q,
    KV,
    W,
    Lengths,
    Table,
    Output,
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
    OUT_ROW,
    REGIONS,
    NEXT_N: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    TILE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    READ_FNUZ: tl.constexpr,
    FN_AS_FNUZ: tl.constexpr,
    DOT_M: tl.constexpr,
):
    """Rescore, one row at a time, the tiles where a row of the block maps a
    visible key to a different page than the longest row."""
    row_start = tl.program_id(0) * BLOCK_R
    first_tile = tl.program_id(1) * TILES_PER_PROGRAM
    _, _, batch, limit, max_len, lead_batch = _block_rows(
        row_start, Lengths, ROWS, LENGTH_STRIDE, REGIONS, NEXT_N, BLOCK_R
    )
    d = tl.arange(0, HEAD_DIM)
    h = tl.arange(0, DOT_M)
    h_ok = h < HEADS

    for t in range(first_tile, first_tile + TILES_PER_PROGRAM):
        if t * TILE < max_len:
            region = t * TILE + tl.arange(0, TILE)
            if TILE % PAGE_SIZE == 0:
                page = t * (TILE // PAGE_SIZE) + tl.arange(0, TILE // PAGE_SIZE)
                sees = page[None, :] * PAGE_SIZE < limit[:, None]
                lead_valid = page * PAGE_SIZE < max_len
            else:
                page = region // PAGE_SIZE
                sees = region[None, :] < limit[:, None]
                lead_valid = region < max_len
            row_pages = tl.load(
                Table + batch[:, None] * TABLE_ROW + page[None, :] * TABLE_COL,
                sees,
                other=0,
            )
            lead_pages = tl.load(
                Table + lead_batch * TABLE_ROW + page * TABLE_COL,
                lead_valid,
                other=0,
            )
            differ = sees & (row_pages != lead_pages[None, :])
            if tl.max(tl.max(differ.to(tl.int32), 1), 0) != 0:
                for r in range(BLOCK_R):
                    row = row_start + r
                    if row < ROWS:
                        b = row // NEXT_N
                        length = _visible_keys(
                            Lengths, b, row, True, LENGTH_STRIDE, REGIONS, NEXT_N
                        )
                        valid = region < length
                        pages = tl.load(
                            Table + b * TABLE_ROW + (region // PAGE_SIZE) * TABLE_COL,
                            valid,
                            other=0,
                        )
                        key = _load_fp8(
                            KV
                            + pages.to(tl.int64)[None, :] * KV_PAGE
                            + (region % PAGE_SIZE)[None, :] * HEAD_DIM
                            + d[:, None],
                            valid[None, :],
                            READ_FNUZ,
                            FN_AS_FNUZ,
                        )
                        query = _load_fp8(
                            Q
                            + b.to(tl.int64) * Q_BATCH
                            + (row % NEXT_N) * Q_NEXT
                            + h[:, None] * Q_HEAD
                            + d[None, :],
                            h_ok[:, None],
                            READ_FNUZ,
                            FN_AS_FNUZ,
                        )
                        weights = tl.load(
                            W + row.to(tl.int64) * W_ROW + h * W_HEAD, h_ok, 0.0
                        )
                        dot = tl.dot(query, key, out_dtype=tl.float32)
                        if FN_AS_FNUZ:
                            dot = dot * 4.0
                        summed = tl.sum(tl.maximum(dot, 0.0) * weights[:, None], 0)
                        scales = _load_scales(
                            KV, pages, region, valid, KV_PAGE, PAGE_SIZE, HEAD_DIM
                        )
                        tl.store(
                            Output + row.to(tl.int64) * OUT_ROW + region,
                            summed * scales,
                            valid,
                        )
