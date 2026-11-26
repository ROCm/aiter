# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU grouping planner for cascade (shared-prefix) decode.

Per step, groups decode rows whose block tables share a block-aligned
prefix and fills a
:class:`~aiter.ops.flydsl.pa_decode_cascade.CascadeWorkspace` with the
metadata :func:`~aiter.ops.flydsl.pa_decode_cascade.pa_decode_cascade`
consumes. Everything runs on device from the current block table: block
IDs are only meaningful within one step (eviction and reuse recycle
them), so groups are rebuilt every step and never cached.

Pipeline per fill (all device kernels / collectives, static shapes):

1. Stable radix order: rows sorted lexicographically over the first
   ``LEAD_KEY_COLUMNS`` block-table columns (stable int64 argsorts, two
   physical block ids per key). Deep sibling trees collide on all six
   lead columns and stay a single run; the LCP pass below then measures
   real sharing. Rows sharing fewer than the sorted depth can in
   principle interleave deep families and split a group into two -- a
   grouping-quality issue, never a correctness one.
2. ``_cascade_lcp_kernel`` compares every adjacent pair in sorted order
   and emits its longest shared full-block prefix (LCP), clamped by both
   contexts' whole blocks (the in-progress tail block is private by
   construction).
3. ``_cascade_plan_items_kernel`` -- one CTA, one tile of up to
   ``TILE_LIMIT`` rows -- turns pair LCPs into the full cascade
   metadata with segmented scans: runs of joined pairs, per-run shared
   block counts (min pair LCP), run-local member indices, items of at
   most ``members_per_item`` members, B/single ranks, and every scatter
   and count. Overflow of the item table or B batch degrades members to
   singles, never to failure.
4. ``_cascade_tables_kernel`` builds the shifted tail block tables.

All buffers are allocated once per batch bucket by
:func:`plan_pa_decode_cascade_groups`; :func:`fill_pa_decode_cascade_plan`
is a fixed sequence of launches with static shapes -- no host
synchronization -- suitable for CUDA-graph capture.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

LEAD_KEY_COLUMNS = 6
LCP_CHUNK = 128
TILE_LIMIT = 4096


@triton.jit
def _cascade_lcp_kernel(
    bt_ptr,  # [n, table_width] int32 block table
    len_ptr,  # [n] int32 context lengths
    order_ptr,  # [n] int32 sorted row order; adjacent pairs are compared
    lcp_ptr,  # [n - 1] int32 out: shared full blocks of pair (p, p + 1)
    n_rows,
    table_width,
    PAGE: tl.constexpr,
    CHUNK: tl.constexpr,
):
    p = tl.program_id(0)
    if p >= n_rows - 1:
        return
    a = tl.load(order_ptr + p)
    b = tl.load(order_ptr + p + 1)
    la = tl.load(len_ptr + a)
    lb = tl.load(len_ptr + b)
    # Only whole blocks may be shared: the last partial block is private.
    valid = tl.minimum(la, lb) // PAGE
    valid = tl.minimum(valid, table_width)
    c = 0
    result = 0
    done = 0
    while done == 0:
        cols = c + tl.arange(0, CHUNK)
        inb = cols < valid
        ka = tl.load(bt_ptr + a * table_width + cols, mask=inb, other=0)
        kb = tl.load(bt_ptr + b * table_width + cols, mask=inb, other=1)
        diff = ka != kb
        arange = tl.arange(0, CHUNK)
        first = tl.min(tl.where(diff, arange, CHUNK))
        result = c + first
        done = tl.where(first < CHUNK, 1, 0)
        done = tl.where(c + CHUNK >= valid, 1, done)
        c += CHUNK
    tl.store(lcp_ptr + p, result)


@triton.jit
def _scan_chain_min(seg_a, val_a, seg_b, val_b):
    # Segmented-min combine: newer segments reset the accumulator.
    return seg_b, tl.where(seg_a == seg_b, tl.minimum(val_a, val_b), val_b)


@triton.jit
def _scan_max(a, b):
    return tl.maximum(a, b)


@triton.jit
def _cascade_plan_items_kernel(
    lcp_ptr,  # [n - 1] int32 pair LCPs between sorted rows
    order_ptr,  # [n] int64 sorted row order
    len_ptr,  # [n] int32 context lengths
    qmap_ptr,  # [n] int32 request row -> token row, or an inert pointer
    items_ptr,  # [max_items, 2 + M] int32, pre-zeroed
    bslot_ptr,  # [max_items, MEMBERS] int32 member's B-pass row
    b_map_ptr,
    b_tok_ptr,
    b_item_ptr,
    b_slot_ptr,
    tail_lens_ptr,
    sblk_b_ptr,
    singles_rows_ptr,
    singles_tok_ptr,
    singles_lens_ptr,
    counts_ptr,  # [3]: items with live members, B rows, single rows
    n_rows,
    min_shared,
    MAX_ITEMS,
    MAX_B,
    M: tl.constexpr,
    PAGE: tl.constexpr,
    HAS_QMAP: tl.constexpr,
    TILE: tl.constexpr,
):
    BIG: tl.constexpr = 1 << 30
    row_idx = tl.arange(0, TILE)
    inb = row_idx < n_rows

    # --- joins: join_next[p] means sorted rows p and p + 1 are siblings,
    #     join_prev[p] means p - 1 and p are.
    lcp_next = tl.load(lcp_ptr + row_idx, mask=inb & (row_idx <= n_rows - 2), other=0)
    lcp_prev = tl.load(lcp_ptr + row_idx - 1, mask=inb & (row_idx >= 1), other=0)
    join_next = row_idx <= n_rows - 2
    join_next = join_next & (lcp_next >= min_shared)
    join_prev = inb & (row_idx >= 1) & (lcp_prev >= min_shared)
    member = (join_next | join_prev).to(tl.int32)

    # --- run structure via segmented scans (all register-resident).
    # first[p]: sorted index of the first row of p's run (reset-flagged max).
    reset = inb & (~join_prev)
    reset_i = reset.to(tl.int32)
    first = tl.associative_scan(tl.where(reset, row_idx, -1).to(tl.int32), 0, _scan_max)
    local = row_idx - first  # within-run position
    # seg: monotone run id, constant within a run (inclusive cumsum
    # counts the run's own start boundary, so every row of the run ties).
    seg = tl.cumsum(reset_i, 0)
    # chainv on ROW indices: pair (p, p+1) lcp where the chain is alive.
    chainv = tl.where(join_next, lcp_next, BIG).to(tl.int32)
    _seg_fwd, segmin_fwd = tl.associative_scan((seg, chainv), 0, _scan_chain_min)
    # The run's sblk is the min over its whole pair chain: the forward
    # scan covers the chain from its start through each row, a mirrored
    # scan over flipped indices covers each row through the chain's end,
    # and pair chains partition (with idempotent overlap) around the row.
    _seg_rev, segmin_rev = tl.associative_scan(
        (tl.flip(seg, 0), tl.flip(chainv, 0)), 0, _scan_chain_min
    )
    sblk = tl.minimum(segmin_fwd, tl.flip(segmin_rev, 0))

    # --- items: one start per run and per M members inside it. Singleton
    # runs (valid rows that joined no run) become items of one member with
    # their own full-block context as the shared range, so they take the
    # shared-pass engine too: the v2 tail engine sees only their partial
    # tail block.
    order = tl.load(order_ptr + row_idx, mask=inb, other=0)
    rows = order.to(tl.int32)
    lengths_all = tl.load(len_ptr + rows, mask=inb, other=0)
    run_single = (~join_next) & (~join_prev) & (lengths_all >= PAGE)
    member = (member != 0) | run_single
    own_blocks = lengths_all // PAGE
    sblk = tl.where(run_single, own_blocks, sblk)

    item_start = (member != 0) & ((local % M) == 0)
    item_id = tl.cumsum(item_start.to(tl.int32), 0) - 1
    ok1 = (member != 0) & (item_id < MAX_ITEMS)
    rank_ok1 = tl.cumsum(ok1.to(tl.int32), 0) - ok1.to(tl.int32)
    ok = ok1 & (rank_ok1 < MAX_B)
    single = inb & (~ok)

    lengths = lengths_all
    slot = (local % M).to(tl.int32)

    # --- emit: items table, B metadata, singles metadata.
    item_flat = item_id.to(tl.int64) * (2 + M)
    tl.store(
        items_ptr + item_flat,
        sblk,
        mask=ok,
    )  # every member of an item writes the same value
    tl.atomic_add(items_ptr + item_flat + 1, 1, mask=ok)
    tl.store(items_ptr + item_flat + 2 + slot, rows, mask=ok)  # unique

    mrank = rank_ok1.to(tl.int64)
    if HAS_QMAP:
        tok = tl.load(qmap_ptr + rows, mask=ok | single, other=0)
    else:
        tok = rows
    tl.store(bslot_ptr + item_id * M + slot, mrank.to(tl.int32), mask=ok)
    tl.store(b_map_ptr + mrank, rows, mask=ok)
    tl.store(b_tok_ptr + mrank, tok, mask=ok)
    tl.store(b_item_ptr + mrank, item_id.to(tl.int32), mask=ok)
    tl.store(b_slot_ptr + mrank, slot, mask=ok)
    tl.store(tail_lens_ptr + mrank, lengths - sblk * PAGE, mask=ok)
    tl.store(sblk_b_ptr + mrank, sblk, mask=ok)

    srank = tl.cumsum(single.to(tl.int32), 0) - single.to(tl.int32)
    s_len = tl.load(len_ptr + rows, mask=single, other=0)
    tl.store(singles_rows_ptr + srank, rows, mask=single)
    tl.store(singles_tok_ptr + srank, tok, mask=single)
    tl.store(singles_lens_ptr + srank, s_len, mask=single)

    # --- counts.
    n_items = tl.max(tl.where(ok, item_id + 1, 0), 0)
    n_members = tl.sum(ok.to(tl.int32), 0)
    n_singles = tl.sum(single.to(tl.int32), 0)
    tl.store(counts_ptr + 0, n_items)
    tl.store(counts_ptr + 1, n_members)
    tl.store(counts_ptr + 2, n_singles)


@triton.jit
def _cascade_tables_kernel(
    bt_ptr,  # [rows, width] int32 full block table
    b_map_ptr,  # [max_b] int32 member rows
    sblk_b_ptr,  # [max_b] int32 shared blocks per member
    n_b_ptr,
    s_rows_ptr,  # [max_s] int32 single rows
    n_s_ptr,
    tail_tabs_ptr,  # [max_b, width] int32 out (shifted tables)
    s_tabs_ptr,  # [max_s, width] int32 out
    width,
    MAX_B,
    MAX_S,
    CHUNK: tl.constexpr,
):
    # One program per (table, row, column chunk).
    which = tl.program_id(0)
    row = tl.program_id(1)
    chunk = tl.program_id(2)
    cols = chunk * CHUNK + tl.arange(0, CHUNK)
    if which == 0:
        n = tl.load(n_b_ptr)
        if row >= n:
            return
        src_row = tl.load(b_map_ptr + row).to(tl.int64)
        sblk = tl.load(sblk_b_ptr + row)
        dst = tail_tabs_ptr + row.to(tl.int64) * width + cols
        src = bt_ptr + src_row * width + sblk + cols
        vals = tl.load(src, mask=(sblk + cols) < width, other=0)
        tl.store(dst, vals, mask=cols < width)
    else:
        n = tl.load(n_s_ptr)
        if row >= n:
            return
        src_row = tl.load(s_rows_ptr + row).to(tl.int64)
        dst = s_tabs_ptr + row.to(tl.int64) * width + cols
        src = bt_ptr + src_row * width + cols
        vals = tl.load(src, mask=cols < width, other=0)
        tl.store(dst, vals, mask=cols < width)


def _pow2(n: int) -> int:
    return 1 << max(n - 1, 1).bit_length()


@dataclass(frozen=True)
class CascadeGroupPlanner:
    """Scratch for one batch bucket of grouping work.

    Created by :func:`plan_pa_decode_cascade_groups`, reused every step.
    """

    max_rows: int
    table_width: int
    max_b_rows: int
    arange_n: torch.Tensor  # int64 [max_rows]
    order: torch.Tensor  # int64 [max_rows]
    lcp: torch.Tensor  # int32 [max_rows] (the last slot is unused padding)
    sblk_b: torch.Tensor  # int32 [max_b_rows]


def plan_pa_decode_cascade_groups(
    device, *, max_rows: int, table_width: int, max_b_rows: int
) -> CascadeGroupPlanner:
    if max_rows > TILE_LIMIT:
        raise ValueError(
            f"cascade planner supports at most {TILE_LIMIT} rows, got {max_rows}"
        )
    dev = torch.device(device)
    return CascadeGroupPlanner(
        max_rows=max_rows,
        table_width=table_width,
        max_b_rows=max_b_rows,
        arange_n=torch.arange(max_rows, dtype=torch.int64, device=dev),
        order=torch.empty(max_rows, dtype=torch.int64, device=dev),
        lcp=torch.empty(max_rows, dtype=torch.int32, device=dev),
        sblk_b=torch.empty(max_b_rows, dtype=torch.int32, device=dev),
    )


def fill_pa_decode_cascade_plan(
    workspace,
    planner: CascadeGroupPlanner,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    *,
    min_shared_blocks: int = 1,
    q_map_of_req: torch.Tensor | None = None,
) -> None:
    """Group ``block_tables`` rows into the workspace's cascade metadata.

    ``workspace`` comes from
    :func:`aiter.ops.flydsl.pa_decode_cascade.plan_pa_decode_cascade`;
    ``planner`` from :func:`plan_pa_decode_cascade_groups` for a bucket
    at least as large as ``block_tables.shape[0]``. Rows whose item or B
    slot would exceed the workspace's static caps are routed to the
    singles batch (the op's plain path).
    """
    ws = workspace
    pl = planner
    n, width = block_tables.shape
    if n > pl.max_rows or width > pl.table_width:
        raise ValueError("batch exceeds the planner's allocation")
    if n > ws.max_singles:
        raise ValueError("workspace singles batch must fit the whole batch")
    ws_page = ws.block_size
    M = ws.members_per_item
    order = pl.order[:n]
    order.copy_(pl.arange_n[:n])

    # --- 1. lexicographic order over the first LEAD_KEY_COLUMNS blocks ---
    bt64 = block_tables.long()
    n_pairs_cols = min(LEAD_KEY_COLUMNS, width) // 2
    for col in range(n_pairs_cols * 2 - 2, -1, -2):
        key = bt64[:, col] * (1 << 32) + bt64[:, col + 1]
        order.copy_(order[torch.argsort(key[order], stable=True)])
    if (width % 2 or LEAD_KEY_COLUMNS % 2) and n_pairs_cols * 2 < min(
        LEAD_KEY_COLUMNS, width
    ):
        order.copy_(order[torch.argsort(bt64[:, n_pairs_cols * 2][order], stable=True)])

    # --- 2. adjacent-pair LCP (Triton) ---
    if n > 1:
        _cascade_lcp_kernel[(n - 1,)](
            block_tables,
            context_lengths,
            order.to(torch.int32).contiguous(),
            pl.lcp,
            n,
            width,
            PAGE=ws_page,
            CHUNK=LCP_CHUNK,
            num_warps=1,
        )

    # --- 3. items, B/singles metadata and counts (one fused kernel) ---
    ws.items.zero_()
    for t in (
        ws.b_map,
        ws.tok_of_b,
        ws.b_item,
        ws.b_slot,
        ws.tail_lens,
        pl.sblk_b,
        ws.b_of_item_slot,
    ):
        t.fill_(0)
    ws.tok_of_s.fill_(0)
    ws.singles_rows.fill_(0)
    ws.singles_lens.fill_(0)
    qmap = q_map_of_req if q_map_of_req is not None else context_lengths
    _cascade_plan_items_kernel[(1,)](
        pl.lcp,
        order,
        context_lengths,
        qmap,
        ws.items,
        ws.b_of_item_slot,
        ws.b_map,
        ws.tok_of_b,
        ws.b_item,
        ws.b_slot,
        ws.tail_lens,
        pl.sblk_b,
        ws.singles_rows,
        ws.tok_of_s,
        ws.singles_lens,
        ws.counts,
        n,
        min_shared_blocks,
        ws.max_items,
        ws.max_b_rows,
        M=M,
        PAGE=ws_page,
        HAS_QMAP=q_map_of_req is not None,
        TILE=_pow2(pl.max_rows),
        num_warps=8,
    )

    # --- 4. tail / singles block tables (one kernel) ---
    cols_chunks = -(-width // LCP_CHUNK)
    _cascade_tables_kernel[(2, max(ws.max_b_rows, ws.max_singles), cols_chunks)](
        block_tables,
        ws.b_map,
        pl.sblk_b,
        ws.counts[1:2],
        ws.singles_rows,
        ws.counts[2:3],
        ws.tail_tables,
        ws.singles_tables,
        width,
        ws.max_b_rows,
        ws.max_singles,
        CHUNK=LCP_CHUNK,
        num_warps=2,
    )
