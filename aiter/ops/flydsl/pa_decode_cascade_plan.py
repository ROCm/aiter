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

Algorithm
---------
1. Stable radix order: rows sorted lexicographically over the first
   ``LEAD_KEY_COLUMNS`` block-table columns (three stable int64 argsorts,
   two physical block ids per key). Deep sibling trees collide on all six
   lead columns and stay a single run; the LCP pass below then measures
   real sharing. Rows sharing fewer than the sorted depth can in
   principle interleave deep families and split a group into two -- a
   grouping-quality issue, never a correctness one.
2. A Triton kernel (``_cascade_lcp_kernel``) compares every adjacent pair
   in sorted order and emits its longest shared full-block prefix (LCP),
   clamped by both contexts' whole blocks (the in-progress tail block is
   private by construction).
3. Torch collectives turn pair LCPs into groups (runs of pairs meeting
   ``min_shared_blocks``), per-group shared block counts (min pair LCP),
   items of at most ``members_per_item`` members, and the B/singles
   metadata. Spillover (item table or B batch full) degrades, never
   fails: those members run as singles.

All buffers are allocated once per batch bucket by
:func:`plan_pa_decode_cascade_groups`; :func:`fill_pa_decode_cascade_plan`
is a fixed sequence of kernel launches and collectives with static
shapes -- no host synchronization -- suitable for CUDA-graph capture.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

LEAD_KEY_COLUMNS = 6
LCP_CHUNK = 128


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
def _cascade_emit_kernel(
    order_ptr,  # [n] int64 sorted rows
    lens_ptr,  # [n] int32 context lengths
    mrank_ptr,  # [n] int64 member rank
    srank_ptr,  # [n] int64 single rank
    item_id_ptr,  # [n] int64 item index
    slot_ptr,  # [n] int64 slot within item (member-major order)
    sblk_ptr,  # [n] int32 shared blocks of the row's group
    ok_ptr,  # [n] int32 1 = goes to an item, 0 = single
    items_ptr,  # [max_items, 2 + MEMBERS] int32
    b_map_ptr,
    b_item_ptr,
    b_slot_ptr,
    tail_lens_ptr,
    sblk_b_ptr,
    singles_rows_ptr,
    singles_lens_ptr,
    n_rows,
    MAX_B,
    M: tl.constexpr,
    PAGE: tl.constexpr,
):
    p = tl.program_id(0)
    if p >= n_rows:
        return
    row = tl.load(order_ptr + p).to(tl.int32)
    if tl.load(ok_ptr + p):
        mrank = tl.load(mrank_ptr + p)
        item = tl.load(item_id_ptr + p).to(tl.int32)
        slot = tl.load(slot_ptr + p).to(tl.int32)
        sblk = tl.load(sblk_ptr + p)
        length = tl.load(lens_ptr + row)
        tl.store(b_map_ptr + mrank, row)
        tl.store(b_item_ptr + mrank, item)
        tl.store(b_slot_ptr + mrank, slot)
        tl.store(tail_lens_ptr + mrank, length - sblk * PAGE)
        tl.store(sblk_b_ptr + mrank, sblk)
        base = items_ptr + item * (2 + M)
        tl.store(base + 0, sblk)  # all members write the same value
        tl.atomic_add(base + 1, 1)
        tl.store(base + 2 + slot, row)  # unique target
    else:
        srank = tl.load(srank_ptr + p)
        tl.store(singles_rows_ptr + srank, row)
        tl.store(singles_lens_ptr + srank, tl.load(lens_ptr + row))


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
    dev = block_tables.device
    page = ws.block_size
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
            PAGE=page,
            CHUNK=LCP_CHUNK,
            num_warps=1,
        )
    pair = pl.lcp[: n - 1] if n > 1 else pl.lcp[:0]

    # --- 3. groups, items and row assignments (static collectives) ---
    join = pair >= min_shared_blocks  # bool [n-1]
    arange = pl.arange_n[:n]
    gid = torch.zeros(n, dtype=torch.int64, device=dev)
    if n > 1:
        gid[1:] = torch.cumsum((~join).to(torch.int64), 0)
    gcount = torch.zeros(n, dtype=torch.int64, device=dev).scatter_add_(
        0, gid, torch.ones(n, dtype=torch.int64, device=dev)
    )
    member = gcount[gid] >= 2
    gmin = torch.full((n,), 1 << 30, dtype=torch.int32, device=dev)
    if n > 1:
        gmin.scatter_reduce_(
            0,
            gid[: n - 1],
            torch.where(join, pair, torch.full_like(pair, 1 << 30)),
            reduce="amin",
            include_self=True,
        )
    gstart = torch.full((n,), n, dtype=torch.int64, device=dev)
    gstart.scatter_reduce_(0, gid, arange, reduce="amin", include_self=True)
    local = arange - gstart[gid]
    sblk_at = gmin[gid]  # int32 [n], meaningful where member
    g_items = torch.div(gcount + (M - 1), M, rounding_mode="floor")
    item_base = torch.zeros(n, dtype=torch.int64, device=dev)
    if n > 1:
        item_base[1:] = torch.cumsum(g_items, 0)[: n - 1]
    item_id = item_base[gid] + torch.div(local, M, rounding_mode="floor")
    slot = local % M
    mrank = torch.cumsum(member.to(torch.int64), 0) - 1
    member_ok = member & (item_id < ws.max_items) & (mrank < ws.max_b_rows)
    single = ~member_ok
    srank = torch.cumsum(single.to(torch.int64), 0) - 1
    order.to(torch.int32)

    # --- 4-6. item table, B metadata and singles metadata (one kernel) ---
    ws.items.zero_()
    for t in (ws.b_map, ws.b_item, ws.b_slot, ws.tail_lens, pl.sblk_b):
        t.fill_(0)
    ws.singles_rows.fill_(0)
    ws.singles_lens.fill_(0)
    _cascade_emit_kernel[(n,)](
        order,
        context_lengths,
        mrank,
        srank,
        item_id,
        slot,
        sblk_at,
        member_ok.to(torch.int32),
        ws.items,
        ws.b_map,
        ws.b_item,
        ws.b_slot,
        ws.tail_lens,
        pl.sblk_b,
        ws.singles_rows,
        ws.singles_lens,
        n,
        ws.max_b_rows,
        M=M,
        PAGE=page,
        num_warps=1,
    )

    # --- 7. counts ---
    ws.counts[0] = (ws.items[:, 1] > 0).sum().to(torch.int32)
    ws.counts[1] = member_ok.sum().to(torch.int32)
    ws.counts[2] = single.sum().to(torch.int32)

    # --- 8. tail / singles block tables (one kernel) ---
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
