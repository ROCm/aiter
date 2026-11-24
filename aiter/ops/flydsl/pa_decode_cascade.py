# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cascade (shared-prefix) mode for FlyDSL ``pa_decode`` decode attention.

In agentic traffic, sibling requests share long block-aligned prefixes
(system prompt, tool schemas, a common plan). Prefix caching stores that
KV once, but a plain decode kernel streams it once per request. Cascade
attention streams every shared block once per *item* of up to
``MEMBERS_PER_ITEM`` siblings and merges the result in log-sum-exp space:

* **Shared pass** (``aiter.ops.triton.attention.pa_decode_cascade``): one
  work item per item packs the item's members into one MFMA tile over the
  shared prefix and writes ``(m, l, acc)`` partials per member.
* **Tail pass**: the production FlyDSL ``pa_decode`` runs, unchanged,
  over every member's private tail (its block table shifted by the
  item's shared block count) with partition partials staged in the
  standard ``exp_sums``/``max_logits``/``temporary_output`` buffers.
  Rows with no group (singles) run the same kernel over their full
  block table, exactly as the plain path would.
* **Fold** (a small Triton kernel): merges each member's shared partial
  with its tail partition partials, applying optional attention sinks
  exactly once, and writes its output rows. Singles come from the plain
  ``pa_decode`` call directly.

API contract
------------
Metadata describing the decomposition (which rows group, the shared
length of every item, tails, singles) is built per step, on device,
into a :class:`CascadeWorkspace` allocated once per capture bucket by
:func:`plan_pa_decode_cascade`. :func:`pa_decode_cascade` then performs
the passes above. Grouping itself (``aiter.ops.flydsl.pa_decode_cascade_plan``)
is a separate module so the op stays runnable with any planner that
honors the layout.

All launch grids are static (max-sized) with device counts and padded
slots handled inside the kernels, so capture and replay of the whole op
-- including the per-step plan refreshes -- are CUDA-graph safe as long
as every tensor in the workspace is reused across steps.

Scope: gfx942/gfx950 BF16 K/V in the vectorized 5D layout, page 16,
head dims 64..512 (multiples of 64), 1..8 query heads per KV head,
qlen-1 decode. Attention sinks are supported and applied in the fold
only. No FP8 KV, sliding window, ALiBi, MTP or speculative decode yet.
"""

from dataclasses import dataclass

import torch

from aiter.ops.flydsl.kernels.pa_decode_plan import PADecodePlan, plan_pa_decode
from aiter.ops.flydsl.pa_decode import pa_decode

MEMBERS_PER_ITEM = 4
N_COUNTS = 3  # (n_items, n_members, n_singles)


@dataclass(frozen=True)
class CascadeWorkspace:
    """Static buffers and work plans for :func:`pa_decode_cascade`.

    Metadata tensors (``items`` .. ``singles_tables`` and ``counts``) are
    written per step by the grouping planner; the rest are scratch or
    work plans refreshed by ``pa_decode_cascade`` itself. Everything is
    allocated once so CUDA-graph capture replays pointer-identical
    launches.
    """

    # Shapes.
    block_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    members_per_item: int
    max_items: int
    max_b_rows: int  # max_items * members_per_item
    max_singles: int
    table_width: int  # max blocks per sequence
    a_segments: int  # segments per item for shared-pass occupancy
    dtype: torch.dtype

    # -- Group metadata, device int32, written per step by the planner. --
    items: torch.Tensor  # [max_items, 2 + members_per_item] (sblk, count, rows...)
    b_map: torch.Tensor  # [max_b_rows] original batch row of tail-pass row
    b_item: torch.Tensor  # [max_b_rows] item index
    b_slot: torch.Tensor  # [max_b_rows] member slot within the item
    tail_lens: torch.Tensor  # [max_b_rows] private tail lengths
    tail_tables: torch.Tensor  # [max_b_rows, table_width] shifted block tables
    singles_rows: torch.Tensor  # [max_singles]
    singles_lens: torch.Tensor  # [max_singles]
    singles_tables: torch.Tensor  # [max_singles, table_width]
    counts: torch.Tensor  # [3] (n_items, n_members, n_singles)

    # -- Scratch, device, reused per step. --
    a_m: torch.Tensor  # fp32 [max_items, segments, members, num_q_heads]
    a_l: torch.Tensor  # fp32 [max_items, segments, members, num_q_heads]
    a_acc: torch.Tensor  # fp32 [max_items, segments, members, num_q_heads, head_dim]
    p_max: torch.Tensor  # fp32 [num_kv_heads, b_capacity, group]
    p_sum: torch.Tensor  # fp32 [num_kv_heads, b_capacity, group]
    p_out: torch.Tensor  # dtype [num_kv_heads, b_capacity, group, head_dim]
    q_gather_b: torch.Tensor  # dtype [max_b_rows, num_q_heads, head_dim]
    q_gather_s: torch.Tensor  # dtype [max_singles, num_q_heads, head_dim]
    out_b: torch.Tensor  # dtype [max_b_rows, num_q_heads, head_dim] (discarded)
    out_s: torch.Tensor  # dtype [max_singles, num_q_heads, head_dim]

    b_plan: PADecodePlan  # tail-pass plan (max_partitions = 2)
    s_plan: PADecodePlan  # singles plan


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def plan_pa_decode_cascade(
    device,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int = 16,
    members_per_item: int = MEMBERS_PER_ITEM,
    max_items: int,
    max_b_rows: int | None = None,
    max_singles: int,
    table_width: int,
    a_segments: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    singles_max_partitions: int | None = None,
) -> CascadeWorkspace:
    """Allocate a :class:`CascadeWorkspace` and its two ``PADecodePlan``.

    Call once per capture bucket (outside graph capture).
    ``max_b_rows`` defaults to ``max_items * members_per_item``; set it
    smaller to bound the tail batch when a summand spillover policy makes
    distinct member sets impossible. ``table_width`` is
    ``block_tables.shape[1]``. The tail plan uses ``max_partitions=2`` so
    ``pa_decode`` always takes the packed-partials path the fold reads;
    the singles plan mirrors a plain production call.
    """
    if max_b_rows is None:
        max_b_rows = max_items * members_per_item
    dev = torch.device(device)
    int32 = {"dtype": torch.int32, "device": dev}
    f32 = {"dtype": torch.float32, "device": dev}
    out = {"dtype": dtype, "device": dev}
    group = num_q_heads // num_kv_heads

    def t(shape, **kw):
        return torch.empty(shape, **kw)

    b_lens = t((max_b_rows,), **int32)
    s_lens = t((max_singles,), **int32)
    b_lens.zero_()
    s_lens.zero_()
    b_plan = plan_pa_decode(
        b_lens,
        num_kv_heads,
        max_partitions=2,
        max_batch_size=max_b_rows,
    )
    s_plan = plan_pa_decode(
        s_lens,
        num_kv_heads,
        max_partitions=singles_max_partitions,
        max_batch_size=max_singles,
    )
    cap = b_plan.capacity
    return CascadeWorkspace(
        block_size=block_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        members_per_item=members_per_item,
        max_items=max_items,
        max_b_rows=max_b_rows,
        max_singles=max_singles,
        table_width=table_width,
        a_segments=a_segments,
        dtype=dtype,
        items=t((max_items, 2 + members_per_item), **int32),
        b_map=t((max_b_rows,), **int32),
        b_item=t((max_b_rows,), **int32),
        b_slot=t((max_b_rows,), **int32),
        tail_lens=b_lens,
        tail_tables=t((max_b_rows, table_width), **int32),
        singles_rows=t((max_singles,), **int32),
        singles_lens=s_lens,
        singles_tables=t((max_singles, table_width), **int32),
        counts=t((N_COUNTS,), **int32),
        a_m=t((max_items, a_segments, members_per_item, num_q_heads), **f32),
        a_l=t((max_items, a_segments, members_per_item, num_q_heads), **f32),
        a_acc=t(
            (max_items, a_segments, members_per_item, num_q_heads, head_dim), **f32
        ),
        p_max=t((num_kv_heads, cap, group), **f32),
        p_sum=t((num_kv_heads, cap, group), **f32),
        p_out=t((num_kv_heads, cap, group, head_dim), **out),
        q_gather_b=t((max_b_rows, num_q_heads, head_dim), **out),
        q_gather_s=t((max_singles, num_q_heads, head_dim), **out),
        out_b=t((max_b_rows, num_q_heads, head_dim), **out),
        out_s=t((max_singles, num_q_heads, head_dim), **out),
        b_plan=b_plan,
        s_plan=s_plan,
    )


def _launch_shared_pass(
    ws: CascadeWorkspace, query, key_cache, value_cache, block_tables, softmax_scale
):
    from aiter.ops.triton.attention.pa_decode_cascade import _cascade_shared_kernel

    group = ws.num_q_heads // ws.num_kv_heads
    block_mg = max(16, next_pow2(ws.members_per_item * group))
    page = ws.block_size
    block_t = 64 if ws.head_dim <= 256 else 32
    block_t = (block_t // page) * page
    if block_t < page:
        raise ValueError(f"block_size {page} too large for the shared pass")
    num_warps = 4
    _cascade_shared_kernel[(ws.max_items, ws.a_segments, ws.num_kv_heads)](
        query,
        key_cache,
        value_cache,
        block_tables,
        ws.items,
        ws.a_m,
        ws.a_l,
        ws.a_acc,
        block_tables.stride(0),
        ws.a_m.stride(0),
        ws.a_m.stride(1),
        ws.a_m.stride(2),
        ws.a_acc.stride(0),
        ws.a_acc.stride(1),
        ws.a_acc.stride(2),
        softmax_scale * 1.4426950408889634,
        PAGE=page,
        GQA=group,
        HEAD_DIM=ws.head_dim,
        MEMBERS=ws.members_per_item,
        BLOCK_MG=block_mg,
        Q_ROW_STRIDE=ws.num_q_heads * ws.head_dim,
        Q_HEAD_STRIDE=ws.head_dim,
        K_BLK_STRIDE=ws.num_kv_heads * ws.head_dim * page,
        K_HEAD_STRIDE=ws.head_dim * page,
        V_BLK_STRIDE=ws.num_kv_heads * ws.head_dim * page,
        V_HEAD_STRIDE=ws.head_dim * page,
        num_warps=num_warps,
        num_stages=1,
    )


def _launch_fold(ws: CascadeWorkspace, output, sinks):
    from aiter.ops.triton.attention.pa_decode_cascade import _cascade_fold_kernel

    group = ws.num_q_heads // ws.num_kv_heads
    _cascade_fold_kernel[(ws.max_b_rows, ws.num_q_heads)](
        output,
        ws.a_m,
        ws.a_l,
        ws.a_acc,
        ws.b_map,
        ws.b_item,
        ws.b_slot,
        ws.counts[1:2],
        ws.p_max,
        ws.p_sum,
        ws.p_out,
        ws.b_plan.reduce_info,
        sinks if sinks is not None else ws.a_m,  # inert when HAS_SINKS=False
        output.stride(0),
        output.stride(1),
        ws.a_m.stride(0),
        ws.a_m.stride(1),
        ws.a_m.stride(2),
        ws.a_acc.stride(0),
        ws.a_acc.stride(1),
        ws.a_acc.stride(2),
        ws.p_max.stride(0),
        ws.p_max.stride(1),
        ws.p_out.stride(0),
        ws.p_out.stride(1),
        GQA=group,
        HEAD_DIM=ws.head_dim,
        A_SEGMENTS=ws.a_segments,
        HAS_SINKS=sinks is not None,
        KERNEL_MAX_PARTITIONS=ws.b_plan.max_partitions,
        num_warps=1,
    )


def _launch_scatter(ws: CascadeWorkspace, output):
    from aiter.ops.triton.attention.pa_decode_cascade import _cascade_scatter_kernel

    _cascade_scatter_kernel[(ws.max_singles, ws.num_q_heads)](
        ws.out_s,
        output,
        ws.singles_rows,
        ws.counts[2:3],
        ws.out_s.stride(0),
        output.stride(0),
        output.stride(1),
        HEAD_DIM=ws.head_dim,
        num_warps=1,
    )


def pa_decode_cascade(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
    *,
    sinks: torch.Tensor | None = None,
    workspace: CascadeWorkspace,
) -> None:
    """Decode rows decomposed by the workspace's cascade plan.

    Signature mirrors ``pa_decode`` (qlen-1 decode only); the cascade
    metadata must have been filled into ``workspace`` for this step (see
    the module docstring). ``output`` and ``query`` cover the full decode
    batch; grouped rows are written by the fold, single rows by the plain
    ``pa_decode`` path (plus a scatter).
    """
    ws = workspace
    if query.shape[-1] != ws.head_dim:
        raise ValueError("query head_dim does not match workspace")
    if not query.is_contiguous():
        raise NotImplementedError("cascade decode needs a contiguous query")
    if key_cache.dim() != 5 or key_cache.dtype != torch.bfloat16:
        raise NotImplementedError("cascade decode needs 5D BF16 K/V caches")
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        raise NotImplementedError("cascade decode needs contiguous K/V caches")
    # 1) Shared pass over all items (padded items are no-ops).
    _launch_shared_pass(ws, query, key_cache, value_cache, block_tables, softmax_scale)
    # 2) Tail pass: production pa_decode over members' private tails.
    torch.index_select(query, 0, ws.b_map.long(), out=ws.q_gather_b)
    plan_pa_decode(
        ws.tail_lens,
        ws.num_kv_heads,
        max_partitions=2,
        plan=ws.b_plan,
    )
    pa_decode(
        ws.out_b,
        ws.q_gather_b,
        key_cache,
        value_cache,
        ws.tail_lens,
        ws.tail_tables,
        softmax_scale,
        1,
        work_plan=ws.b_plan,
        max_logits=ws.p_max,
        exp_sums=ws.p_sum,
        temporary_output=ws.p_out,
    )
    # 3) Singles: exactly the plain pa_decode call on their full rows.
    torch.index_select(query, 0, ws.singles_rows.long(), out=ws.q_gather_s)
    plan_pa_decode(
        ws.singles_lens,
        ws.num_kv_heads,
        plan=ws.s_plan,
    )
    pa_decode(
        ws.out_s,
        ws.q_gather_s,
        key_cache,
        value_cache,
        ws.singles_lens,
        ws.singles_tables,
        softmax_scale,
        1,
        sinks=sinks,
        work_plan=ws.s_plan,
    )
    # 4) Fold members (shared + tail partials, sinks once), 5) scatter singles.
    _launch_fold(ws, output, sinks)
    _launch_scatter(ws, output)
