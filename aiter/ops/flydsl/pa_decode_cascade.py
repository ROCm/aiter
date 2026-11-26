# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cascade (shared-prefix) mode for paged decode attention.

In agentic traffic, sibling requests share long block-aligned prefixes
(system prompt, tool schemas, a common plan). Prefix caching stores that
KV once, but a plain decode kernel streams it once per request. Cascade
attention streams every shared block once per *item* of up to
``MEMBERS_PER_ITEM`` siblings and merges the result in log-sum-exp space:

* **Shared pass**: one work item per item packs the item's members into
  one MFMA tile over the shared prefix and writes ``(m, l, acc)`` fp32
  partials per member and segment.
* **Tail pass**: production split-KV decode runs, unchanged, over every
  member's private tail (its block table shifted by the shared count).
  Two staging modes cover today's serving layouts: the vectorized 5D
  BF16 layout through FlyDSL ``pa_decode`` (``kv_layout="vec5d"``) and
  the plain vLLM layout through AITER's Triton v2 decode tile kernel
  (``kv_layout="plain4d"``). Singles run the full context through the
  same layout's plain path, exactly as a stock backend would.
* **Fold** (a small Triton kernel): merges shared and tail partials,
  applying optional attention sinks exactly once.

Row spaces
----------
Metadata is built over *request* rows (where block tables and sequence
lengths live); query and output are indexed by *token* rows. On a
pure-decode step they coincide. On a mixed (prefill + decode) step the
decode rows are a token-space subset, and the workspace's ``tok_of_b`` /
``tok_of_s`` maps plus ``b_of_item_slot`` keep the spaces straight:
group members keep Q/output in token space while KV stays in request
space -- the same three-index-space routing the serving backends use.

API contract
------------
:func:`plan_pa_decode_cascade` allocates a :class:`CascadeWorkspace`
once per capture bucket. A grouping planner (see
``aiter.ops.flydsl.pa_decode_cascade_plan``) fills the metadata per
step; :func:`pa_decode_cascade` runs the three passes. Every launch
grid is static with device counts and padded slots handled inside the
kernels, so the whole op (plan refreshes included) captures in one CUDA
graph and replays under regrouping.

Scope: gfx942/gfx950 BF16 K/V, page 16, head dims 64..512 (multiples of
64), 1..8 query heads per KV head, qlen-1 decode rows, sinks. No FP8 KV,
sliding window, ALiBi, MTP/verify queries.
"""

from dataclasses import dataclass

import os

import torch
import triton.language as tl

from aiter.ops.flydsl.kernels.pa_decode_plan import PADecodePlan, plan_pa_decode
from aiter.ops.flydsl.pa_decode import pa_decode

MEMBERS_PER_ITEM = 4
N_COUNTS = 3  # (n_items, n_members, n_singles)
V2_PARTITION_SZ = 1024


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
    a_segments: int  # segments per item per range (prefix and tails)
    kv_layout: str  # "vec5d" | "plain4d"
    b_partitions: int  # plain4d static partition cap per tail
    dtype: torch.dtype

    # -- Group metadata, device int32, written per step by the planner. --
    items: torch.Tensor  # [max_items, 2 + members_per_item] (sblk, count, rows...)
    b_of_item_slot: torch.Tensor  # [max_items, members_per_item] member's B row
    b_map: torch.Tensor  # [max_b_rows] request row of tail-pass row
    tok_of_b: torch.Tensor  # [max_b_rows] token/Q row of tail-pass row
    b_item: torch.Tensor  # [max_b_rows] item index
    b_slot: torch.Tensor  # [max_b_rows] member slot within the item
    tail_lens: torch.Tensor  # [max_b_rows] private tail lengths
    tail_tables: torch.Tensor  # [max_b_rows, table_width] shifted block tables
    bend_zeros: torch.Tensor  # [1] int32 zeros used to null B partition counts
    singles_rows: torch.Tensor  # [max_singles] request rows
    tok_of_s: torch.Tensor  # [max_singles] token/Q rows
    singles_lens: torch.Tensor  # [max_singles]
    singles_tables: torch.Tensor  # [max_singles, table_width]
    counts: torch.Tensor  # [3] (n_items, n_members, n_singles)

    # -- Scratch, device, reused per step. --
    a_m: torch.Tensor  # fp32 [max_items, segments, members, num_q_heads]
    a_l: torch.Tensor  # fp32 [max_items, segments, members, num_q_heads]
    a_acc: torch.Tensor  # fp32 [max_items, segments, members, num_q_heads, head_dim]
    # vec5d staging: [kvh, capacity, group(, head_dim)]
    p_max: torch.Tensor
    p_sum: torch.Tensor
    p_out: torch.Tensor
    # plain4d staging: [b, kvh, b_partitions, group(, head_dim)]
    p2_max: torch.Tensor
    p2_sum: torch.Tensor
    p2_out: torch.Tensor
    q_gather_b: torch.Tensor  # dtype [max_b_rows, num_q_heads, head_dim]
    q_gather_s: torch.Tensor  # dtype [max_singles, num_q_heads, head_dim]
    out_b: torch.Tensor  # dtype [max_b_rows, num_q_heads, head_dim] (discarded)
    out_s: torch.Tensor  # dtype [max_singles, num_q_heads, head_dim]

    b_plan: PADecodePlan | None  # vec5d tail-pass plan (max_partitions = 2)
    s_plan: PADecodePlan | None  # vec5d singles plan


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
    kv_layout: str = "vec5d",
    dtype: torch.dtype = torch.bfloat16,
    singles_max_partitions: int | None = None,
) -> CascadeWorkspace:
    """Allocate a :class:`CascadeWorkspace` and its tail/singles plans.

    Call once per capture bucket (outside graph capture).
    ``max_b_rows`` defaults to ``max_items * members_per_item``.
    ``table_width`` is ``block_tables.shape[1]``. With
    ``kv_layout="vec5d"`` the tail plan is FlyDSL ``pa_decode`` with
    ``max_partitions=2`` (always taking the packed-partials path the
    fold reads); with ``"plain4d"`` the tail is AITER's Triton v2 tile
    kernel and ``b_partitions`` is its static partition cap, sized to
    cover the longest possible tail.
    """
    if kv_layout not in ("vec5d", "plain4d"):
        raise ValueError(f"kv_layout must be vec5d or plain4d, got {kv_layout}")
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
    b_partitions = max(1, -((-table_width * block_size) // V2_PARTITION_SZ))
    vec = kv_layout == "vec5d"
    b_plan = (
        plan_pa_decode(b_lens, num_kv_heads, max_partitions=2, max_batch_size=max_b_rows)
        if vec
        else None
    )
    s_plan = (
        plan_pa_decode(
            s_lens,
            num_kv_heads,
            max_partitions=singles_max_partitions,
            max_batch_size=max_singles,
        )
        if vec
        else None
    )
    cap = b_plan.capacity if vec else 0
    f32_empty = (0,)
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
        kv_layout=kv_layout,
        b_partitions=b_partitions,
        dtype=dtype,
        items=t((max_items, 2 + members_per_item), **int32).zero_(),
        b_of_item_slot=t((max_items, members_per_item), **int32).zero_(),
        b_map=t((max_b_rows,), **int32).zero_(),
        tok_of_b=t((max_b_rows,), **int32).zero_(),
        b_item=t((max_b_rows,), **int32).zero_(),
        b_slot=t((max_b_rows,), **int32).zero_(),
        tail_lens=b_lens,
        bend_zeros=torch.zeros(max_b_rows, dtype=torch.int32, device=dev),
        tail_tables=t((max_b_rows, table_width), **int32),
        singles_rows=t((max_singles,), **int32).zero_(),
        tok_of_s=t((max_singles,), **int32).zero_(),
        singles_lens=s_lens,
        singles_tables=t((max_singles, table_width), **int32),
        counts=t((N_COUNTS,), **int32),
        # Segment slots: [0, a_segments) cover the shared prefix and
        # [a_segments, 2*a_segments) cover member tails.
        a_m=t((max_items, 2 * a_segments, members_per_item, num_q_heads), **f32),
        a_l=t((max_items, 2 * a_segments, members_per_item, num_q_heads), **f32),
        a_acc=t(
            (max_items, 2 * a_segments, members_per_item, num_q_heads, head_dim), **f32
        ),
        p_max=t((num_kv_heads, cap, group), **f32) if vec else t(f32_empty, **f32),
        p_sum=t((num_kv_heads, cap, group), **f32) if vec else t(f32_empty, **f32),
        p_out=t((num_kv_heads, cap, group, head_dim), **out) if vec else t(f32_empty, **out),
        p2_max=(
            t((max_b_rows, num_kv_heads, b_partitions, group), **f32)
            if not vec
            else t(f32_empty, **f32)
        ),
        p2_sum=(
            t((max_b_rows, num_kv_heads, b_partitions, group), **f32)
            if not vec
            else t(f32_empty, **f32)
        ),
        p2_out=(
            t((max_b_rows, num_kv_heads, b_partitions, group, head_dim), **out)
            if not vec
            else t(f32_empty, **out)
        ),
        q_gather_b=t((max_b_rows, num_q_heads, head_dim), **out),
        q_gather_s=t((max_singles, num_q_heads, head_dim), **out),
        out_b=t((max_b_rows, num_q_heads, head_dim), **out),
        out_s=t((max_singles, num_q_heads, head_dim), **out),
        b_plan=b_plan,
        s_plan=s_plan,
    )


def _launch_shared_pass(
    ws: CascadeWorkspace, key_cache, value_cache, block_tables, softmax_scale
):
    from aiter.ops.triton.attention.pa_decode_cascade import _cascade_shared_kernel

    group = ws.num_q_heads // ws.num_kv_heads
    block_mg = max(16, next_pow2(ws.members_per_item * group))
    page = ws.block_size
    plain = ws.kv_layout == "plain4d"
    if plain:
        k_strides = (key_cache.stride(0), key_cache.stride(1))
        v_strides = (value_cache.stride(0), value_cache.stride(1))
    else:
        k_strides = (ws.num_kv_heads * ws.head_dim * page, ws.head_dim * page)
        v_strides = k_strides
    _cascade_shared_kernel[(ws.max_items, ws.a_segments, ws.num_kv_heads)](
        ws.q_gather_b,
        key_cache,
        value_cache,
        block_tables,
        ws.items,
        ws.b_of_item_slot,
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
        K_BLK_STRIDE=k_strides[0],
        K_HEAD_STRIDE=k_strides[1],
        V_BLK_STRIDE=v_strides[0],
        V_HEAD_STRIDE=v_strides[1],
        PLAIN_LAYOUT=plain,
        K_TOK_STRIDE=key_cache.stride(2) if plain else 0,
        K_DIM_STRIDE=key_cache.stride(3) if plain else 0,
        V_TOK_STRIDE=value_cache.stride(2) if plain else 0,
        V_DIM_STRIDE=value_cache.stride(3) if plain else 0,
        num_warps=4,
        num_stages=1,
    )


def _launch_tail_pass(ws: CascadeWorkspace, key_cache, value_cache, block_tables, softmax_scale):
    """Member-tail engine: the same packed segment kernel covering each
    member row's private tail [shared prefix end, own context end), with
    partials written to the alloc_extended segment slots."""
    from aiter.ops.triton.attention.pa_decode_cascade import _cascade_tail_kernel

    group = ws.num_q_heads // ws.num_kv_heads
    block_mg = max(16, next_pow2(ws.members_per_item * group))
    page = ws.block_size
    plain = ws.kv_layout == "plain4d"
    if plain:
        k_strides = (key_cache.stride(0), key_cache.stride(1))
        v_strides = (value_cache.stride(0), value_cache.stride(1))
    else:
        k_strides = (ws.num_kv_heads * ws.head_dim * page, ws.head_dim * page)
        v_strides = k_strides
    _cascade_tail_kernel[(ws.max_items, ws.a_segments, ws.num_kv_heads)](
        ws.q_gather_b,
        key_cache,
        value_cache,
        block_tables,
        ws.items,
        ws.b_of_item_slot,
        ws.a_m,
        ws.a_l,
        ws.a_acc,
        ws.tail_lens,
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
        K_BLK_STRIDE=k_strides[0],
        K_HEAD_STRIDE=k_strides[1],
        V_BLK_STRIDE=v_strides[0],
        V_HEAD_STRIDE=v_strides[1],
        PLAIN_LAYOUT=plain,
        K_TOK_STRIDE=key_cache.stride(2) if plain else 0,
        K_DIM_STRIDE=key_cache.stride(3) if plain else 0,
        V_TOK_STRIDE=value_cache.stride(2) if plain else 0,
        V_DIM_STRIDE=value_cache.stride(3) if plain else 0,
        SEG_BASE=ws.a_segments,
        num_warps=4,
        num_stages=1,
    )


def _launch_v2_tails(ws: CascadeWorkspace, key_cache, value_cache, softmax_scale):
    from aiter.ops.triton._triton_kernels.attention.pa_decode import (
        _paged_attn_decode_v2_w_dot_kernel,
    )
    from aiter.ops.triton.attention.pa_decode import _get_launch_config
    import triton.language as tl

    group = ws.num_q_heads // ws.num_kv_heads
    parts = ws.b_partitions
    launch = _get_launch_config("v2_w_dot", key_cache.shape[2]) or {}
    _paged_attn_decode_v2_w_dot_kernel[
        (ws.max_b_rows, ws.num_kv_heads, parts)
    ](
        ws.p2_sum,
        ws.p2_max,
        ws.p2_out,
        ws.q_gather_b,
        key_cache,
        value_cache,
        ws.tail_tables,
        ws.tail_lens,
        softmax_scale,
        1.0,  # k_scale
        1.0,  # v_scale
        None,  # alibi_slopes
        ws.p2_sum.stride(0),
        ws.p2_sum.stride(1),
        ws.p2_sum.stride(2),
        ws.p2_out.stride(0),
        ws.p2_out.stride(1),
        ws.p2_out.stride(2),
        ws.p2_out.stride(3),
        ws.q_gather_b.stride(0),
        ws.q_gather_b.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        ws.tail_tables.stride(0),
        compute_type=(tl.bfloat16 if key_cache.dtype == torch.bfloat16 else tl.float16),
        HEAD_SZ=ws.head_dim,
        HEAD_SZ_POW2=next_pow2(ws.head_dim),
        QUERY_GRP_SZ=group,
        QUERY_GRP_SZ_POW2=16 if group <= 16 else next_pow2(group),
        KV_BLK_SZ=ws.block_size,
        KV_BLK_SZ_POW2=next_pow2(ws.block_size),
        SEQ_PARTITION_SZ=V2_PARTITION_SZ,
        **launch,
    )


def _launch_v2_staging_kernels(
    ws,
    key_cache,
    value_cache,
    q_gather,
    tables,
    lens,
    out_s,
    p_max,
    p_sum,
    p_out,
    softmax_scale,
):
    """v2 tile + reduce kernels into resident staging (no per-call scratch).

    Bit-identical to ``paged_attn_decode_v2`` for the GQA shapes cascade
    uses (verified by an equality test): only the wrapper's per-call
    ``exp_sums``/``max_logits``/``tmp_output`` allocations are replaced
    with resident workspace buffers.
    """
    from aiter.ops.triton._triton_kernels.attention.pa_decode import (
        _paged_attn_decode_v2_w_dot_kernel,
        _paged_attn_decode_v2_w_dot_reduce_kernel,
    )
    from aiter.ops.triton.attention.pa_decode import _get_launch_config
    import triton.language as tl

    group = ws.num_q_heads // ws.num_kv_heads
    parts = ws.b_partitions
    n_rows = q_gather.shape[0]
    launch = _get_launch_config("v2_w_dot", key_cache.shape[2]) or {}
    _paged_attn_decode_v2_w_dot_kernel[(n_rows, ws.num_kv_heads, parts)](
        p_sum,
        p_max,
        p_out,
        q_gather,
        key_cache,
        value_cache,
        tables,
        lens,
        softmax_scale,
        1.0,
        1.0,
        None,
        p_sum.stride(0),
        p_sum.stride(1),
        p_sum.stride(2),
        p_out.stride(0),
        p_out.stride(1),
        p_out.stride(2),
        p_out.stride(3),
        q_gather.stride(0),
        q_gather.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        tables.stride(0),
        compute_type=(tl.bfloat16 if key_cache.dtype == torch.bfloat16 else tl.float16),
        HEAD_SZ=ws.head_dim,
        HEAD_SZ_POW2=next_pow2(ws.head_dim),
        QUERY_GRP_SZ=group,
        QUERY_GRP_SZ_POW2=16 if group <= 16 else next_pow2(group),
        KV_BLK_SZ=ws.block_size,
        KV_BLK_SZ_POW2=next_pow2(ws.block_size),
        SEQ_PARTITION_SZ=V2_PARTITION_SZ,
        **launch,
    )
    _paged_attn_decode_v2_w_dot_reduce_kernel[(n_rows, ws.num_kv_heads, 1)](
        out_s,
        p_sum,
        p_max,
        p_out,
        lens,
        out_s.stride(0),
        out_s.stride(1),
        p_sum.stride(0),
        p_sum.stride(1),
        p_sum.stride(2),
        p_out.stride(0),
        p_out.stride(1),
        p_out.stride(2),
        p_out.stride(3),
        HEAD_SZ=ws.head_dim,
        HEAD_SZ_POW2=next_pow2(ws.head_dim),
        QUERY_GRP_SZ=group,
        QUERY_GRP_SZ_POW2=16 if group <= 16 else next_pow2(group),
        SEQ_PARTITION_SZ=V2_PARTITION_SZ,
        MAX_NUM_SEQ_PARTITIONS_POW2=max(1, 2 ** (parts - 1).bit_length()),
    )


def _launch_fold(ws: CascadeWorkspace, output, sinks):
    from aiter.ops.triton.attention.pa_decode_cascade import _cascade_fold_kernel

    group = ws.num_q_heads // ws.num_kv_heads
    vec = ws.kv_layout == "vec5d"
    if vec:
        pmax, psum, pout = ws.p_max, ws.p_sum, ws.p_out
        info = ws.b_plan.reduce_info
        ps_seq, ps_head, ps_part = 0, pmax.stride(0), pmax.stride(1)
        po_seq, po_head, po_part = 0, pout.stride(0), pout.stride(1)
        parts = ws.b_plan.max_partitions
    else:
        pmax, psum, pout = ws.p2_max, ws.p2_sum, ws.p2_out
        # Tail-side partials now come from the A engine, so every B row
        # has zero v2 partitions (keeps the staging inert by contract).
        info = ws.bend_zeros
        ps_seq, ps_head, ps_part = pmax.stride(0), pmax.stride(1), pmax.stride(2)
        po_seq, po_head, po_part = pout.stride(0), pout.stride(1), pout.stride(2)
        parts = ws.b_partitions
    _cascade_fold_kernel[(ws.max_b_rows, ws.num_q_heads)](
        output,
        ws.a_m,
        ws.a_l,
        ws.a_acc,
        ws.tok_of_b,
        ws.b_item,
        ws.b_slot,
        ws.counts[1:2],
        pmax,
        psum,
        pout,
        info,
        sinks if sinks is not None else ws.a_m,  # inert when HAS_SINKS=False
        output.stride(0),
        output.stride(1),
        ws.a_m.stride(0),
        ws.a_m.stride(1),
        ws.a_m.stride(2),
        ws.a_acc.stride(0),
        ws.a_acc.stride(1),
        ws.a_acc.stride(2),
        ps_seq,
        ps_head,
        ps_part,
        po_seq,
        po_head,
        po_part,
        GQA=group,
        HEAD_DIM=ws.head_dim,
        A_SEGMENTS=ws.a_segments,
        STAGING_FLYDSL=vec,
        PARTITION_SZ=V2_PARTITION_SZ,
        HAS_SINKS=sinks is not None,
        KERNEL_MAX_PARTITIONS=parts,
        num_warps=1,
    )


def _launch_scatter(ws: CascadeWorkspace, output):
    from aiter.ops.triton.attention.pa_decode_cascade import _cascade_scatter_kernel

    _cascade_scatter_kernel[(ws.max_singles, ws.num_q_heads)](
        ws.out_s,
        output,
        ws.tok_of_s,
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

    Signature mirrors a decode op's (qlen-1 rows only); the cascade
    metadata must have been filled into ``workspace`` for this step (see
    the module docstring). ``output`` and ``query`` rows are token rows;
    ``context_lengths`` and ``block_tables`` are request rows.
    """
    ws = workspace
    if query.shape[-1] != ws.head_dim:
        raise ValueError("query head_dim does not match workspace")
    vec = ws.kv_layout == "vec5d"
    if vec:
        if key_cache.dim() != 5 or key_cache.dtype != torch.bfloat16:
            raise NotImplementedError("vec5d cascade needs 5D BF16 K/V caches")
        if not key_cache.is_contiguous() or not value_cache.is_contiguous():
            raise NotImplementedError("vec5d cascade needs contiguous K/V caches")
    elif key_cache.dim() < 4:
        raise NotImplementedError("plain4d cascade needs 4D K/V caches")
    prof = None
    if os.environ.get("AITER_CASCADE_PROF"):
        import time

        prof = {"fill_start": time.perf_counter()}
    # Query gathers first: every later pass reads q_gather rows.
    torch.index_select(query, 0, ws.tok_of_b.long(), out=ws.q_gather_b)
    # 1) Shared pass over all items (padded items are no-ops).
    _launch_shared_pass(ws, key_cache, value_cache, block_tables, softmax_scale)
    if prof is not None:
        torch.cuda.synchronize()
        prof["shared"] = time.perf_counter()
    if prof is not None:
        torch.cuda.synchronize()
        prof["pre_tail"] = time.perf_counter()
    folded = False
    if vec:
        # 2) Tail pass: member-tail engine over each item's private tail.
        _launch_tail_pass(ws, key_cache, value_cache, block_tables, softmax_scale)
        # 3) Singles: exactly the plain pa_decode call on their full rows.
        torch.index_select(query, 0, ws.tok_of_s.long(), out=ws.q_gather_s)
        plan_pa_decode(ws.singles_lens, ws.num_kv_heads, plan=ws.s_plan)
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
    else:
        # 2) Tail pass: member-tail engine over each item's private tail.
        _launch_tail_pass(ws, key_cache, value_cache, block_tables, softmax_scale)
        if prof is not None:
            torch.cuda.synchronize()
            prof["tails_done"] = time.perf_counter()
        # Fold members now, before the singles kernels reuse the staging
        # buffers the tails wrote.
        if prof is not None:
            torch.cuda.synchronize()
            prof["pre_fold"] = time.perf_counter()
        _launch_fold(ws, output, sinks)
        folded = True
        if prof is not None:
            torch.cuda.synchronize()
            prof["fold_done"] = time.perf_counter()
        # 3) Singles: full production Triton v2 decode call.
        from aiter.ops.triton.attention.pa_decode import paged_attn_decode_v2

        torch.index_select(query, 0, ws.tok_of_s.long(), out=ws.q_gather_s)
        if sinks is None and ws.max_b_rows >= ws.max_singles:
            _launch_v2_staging_kernels(
                ws,
                key_cache,
                value_cache,
                ws.q_gather_s,
                ws.singles_tables,
                ws.singles_lens,
                ws.out_s,
                ws.p2_max,
                ws.p2_sum,
                ws.p2_out,
                softmax_scale,
            )
        else:
            paged_attn_decode_v2(
            ws.out_s,
            ws.q_gather_s,
            key_cache,
            value_cache,
            ws.singles_tables,
            ws.singles_lens,
            ws.table_width * ws.block_size,
            tl.bfloat16 if key_cache.dtype == torch.bfloat16 else tl.float16,
            ws.num_kv_heads,
            softmax_scale,
            None,
            1.0,
            1.0,
            max_num_partitions=max(
                1, -((-ws.table_width * ws.block_size) // V2_PARTITION_SZ)
            ),
        )
    # 4) Fold members (shared + tail partials, sinks once), 5) scatter singles.
    if not folded:
        _launch_fold(ws, output, sinks)
    _launch_scatter(ws, output)
    if prof is not None:
        torch.cuda.synchronize()
        fill_end = time.perf_counter()
        t0 = prof
        td = t0.get("tails_done", t0["pre_tail"])
        fd = t0.get("fold_done", t0["pre_fold"])
        td_pre = (td - t0['pre_tail']) * 1000
        fd_pre = fd
        print(
            f"[cascade-prof] shared={(t0['shared']-t0['fill_start'])*1000:6.2f} "
            f"tail={(t0['pre_tail']-t0['shared'])*1000:6.2f} singles+scat={(t0['pre_fold']-t0['pre_tail'])*1000:6.2f} "
            f"fold={(fill_end-t0['pre_fold'])*1000:6.2f}"
        )
