# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-stage MiniMax-M3 launch adapters for the optional ATOM kernels."""

import importlib

import triton


def minimax_m3_prefill_score(
    idx_q,
    index_cache,
    block_table,
    seq_lens,
    context,
    query,
    cu_seqlens_q,
    prefix_lens,
    score,
):
    """Compute MiniMax-M3 block scores with the optional ATOM Triton kernel.

    Inputs use ATOM's prefill layout. ``score`` is a preallocated FP32 tensor
    of shape [heads, total_q, ceil(context / 128)]. ``query`` is the largest
    query length. The result is ``score``; only causal blocks are defined.
    Requires the ATOM revision documented by the MiniMax-M3 capture suite.
    No scorer fallback or tensor allocation occurs in this wrapper.
    """
    op = importlib.import_module("atom.model_ops.minimax_m3.index_topk")
    heads, _, blocks = score.shape
    batch = cu_seqlens_q.numel() - 1
    tiles = triton.cdiv(query, op.SCORE_BLOCK_SIZE_Q)
    chunk = op._score_chunk_blocks(blocks, tiles, batch, heads, idx_q.device)
    warps, stages = op._prefill_score_launch_config(index_cache.dtype)
    op._index_block_score_kernel[(tiles, batch * heads, triton.cdiv(blocks, chunk))](
        idx_q,
        index_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        heads,
        idx_q.shape[-1],
        128**-0.5,
        chunk,
        *idx_q.stride(),
        *index_cache.stride(),
        *score.stride(),
        block_table.stride(0),
        BLOCK_SIZE_Q=op.SCORE_BLOCK_SIZE_Q,
        BLOCK_SIZE_K=op.SPARSE_BLOCK_SIZE,
        num_warps=warps,
        num_stages=stages,
    )
    return score


def minimax_m3_packed_topk(
    score, topk_idx, cu_seqlens_q, prefix_lens, block_table, emit_args, query
):
    """Select 16 blocks per row and emit the sparse KV table with ATOM.

    ``score`` has shape [heads, total_q, blocks]. ``topk_idx`` and the first
    two entries in ``emit_args`` are preallocated outputs. The remaining
    entries are table strides, as returned by ATOM's allocation helper.
    ``cu_seqlens_q`` and ``prefix_lens`` describe the ragged prefill rows.
    Return (topk_idx, sparse_table, sparse_lengths). Requires the pinned ATOM
    API and eight page-16 KV pages per 128-token block. Always use Triton.
    """
    op = importlib.import_module("atom.model_ops.minimax_m3.index_topk")
    op._launch_select(
        score,
        topk_idx,
        cu_seqlens_q,
        prefix_lens,
        cu_seqlens_q.numel() - 1,
        16,
        0,
        0,
        block_table,
        emit_args,
        score.shape[0],
        rows_per_req=query,
        decode_max_q=0,
        block_size_k=op._prefill_topk_block_size_k(score.shape[-1]),
        num_warps=op.PREFILL_TOPK_NUM_WARPS,
        emit=True,
        block_page_stride=op.PAGES_PER_SPARSE_BLOCK,
        n_valid_column_per_row=None,
        pages_per_block=op.PAGES_PER_SPARSE_BLOCK,
    )
    return topk_idx, emit_args[0], emit_args[1]
