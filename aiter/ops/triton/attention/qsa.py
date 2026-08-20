# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen Sparse Attention operators for BF16 paged KV caches."""

from __future__ import annotations

import math

import torch
import triton

from aiter.ops.topk import top_k_per_row_prefill
from aiter.ops.triton._triton_kernels.attention.qsa_expand_indices import (
    _qsa_expand_block_indices_kernel,
)
from aiter.ops.triton._triton_kernels.attention.qsa_paged_mqa_logits import (
    _qsa_paged_mqa_logits_kernel,
)
from aiter.ops.triton._triton_kernels.attention.qsa_sparse_paged_gqa import (
    _qsa_sparse_paged_gqa_kernel,
)

_DEFAULT_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024


def _require_hip_tensor(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA/HIP tensor")


def _validate_integer_vector(
    name: str,
    tensor: torch.Tensor,
    length: int | None = None,
) -> None:
    _require_hip_tensor(name, tensor)
    if tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a one-dimensional int32/int64 tensor")
    if length is not None and tensor.shape[0] != length:
        raise ValueError(f"{name} must contain {length} entries")


def qsa_paged_mqa_logits(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    compress_ratio: int = 4,
    num_columns: int | None = None,
    score_divisor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QSA ReLU-sum scores directly from a compressed paged cache.

    ``q`` is ``[tokens, index_heads, head_dim]`` and ``compressed_k_cache`` is
    ``[pages, page_size, 1, head_dim]``. The result contains FP32 logits and the
    number of causally visible complete compression groups for every token.
    """
    _require_hip_tensor("q", q)
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("q must have shape [tokens, heads, head_dim]")
    if q.dtype != torch.bfloat16:
        raise ValueError(f"q must be bfloat16, got {q.dtype}")
    _require_hip_tensor("compressed_k_cache", compressed_k_cache)
    if (
        compressed_k_cache.ndim != 4
        or compressed_k_cache.shape[2] != 1
        or compressed_k_cache.shape[3] != q.shape[2]
    ):
        raise ValueError(
            "compressed_k_cache must have shape [pages, page_size, 1, head_dim]"
        )
    if compressed_k_cache.dtype != q.dtype:
        raise ValueError("q and compressed_k_cache must have the same dtype")
    _require_hip_tensor("page_table", page_table)
    if page_table.ndim != 2 or page_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("page_table must be a two-dimensional integer tensor")
    _validate_integer_vector("token_to_request", token_to_request, q.shape[0])
    _validate_integer_vector("query_positions", query_positions, q.shape[0])
    _validate_integer_vector("context_lens", context_lens, page_table.shape[0])
    if q.shape[0] and (
        not all(compressed_k_cache.shape[:2]) or not all(page_table.shape)
    ):
        raise ValueError("paged QSA cache and page_table must be nonempty")
    if compress_ratio <= 0:
        raise ValueError("compress_ratio must be positive")

    divisor = math.sqrt(q.shape[2]) if score_divisor is None else score_divisor
    if divisor <= 0:
        raise ValueError("score_divisor must be positive")
    capacity = page_table.shape[1] * compressed_k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if columns < 0 or columns > capacity:
        raise ValueError(f"num_columns must be in [0, {capacity}]")

    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_groups = torch.zeros(q.shape[0], dtype=torch.int32, device=q.device)
    if q.shape[0] == 0 or columns == 0:
        return logits, visible_groups

    block_n = 32
    _qsa_paged_mqa_logits_kernel[
        (q.shape[0], triton.cdiv(columns, block_n))
    ](
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        visible_groups,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        compressed_k_cache.stride(0),
        compressed_k_cache.stride(1),
        compressed_k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        compressed_k_cache.shape[0],
        page_table.shape[0],
        float(divisor),
        PAGE_SIZE=compressed_k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        COMPRESS_RATIO=compress_ratio,
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(q.shape[2]),
        num_warps=4,
    )
    return logits, visible_groups


def qsa_expand_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    token_to_request: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand compressed group indices and append each query's causal tail."""
    _require_hip_tensor("block_indices", block_indices)
    if block_indices.ndim != 2 or block_indices.dtype != torch.int32:
        raise ValueError("block_indices must be a two-dimensional int32 tensor")
    _validate_integer_vector(
        "query_positions", query_positions, block_indices.shape[0]
    )
    _validate_integer_vector(
        "token_to_request", token_to_request, block_indices.shape[0]
    )
    _validate_integer_vector("context_lens", context_lens)
    if context_lens.shape[0] == 0:
        raise ValueError("context_lens must be nonempty")
    if compress_ratio <= 0 or token_topk <= 0:
        raise ValueError("compress_ratio and token_topk must be positive")
    if token_topk % compress_ratio:
        raise ValueError("token_topk must be divisible by compress_ratio")

    block_topk = token_topk // compress_ratio
    if block_indices.shape[1] != block_topk:
        raise ValueError(f"block_indices must have {block_topk} columns")
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.shape != (block_indices.shape[0], output_width):
        raise ValueError("out has an invalid shape")
    elif out.dtype != torch.int32 or not out.is_cuda:
        raise ValueError("out must be an int32 CUDA/HIP tensor")
    if block_indices.shape[0] == 0:
        return out

    block_n = 256
    _qsa_expand_block_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, block_n))
    ](
        block_indices,
        query_positions,
        context_lens,
        token_to_request,
        out,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        block_indices.shape[0],
        context_lens.shape[0],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        OUTPUT_WIDTH=output_width,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out


def qsa_select_paged_tokens(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    token_topk: int,
    compress_ratio: int = 4,
    out: torch.Tensor | None = None,
    *,
    stable: bool = False,
    logits_workspace_bytes: int = _DEFAULT_LOGITS_WORKSPACE_BYTES,
) -> torch.Tensor:
    """Run paged scoring, AITER radix top-k, and group-to-token expansion."""
    if token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("token_topk must be positive and divisible by compress_ratio")
    if logits_workspace_bytes <= 0:
        raise ValueError("logits_workspace_bytes must be positive")

    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    elif out.shape != (rows, output_width):
        raise ValueError("out has an invalid shape")
    elif out.dtype != torch.int32 or not out.is_cuda:
        raise ValueError("out must be an int32 CUDA/HIP tensor")
    if rows == 0:
        return out

    columns = page_table.shape[1] * compressed_k_cache.shape[1]
    block_topk = token_topk // compress_ratio
    if block_topk > columns:
        raise ValueError("compressed top-k exceeds paged-cache capacity")
    rows_per_chunk = max(1, logits_workspace_bytes // max(columns * 4, 1))
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        logits, visible_groups = qsa_paged_mqa_logits(
            q[row_slice],
            compressed_k_cache,
            page_table,
            token_to_request[row_slice],
            query_positions[row_slice],
            context_lens,
            compress_ratio,
        )
        selected_groups = torch.empty(
            (row_end - row_start, block_topk),
            dtype=torch.int32,
            device=q.device,
        )
        row_starts = torch.zeros_like(visible_groups)
        topk_args = (
            logits,
            row_starts,
            visible_groups,
            selected_groups,
            None,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            block_topk,
        )
        if stable:
            top_k_per_row_prefill(*topk_args, stable=True)
        else:
            # Omitting the optional argument also supports AITER releases from
            # before deterministic radix top-k was added.
            top_k_per_row_prefill(*topk_args)
        qsa_expand_block_indices(
            selected_groups,
            query_positions[row_slice],
            context_lens,
            token_to_request[row_slice],
            compress_ratio,
            token_topk,
            out[row_slice],
        )
    return out


def qsa_sparse_paged_gqa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply sparse grouped-query attention over token-indexed paged BF16 K/V."""
    _require_hip_tensor("q", q)
    if q.ndim != 3 or q.dtype != torch.bfloat16:
        raise ValueError("q must be bfloat16 [tokens, query_heads, head_dim]")
    _require_hip_tensor("k_cache", k_cache)
    _require_hip_tensor("v_cache", v_cache)
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("K/V caches must have matching [pages, page, heads, dim]")
    if k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        raise ValueError("q, k_cache, and v_cache must have the same dtype")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("query heads must form equal groups over KV heads")
    _require_hip_tensor("logical_indices", logical_indices)
    if (
        logical_indices.ndim != 2
        or logical_indices.shape[0] != q.shape[0]
        or logical_indices.shape[1] <= 0
        or logical_indices.dtype != torch.int32
    ):
        raise ValueError("logical_indices must be int32 [tokens, selection_width]")
    _require_hip_tensor("block_table", block_table)
    if block_table.ndim != 2 or block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("block_table must be a two-dimensional integer tensor")
    _validate_integer_vector("token_to_request", token_to_request, q.shape[0])
    if q.shape[0] and (not all(k_cache.shape[:3]) or not all(block_table.shape)):
        raise ValueError("paged K/V caches and block_table must be nonempty")

    scale = q.shape[2] ** -0.5 if softmax_scale is None else softmax_scale
    if scale <= 0:
        raise ValueError("softmax_scale must be positive")
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or not out.is_cuda:
        raise ValueError("out must match q")
    if q.shape[0] == 0:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = max(16, triton.next_power_of_2(group_size))
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    _qsa_sparse_paged_gqa_kernel[(q.shape[0], k_cache.shape[2])](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_request,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        logical_indices.stride(0),
        logical_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        float(scale),
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        NUM_KV_HEADS=k_cache.shape[2],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=16,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return out
