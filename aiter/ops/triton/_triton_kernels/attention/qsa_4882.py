# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pinned AITER #4882 Triton QSA operators (no Gluon).

Pin: ROCm/aiter#4882 @ 150c7bc12b45ced1529a5512bf4ac30ecf9f35ba
(parent 2462d5b6427b71619f2d6f09e68a9ce3ea2e9d2f). Portable Triton launchers
from ``aiter/ops/triton/attention/qsa.py`` with Gluon dispatch stripped.
HIP top-k matches the live AMD harness: ``top_k_per_row_decode`` when
``module_top_k_per_row.so`` exists, else oracle smaller-index tie-break.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import triton

from aiter.ops.flydsl.kernels.qsa.oracle import qsa_topk_blocks
from aiter.ops.topk import _hip_top_k_per_row_decode
from aiter.ops.triton._triton_kernels.attention.qsa_expand_indices import (
    _qsa_expand_block_indices_kernel,
)
from aiter.ops.triton._triton_kernels.attention.qsa_paged_mqa_logits import (
    _qsa_paged_mqa_logits_kernel,
)
from aiter.ops.triton._triton_kernels.attention.qsa_sparse_paged_gqa import (
    _qsa_sparse_paged_gqa_kernel,
)

AITER_4882_QSA_PIN = "150c7bc12b45ced1529a5512bf4ac30ecf9f35ba"
_DEFAULT_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
_HIP_TOPK_OK: bool | None = None


def _hip_topk_available() -> bool:
    global _HIP_TOPK_OK
    if _HIP_TOPK_OK is not None:
        return _HIP_TOPK_OK
    import aiter as _aiter

    so = Path(_aiter.__file__).resolve().parent / "jit" / "module_top_k_per_row.so"
    _HIP_TOPK_OK = so.is_file()
    return _HIP_TOPK_OK


def _topk_per_row_amd(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    blocks: torch.Tensor,
    k: int,
) -> None:
    if _hip_topk_available():
        _hip_top_k_per_row_decode(
            logits,
            1,
            visible_blocks,
            blocks,
            blocks.shape[0],
            logits.stride(0),
            logits.stride(1),
            k,
            False,
            None,
        )
        return
    if not getattr(_topk_per_row_amd, "_warned", False):
        import aiter as _aiter

        _aiter.logger.warning(
            "HIP module_top_k_per_row.so missing; #4882 Triton select uses "
            "oracle tie-break on paged MQA logits (not the production HIP kernel)"
        )
        _topk_per_row_amd._warned = True
    n_col = logits.shape[1]
    cols = torch.arange(n_col, device=logits.device)
    masked = logits.masked_fill(
        cols.unsqueeze(0) >= visible_blocks.unsqueeze(1), float("-inf")
    )
    blocks.copy_(qsa_topk_blocks(masked, k))


def _require_hip_tensor(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA/HIP tensor")


def _validate_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_integer_vector(
    name: str, tensor: torch.Tensor, length: int | None = None
) -> None:
    _require_hip_tensor(name, tensor)
    if tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a one-dimensional int32/int64 tensor")
    if length is not None and tensor.shape[0] != length:
        raise ValueError(f"{name} must contain {length} entries")
    if tensor.stride(0) != 1:
        raise ValueError(f"{name} must be contiguous")


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
    """Triton paged ReLU-sum scores. Cache is ``[pages, page_size, 1, head_dim]``."""
    _validate_positive_integer("compress_ratio", compress_ratio)
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
    _qsa_paged_mqa_logits_kernel[(q.shape[0], triton.cdiv(columns, block_n))](
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
    _validate_positive_integer("compress_ratio", compress_ratio)
    _validate_positive_integer("token_topk", token_topk)
    _require_hip_tensor("block_indices", block_indices)
    if block_indices.ndim != 2 or block_indices.dtype != torch.int32:
        raise ValueError("block_indices must be a two-dimensional int32 tensor")
    _validate_integer_vector("query_positions", query_positions, block_indices.shape[0])
    _validate_integer_vector(
        "token_to_request", token_to_request, block_indices.shape[0]
    )
    _validate_integer_vector("context_lens", context_lens)
    if context_lens.shape[0] == 0:
        raise ValueError("context_lens must be nonempty")
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
    logits_workspace_bytes: int = _DEFAULT_LOGITS_WORKSPACE_BYTES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paged MQA + HIP/oracle top-k + expand. Returns ``(indices, block_ids)``."""
    _validate_positive_integer("token_topk", token_topk)
    _validate_positive_integer("compress_ratio", compress_ratio)
    _validate_positive_integer("logits_workspace_bytes", logits_workspace_bytes)
    if token_topk % compress_ratio:
        raise ValueError("token_topk must be divisible by compress_ratio")

    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    elif out.shape != (rows, output_width):
        raise ValueError("out has an invalid shape")
    elif out.dtype != torch.int32 or not out.is_cuda:
        raise ValueError("out must be an int32 CUDA/HIP tensor")
    block_topk = token_topk // compress_ratio
    if rows == 0:
        return out, torch.empty((0, block_topk), dtype=torch.int32, device=q.device)

    columns = page_table.shape[1] * compressed_k_cache.shape[1]
    score_columns_per_row = columns + block_topk if columns < block_topk else columns
    rows_per_chunk = max(1, logits_workspace_bytes // max(score_columns_per_row * 4, 1))
    all_blocks = torch.empty((rows, block_topk), dtype=torch.int32, device=q.device)
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
        if columns < block_topk:
            padded_logits = torch.full(
                (row_end - row_start, block_topk),
                float("-inf"),
                dtype=logits.dtype,
                device=logits.device,
            )
            padded_logits[:, :columns].copy_(logits)
            logits = padded_logits
        selected_groups = all_blocks[row_slice]
        _topk_per_row_amd(logits, visible_groups, selected_groups, block_topk)
        qsa_expand_block_indices(
            selected_groups,
            query_positions[row_slice],
            context_lens,
            token_to_request[row_slice],
            compress_ratio,
            token_topk,
            out[row_slice],
        )
    return out, all_blocks


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
    """#4882 Triton sparse GQA (``num_stages=2``)."""
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
