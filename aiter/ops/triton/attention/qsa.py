# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA operators for index scoring, selection and sparse paged GQA."""

import torch
import triton

from aiter.jit.utils.chip_info import get_cu_num
from aiter.ops.topk import top_k_per_row_prefill
from aiter.ops.triton._triton_kernels.attention.qsa import (
    _expand_selected_groups,
    _qsa_paged_mqa_logits_kernel,
    _qsa_paged_mqa_logits_mfma_kernel,
    _qsa_sparse_paged_gqa_kernel,
)

try:
    from aiter.ops.flydsl.kernels.mqa_logits.qsa_flydsl_layout import (
        PAGE_SIZE as _FLYDSL_PAGE_SIZE,
    )
    from aiter.ops.flydsl.kernels.mqa_logits.qsa_flydsl_layout import (
        qsa_paged_mqa_logits_flydsl,
    )

    _HAS_FLYDSL = True
except ImportError:
    _HAS_FLYDSL = False


def _prev_pow2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _table_width_bucket(width: int) -> int:
    """Page-table width as seen by the kernels, rounded up to a power of two."""
    return 1 << (max(int(width), 1) - 1).bit_length()


def _kv_splits_heuristic(
    T: int,
    kv_heads: int,
    topk: int,
    num_cu: int | None = None,
    target_wg_per_cu: float = 4.0,
    max_kv_splits: int = 64,
) -> int:
    if topk < 512:
        return 1
    if num_cu is None:
        num_cu = get_cu_num()
    target_wg = max(1, int(target_wg_per_cu * num_cu))
    base_ctas = max(1, T * kv_heads)
    if base_ctas >= target_wg:
        return 1
    return _prev_pow2(min(target_wg // base_ctas, max_kv_splits))


def _kernel_config(
    T: int,
    kv_heads: int,
    kv_splits: int,
    group_size: int,
    num_cu: int | None = None,
) -> tuple[int, int, int, int, int]:
    """Pick (BLOCK_N, num_warps, num_stages, waves_per_eu, sub_group)."""
    if num_cu is None:
        num_cu = get_cu_num()
    sub_group = group_size
    if group_size > 16:
        sub_group = 8
    num_head_groups = (group_size + sub_group - 1) // sub_group
    grid_size = T * kv_heads * num_head_groups * kv_splits
    if grid_size <= num_cu:
        return 64, 4, 1, 0, group_size
    if grid_size >= num_cu * 8:
        return 32, 2, 1, 0, sub_group
    return 32, 4, 1, 3, sub_group


DEFAULT_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024

_SCORING_BLOCK_N = 32
_SCORING_COLUMN_PROGRAMS = 64


def _check_vector(name: str, tensor: torch.Tensor, length: int | None = None) -> None:
    if tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a one-dimensional int32/int64 tensor")
    if length is not None and tensor.shape[0] != length:
        raise ValueError(f"{name} must contain {length} entries")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def qsa_paged_mqa_logits(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    compress_ratio: int = 4,
    score_divisor: float | None = None,
    max_columns: int | None = None,
    row_starts: torch.Tensor | None = None,
    allow_flydsl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score compressed key groups.

    `q` is `[tokens, index_heads, head_dim]` and `compressed_k_cache` is
    `[pages, page_size, 1, head_dim]`. Returns FP32 logits `[tokens, columns]`
    and the per-row count of causally visible groups.
    """
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("q must have shape [tokens, heads, head_dim]")
    if (
        compressed_k_cache.ndim != 4
        or compressed_k_cache.shape[2] != 1
        or compressed_k_cache.shape[3] != q.shape[2]
    ):
        raise ValueError(
            "compressed_k_cache must have shape [pages, page_size, 1, head_dim]"
        )
    if q.dtype != compressed_k_cache.dtype:
        raise ValueError("q and compressed_k_cache must have the same dtype")
    if page_table.ndim != 2 or page_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("page_table must be a two-dimensional integer tensor")
    _check_vector("token_to_request", token_to_request, q.shape[0])
    _check_vector("query_positions", query_positions, q.shape[0])
    _check_vector("context_lens", context_lens, page_table.shape[0])
    if row_starts is not None:
        _check_vector("row_starts", row_starts, q.shape[0])

    divisor = q.shape[2] ** 0.5 if score_divisor is None else score_divisor
    if divisor <= 0:
        raise ValueError("score_divisor must be positive")
    columns = page_table.shape[1] * compressed_k_cache.shape[1]
    if max_columns is not None:
        columns = max(1, min(columns, int(max_columns)))
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_groups = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if q.shape[0] == 0 or columns == 0:
        visible_groups.zero_()
        if row_starts is not None:
            row_starts.zero_()
        return logits, visible_groups

    if (
        q.shape[0] > 256
        and q.stride(2) == 1
        and compressed_k_cache.stride(3) == 1
        and page_table.stride(1) == 1
        and q.shape[2] >= 16
    ):
        if (
            allow_flydsl
            and _HAS_FLYDSL
            and compressed_k_cache.shape[1] == _FLYDSL_PAGE_SIZE
        ):
            qsa_paged_mqa_logits_flydsl(
                q,
                compressed_k_cache,
                page_table,
                token_to_request,
                query_positions,
                context_lens,
                compress_ratio,
                float(divisor),
                logits,
                visible_groups,
                row_starts,
            )
            return logits, visible_groups
        bmt, bn = 16, 128
        _qsa_paged_mqa_logits_mfma_kernel[
            (triton.cdiv(q.shape[0], bmt), triton.cdiv(columns, bn))
        ](
            q,
            compressed_k_cache,
            page_table,
            token_to_request,
            query_positions,
            context_lens,
            visible_groups,
            row_starts,
            logits,
            q.stride(0),
            q.stride(1),
            compressed_k_cache.stride(0),
            compressed_k_cache.stride(1),
            page_table.stride(0),
            logits.stride(0),
            q.shape[0],
            columns,
            compressed_k_cache.shape[0],
            page_table.shape[0],
            float(divisor),
            PAGE_SIZE=compressed_k_cache.shape[1],
            PAGE_TABLE_WIDTH=_table_width_bucket(page_table.shape[1]),
            NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2],
            COMPRESS_RATIO=compress_ratio,
            BMT=bmt,
            BN=bn,
            num_warps=4,
        )
        return logits, visible_groups
    column_programs = min(
        triton.cdiv(columns, _SCORING_BLOCK_N), _SCORING_COLUMN_PROGRAMS
    )
    _qsa_paged_mqa_logits_kernel[(q.shape[0], column_programs)](
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        visible_groups,
        row_starts,
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
        PAGE_TABLE_WIDTH=_table_width_bucket(page_table.shape[1]),
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        COMPRESS_RATIO=compress_ratio,
        BLOCK_N=_SCORING_BLOCK_N,
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
    output_width: int | None = None,
) -> torch.Tensor:
    """Expand selected groups to token ids and append the causal tail."""
    if compress_ratio <= 0 or token_topk <= 0:
        raise ValueError("compress_ratio and token_topk must be positive")
    if block_indices.ndim != 2 or block_indices.dtype != torch.int32:
        raise ValueError("block_indices must be a two-dimensional int32 tensor")
    if token_topk % compress_ratio:
        raise ValueError("token_topk must be divisible by compress_ratio")
    _check_vector("query_positions", query_positions, block_indices.shape[0])
    _check_vector("token_to_request", token_to_request, block_indices.shape[0])
    _check_vector("context_lens", context_lens)

    block_topk = block_indices.shape[1]
    if block_topk > token_topk // compress_ratio:
        raise ValueError(
            f"block_indices must have at most {token_topk // compress_ratio} columns"
        )
    if output_width is None:
        output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.dtype != torch.int32 or out.shape != (
        block_indices.shape[0],
        output_width,
    ):
        raise ValueError("out has an invalid shape")
    if block_indices.shape[0] == 0:
        return out

    _expand_selected_groups[(block_indices.shape[0], triton.cdiv(output_width, 128))](
        block_indices,
        context_lens,
        query_positions,
        token_to_request,
        out,
        context_lens.numel(),
        block_topk,
        output_width,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        RATIO=compress_ratio,
        BLOCK=128,
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
    logits_workspace_bytes: int = DEFAULT_LOGITS_WORKSPACE_BYTES,
    max_seq_len: int | None = None,
    allow_flydsl: bool = False,
) -> torch.Tensor:
    """Score, select and expand in one call."""
    if compress_ratio <= 0 or token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("token_topk must be positive and divisible by compress_ratio")
    rows = q.shape[0]

    columns = page_table.shape[1] * compressed_k_cache.shape[1]
    if columns < 1:
        raise ValueError("compressed paged-cache capacity must be positive")
    block_topk = min(token_topk // compress_ratio, columns)
    if max_seq_len is not None:
        reachable = (int(max_seq_len) + compress_ratio - 1) // compress_ratio
        bucket = 1 << (max(reachable, 1) - 1).bit_length()
        columns = max(1, min(columns, bucket))
        block_topk = max(1, min(block_topk, bucket))
    output_width = block_topk * compress_ratio + compress_ratio - 1

    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    elif out.shape[0] != rows or out.shape[1] < output_width:
        raise ValueError("out has an invalid shape")
    else:
        out = out[:, :output_width]
    if rows == 0:
        return out

    rows_per_chunk = max(1, logits_workspace_bytes // max(columns * 4, 1))
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        row_starts = torch.empty(
            row_end - row_start, dtype=torch.int32, device=q.device
        )
        logits, visible_groups = qsa_paged_mqa_logits(
            q[row_slice],
            compressed_k_cache,
            page_table,
            token_to_request[row_slice],
            query_positions[row_slice],
            context_lens,
            compress_ratio,
            max_columns=columns,
            row_starts=row_starts,
            allow_flydsl=allow_flydsl,
        )
        selected_groups = torch.empty(
            (row_end - row_start, block_topk), dtype=torch.int32, device=q.device
        )
        top_k_per_row_prefill(
            logits,
            row_starts,
            visible_groups,
            selected_groups,
            None,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            block_topk,
            stable=True,
        )
        qsa_expand_block_indices(
            selected_groups,
            query_positions[row_slice],
            context_lens,
            token_to_request[row_slice],
            compress_ratio,
            token_topk,
            out[row_slice],
            output_width=output_width,
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
    kv_splits: int | None = None,
    num_decode_requests: int | None = None,
) -> torch.Tensor:
    """Grouped-query attention restricted to `logical_indices` (-1 = padding)."""
    if q.ndim != 3:
        raise ValueError("q must be [tokens, query_heads, head_dim]")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("K/V caches must have matching [pages, page, heads, dim]")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q, k_cache and v_cache must have the same dtype")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("query heads must form equal groups over KV heads")
    if (
        logical_indices.ndim != 2
        or logical_indices.shape[0] != q.shape[0]
        or logical_indices.dtype != torch.int32
    ):
        raise ValueError("logical_indices must be int32 [tokens, selection_width]")
    if block_table.ndim != 2:
        raise ValueError("block_table must be a two-dimensional integer tensor")
    if (
        token_to_request.shape != (q.shape[0],)
        or token_to_request.dtype not in (torch.int32, torch.int64)
        or not token_to_request.is_contiguous()
    ):
        raise ValueError("token_to_request must be contiguous int32/int64 [tokens]")

    scale = q.shape[2] ** -0.5 if softmax_scale is None else softmax_scale
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if q.shape[0] == 0:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    if kv_splits is None:
        split_rows = q.shape[0]
        if split_rows <= 0:
            raise ValueError("num_decode_requests must be positive")
        kv_splits = _kv_splits_heuristic(
            split_rows, k_cache.shape[2], logical_indices.shape[1]
        )
    if kv_splits < 1 or kv_splits & (kv_splits - 1):
        raise ValueError("kv_splits must be a positive power of two")
    block_n, num_warps, num_stages, waves_per_eu, sub_group = _kernel_config(
        q.shape[0], k_cache.shape[2], kv_splits, group_size
    )
    num_head_groups = (group_size + sub_group - 1) // sub_group
    block_m = max(16, triton.next_power_of_2(sub_group))
    if logical_indices.shape[1] == 0:
        return out.zero_()
    target = out
    partial_max = partial_sum = out
    widths = None
    if kv_splits > 1:
        shape = (q.shape[0], q.shape[1], kv_splits)
        partial_max = torch.empty(shape, device=q.device, dtype=torch.float32)
        partial_sum = torch.empty_like(partial_max)
        target = torch.empty((*shape, q.shape[2]), device=q.device, dtype=torch.float32)
        widths = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    grid_y = k_cache.shape[2] * num_head_groups
    _qsa_sparse_paged_gqa_kernel[(q.shape[0], grid_y, kv_splits)](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_request,
        target,
        partial_max,
        partial_sum,
        widths,
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
        target.stride(0),
        target.stride(1),
        target.stride(-1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        float(scale),
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=_table_width_bucket(block_table.shape[1]),
        NUM_KV_HEADS=k_cache.shape[2],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        KV_SPLITS=kv_splits,
        SUB_GROUP=sub_group if sub_group < group_size else 0,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )
    if kv_splits > 1:
        from aiter.ops.triton._triton_kernels.attention.mla import (
            _mla_decode_fwd_reduce_kernel,
        )

        _mla_decode_fwd_reduce_kernel[(q.shape[0], q.shape[1])](
            out,
            target,
            partial_max,
            partial_sum,
            widths,
            None,
            q.shape[0],
            q.shape[1],
            out.stride(0),
            out.stride(1),
            1,
            1,
            q.shape[0],
            TILE_SIZE=block_n,
            KV_LORA_RANK=q.shape[2],
            query_start_len_ptr=None,
            BLOCK_Q=1,
            NUM_SEGMENTS_PER_SEQ=kv_splits,
            ALL_DECODE=True,
        )
    return out
