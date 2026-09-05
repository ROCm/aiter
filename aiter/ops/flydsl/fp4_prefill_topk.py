# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bounded FP4 paged-MQA score + stable TopK for gfx950 prefill.

This is deliberately an operator-level fusion, not a claim that score
production and local selection share one GPU kernel.  Each pass scores one
fixed-width logical-token tile, selects that tile's stable TopK, and radix
merges it into a K-wide accumulator.  Workspace is therefore O(rows *
(tile_tokens + K)), independent of context length; no full-logit tensor is
materialized. The stable TopK primitive's transient scratch is sized from the
same tile or 2K merge width and is context-independent as well.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, fields
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from .kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
    CTA_INFO_WIDTH,
    flydsl_pa_mqa_logits_fp4_prefill,
)

_SUPPORTED_TOP_K = (512, 1024)
_HEADS = 64
_HEAD_DIM = 128
_KV_BLOCK_SIZE = 64
_BLOCK_K = 256
_NUM_WARPS = 4
_DEFAULT_TILE_TOKENS = 4096

# The tiled path below is exact and bounded, but its tile-local stable radix
# selection is still a second launch over a bounded score tile. This flag says
# nothing about the separate canonical score-CTA implementation.
FP4_TILED_PREFILL_TOPK_IN_KERNEL_FUSION_COMPLETE = False
FP4_PREFILL_TOPK_IN_KERNEL_FUSION_COMPLETE = (
    FP4_TILED_PREFILL_TOPK_IN_KERNEL_FUSION_COMPLETE
)


class FP4PrefillTopKCandidates(NamedTuple):
    """One score tile's compact, stable candidates."""

    values: torch.Tensor
    raw_indices: torch.Tensor
    valid_counts: torch.Tensor


class FP4PrefillTopKResult(NamedTuple):
    """Final stable TopK in logical-token and physical-page coordinates."""

    values: torch.Tensor
    raw_indices: torch.Tensor
    kv_indices: torch.Tensor
    valid_counts: torch.Tensor


@dataclass
class FP4PrefillTopKWorkspace:
    """Caller-reusable bounded workspace.

    Result and candidate tensors alias this workspace and are overwritten by a
    later invocation that reuses it.
    """

    rows: int
    k: int
    tile_tokens: int
    cta_info: torch.Tensor
    tile_row_starts: torch.Tensor
    tile_row_ends: torch.Tensor
    tile_scores: torch.Tensor
    selection_positions: torch.Tensor
    candidate_values: torch.Tensor
    candidate_raw_indices: torch.Tensor
    candidate_valid_counts: torch.Tensor
    accum_values_a: torch.Tensor
    accum_values_b: torch.Tensor
    accum_raw_indices_a: torch.Tensor
    accum_raw_indices_b: torch.Tensor
    accum_valid_counts: torch.Tensor
    merge_values: torch.Tensor
    merge_raw_indices: torch.Tensor
    merge_row_starts: torch.Tensor
    merge_row_ends: torch.Tensor
    mapped_kv_indices: torch.Tensor

    @property
    def nbytes(self) -> int:
        """Total tensor storage owned by this workspace."""

        return sum(
            value.numel() * value.element_size()
            for field in fields(self)
            if isinstance((value := getattr(self, field.name)), torch.Tensor)
        )


def allocate_fp4_prefill_topk_workspace(
    rows: int,
    k: int,
    device: torch.device | str,
    *,
    tile_tokens: int = _DEFAULT_TILE_TOKENS,
) -> FP4PrefillTopKWorkspace:
    """Allocate workspace whose size does not depend on context length."""

    _validate_shape_parameters(rows, k, tile_tokens)
    device = torch.device(device)

    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)

    return FP4PrefillTopKWorkspace(
        rows=rows,
        k=k,
        tile_tokens=tile_tokens,
        cta_info=empty((rows, CTA_INFO_WIDTH), torch.int32),
        tile_row_starts=empty((rows,), torch.int32),
        tile_row_ends=empty((rows,), torch.int32),
        tile_scores=empty((rows, tile_tokens), torch.float32),
        selection_positions=empty((rows, k), torch.int32),
        candidate_values=empty((rows, k), torch.float32),
        candidate_raw_indices=empty((rows, k), torch.int32),
        candidate_valid_counts=empty((rows,), torch.int32),
        accum_values_a=empty((rows, k), torch.float32),
        accum_values_b=empty((rows, k), torch.float32),
        accum_raw_indices_a=empty((rows, k), torch.int32),
        accum_raw_indices_b=empty((rows, k), torch.int32),
        accum_valid_counts=empty((rows,), torch.int32),
        merge_values=empty((rows, 2 * k), torch.float32),
        merge_raw_indices=empty((rows, 2 * k), torch.int32),
        merge_row_starts=torch.zeros((rows,), dtype=torch.int32, device=device),
        merge_row_ends=empty((rows,), torch.int32),
        mapped_kv_indices=empty((rows, k), torch.int32),
    )


@triton.jit(do_not_specialize=["rows", "tile_start", "max_seq_len"])
def _build_tile_metadata_kernel(
    row_to_batch,
    local_starts,
    local_ends,
    cta_info,
    tile_row_starts,
    tile_row_ends,
    rows,
    tile_start,
    max_seq_len,
    BLOCK_K: tl.constexpr,
    TILE_TOKENS: tl.constexpr,
    INFO_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = row < rows

    raw_start = tl.load(local_starts + row, mask=mask, other=0)
    raw_end = tl.load(local_ends + row, mask=mask, other=0)
    batch = tl.load(row_to_batch + row, mask=mask, other=0)

    tile_end = tl.minimum(tile_start + TILE_TOKENS, max_seq_len)
    win_start = tl.maximum(tl.maximum(raw_start, tile_start), 0)
    win_end = tl.minimum(raw_end, tile_end)
    live = mask & (win_end > win_start)

    # The score kernel assumes chunk_count >= 1 because it has a mandatory
    # prologue/epilogue. Empty rows point at safe logical chunk zero and carry
    # an empty window, so they execute no store and never touch an invalid page.
    first_chunk = win_start // BLOCK_K
    chunk_start = tl.where(live, first_chunk, 0)
    chunk_count = (win_end + BLOCK_K - 1) // BLOCK_K - first_chunk
    chunk_count = tl.where(live, tl.maximum(chunk_count, 1), 1)
    cta_base = row * INFO_WIDTH
    tl.store(cta_info + cta_base + 0, row, mask=mask)
    tl.store(cta_info + cta_base + 1, tl.where(live, batch, 0), mask=mask)
    tl.store(cta_info + cta_base + 2, chunk_start, mask=mask)
    tl.store(cta_info + cta_base + 3, chunk_count, mask=mask)
    tl.store(cta_info + cta_base + 4, tl.where(live, win_start, 0), mask=mask)
    tl.store(cta_info + cta_base + 5, tl.where(live, win_end, 0), mask=mask)

    tl.store(
        tile_row_starts + row,
        tl.where(live, win_start - tile_start, 0),
        mask=mask,
    )
    tl.store(
        tile_row_ends + row,
        tl.where(live, win_end - tile_start, 0),
        mask=mask,
    )


@triton.jit(do_not_specialize=["rows", "tile_start"])
def _materialize_tile_candidates_kernel(
    positions,
    tile_row_starts,
    tile_row_ends,
    raw_indices,
    valid_counts,
    rows,
    tile_start,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    col_mask = col < K
    position = tl.load(positions + row * K + col, mask=col_mask, other=-1)
    raw = tl.where(position >= 0, position + tile_start, -1)
    tl.store(raw_indices + row * K + col, raw, mask=col_mask)

    if tl.program_id(1) == 0:
        start = tl.load(tile_row_starts + row)
        end = tl.load(tile_row_ends + row)
        count = tl.minimum(tl.maximum(end - start, 0), K)
        tl.store(valid_counts + row, count)


@triton.jit(do_not_specialize=["rows"])
def _pack_merge_candidates_kernel(
    accum_values,
    accum_raw_indices,
    accum_valid_counts,
    candidate_values,
    candidate_raw_indices,
    candidate_valid_counts,
    merge_values,
    merge_raw_indices,
    merge_row_ends,
    rows,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < 2 * K
    old_count = tl.load(accum_valid_counts + row)
    new_count = tl.load(candidate_valid_counts + row)
    total = old_count + new_count

    from_old = col < old_count
    new_col = col - old_count
    from_new = (col >= old_count) & (new_col < new_count)
    safe_old = tl.minimum(col, K - 1)
    safe_new = tl.minimum(tl.maximum(new_col, 0), K - 1)

    old_value = tl.load(
        accum_values + row * K + safe_old,
        mask=mask & from_old,
        other=-float("inf"),
    )
    new_value = tl.load(
        candidate_values + row * K + safe_new,
        mask=mask & from_new,
        other=-float("inf"),
    )
    old_raw = tl.load(
        accum_raw_indices + row * K + safe_old,
        mask=mask & from_old,
        other=-1,
    )
    new_raw = tl.load(
        candidate_raw_indices + row * K + safe_new,
        mask=mask & from_new,
        other=-1,
    )
    value = tl.where(from_old, old_value, tl.where(from_new, new_value, -float("inf")))
    raw = tl.where(from_old, old_raw, tl.where(from_new, new_raw, -1))
    tl.store(merge_values + row * (2 * K) + col, value, mask=mask)
    tl.store(merge_raw_indices + row * (2 * K) + col, raw, mask=mask)

    if tl.program_id(1) == 0:
        tl.store(merge_row_ends + row, total)


@triton.jit(do_not_specialize=["rows"])
def _gather_merged_raw_indices_kernel(
    merge_positions,
    merge_raw_indices,
    merge_row_ends,
    out_raw_indices,
    out_valid_counts,
    rows,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < K
    position = tl.load(merge_positions + row * K + col, mask=mask, other=-1)
    safe_position = tl.minimum(tl.maximum(position, 0), 2 * K - 1)
    raw = tl.load(
        merge_raw_indices + row * (2 * K) + safe_position,
        mask=mask & (position >= 0),
        other=-1,
    )
    tl.store(out_raw_indices + row * K + col, raw, mask=mask)

    if tl.program_id(1) == 0:
        total = tl.load(merge_row_ends + row)
        tl.store(out_valid_counts + row, tl.minimum(total, K))


def _validate_shape_parameters(rows: int, k: int, tile_tokens: int) -> None:
    if rows < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")
    if k not in _SUPPORTED_TOP_K:
        raise ValueError(f"k must be one of {_SUPPORTED_TOP_K}, got {k}")
    if tile_tokens < k:
        raise ValueError(f"tile_tokens={tile_tokens} must be >= k={k}")
    if tile_tokens % _BLOCK_K:
        raise ValueError(
            f"tile_tokens={tile_tokens} must be divisible by block_k={_BLOCK_K}"
        )


def _validate_inputs(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    max_seq_len: int,
    k: int,
    tile_tokens: int,
) -> None:
    rows = q_fp4.shape[0] if q_fp4.ndim else -1
    _validate_shape_parameters(rows, k, tile_tokens)
    if max_seq_len < 0:
        raise ValueError(f"max_seq_len must be non-negative, got {max_seq_len}")
    if q_fp4.shape != (rows, _HEADS, _HEAD_DIM // 2):
        raise ValueError(
            "q_fp4 must have shape "
            f"[rows, {_HEADS}, {_HEAD_DIM // 2}], got {tuple(q_fp4.shape)}"
        )
    if q_scale.shape != (rows, 1, 4, 16, 4):
        raise ValueError(
            f"q_scale must have shape [rows, 1, 4, 16, 4], got {tuple(q_scale.shape)}"
        )
    if weights.shape != (rows, _HEADS):
        raise ValueError(
            f"weights must have shape [rows, {_HEADS}], got {tuple(weights.shape)}"
        )
    if kv_cache.ndim != 5 or tuple(kv_cache.shape[1:]) != (1, 4, 64, 16):
        raise ValueError(
            "kv_cache must have shape [num_blocks, 1, 4, 64, 16], "
            f"got {tuple(kv_cache.shape)}"
        )
    if kv_scale.ndim != 4 or tuple(kv_scale.shape[1:]) != (1, 4, 64):
        raise ValueError(
            "kv_scale must have shape [num_blocks, 1, 4, 64], "
            f"got {tuple(kv_scale.shape)}"
        )
    if (
        block_tables.ndim != 2
        or block_tables.dtype != torch.int32
        or not block_tables.is_contiguous()
    ):
        raise ValueError("block_tables must be a contiguous 2D int32 tensor")
    if kv_cache.shape[0] != kv_scale.shape[0]:
        raise ValueError("kv_cache and kv_scale must have the same num_blocks")
    if max_seq_len > block_tables.shape[1] * _KV_BLOCK_SIZE:
        raise ValueError(
            f"max_seq_len={max_seq_len} exceeds block-table capacity "
            f"{block_tables.shape[1] * _KV_BLOCK_SIZE}"
        )

    expected_vectors = {
        "row_to_batch": row_to_batch,
        "local_starts": local_starts,
        "local_ends": local_ends,
    }
    for name, tensor in expected_vectors.items():
        if tensor.shape != (rows,) or tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be contiguous int32 [rows]")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous int32 [rows]")

    expected_dtypes = {
        "q_fp4": (q_fp4, torch.uint8),
        "q_scale": (q_scale, torch.uint8),
        "kv_cache": (kv_cache, torch.uint8),
        "kv_scale": (kv_scale, torch.uint8),
        "weights": (weights, torch.bfloat16),
    }
    for name, (tensor, dtype) in expected_dtypes.items():
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    device = q_fp4.device
    tensors = (
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        local_starts,
        local_ends,
    )
    if device.type != "cuda" or any(tensor.device != device for tensor in tensors):
        raise ValueError("all inputs must be contiguous tensors on one CUDA device")
    arch = str(torch.cuda.get_device_properties(device).gcnArchName).split(":")[0]
    if arch != "gfx950":
        raise ValueError(f"FP4 prefill TopK requires gfx950, got {arch}")


def _validate_workspace(
    workspace: FP4PrefillTopKWorkspace,
    rows: int,
    k: int,
    tile_tokens: int,
    device: torch.device,
) -> None:
    if (
        workspace.rows != rows
        or workspace.k != k
        or workspace.tile_tokens != tile_tokens
    ):
        raise ValueError(
            "workspace geometry mismatch: expected "
            f"(rows={rows}, k={k}, tile_tokens={tile_tokens}), got "
            f"(rows={workspace.rows}, k={workspace.k}, "
            f"tile_tokens={workspace.tile_tokens})"
        )
    expected = {
        "cta_info": ((rows, CTA_INFO_WIDTH), torch.int32),
        "tile_row_starts": ((rows,), torch.int32),
        "tile_row_ends": ((rows,), torch.int32),
        "tile_scores": ((rows, tile_tokens), torch.float32),
        "selection_positions": ((rows, k), torch.int32),
        "candidate_values": ((rows, k), torch.float32),
        "candidate_raw_indices": ((rows, k), torch.int32),
        "candidate_valid_counts": ((rows,), torch.int32),
        "accum_values_a": ((rows, k), torch.float32),
        "accum_values_b": ((rows, k), torch.float32),
        "accum_raw_indices_a": ((rows, k), torch.int32),
        "accum_raw_indices_b": ((rows, k), torch.int32),
        "accum_valid_counts": ((rows,), torch.int32),
        "merge_values": ((rows, 2 * k), torch.float32),
        "merge_raw_indices": ((rows, 2 * k), torch.int32),
        "merge_row_starts": ((rows,), torch.int32),
        "merge_row_ends": ((rows,), torch.int32),
        "mapped_kv_indices": ((rows, k), torch.int32),
    }
    for name, (shape, dtype) in expected.items():
        tensor = getattr(workspace, name)
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"workspace.{name} must be contiguous {dtype} {shape} on {device}"
            )


def _launch_grid(rows: int, width: int, block: int) -> tuple[int, int]:
    return rows, triton.cdiv(width, block)


@triton.jit
def _canonicalize_nan_low_kernel(scores, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    values = tl.load(scores + offsets, mask=mask)
    bits = values.to(tl.int32, bitcast=True)
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    negative_nan = tl.full((BLOCK,), -1, tl.int32).to(tl.float32, bitcast=True)
    tl.store(scores + offsets, tl.where(is_nan, negative_nan, values), mask=mask)


def _score_tile_topk_into(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    max_seq_len: int,
    tile_start: int,
    workspace: FP4PrefillTopKWorkspace,
    *,
    weight_scale: float,
    stream: torch.cuda.Stream,
) -> FP4PrefillTopKCandidates:
    from aiter.ops.topk import top_k_per_row_prefill

    rows, k, tile_tokens = workspace.rows, workspace.k, workspace.tile_tokens
    if rows == 0:
        return FP4PrefillTopKCandidates(
            workspace.candidate_values,
            workspace.candidate_raw_indices,
            workspace.candidate_valid_counts,
        )

    metadata_block = 256
    _build_tile_metadata_kernel[(triton.cdiv(rows, metadata_block),)](
        row_to_batch,
        local_starts,
        local_ends,
        workspace.cta_info,
        workspace.tile_row_starts,
        workspace.tile_row_ends,
        rows,
        tile_start,
        max_seq_len,
        BLOCK_K=_BLOCK_K,
        TILE_TOKENS=tile_tokens,
        INFO_WIDTH=CTA_INFO_WIDTH,
        BLOCK=metadata_block,
    )
    workspace.tile_scores.fill_(float("-inf"))

    flydsl_pa_mqa_logits_fp4_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        local_starts,
        local_ends,
        max_seq_len,
        weight_scale=weight_scale,
        block_k=_BLOCK_K,
        kv_block_size=_KV_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
        out=workspace.tile_scores,
        cta_info=workspace.cta_info,
        n_ctas=rows,
        output_token_base=tile_start,
        stream=stream,
    )
    elements = rows * tile_tokens
    _canonicalize_nan_low_kernel[(triton.cdiv(elements, 256),)](
        workspace.tile_scores,
        elements,
        BLOCK=256,
    )

    # KERNEL-FUSION TODO(gfx950): reserve tile_tokens*f32 LDS in the score CTA,
    # write `_post_process_nt` results there, and run the stable local radix
    # threshold/scatter before the CTA exits. That removes tile_scores and this
    # top_k_per_row_prefill launch. Until that kernel is compiled and measured,
    # this path intentionally remains bounded operator-level fusion.
    top_k_per_row_prefill(
        workspace.tile_scores,
        workspace.tile_row_starts,
        workspace.tile_row_ends,
        workspace.selection_positions,
        workspace.candidate_values,
        rows,
        workspace.tile_scores.stride(0),
        workspace.tile_scores.stride(1),
        k=k,
        stable=True,
    )

    block = 256
    _materialize_tile_candidates_kernel[_launch_grid(rows, k, block)](
        workspace.selection_positions,
        workspace.tile_row_starts,
        workspace.tile_row_ends,
        workspace.candidate_raw_indices,
        workspace.candidate_valid_counts,
        rows,
        tile_start,
        K=k,
        BLOCK=block,
    )
    return FP4PrefillTopKCandidates(
        workspace.candidate_values,
        workspace.candidate_raw_indices,
        workspace.candidate_valid_counts,
    )


def flydsl_pa_mqa_fp4_score_tile_topk(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    max_seq_len: int,
    tile_start: int,
    *,
    k: int = 1024,
    tile_tokens: int = _DEFAULT_TILE_TOKENS,
    weight_scale: float = 1.0,
    workspace: FP4PrefillTopKWorkspace | None = None,
    stream: torch.cuda.Stream | None = None,
) -> FP4PrefillTopKCandidates:
    """Score one aligned tile and return values, raw indices, and valid counts."""

    _validate_inputs(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        local_starts,
        local_ends,
        max_seq_len,
        k,
        tile_tokens,
    )
    if tile_start < 0 or tile_start % tile_tokens:
        raise ValueError(
            f"tile_start={tile_start} must be a non-negative multiple of "
            f"tile_tokens={tile_tokens}"
        )
    if workspace is None:
        workspace = allocate_fp4_prefill_topk_workspace(
            q_fp4.shape[0], k, q_fp4.device, tile_tokens=tile_tokens
        )
    _validate_workspace(workspace, q_fp4.shape[0], k, tile_tokens, q_fp4.device)
    if stream is None:
        stream = torch.cuda.current_stream(q_fp4.device)
        stream_context = nullcontext()
    else:
        stream_context = torch.cuda.stream(stream)
    with stream_context:
        return _score_tile_topk_into(
            q_fp4,
            q_scale,
            kv_cache,
            kv_scale,
            block_tables,
            weights,
            row_to_batch,
            local_starts,
            local_ends,
            max_seq_len,
            tile_start,
            workspace,
            weight_scale=weight_scale,
            stream=stream,
        )


def flydsl_pa_mqa_fp4_prefill_topk(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    max_seq_len: int,
    *,
    k: int = 1024,
    tile_tokens: int = _DEFAULT_TILE_TOKENS,
    weight_scale: float = 1.0,
    workspace: FP4PrefillTopKWorkspace | None = None,
    stream: torch.cuda.Stream | None = None,
) -> FP4PrefillTopKResult:
    """Exact bounded FP4 score + stable TopK + page mapping on gfx950.

    Long rows are returned in total order (score descending, raw logical index
    ascending). Rows no longer than K preserve their complete sequential
    window. Empty/short rows are padded with ``(-inf, -1, -1)``.
    """

    _validate_inputs(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        local_starts,
        local_ends,
        max_seq_len,
        k,
        tile_tokens,
    )
    rows = q_fp4.shape[0]
    from aiter.ops.topk import top_k_per_row_prefill

    if workspace is None:
        workspace = allocate_fp4_prefill_topk_workspace(
            rows, k, q_fp4.device, tile_tokens=tile_tokens
        )
    _validate_workspace(workspace, rows, k, tile_tokens, q_fp4.device)
    if rows == 0:
        return FP4PrefillTopKResult(
            workspace.accum_values_a,
            workspace.accum_raw_indices_a,
            workspace.mapped_kv_indices,
            workspace.accum_valid_counts,
        )

    if stream is None:
        stream = torch.cuda.current_stream(q_fp4.device)
        stream_context = nullcontext()
    else:
        stream_context = torch.cuda.stream(stream)

    with stream_context:
        workspace.accum_valid_counts.zero_()
        workspace.accum_values_a.fill_(float("-inf"))
        workspace.accum_raw_indices_a.fill_(-1)
        accum_values = workspace.accum_values_a
        next_values = workspace.accum_values_b
        accum_raw = workspace.accum_raw_indices_a
        next_raw = workspace.accum_raw_indices_b

        for tile_start in range(0, max_seq_len, tile_tokens):
            _score_tile_topk_into(
                q_fp4,
                q_scale,
                kv_cache,
                kv_scale,
                block_tables,
                weights,
                row_to_batch,
                local_starts,
                local_ends,
                max_seq_len,
                tile_start,
                workspace,
                weight_scale=weight_scale,
                stream=stream,
            )

            # Exact tie stability follows by induction. Stable local selection
            # emits each tile's winners in ascending raw-index order. The prior
            # accumulator is ordered the same way, and every prior-tile raw
            # index is smaller than every current-tile raw index. Packing those
            # two prefixes therefore presents the merge radix with globally
            # ascending raw indices; its smallest-position tie rule is exactly
            # the required smallest-raw-index rule, and its stable emit keeps
            # the invariant for the next pass.
            block = 256
            _pack_merge_candidates_kernel[_launch_grid(rows, 2 * k, block)](
                accum_values,
                accum_raw,
                workspace.accum_valid_counts,
                workspace.candidate_values,
                workspace.candidate_raw_indices,
                workspace.candidate_valid_counts,
                workspace.merge_values,
                workspace.merge_raw_indices,
                workspace.merge_row_ends,
                rows,
                K=k,
                BLOCK=block,
            )
            top_k_per_row_prefill(
                workspace.merge_values,
                workspace.merge_row_starts,
                workspace.merge_row_ends,
                workspace.selection_positions,
                next_values,
                rows,
                workspace.merge_values.stride(0),
                workspace.merge_values.stride(1),
                k=k,
                stable=True,
            )
            _gather_merged_raw_indices_kernel[_launch_grid(rows, k, block)](
                workspace.selection_positions,
                workspace.merge_raw_indices,
                workspace.merge_row_ends,
                next_raw,
                workspace.accum_valid_counts,
                rows,
                K=k,
                BLOCK=block,
            )
            accum_values, next_values = next_values, accum_values
            accum_raw, next_raw = next_raw, accum_raw

        if rows:
            from .mqa_topk_finalize import order_and_map_mqa_topk

            order_and_map_mqa_topk(
                accum_values,
                accum_raw,
                workspace.accum_valid_counts,
                local_starts,
                local_ends,
                row_to_batch,
                block_tables,
                next_values,
                next_raw,
                workspace.mapped_kv_indices,
                max_seq_len,
                k,
                _KV_BLOCK_SIZE,
            )
            accum_values = next_values
            accum_raw = next_raw

    return FP4PrefillTopKResult(
        accum_values,
        accum_raw,
        workspace.mapped_kv_indices,
        workspace.accum_valid_counts,
    )


# Distinct aliases keep the bounded operator-level workspace/result API
# available beside the true score-CTA implementation, whose public types use
# the shorter FP4PrefillTopK names.
FP4BoundedPrefillTopKResult = FP4PrefillTopKResult
FP4BoundedPrefillTopKWorkspace = FP4PrefillTopKWorkspace
allocate_fp4_bounded_prefill_topk_workspace = allocate_fp4_prefill_topk_workspace
flydsl_pa_mqa_topk_fp4_prefill_tiled = flydsl_pa_mqa_fp4_prefill_topk


__all__ = [
    "FP4_PREFILL_TOPK_IN_KERNEL_FUSION_COMPLETE",
    "FP4_TILED_PREFILL_TOPK_IN_KERNEL_FUSION_COMPLETE",
    "FP4BoundedPrefillTopKResult",
    "FP4BoundedPrefillTopKWorkspace",
    "FP4PrefillTopKCandidates",
    "FP4PrefillTopKResult",
    "FP4PrefillTopKWorkspace",
    "allocate_fp4_bounded_prefill_topk_workspace",
    "allocate_fp4_prefill_topk_workspace",
    "flydsl_pa_mqa_fp4_prefill_topk",
    "flydsl_pa_mqa_fp4_score_tile_topk",
    "flydsl_pa_mqa_topk_fp4_prefill_tiled",
]
