# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public contract for experimental FP8 paged-MQA Stage A local TopK."""

from functools import lru_cache

import torch

from aiter.ops.flydsl.topk_per_row import flydsl_top_k_per_row_decode

from .kernels.mqa_logits.fp8_paged_mqa_local_topk import (
    HEAD_DIM,
    INDEX_DIM,
    NUM_XCD,
    SUPPORTED_K,
    WORKGROUPS_PER_CU,
    launch_fp8_paged_mqa_local_topk,
)
from .split_topk_merge import split_topk_merge, split_topk_merge_workspace

SUPPORTED_ARCHES = ("gfx950",)


@lru_cache(maxsize=32)
def _full_merge_widths(
    device_type: str, device_index: int, rows: int, width: int
) -> torch.Tensor:
    return torch.full(
        (rows,),
        width,
        dtype=torch.int32,
        device=torch.device(device_type, device_index),
    )


def _arch_name(device: torch.device) -> str:
    props = torch.cuda.get_device_properties(device)
    return props.gcnArchName.split(":", 1)[0]


def _require_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be on a CUDA/HIP device")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _packed_page_shape(kv_cache: torch.Tensor) -> tuple[int, int] | None:
    if kv_cache.ndim == 4 and kv_cache.shape[2:] == (1, INDEX_DIM):
        return int(kv_cache.shape[0]), int(kv_cache.shape[1])
    if kv_cache.ndim == 3 and kv_cache.shape[-1] == INDEX_DIM:
        return int(kv_cache.shape[0]), int(kv_cache.shape[1])
    return None


def _packed_scale_view(kv_cache: torch.Tensor) -> torch.Tensor:
    return kv_cache.view(torch.float32).reshape(-1)


def _normalize_kv_inputs(kv_cache, k_scales, q_dtype):
    packed_shape = _packed_page_shape(kv_cache)
    if packed_shape is not None:
        num_pages, page_size = packed_shape
        if kv_cache.dtype not in (q_dtype, torch.uint8):
            raise ValueError(
                "packed kv_cache must be uint8 or the query fp8 dtype, "
                f"got {kv_cache.dtype}"
            )
        return kv_cache, _packed_scale_view(kv_cache), True, num_pages, page_size
    if kv_cache.ndim != 3 or kv_cache.shape[2] != HEAD_DIM:
        raise ValueError(
            "kv_cache must have shape [num_pages,page_size,128] or packed "
            f"[num_pages,page_size,1,{INDEX_DIM}], got {tuple(kv_cache.shape)}"
        )
    if kv_cache.dtype != q_dtype:
        raise ValueError(
            f"kv_cache dtype must match q_fp8 ({q_dtype}), got {kv_cache.dtype}"
        )
    num_pages, page_size, _ = kv_cache.shape
    if k_scales is None:
        raise ValueError("k_scales is required for split (non-packed) kv_cache")
    if k_scales.shape != (num_pages, page_size) or k_scales.dtype != torch.float32:
        raise ValueError(
            "k_scales must be contiguous float32 with shape "
            f"{(num_pages, page_size)}, got {tuple(k_scales.shape)} {k_scales.dtype}"
        )
    return kv_cache, k_scales, False, num_pages, page_size


def _normalize_context_lens(
    context_lens: torch.Tensor, batch: int, next_n: int
) -> torch.Tensor:
    if context_lens.shape == (batch,):
        if next_n == 1:
            return context_lens
        offsets = torch.arange(
            1 - next_n,
            1,
            dtype=torch.int32,
            device=context_lens.device,
        )
        return (context_lens[:, None] + offsets).clamp_min_(0).reshape(-1)
    if context_lens.shape == (batch, next_n):
        return context_lens.reshape(batch * next_n)
    raise ValueError(
        "context_lens must have shape "
        f"{(batch,)} or {(batch, next_n)}, got {tuple(context_lens.shape)}"
    )


_MAX_AUTO_SPLIT_SPAN = 32768


def _plan_num_splits(rows: int, max_history: int, k: int, cu_count: int) -> int:
    row_groups = max(1, rows)
    workgroups_per_cu = WORKGROUPS_PER_CU
    target_blocks = workgroups_per_cu * cu_count
    occupancy_splits = max(
        1,
        (target_blocks + row_groups - 1) // row_groups,
    )
    useful_splits = max(1, (max_history + k - 1) // k)
    splits = min(128, occupancy_splits, useful_splits)
    while splits < 128:
        span = (max_history + splits - 1) // splits
        if span <= _MAX_AUTO_SPLIT_SPAN:
            break
        splits += 1
    if splits < NUM_XCD:
        return splits
    aligned = min(128, (splits + NUM_XCD - 1) // NUM_XCD * NUM_XCD)
    return aligned


def _auto_num_splits(q_fp8: torch.Tensor, max_history: int, k: int) -> int:
    rows = q_fp8.shape[0] * q_fp8.shape[1]
    cu_count = torch.cuda.get_device_properties(q_fp8.device).multi_processor_count
    return _plan_num_splits(rows, max_history, k, cu_count)


def flydsl_fp8_paged_mqa_local_topk(
    q_fp8,
    kv_cache,
    k_scales,
    weights,
    context_lens,
    block_tables,
    *,
    k=2048,
    num_splits=None,
    preshuffled=False,
):
    """Compute exact H32D128 FP8 scores and retain local TopK per history split.

    Live candidates are emitted unordered. Positions are logical history positions.
    Set ``preshuffled=True`` when ``kv_cache`` uses
    ``shuffle_weight(..., layout=(16,16))`` within each page. This experimental
    API never allocates a full-width logits tensor.

    Row lengths larger than ``block_tables.shape[1] * page_size`` are clamped
    to that table span. Block-table entries outside ``[0, num_pages)`` are not
    scored (they contribute ``-inf``); the kernel will not index the KV cache
    with those ids.
    """
    for name, tensor in (
        ("q_fp8", q_fp8),
        ("kv_cache", kv_cache),
        ("weights", weights),
        ("context_lens", context_lens),
        ("block_tables", block_tables),
    ):
        _require_cuda_contiguous(name, tensor)
    packed_shape = _packed_page_shape(kv_cache)
    if packed_shape is None:
        _require_cuda_contiguous("k_scales", k_scales)

    device = q_fp8.device
    kv_cache, k_scales, packed, num_pages, page_size = _normalize_kv_inputs(
        kv_cache, k_scales, q_fp8.dtype
    )
    if num_pages < 1:
        raise ValueError("kv_cache must contain at least one page")
    tensors = (kv_cache, k_scales, weights, context_lens, block_tables)
    if any(t.device != device for t in tensors):
        raise ValueError("all inputs must be on the same device")

    arch = _arch_name(device)
    if arch not in SUPPORTED_ARCHES:
        raise RuntimeError(
            f"flydsl_fp8_paged_mqa_local_topk is unsupported on {arch}; "
            f"supported architectures: {SUPPORTED_ARCHES}"
        )
    if q_fp8.ndim != 4 or q_fp8.shape[2:] != (32, 128):
        raise ValueError(
            f"q_fp8 must have contiguous shape [B,next_n,32,128], got {tuple(q_fp8.shape)}"
        )
    if q_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"gfx950 requires torch.float8_e4m3fn q_fp8, got {q_fp8.dtype}"
        )
    if not isinstance(preshuffled, bool):
        raise TypeError(f"preshuffled must be bool, got {type(preshuffled).__name__}")

    batch, next_n, _, _ = q_fp8.shape
    rows = batch * next_n
    if weights.shape != (rows, 32) or weights.dtype != torch.float32:
        raise ValueError(
            f"weights must be contiguous float32 with shape {(rows, 32)}, "
            f"got {tuple(weights.shape)} {weights.dtype}"
        )
    if context_lens.dtype != torch.int32:
        raise ValueError(f"context_lens must be int32, got {context_lens.dtype}")
    if (
        block_tables.ndim != 2
        or block_tables.shape[0] != batch
        or block_tables.dtype != torch.int32
    ):
        raise ValueError(
            f"block_tables must be contiguous int32 [B,max_pages], got "
            f"{tuple(block_tables.shape)} {block_tables.dtype}"
        )
    if k not in SUPPORTED_K:
        raise ValueError(f"k must be one of {SUPPORTED_K}, got {k}")

    max_history = block_tables.shape[1] * page_size
    if preshuffled and page_size % 16:
        raise ValueError(
            f"preshuffled kv_cache requires page_size divisible by 16, got {page_size}"
        )
    if num_splits is None:
        num_splits = _auto_num_splits(q_fp8, max_history, k)
    if not isinstance(num_splits, int) or num_splits <= 0:
        raise ValueError(f"num_splits must be a positive integer, got {num_splits}")
    max_split_span = (max_history + num_splits - 1) // num_splits
    if max_split_span > 65535:
        raise ValueError(
            "each Stage-A split must span at most 65535 positions for the "
            f"uint16 local-position reservoir; got at most {max_split_span}"
        )

    row_context_lens = _normalize_context_lens(context_lens, batch, next_n)
    candidate_scores = torch.empty(
        (rows, num_splits, k), dtype=torch.float32, device=device
    )
    candidate_positions = torch.empty(
        (rows, num_splits, k), dtype=torch.int32, device=device
    )
    candidate_counts = torch.empty((rows, num_splits), dtype=torch.int32, device=device)

    stream = torch.cuda.current_stream(device)
    with torch.cuda.device(device):
        launch_fp8_paged_mqa_local_topk(
            q_fp8,
            kv_cache,
            k_scales,
            weights,
            row_context_lens,
            block_tables,
            candidate_scores,
            candidate_positions,
            candidate_counts,
            topk=int(k),
            num_splits=int(num_splits),
            preshuffled=preshuffled,
            arch=arch,
            stream=stream,
            packed=packed,
        )
    return candidate_scores, candidate_positions, candidate_counts


def flydsl_fp8_paged_mqa_topk(
    q_fp8,
    kv_cache,
    k_scales,
    weights,
    context_lens,
    block_tables,
    *,
    k=2048,
    num_splits=None,
):
    """Compute exact TopK through compact split-local candidate bags.

    Packed pages may pass ``k_scales=None``. Lengths and block-table ids follow
    the same safety contract as ``flydsl_fp8_paged_mqa_local_topk``.
    """
    for name, tensor in (
        ("q_fp8", q_fp8),
        ("kv_cache", kv_cache),
        ("weights", weights),
        ("context_lens", context_lens),
        ("block_tables", block_tables),
    ):
        _require_cuda_contiguous(name, tensor)
    packed_shape = _packed_page_shape(kv_cache)
    if packed_shape is None:
        _require_cuda_contiguous("k_scales", k_scales)

    device = q_fp8.device
    kv_cache, k_scales, packed, num_pages, page_size = _normalize_kv_inputs(
        kv_cache, k_scales, q_fp8.dtype
    )
    if num_pages < 1:
        raise ValueError("kv_cache must contain at least one page")
    if any(
        tensor.device != device
        for tensor in (
            kv_cache,
            k_scales,
            weights,
            context_lens,
            block_tables,
        )
    ):
        raise ValueError("all inputs must be on the same device")
    if q_fp8.ndim != 4:
        raise ValueError(
            "this specialization requires q_fp8 shape [B,next_n,32,128], "
            f"got {tuple(q_fp8.shape)}"
        )
    batch, next_n, heads, head_dim = q_fp8.shape
    if next_n < 1 or (heads, head_dim) != (32, 128):
        raise ValueError(
            "this specialization requires q_fp8 shape [B,next_n,32,128] "
            "with next_n >= 1, "
            f"got {tuple(q_fp8.shape)}"
        )
    rows = batch * next_n
    if q_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(f"q_fp8 must be float8_e4m3fn, got {q_fp8.dtype}")
    if page_size != 64:
        raise ValueError(
            "this specialization requires page_size 64, " f"got {page_size}"
        )
    if weights.shape != (rows, 32) or weights.dtype != torch.float32:
        raise ValueError(f"weights must be float32 with shape {(rows, 32)}")
    if context_lens.dtype != torch.int32:
        raise ValueError(f"context_lens must be int32, got {context_lens.dtype}")
    if (
        block_tables.ndim != 2
        or block_tables.shape[0] != batch
        or block_tables.dtype != torch.int32
    ):
        raise ValueError("block_tables must be int32 [B,max_pages]")
    if k not in SUPPORTED_K:
        raise ValueError(f"k must be one of {SUPPORTED_K}, got {k}")

    arch = _arch_name(device)
    row_context_lens = _normalize_context_lens(
        context_lens,
        batch,
        next_n,
    )
    max_history = block_tables.shape[1] * 64
    if num_splits is None:
        num_splits = _auto_num_splits(q_fp8, max_history, k)
    if not isinstance(num_splits, int) or num_splits <= 0:
        raise ValueError(f"num_splits must be a positive integer, got {num_splits}")
    max_split_span = (max_history + num_splits - 1) // num_splits
    if max_split_span > 65535:
        raise ValueError(
            "each Stage-A split must span at most 65535 positions for the "
            f"uint16 local-position reservoir; got at most {max_split_span}"
        )

    candidate_scores = torch.empty(
        (rows, num_splits, k),
        dtype=torch.float32,
        device=device,
    )
    candidate_positions = torch.empty(
        (rows, num_splits, k),
        dtype=torch.int32,
        device=device,
    )
    candidate_counts = torch.empty(
        (rows, num_splits),
        dtype=torch.int32,
        device=device,
    )
    # Stage A holds every candidate's key in registers anyway, so it can fill
    # the merge's pass-0 histogram on the way out and let the merge start at
    # pass 1. Only the multi-split path has a merge to prepare for.
    workspace = split_topk_merge_workspace(device, rows) if num_splits > 1 else None
    stream = torch.cuda.current_stream(device)
    with torch.cuda.device(device):
        launch_fp8_paged_mqa_local_topk(
            q_fp8,
            kv_cache,
            k_scales,
            weights,
            row_context_lens,
            block_tables,
            candidate_scores,
            candidate_positions,
            candidate_counts,
            topk=k,
            num_splits=num_splits,
            preshuffled=True,
            arch=arch,
            stream=stream,
            packed=packed,
            prepare_merge=workspace is not None,
            merge_histogram=workspace[0] if workspace is not None else None,
            merge_state=workspace[1] if workspace is not None else None,
        )
        if num_splits == 1:
            values = candidate_scores[:, 0]
            positions = candidate_positions[:, 0]
        else:
            values, positions = split_topk_merge(
                candidate_scores,
                candidate_positions,
                candidate_counts,
                k=k,
                precomputed_first_pass=True,
                workspace=workspace,
            )
    return values, positions


def merge_local_topk_candidates(
    candidate_scores: torch.Tensor,
    candidate_positions: torch.Tensor,
    candidate_counts: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select directly from the fixed-width Stage-A candidate buffer.

    Invalid local slots are already ``(-inf, -1)``, so they can remain in the
    row. This avoids a stable partition and preserves a graph-static ``S*k``
    merge width.
    """
    if candidate_scores.ndim != 3:
        raise ValueError("candidate_scores must have shape [rows,splits,local_k]")
    if candidate_positions.shape != candidate_scores.shape:
        raise ValueError("candidate_positions must match candidate_scores shape")
    rows, splits, local_k = candidate_scores.shape
    if candidate_counts.shape != (rows, splits):
        raise ValueError("candidate_counts must have shape [rows,splits]")
    if k > splits * local_k:
        raise ValueError(f"k={k} exceeds candidate width {splits * local_k}")
    if splits > 1 and local_k == k:
        return split_topk_merge(
            candidate_scores,
            candidate_positions,
            candidate_counts,
            k=k,
        )
    if splits == 1:
        return candidate_scores[:, 0, :k], candidate_positions[:, 0, :k]

    merge_scores = candidate_scores.reshape(rows, splits * local_k)
    merge_positions = candidate_positions.reshape(rows, splits * local_k)
    device = candidate_scores.device
    merge_widths = _full_merge_widths(
        device.type,
        device.index if device.index is not None else torch.cuda.current_device(),
        rows,
        splits * local_k,
    )

    merge_slots = torch.empty((rows, k), dtype=torch.int32, device=device)
    selected_scores = torch.empty((rows, k), dtype=torch.float32, device=device)
    flydsl_top_k_per_row_decode(
        logits=merge_scores,
        next_n=1,
        seq_lens=merge_widths,
        indices=merge_slots,
        num_rows=rows,
        stride0=merge_scores.stride(0),
        stride1=1,
        k=k,
        stable=False,
        values=selected_scores,
    )
    selected_positions = merge_positions.gather(1, merge_slots.to(torch.int64))
    return selected_scores, selected_positions
