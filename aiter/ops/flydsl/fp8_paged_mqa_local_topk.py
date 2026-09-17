# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public contract for experimental FP8 paged-MQA Stage A local TopK."""

from dataclasses import dataclass

import torch

from .kernels.mqa_logits.fp8_paged_mqa_local_topk import (
    HEAD_DIM,
    INDEX_DIM,
    NUM_XCD,
    SUPPORTED_K,
    WORKGROUPS_PER_CU,
    launch_fp8_paged_mqa_local_topk,
)
from .split_topk_merge import (
    _require_merge_workspace,
    split_topk_merge,
    split_topk_merge_workspace,
)

SUPPORTED_ARCHES = ("gfx950",)


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


def _auto_num_splits(rows: int, device: torch.device, max_history: int, k: int) -> int:
    cu_count = torch.cuda.get_device_properties(device).multi_processor_count
    return _plan_num_splits(rows, max_history, k, cu_count)


@dataclass
class _StageALaunch:
    q_fp8: torch.Tensor
    kv_cache: torch.Tensor
    k_scales: torch.Tensor
    weights: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    packed: bool
    page_size: int
    rows: int
    k: int
    num_splits: int
    arch: str
    stream: torch.cuda.Stream
    candidate_scores: torch.Tensor
    candidate_positions: torch.Tensor
    candidate_counts: torch.Tensor


def _prepare_stage_a(
    q_fp8,
    kv_cache,
    k_scales,
    weights,
    context_lens,
    block_tables,
    *,
    k,
    num_splits,
    preshuffled,
    require_page64=False,
):
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
        for tensor in (kv_cache, k_scales, weights, context_lens, block_tables)
    ):
        raise ValueError("all inputs must be on the same device")

    arch = _arch_name(device)
    if arch not in SUPPORTED_ARCHES:
        raise RuntimeError(
            f"flydsl_fp8_paged_mqa_local_topk is unsupported on {arch}; "
            f"supported architectures: {SUPPORTED_ARCHES}"
        )
    if q_fp8.ndim != 4 or q_fp8.shape[1:] != (1, 32, 128):
        raise ValueError(
            f"q_fp8 must have contiguous shape [B,1,32,128], got {tuple(q_fp8.shape)}"
        )
    if q_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"gfx950 requires torch.float8_e4m3fn q_fp8, got {q_fp8.dtype}"
        )
    if not isinstance(preshuffled, bool):
        raise TypeError(f"preshuffled must be bool, got {type(preshuffled).__name__}")

    rows = q_fp8.shape[0]
    if weights.shape != (rows, 32) or weights.dtype != torch.float32:
        raise ValueError(
            f"weights must be contiguous float32 with shape {(rows, 32)}, "
            f"got {tuple(weights.shape)} {weights.dtype}"
        )
    if context_lens.dtype != torch.int32 or context_lens.shape != (rows,):
        raise ValueError(
            f"context_lens must be contiguous int32 with shape {(rows,)}, "
            f"got {tuple(context_lens.shape)} {context_lens.dtype}"
        )
    if (
        block_tables.ndim != 2
        or block_tables.shape[0] != rows
        or block_tables.dtype != torch.int32
    ):
        raise ValueError(
            f"block_tables must be contiguous int32 [B,max_pages], got "
            f"{tuple(block_tables.shape)} {block_tables.dtype}"
        )
    if k not in SUPPORTED_K:
        raise ValueError(f"k must be one of {SUPPORTED_K}, got {k}")
    if require_page64 and page_size != 64:
        raise ValueError(f"this specialization requires page_size 64, got {page_size}")
    if preshuffled and page_size % 16:
        raise ValueError(
            f"preshuffled kv_cache requires page_size divisible by 16, got {page_size}"
        )

    max_history = block_tables.shape[1] * page_size
    if num_splits is None:
        num_splits = _auto_num_splits(rows, device, max_history, k)
    if not isinstance(num_splits, int) or num_splits <= 0:
        raise ValueError(f"num_splits must be a positive integer, got {num_splits}")
    max_split_span = (max_history + num_splits - 1) // num_splits
    if max_split_span > 65535:
        raise ValueError(
            "each Stage-A split must span at most 65535 positions for the "
            f"uint16 local-position reservoir; got at most {max_split_span}"
        )

    return _StageALaunch(
        q_fp8=q_fp8,
        kv_cache=kv_cache,
        k_scales=k_scales,
        weights=weights,
        context_lens=context_lens,
        block_tables=block_tables,
        packed=packed,
        page_size=page_size,
        rows=rows,
        k=int(k),
        num_splits=int(num_splits),
        arch=arch,
        stream=torch.cuda.current_stream(device),
        candidate_scores=torch.empty(
            (rows, num_splits, k), dtype=torch.float32, device=device
        ),
        candidate_positions=torch.empty(
            (rows, num_splits, k), dtype=torch.int32, device=device
        ),
        candidate_counts=torch.empty(
            (rows, num_splits), dtype=torch.int32, device=device
        ),
    )


def _merge_workspace(workspace, *, rows, device, num_splits, default_cache: bool):
    if num_splits <= 1:
        if workspace is not None:
            raise ValueError("workspace is only used when num_splits > 1")
        return None
    if workspace is not None:
        return _require_merge_workspace(workspace, rows=rows, device=device)
    if default_cache:
        return split_topk_merge_workspace(device, rows)
    return None


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
    workspace=None,
):
    """Compute exact H32D128 FP8 scores and retain local TopK per history split.

    Live candidates are emitted unordered. Positions are logical history positions.
    Set ``preshuffled=True`` when ``kv_cache`` uses
    ``shuffle_weight(..., layout=(16,16))`` within each page. This experimental
    API never allocates a full-width logits tensor.

    ``q_fp8`` is ``[B,1,32,128]``: one decode query per request. Row lengths
    larger than ``block_tables.shape[1] * page_size`` are clamped to that table
    span. Block-table entries outside ``[0, num_pages)`` are not scored (they
    contribute ``-inf``); the kernel will not index the KV cache with those ids.

    If ``workspace`` is set and ``num_splits > 1``, Stage A fills that pair's
    pass-0 histogram for ``split_topk_merge(..., precomputed_first_pass=True)``.
    The pair is not zeroed here; it must already be zeros.
    """
    launch = _prepare_stage_a(
        q_fp8,
        kv_cache,
        k_scales,
        weights,
        context_lens,
        block_tables,
        k=k,
        num_splits=num_splits,
        preshuffled=preshuffled,
    )
    merge_ws = _merge_workspace(
        workspace,
        rows=launch.rows,
        device=launch.q_fp8.device,
        num_splits=launch.num_splits,
        default_cache=False,
    )
    with torch.cuda.device(launch.q_fp8.device):
        launch_fp8_paged_mqa_local_topk(
            launch.q_fp8,
            launch.kv_cache,
            launch.k_scales,
            launch.weights,
            launch.context_lens,
            launch.block_tables,
            launch.candidate_scores,
            launch.candidate_positions,
            launch.candidate_counts,
            topk=launch.k,
            num_splits=launch.num_splits,
            preshuffled=preshuffled,
            arch=launch.arch,
            stream=launch.stream,
            packed=launch.packed,
            prepare_merge=merge_ws is not None,
            merge_histogram=None if merge_ws is None else merge_ws[0],
            merge_state=None if merge_ws is None else merge_ws[1],
        )
    return (
        launch.candidate_scores,
        launch.candidate_positions,
        launch.candidate_counts,
    )


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
    workspace=None,
    out_scores=None,
    out_positions=None,
):
    """Compute exact TopK through compact split-local candidate bags.

    Packed pages may pass ``k_scales=None``. Lengths and block-table ids follow
    the same safety contract as ``flydsl_fp8_paged_mqa_local_topk``.

    Serving / CUDA-graph caller
    ---------------------------
    Allocate the merge workspace **once**, outside capture, then pass the same
    pair on every step. Do not ``zero_()`` between steps.

        from aiter.ops.flydsl.split_topk_merge import (
            alloc_split_topk_merge_workspace,
            split_topk_merge,
        )

        workspace = alloc_split_topk_merge_workspace(device, rows)  # zeros once
        # capture / replay:
        values, positions = flydsl_fp8_paged_mqa_topk(
            q, kv, None, weights, lens, tables, k=2048, workspace=workspace,
        )

    ``rows`` is ``q.shape[0]``. Shapes: histogram ``[rows, 1, 2048]`` int32,
    state ``[rows, 6]`` int32 (see ``split_topk_merge_workspace_shapes``).

    Stage A atomics the retained bag's 11-bit pass-0 histogram into
    ``workspace[0]``. The merge skips that data pass, then writes every bin
    back to 0. If Stage A runs and merge does not, the histogram is dirty.

    Two-kernel form (caller owns the ``[rows, S, k]`` bags too)::

        bags = flydsl_fp8_paged_mqa_local_topk(
            ..., workspace=workspace, preshuffled=True,
        )
        split_topk_merge(
            *bags, k=k, precomputed_first_pass=True, workspace=workspace,
            out_scores=out_scores, out_positions=out_positions,
        )

    ``num_splits == 1`` has no merge and rejects ``workspace``.
    """
    launch = _prepare_stage_a(
        q_fp8,
        kv_cache,
        k_scales,
        weights,
        context_lens,
        block_tables,
        k=k,
        num_splits=num_splits,
        preshuffled=True,
        require_page64=True,
    )
    merge_ws = _merge_workspace(
        workspace,
        rows=launch.rows,
        device=launch.q_fp8.device,
        num_splits=launch.num_splits,
        default_cache=True,
    )
    with torch.cuda.device(launch.q_fp8.device):
        launch_fp8_paged_mqa_local_topk(
            launch.q_fp8,
            launch.kv_cache,
            launch.k_scales,
            launch.weights,
            launch.context_lens,
            launch.block_tables,
            launch.candidate_scores,
            launch.candidate_positions,
            launch.candidate_counts,
            topk=launch.k,
            num_splits=launch.num_splits,
            preshuffled=True,
            arch=launch.arch,
            stream=launch.stream,
            packed=launch.packed,
            prepare_merge=merge_ws is not None,
            merge_histogram=None if merge_ws is None else merge_ws[0],
            merge_state=None if merge_ws is None else merge_ws[1],
        )
        if launch.num_splits == 1:
            values = launch.candidate_scores[:, 0]
            positions = launch.candidate_positions[:, 0]
            if out_scores is not None:
                out_scores.copy_(values)
                values = out_scores
            if out_positions is not None:
                out_positions.copy_(positions)
                positions = out_positions
        else:
            values, positions = split_topk_merge(
                launch.candidate_scores,
                launch.candidate_positions,
                launch.candidate_counts,
                k=launch.k,
                precomputed_first_pass=True,
                workspace=merge_ws,
                out_scores=out_scores,
                out_positions=out_positions,
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

    Invalid local slots are already ``(-inf, -1)``. Stage A always emits
    ``local_k == k``, so multi-split rows go through ``split_topk_merge``.
    """
    if candidate_scores.ndim != 3:
        raise ValueError("candidate_scores must have shape [rows,splits,local_k]")
    if candidate_positions.shape != candidate_scores.shape:
        raise ValueError("candidate_positions must match candidate_scores shape")
    rows, splits, local_k = candidate_scores.shape
    if candidate_counts.shape != (rows, splits):
        raise ValueError("candidate_counts must have shape [rows,splits]")
    if splits == 1:
        return candidate_scores[:, 0, :k], candidate_positions[:, 0, :k]
    if local_k != k:
        raise ValueError(f"local_k must equal k, got {local_k} and {k}")
    return split_topk_merge(
        candidate_scores,
        candidate_positions,
        candidate_counts,
        k=k,
    )
