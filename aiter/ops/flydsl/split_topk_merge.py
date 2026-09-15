# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Merge unordered Stage A bags with the existing decode TopK operator."""

from functools import cache

import torch

from aiter.ops.flydsl.topk_per_row import flydsl_top_k_per_row_decode


@cache
def _full_widths(device_index: int, rows: int, width: int) -> torch.Tensor:
    return torch.full(
        (rows,),
        width,
        dtype=torch.int32,
        device=torch.device("cuda", device_index),
    )


def clear_split_topk_merge_workspace_cache() -> None:
    _full_widths.cache_clear()


def _row_ends(device: torch.device, rows: int, width: int) -> torch.Tensor:
    if torch.cuda.is_current_stream_capturing():
        return torch.full(
            (rows,),
            width,
            dtype=torch.int32,
            device=device,
        )
    return _full_widths(device.index, rows, width)


def split_topk_merge(
    candidate_scores: torch.Tensor,
    candidate_positions: torch.Tensor,
    candidate_counts: torch.Tensor,
    *,
    k: int,
    precomputed_first_pass: bool = False,
    workspace=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge unordered split-local TopK pairs into an unordered row TopK.

    Unused slots are masked from ``candidate_counts`` so a caller that left
    padding as ``+inf`` still gets an exact set. Stage A itself already writes
    ``(-inf, -1)`` into those slots. ``precomputed_first_pass`` and
    ``workspace`` are accepted for call-site compatibility and ignored: merge
    uses the public decode TopK on the compact ``S*k`` bag.
    """
    del precomputed_first_pass, workspace
    if candidate_scores.ndim != 3:
        raise ValueError("candidate_scores must have shape [rows,splits,local_k]")
    if candidate_positions.shape != candidate_scores.shape:
        raise ValueError("candidate_positions must match candidate_scores")
    rows, splits, local_k = candidate_scores.shape
    if candidate_counts.shape != (rows, splits):
        raise ValueError("candidate_counts must have shape [rows,splits]")
    if local_k != k:
        raise ValueError(f"local_k must equal k, got {local_k} and {k}")
    if splits <= 1:
        raise ValueError("split-aware merge requires at least two splits")
    if candidate_scores.dtype != torch.float32:
        raise TypeError("candidate_scores must be float32")
    if candidate_positions.dtype != torch.int32:
        raise TypeError("candidate_positions must be int32")
    if candidate_counts.dtype != torch.int32:
        raise TypeError("candidate_counts must be int32")
    if not candidate_scores.is_cuda:
        raise ValueError("candidate tensors must be on a CUDA/HIP device")
    if (
        candidate_positions.device != candidate_scores.device
        or candidate_counts.device != candidate_scores.device
    ):
        raise ValueError("candidate tensors must share one device")
    if not candidate_scores.is_contiguous() or not candidate_positions.is_contiguous():
        raise ValueError("candidate scores and positions must be contiguous")
    if not candidate_counts.is_contiguous():
        raise ValueError("candidate_counts must be contiguous")

    device = candidate_scores.device
    width = splits * k
    live = torch.arange(k, device=device, dtype=torch.int32).view(
        1, 1, k
    ) < candidate_counts.unsqueeze(-1)
    neg_inf = candidate_scores.new_full((), float("-inf"))
    merge_scores = torch.where(live, candidate_scores, neg_inf)
    merge_scores = torch.nan_to_num(merge_scores, nan=float("-inf")).reshape(
        rows, width
    )
    merge_positions = candidate_positions.reshape(rows, width)
    merge_slots = torch.empty((rows, k), dtype=torch.int32, device=device)
    selected_scores = torch.empty((rows, k), dtype=torch.float32, device=device)
    flydsl_top_k_per_row_decode(
        logits=merge_scores,
        next_n=1,
        seq_lens=_row_ends(device, rows, width),
        indices=merge_slots,
        num_rows=rows,
        stride0=merge_scores.stride(0),
        stride1=1,
        k=k,
        stable=False,
        values=selected_scores,
    )
    selected_positions = merge_positions.gather(1, merge_slots.to(torch.int64))
    selected_scores = torch.where(selected_positions < 0, neg_inf, selected_scores)
    return selected_scores, selected_positions
