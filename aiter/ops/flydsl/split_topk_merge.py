# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-aware exact TopK merge for fixed-width local candidate bags."""

from functools import cache

import torch

from .kernels.tensor_shim import _run_compiled
from .kernels.topk_per_row_decode import (
    _STATE_SIZE,
    build_topk_per_row_decode_module,
)

_RADIX_BINS = 1 << 11
_BLOCK_THREADS = 256
_WAVE_SIZE = 64


def split_topk_merge_workspace_shapes(rows: int):
    """``(histogram, state)`` shapes for a caller-owned merge workspace.

    Histogram is ``[rows, 1, 2048]`` int32 (11-bit pass-0 bins). State is
    ``[rows, 6]`` int32. Stage A adds into the histogram; a completed merge
    writes every bin back to 0, so the pair is reusable with no ``zero_()``.
    """
    return (rows, 1, _RADIX_BINS), (rows, _STATE_SIZE)


@cache
def _full_widths(device_index: int, rows: int, width: int) -> torch.Tensor:
    return torch.full(
        (rows,),
        width,
        dtype=torch.int32,
        device=torch.device("cuda", device_index),
    )


@cache
def _split_workspace(
    device_index: int,
    stream_id: int,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del stream_id
    device = torch.device("cuda", device_index)
    return (
        torch.zeros((rows, 1, _RADIX_BINS), dtype=torch.int32, device=device),
        torch.empty((rows, _STATE_SIZE), dtype=torch.int32, device=device),
    )


def clear_split_topk_merge_workspace_cache() -> None:
    _full_widths.cache_clear()
    _split_workspace.cache_clear()


@cache
def _build_split_topk_merge(k: int, splits: int, precomputed_first_pass: bool):
    # One chunk per split, and `split_width` tells the chunk that its slice of
    # the row is a bag of `k` slots whose live prefix is `candidate_counts`.
    # NaN sinks to the bottom so a NaN scale can never displace a real score.
    return build_topk_per_row_decode_module(
        k,
        stable=False,
        wave_size=_WAVE_SIZE,
        write_values=True,
        chunks_per_row=splits,
        block_threads=_BLOCK_THREADS,
        split_width=k,
        payload_indices=True,
        nan_to_bottom=True,
        combine_histograms=True,
        use_split_counts=True,
        precomputed_first_pass=precomputed_first_pass,
    )


def alloc_split_topk_merge_workspace(
    device: torch.device, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate a clean workspace once. Do not call this inside a CUDA graph."""
    hist_shape, state_shape = split_topk_merge_workspace_shapes(rows)
    return (
        torch.zeros(hist_shape, dtype=torch.int32, device=device),
        torch.empty(state_shape, dtype=torch.int32, device=device),
    )


def split_topk_merge_workspace(
    device: torch.device,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cached histogram and state used when the caller passes none.

    First miss allocates zeros. Capture reuses that tensor; it does not
    ``zeros()`` again. Serving should pass its own pair instead of this cache.
    """
    stream = torch.cuda.current_stream(device)
    return _split_workspace(device.index, stream.cuda_stream, rows)


def _require_merge_workspace(
    workspace: tuple[torch.Tensor, torch.Tensor],
    *,
    rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(workspace) != 2:
        raise ValueError("workspace must be (histogram, state)")
    histogram, state = workspace
    hist_shape, state_shape = split_topk_merge_workspace_shapes(rows)
    for name, tensor, expected in (
        ("workspace histogram", histogram, hist_shape),
        ("workspace state", state, state_shape),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous int32")
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
            )
    return histogram, state


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
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    out_scores: torch.Tensor | None = None,
    out_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge unordered split-local TopK pairs into an unordered row TopK.

    ``candidate_counts`` bounds the live prefix of each split's bag, so the
    slots Stage A left as ``(-inf, -1)`` are never read. Pass
    ``precomputed_first_pass`` when Stage A already filled the pass-0
    histogram in ``workspace``; the merge then skips that data pass.

    A caller-owned ``workspace`` is not zeroed here. It must be zeros before
    Stage A; after this merge it is zeros again.
    """
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
    scores = candidate_scores.view(rows, width)
    positions = candidate_positions.view(rows, width)
    # Skipping pass 0 means the gather can leave a slot untouched when a row
    # has fewer than k live candidates, so those rows must start as (-inf, -1).
    if out_scores is None:
        selected_scores = (
            torch.full((rows, k), -float("inf"), dtype=torch.float32, device=device)
            if precomputed_first_pass
            else torch.empty((rows, k), dtype=torch.float32, device=device)
        )
    else:
        selected_scores = out_scores
    if out_positions is None:
        selected_positions = (
            torch.full((rows, k), -1, dtype=torch.int32, device=device)
            if precomputed_first_pass
            else torch.empty((rows, k), dtype=torch.int32, device=device)
        )
    else:
        selected_positions = out_positions
    row_ends = _row_ends(device, rows, width)
    stream = torch.cuda.current_stream(device)
    if workspace is None:
        partial_hist, state = split_topk_merge_workspace(device, rows)
    else:
        partial_hist, state = _require_merge_workspace(
            workspace, rows=rows, device=device
        )
    launcher = _build_split_topk_merge(k, splits, precomputed_first_pass)
    _run_compiled(
        launcher,
        scores,
        positions,
        candidate_counts,
        row_ends,
        selected_positions,
        selected_scores,
        partial_hist,
        state,
        width,
        1,
        width,
        rows,
        stream,
    )
    return selected_scores, selected_positions
