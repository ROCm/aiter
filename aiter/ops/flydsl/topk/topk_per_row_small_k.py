# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL per-row TopK for a small k (block-sparse indexer selection)."""

from functools import lru_cache

import torch

from ..kernels.tensor_shim import _run_compiled, wave_size_of
from ..kernels.topk.topk_per_row_small_k import (
    build_topk_per_row_small_k_module,
    lds_limit,
    topk_per_row_small_k_shape,
)

# The row bound fixes the register tile, so it is a compile-time constant.
# Bucketing to powers of two keeps a serving run to a couple of compilations.
_MIN_ROW_BUCKET = 256


def _row_bucket(width: int) -> int:
    bucket = _MIN_ROW_BUCKET
    while bucket < width:
        bucket *= 2
    return bucket


@lru_cache(maxsize=64)
def topk_per_row_small_k_serves(k: int, width: int, wave_size: int) -> str | None:
    """Why this geometry cannot be built, or None if it can.

    Separate from the tensor contract below because it is what the build
    itself would hit, and because it is a pure function of three integers --
    the hot path asks it through `_plan`, once per shape.
    """
    # One chunk per lane, so k cannot outrun the wave.
    if not 1 <= k <= wave_size:
        return f"k must be in [1, {wave_size}] (one chunk per lane), got {k}"
    if k > width:
        return f"k={k} exceeds the row width {width}"
    # The survivor buffer grows with the bucketed row bound. Over budget the
    # build fails inside the compiler and leaves the HIP context unusable, so
    # this has to be a decline rather than a crash.
    bucket = _row_bucket(width)
    _, _, lds_bytes = topk_per_row_small_k_shape(k, bucket, wave_size)
    if lds_bytes > lds_limit():
        return (
            f"k={k} over a {bucket}-wide row bound needs {lds_bytes} bytes of "
            f"LDS, over the {lds_limit()} limit"
        )
    return None


def _unsupported_reason(
    scores: torch.Tensor,
    row_lens: torch.Tensor,
    indices: torch.Tensor,
    k: int,
) -> str | None:
    """Why this selector cannot serve these tensors, or None if it can.

    The geometry half is shared with the launch path, so the two cannot drift
    into declining a shape that works or accepting one that does not.
    """
    if scores.dim() != 2 or scores.dtype != torch.float32:
        return f"scores must be 2-D float32, got {tuple(scores.shape)} {scores.dtype}"
    if scores.stride(1) != 1:
        return "scores must have inner stride 1"
    rows, width = scores.shape
    if row_lens.shape != (rows,) or row_lens.dtype != torch.int32:
        return f"row_lens must be int32 [{rows}], got {tuple(row_lens.shape)}"
    if indices.shape != (rows, k) or indices.dtype != torch.int32:
        return f"indices must be int32 [{rows}, {k}], got {tuple(indices.shape)}"
    if indices.stride(1) != 1:
        return "indices must have inner stride 1"
    if not (scores.is_cuda and row_lens.is_cuda and indices.is_cuda):
        return "every tensor must be on the GPU"
    return topk_per_row_small_k_serves(k, width, wave_size_of(scores.device.index))


def topk_per_row_small_k_supported(
    scores: torch.Tensor,
    row_lens: torch.Tensor,
    indices: torch.Tensor,
    k: int,
) -> bool:
    """Would :func:`topk_per_row_small_k` serve these tensors?

    Everything it tests is fixed once the buffers are allocated, so a caller
    holding a fallback asks once and keeps the answer -- asking costs 11us,
    against 7-10us of device time for the selection itself.
    """
    return _unsupported_reason(scores, row_lens, indices, k) is None


@lru_cache(maxsize=64)
def _plan(k: int, width: int, wave_size: int, forced_blocks: bool):
    """The launcher for one (k, row-width) shape."""
    reason = topk_per_row_small_k_serves(k, width, wave_size)
    if reason is not None:
        raise ValueError(f"[FlyDSL topk_per_row_small_k] {reason}")
    return build_topk_per_row_small_k_module(
        k,
        _row_bucket(width),
        wave_size,
        # Compile the forced-block pins in only for callers that use them: six
        # instructions per element, and the row read is what a wide row costs --
        # worth 1.07x at 4096 columns rising to 1.29x at 16384. Keyword so every
        # call lands on one `cache` entry.
        forced_blocks=forced_blocks,
    )


def topk_per_row_small_k(
    scores: torch.Tensor,
    row_lens: torch.Tensor,
    indices: torch.Tensor,
    k: int,
    init_blocks: int = 0,
    local_blocks: int = 0,
) -> None:
    """Write each row's top-k column ids, descending by score.

    Args:
        scores: ``[rows, width]`` float32, inner stride 1.
        row_lens: ``[rows]`` int32; columns at or past a row's length are
            invisible to it.
        indices: ``[rows, k]`` int32, written in place. A row with fewer than
            ``k`` live columns is right-padded with -1.
        init_blocks: leading columns forced into the selection.
        local_blocks: trailing columns (relative to each row's length) forced
            into the selection, outranking ``init_blocks``.

    Ties go to the larger column. Only the geometry is checked here, since the
    check is what the cached plan already does; the tensor contract above is
    the caller's, and :func:`topk_per_row_small_k_supported` is how to ask.
    """
    rows, width = scores.shape
    launcher = _plan(
        k, width, wave_size_of(scores.device.index), bool(init_blocks or local_blocks)
    )
    _run_compiled(
        launcher,
        scores,
        row_lens,
        indices,
        int(init_blocks),
        int(local_blocks),
        rows,
        torch.cuda.current_stream(scores.device),
    )
