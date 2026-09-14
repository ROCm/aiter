# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL per-row argmax -- the k=1 selector."""

from functools import lru_cache

import torch

from .kernels.tensor_shim import _run_compiled
from .kernels.topk_per_row_argmax import (
    _VEC,
    build_topk_per_row_argmax_module,
    topk_per_row_argmax_splits,
)

__all__ = ["topk_per_row_argmax", "topk_per_row_argmax_serves"]


@lru_cache(maxsize=8)
def topk_per_row_argmax_serves(k: int) -> str | None:
    """Why this selector cannot serve, or None if it can.

    Nothing here is sized by the row width or the row count -- the split is a
    runtime grid dimension and the partials follow it -- so k is the whole
    question.
    """
    if k != 1:
        return f"this selector is the k=1 reduction, got k={k}"
    return None


def topk_per_row_argmax(
    scores: torch.Tensor, row_lens: torch.Tensor, indices: torch.Tensor
) -> None:
    """Write each row's argmax column.

    Args:
        scores: ``[rows, width]`` float32, inner stride 1.
        row_lens: ``[rows]`` int32; columns at or past a row's length are
            invisible to it. A row of length 0 yields -1.
        indices: ``[rows, 1]`` int32, written in place.

    Ties go to the smallest column, as ``torch.argmax`` does. NaN outranks
    +inf, which ``torch.argmax`` does not promise.
    """
    rows, width = scores.shape
    splits = topk_per_row_argmax_splits(rows, width)
    slice_launch, fold_launch = build_topk_per_row_argmax_module(splits)
    stream = torch.cuda.current_stream(scores.device)
    vectors = (width + _VEC - 1) // _VEC

    if fold_launch is None:
        # One split writes the answer directly; the partials are unread, so pass
        # the output buffer in their place rather than allocating a pair to
        # ignore.
        _run_compiled(
            slice_launch,
            scores,
            row_lens,
            indices,
            indices,
            indices,
            vectors,
            rows,
            stream,
        )
        return

    # Write-only scratch the caller never sees. Left to the caching allocator
    # rather than kept, the way `get_topk_scratch_workspace` argues for -- a kept
    # buffer would be shared across streams.
    part_key = torch.empty((rows, splits), dtype=torch.int32, device=scores.device)
    part_col = torch.empty_like(part_key)
    _run_compiled(
        slice_launch,
        scores,
        row_lens,
        indices,
        part_key,
        part_col,
        vectors,
        rows,
        stream,
    )
    _run_compiled(
        fold_launch,
        scores,
        row_lens,
        indices,
        part_key,
        part_col,
        vectors,
        rows,
        stream,
    )
