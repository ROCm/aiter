# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL per-row argmax -- the k=1 selector."""

from functools import lru_cache

import torch
from flydsl.expr import BFloat16, Float16, Float32

from ..kernels.tensor_shim import _run_compiled
from ..kernels.topk.topk_per_row_argmax import (
    VEC_BY_ELEM,
    build_topk_per_row_argmax_module,
    topk_per_row_argmax_splits,
)

__all__ = [
    "ARGMAX_DTYPES",
    "topk_per_row_argmax",
    "topk_per_row_argmax_serves",
]

# The dtypes with a build. The half formats are here and nowhere else among the
# selectors, because only this one reduces rather than selects: its partials are
# int32 ordering keys, so widening each element as it is read costs nothing that
# survives the load -- and the row stays half the bytes that casting the tensor
# to fp32 first would make it.
_ELEM_BY_DTYPE = {
    torch.float32: Float32,
    torch.bfloat16: BFloat16,
    torch.float16: Float16,
}
ARGMAX_DTYPES = frozenset(_ELEM_BY_DTYPE)


@lru_cache(maxsize=8)
def topk_per_row_argmax_serves(k: int) -> str | None:
    """Why this selector cannot serve, or None if it can.

    Nothing here is sized by the row width or the row count -- the split is a
    runtime grid dimension and the partials follow it -- so k is the whole
    question. The dtype is not: `ARGMAX_DTYPES` is a fact about which builds
    exist, and the caller checks it before the geometry.
    """
    if k != 1:
        return f"this selector is the k=1 reduction, got k={k}"
    return None


def topk_per_row_argmax(
    scores: torch.Tensor, row_lens: torch.Tensor, indices: torch.Tensor
) -> None:
    """Write each row's argmax column.

    Args:
        scores: ``[rows, width]``, inner stride 1, dtype in `ARGMAX_DTYPES`
            (float32, bfloat16 or float16).
        row_lens: ``[rows]`` int32; columns at or past a row's length are
            invisible to it. A row of length 0 yields -1.
        indices: ``[rows, 1]`` int32, written in place.

    Ties go to the smallest column, as ``torch.argmax`` does. NaN outranks
    +inf, which ``torch.argmax`` does not promise.

    Both hold for the half formats without a second ordering rule to keep in
    step with the first: fp32 covers the range AND the precision of bf16 and of
    fp16, subnormals included, so widening either one is exact, hence injective
    and order-preserving. The one rule therefore lands on the same answer, and
    the elements are widened as they are read rather than the tensor being cast.
    """
    elem = _ELEM_BY_DTYPE.get(scores.dtype)
    if elem is None:
        raise ValueError(
            f"scores must be one of {sorted(str(d) for d in ARGMAX_DTYPES)}; "
            f"got {scores.dtype}"
        )
    rows, width = scores.shape
    vec = VEC_BY_ELEM[elem]
    splits = topk_per_row_argmax_splits(rows, width, vec)
    # The count goes in as an argument, not as a build parameter: it follows the
    # row count, so keying the build on it is keying it on the batch size.
    slice_launch, fold_launch = build_topk_per_row_argmax_module(splits == 1, elem)
    stream = torch.cuda.current_stream(scores.device)
    vectors = (width + vec - 1) // vec

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
            splits,
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
        splits,
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
        splits,
        rows,
        stream,
    )
