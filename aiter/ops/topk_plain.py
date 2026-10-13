# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

# user interface

import functools
import math

import torch

from ..jit.core import (
    compile_ops,
)
from .topk import get_topk_scratch_workspace


@compile_ops("module_topk_plain", fc_name="topk_plain", develop=True)
def _topk_plain(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_out: torch.Tensor | None,
    topk: int,
    largest: bool = True,
    rowStarts: torch.Tensor = None,
    rowEnds: torch.Tensor = None,
    stride0: int = -1,
    stride1: int = 1,
    workspace: torch.Tensor | None = None,
    nan_high: bool = False,
) -> None: ...


@compile_ops("module_topk_plain")
def topk_plain_use_mulblocks(
    numRows: int, stride0: int, k: int, ragged: bool
) -> bool: ...


@compile_ops("module_topk_plain")
def topk_plain_workspace_size(numRows: int, stride0: int, k: int) -> int: ...


@compile_ops("module_topk_plain")
def topk_plain_gfx950_width(name: str) -> int: ...


@functools.lru_cache(maxsize=1024)
def _workspace_plan(device, rows: int, stride0: int, width: int, topk: int):
    # A size query costs a call several microseconds. The size depends on
    # nothing but these; the device is in the key because the radix plan reads
    # its CU count. Beside it, whether the radix path serves the width, which
    # decides whether `topk_out` may be None.
    return topk_plain_workspace_size(rows, stride0, topk), _uses_radix(width, topk)


# Mirrors buffer_load_helpers::MAX_CAPACITY in csrc/kernels/topk_plain_kernels.cu.
_MAX_CAPACITY = 2048


@functools.lru_cache(maxsize=1024)
def _uses_radix(width: int, topk: int) -> bool:
    """Mirrors `should_use_topk_radix` in csrc/kernels/topk_plain_kernels.cu.

    Both `AdaptiveTopK` overloads consult it to choose between the radix
    selector and the block-sort fallback, and the two questions below are both
    really this one.
    """
    if topk <= 1:
        return False
    denom = max(0.0001, math.log2(width) - 9.5)
    log_k = math.log2(topk)
    return topk * log_k * log_k >= (4.8 * width) / denom


def topk_plain_batches_ragged_rows(width: int, topk: int) -> bool:
    """Would a `rowStarts`/`rowEnds` call stay on the batched launcher?

    When the radix path holds, one batched launch serves every row. When it does
    not, the variable-length `AdaptiveTopK` falls back to a host-side loop
    calling the single-row selector once per row -- measured at k=16, N=32768,
    M=16384 as 294918 profiler events per call against 25 for the uniform form,
    and the cost grows with the row count while this test does not look at the
    row count at all.

    Exposed so a caller choosing between kernels can avoid that fallback rather
    than discover it.
    """
    return _uses_radix(width, topk)


def topk_plain_values_optional(width: int, topk: int) -> bool:
    """May `topk_out` be None for this geometry?

    Only the radix path has a build that leaves the selected values unwritten;
    the block-sort fallback writes them unconditionally. The other two
    conditions -- fp32, largest-first -- are properties of the call rather than
    the shape, so a caller doing a half-format or smallest-first selection has
    to pass a buffer whatever this says.

    Worth asking: writing values a caller discards costs 52us of the 741us
    `topk_plain` takes at 4096 rows of 131072 at k=2048.
    """
    return _uses_radix(width, topk)


def _refuse_no_values(x, largest, topk):
    # The C++ side refuses it too, but its check aborts the process under
    # pybind; refused here, the caller gets an exception it can handle.
    raise TypeError(
        "topk_plain: topk_out may be None only for float32 input with "
        "largest=True where topk_plain_values_optional() holds; got "
        f"{x.dtype}, largest={largest}, width={x.shape[-1]}, topk={topk}"
    )


def topk_plain(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_out: torch.Tensor | None,
    topk: int,
    largest: bool = True,
    rowStarts: torch.Tensor = None,
    rowEnds: torch.Tensor = None,
    stride0: int = -1,
    stride1: int = 1,
    nan_high: bool = False,
) -> None:
    """Plain top-k over the last dim.

    The fp32 radix path needs a device scratch workspace; it is allocated (and
    cached) here on the Python side via torch's caching allocator and passed into
    the kernel, so the C++ side never allocates device memory itself. Non-fp32
    inputs never reach the radix path, so no workspace is allocated for them.

    `topk_out` may be None for fp32 input with `largest=True` where
    `topk_plain_values_optional` holds; elsewhere it raises TypeError, as a
    None did before the argument became optional.

    `x` may have any layout. With `stride0 < 0` the row pitch is taken from `x`
    itself, not assumed to be the width.

    NaN: on the radix path (fp32, `largest=True`, where
    `topk_plain_values_optional` holds) +nan ranks above +inf, as in
    `torch.topk`, and a NaN with the sign bit set ranks below -inf, so it is
    selected only when fewer than `topk` other values exist. The block-sort
    fallback, which serves every other call, compares scores as floats and
    gives NaN no defined rank.

    `nan_high` ranks a NaN of either sign above +inf instead, as `topk_select`
    does. It is served by the radix build that writes no values, so it needs
    `topk_out=None`, and with it a geometry `topk_plain_values_optional` holds
    for.
    """
    if topk > _MAX_CAPACITY:
        # `AdaptiveTopK` asserts this at its entry, ahead of any dtype branch,
        # so it bounds every path. The assert also compiles out under NDEBUG,
        # where the same call reads past the buffer instead of aborting -- which
        # is why the bound is restated here rather than left to the C++ side.
        raise ValueError(
            f"topk={topk} exceeds this kernel's capacity of {_MAX_CAPACITY}"
        )
    if nan_high and topk_out is not None:
        raise ValueError(
            "topk_plain: nan_high selects indices only; pass topk_out=None"
        )
    if not x.is_contiguous():
        # Only the ranged form reads row r at `r * stride0`, and it also sizes
        # its launch and scratch by `stride0`, so a strided row is read in place
        # only there, at unit inner stride and a pitch at least the width. The
        # uniform form reads rows `width` apart and takes no pitch at all; it,
        # and every other layout, gets the rows densely.
        ranged = (
            rowStarts is not None
            and rowEnds is not None
            and rowStarts.numel() > 0
            and rowEnds.numel() > 0
        )
        if ranged and x.stride(-1) == 1 and x.stride(0) >= x.shape[-1]:
            if stride0 < 0:
                stride0 = x.stride(0)
        else:
            x = x.contiguous()
            stride0 = -1
    workspace = None
    if x.dtype == torch.float32:
        device, width = x.device, x.shape[-1]
        # Mirror the C++ default: stride0 < 0 means a contiguous last dim.
        s0 = stride0 if stride0 >= 0 else width
        size, radix = _workspace_plan(device, x.shape[0], s0, width, topk)
        if topk_out is None and not (largest and radix):
            _refuse_no_values(x, largest, topk)
        workspace = get_topk_scratch_workspace(device, size)
    elif topk_out is None:
        _refuse_no_values(x, largest, topk)
    return _topk_plain(
        x,
        topk_ids,
        topk_out,
        topk,
        largest,
        rowStarts,
        rowEnds,
        stride0,
        stride1,
        workspace,
        nan_high,
    )
