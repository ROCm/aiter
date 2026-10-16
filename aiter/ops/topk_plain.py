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
    topk_out: torch.Tensor,
    topk: int,
    largest: bool = True,
    rowStarts: torch.Tensor = None,
    rowEnds: torch.Tensor = None,
    stride0: int = -1,
    stride1: int = 1,
    workspace: torch.Tensor | None = None,
) -> None: ...


@compile_ops("module_topk_plain")
def topk_plain_workspace_size(numRows: int, stride0: int, k: int) -> int: ...


@compile_ops("module_topk_plain")
def topk_plain_merge_workspace_size(k: int) -> int: ...


@functools.cache
def _merge_workspace_size(device: torch.device, k: int) -> int:
    # The bound comes from occupancy queries against the current device, so it
    # is cached per device rather than recomputed on every call.
    with torch.cuda.device(device):
        return topk_plain_merge_workspace_size(k)


# Mirrors buffer_load_helpers::MAX_CAPACITY in csrc/kernels/topk_plain_kernels.cu.
_MAX_CAPACITY = 2048


def topk_plain_batches_ragged_rows(width: int, topk: int) -> bool:
    """Would a `rowStarts`/`rowEnds` call stay on the batched launcher?

    Mirrors `should_use_topk_radix` in csrc/kernels/topk_plain_kernels.cu, which
    the variable-length `AdaptiveTopK` consults. When it holds, one batched
    launch serves every row. When it does not, that overload falls back to a
    host-side loop calling the single-row selector once per row -- measured at
    k=16, N=32768, M=16384 as 294918 profiler events per call against 25 for the
    uniform form, and the cost grows with the row count while this test does not
    look at the row count at all.

    Exposed so a caller choosing between kernels can avoid that fallback rather
    than discover it.
    """
    if topk <= 1:
        return False
    denom = max(0.0001, math.log2(width) - 9.5)
    log_k = math.log2(topk)
    return topk * log_k * log_k >= (4.8 * width) / denom


def topk_plain(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_out: torch.Tensor,
    topk: int,
    largest: bool = True,
    rowStarts: torch.Tensor = None,
    rowEnds: torch.Tensor = None,
    stride0: int = -1,
    stride1: int = 1,
) -> None:
    """Plain top-k over the last dim.

    Two kernel paths need device scratch: the fp32 radix path, and the
    multi-block path (any dtype) that splits each row across several blocks when
    there are fewer rows than the grid holds and merges their partial results.
    The workspace is allocated here on the Python side via torch's caching
    allocator and passed into the kernel, so the C++ side never allocates device
    memory itself and the call stays capturable in a CUDA graph. One call takes
    at most one of the two paths, so they share the buffer.
    """
    if topk > _MAX_CAPACITY:
        # `AdaptiveTopK` asserts this at its entry, ahead of any dtype branch,
        # so it bounds every path. The assert also compiles out under NDEBUG,
        # where the same call reads past the buffer instead of aborting -- which
        # is why the bound is restated here rather than left to the C++ side.
        raise ValueError(
            f"topk={topk} exceeds this kernel's capacity of {_MAX_CAPACITY}"
        )
    size = _merge_workspace_size(x.device, topk)
    if x.dtype == torch.float32:
        # Mirror the C++ default: stride0 < 0 means a contiguous last dim.
        s0 = stride0 if stride0 >= 0 else x.shape[-1]
        size = max(size, topk_plain_workspace_size(x.shape[0], s0, topk))
    workspace = get_topk_scratch_workspace(x.device, size)
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
    )
