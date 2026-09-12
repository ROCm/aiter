# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

# user interface

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

    The fp32 radix path needs a device scratch workspace; it is allocated (and
    cached) here on the Python side via torch's caching allocator and passed into
    the kernel, so the C++ side never allocates device memory itself. Non-fp32
    inputs never reach the radix path, so no workspace is allocated for them.
    """
    workspace = None
    if x.dtype == torch.float32:
        # Mirror the C++ default: stride0 < 0 means a contiguous last dim.
        s0 = stride0 if stride0 >= 0 else x.shape[-1]
        size = topk_plain_workspace_size(x.shape[0], s0, topk)
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
