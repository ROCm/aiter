# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Experimental single-phase INT4 reduce-scatter using QuickReduce IPC state."""

import torch

from ..jit.core import compile_ops


@compile_ops("module_quick_reduce_scatter", develop=True)
def qr_reduce_scatter(
    handle: int,
    input: torch.Tensor,
    output: torch.Tensor,
    cast_bf2half: bool = True,
) -> None: ...
