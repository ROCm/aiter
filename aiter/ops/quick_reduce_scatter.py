# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Experimental single-phase INT4 reduce-scatter using QuickReduce IPC state."""

import torch

from aiter.jit.core import compile_ops


@compile_ops("module_quick_reduce_scatter", develop=True)
def qr_reduce_scatter(
    handle: int,
    input: torch.Tensor,
    output: torch.Tensor,
    cast_bf2half: bool = True,
) -> None:
    """Write this rank's contiguous sum shard using the QuickReduce INT4 codec.

    ``handle`` is an initialized four-rank QuickReduce communicator. ``input``
    and ``output`` are contiguous FP16/BF16 tensors on the same GPU, with four
    input elements per output element and a 16-byte-aligned output byte count.
    BF16 inputs use FP16 communication arithmetic when ``cast_bf2half`` is set.
    Calls sharing the communicator must be serialized on its current stream.
    """
