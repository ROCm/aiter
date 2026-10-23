# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The row-length convention shared by the per-row TopK launchers."""

from functools import cache

import torch


@cache
def no_row_lens(device: torch.device) -> torch.Tensor:
    """The "no per-row length" sentinel: an empty 1-D int32 tensor.

    Empty, so it owns no memory -- nothing for a captured graph to hold an
    address of. 1-D, so it costs a kernel the 4-byte layout of a real length
    tensor rather than a matrix's 16: kernel arguments are fetched in 64-byte
    lines, and 12 bytes is what took the argmax slice past 128.
    """
    return torch.empty(0, dtype=torch.int32, device=device)


def row_len_slot(row_lens: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    """The tensor to hand a kernel's row-length argument.

    `row_lens` None means every row is full-width: the launcher then picks a
    full-rows build, which takes each length from the row width and never reads
    this argument, so the sentinel stands in.
    """
    return no_row_lens(device) if row_lens is None else row_lens
