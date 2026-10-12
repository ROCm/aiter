# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""SP4 head exchange into contiguous token-major rows."""

import torch

from aiter.jit.core import compile_ops


@compile_ops("module_sp_head_exchange", develop=True)
def sp_head_exchange(
    handle: int,
    input: torch.Tensor,
    output: torch.Tensor,
    registered_buffer: int,
    registered_bytes: int,
    stage: bool,
    blocks: int,
) -> None:
    """Exchange head shards using a four-rank custom-allreduce communicator.

    Contiguous matrices map ``[tokens, heads]`` to ``[tokens / 4, heads * 4]``.
    With ``stage=True``, copy input into the registered buffer before exchange;
    otherwise input must already be registered with the communicator. Calls
    sharing the communicator must be serialized on its current stream.
    """
