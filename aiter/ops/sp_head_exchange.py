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


@compile_ops("module_sp_head_exchange", develop=True)
def all_to_all(
    handle: int,
    input: torch.Tensor,
    output: torch.Tensor,
    registered_buffer: int,
    registered_bytes: int,
    stage: bool,
    gather_last: bool,
    blocks: int,
    scales: torch.Tensor | None = None,
    scales_out: torch.Tensor | None = None,
) -> None:
    """All-to-all of row blocks over a custom-allreduce communicator.

    ``input`` is a contiguous ``[W * rows, C]`` matrix, ``W`` the world size
    (2, 4 or 8) and ``C * element_size`` a multiple of 16. Row block ``j`` goes
    to rank ``j``. ``gather_last=True`` writes the blocks received from ranks
    0..W-1 side by side into ``output`` ``[rows, W * C]``; ``False`` stacks them
    into ``[W * rows, C]``, the ``all_to_all_single`` layout. ``scales`` and
    ``scales_out`` (same rules, own width) move in the same launch.

    With ``stage=True`` the inputs are first copied into the registered pool
    (``registered_buffer``, ``registered_bytes``); otherwise they must already
    be registered: views of that pool, tensors registered with the communicator,
    or tensors captured in a graph. Calls sharing the communicator must be
    serialized on its current stream; when the launch is done on a rank no peer
    reads its inputs any more.
    """
