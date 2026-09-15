# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
from contextlib import contextmanager

import torch


@functools.cache
def _persistent_pool(index: int) -> "torch.cuda.MemPool":
    # The cache also owns the pool: destroying the MemPool frees its memory and
    # leaves every pointer handed out from it dangling.
    with torch.cuda.device(index):
        return torch.cuda.MemPool()


@contextmanager
def persistent_alloc(device: torch.device):
    """Allocate buffers that outlive a CUDA graph capture.

    A buffer allocated inside a capture is served from that graph's private
    mempool, where it can inherit the address of an intermediate tensor the same
    capture freed. The kernel writing that address is already a node in the
    graph, so every replay overwrites the buffer. A separate MemPool is not part
    of that reuse chain.
    """
    index = torch.cuda.current_device() if device.index is None else device.index
    with torch.cuda.use_mem_pool(_persistent_pool(index)):
        yield
