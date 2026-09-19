# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import logging
from contextlib import contextmanager

import torch

logger = logging.getLogger("aiter")

# torch < 2.10 scans captures_underway in registration order, so the graph pool
# registered at capture begin always wins and use_mem_pool is ignored inside a
# capture; 2.10 scans it in LIFO order (c10/cuda/CUDACachingAllocator.cpp).
ROUTES_INSIDE_CAPTURE = torch.__version__ >= "2.10"


@functools.cache
def _persistent_pool(index: int) -> "torch.cuda.MemPool":
    # The cache also owns the pool: destroying the MemPool frees its memory and
    # leaves every pointer handed out from it dangling.
    with torch.cuda.device(index):
        return torch.cuda.MemPool()


@functools.cache
def _warn_capture_routing_unavailable() -> None:
    logger.warning(
        "torch %s cannot route an allocation out of a CUDA graph capture; "
        "scratch buffers first allocated during capture may be overwritten on "
        "replay. Upgrade to torch 2.10 or later.",
        torch.__version__,
    )


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
    if not ROUTES_INSIDE_CAPTURE and torch.cuda.is_current_stream_capturing():
        _warn_capture_routing_unavailable()
    with torch.cuda.use_mem_pool(_persistent_pool(index), device=index):
        yield
