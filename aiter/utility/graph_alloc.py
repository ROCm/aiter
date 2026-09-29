# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
from contextlib import contextmanager

import torch
from packaging.version import Version

# Before 2.10 the allocator scans captures_underway in registration order, so
# the graph pool wins and use_mem_pool is ignored inside a capture. Parsed, not
# string-compared: "2.9.1" sorts above "2.10".
ROUTES_INSIDE_CAPTURE = Version(torch.__version__.split("+")[0]) >= Version("2.10")


@functools.cache
def _persistent_pool(index: int) -> "torch.cuda.MemPool":
    # Concurrent first callers can each build one and only one is kept: the cost
    # is a duplicate set of segments, not a dangling pointer.
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
    if not ROUTES_INSIDE_CAPTURE and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"aiter: a cached scratch buffer was first allocated on device {index} "
            f"while a CUDA graph was capturing, and torch {torch.__version__} "
            "ignores use_mem_pool inside a capture, so the buffer would come from "
            "the graph's private pool and could inherit the address of an "
            "intermediate that every replay overwrites. Upgrade to torch 2.10 or "
            "later, or run this op once on the stream you capture on so the "
            "buffer is already cached."
        )
    with torch.cuda.use_mem_pool(_persistent_pool(index), device=index):
        yield
