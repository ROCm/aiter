# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import functools
from contextlib import contextmanager

import torch
from packaging.version import Version

from aiter.jit.utils.hip_runtime import load_hip_runtime

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


def _capture_id() -> int | None:
    """ID of the graph capture the current stream is part of, or None."""
    if not torch.cuda.is_current_stream_capturing():
        return None
    status, capture_id = ctypes.c_int(0), ctypes.c_ulonglong(0)
    err = load_hip_runtime().hipStreamGetCaptureInfo(
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        ctypes.byref(status),
        ctypes.byref(capture_id),
    )
    if err != 0:
        raise RuntimeError(f"hipStreamGetCaptureInfo failed with error {err}")
    return capture_id.value


def capture_scoped_cache(fn):
    """``functools.cache`` for scratch buffers a kernel initializes, e.g. zeroed
    semaphores built with ``torch.zeros`` under ``persistent_alloc``.

    Inside a capture the initializing kernel is recorded into that graph instead
    of running, so the buffer holds whatever the memory held before until the
    graph replays. Sharing it with eager launches or other graphs would run them
    on that garbage (a split-K ticket protocol then spins forever), so a buffer
    first built inside a capture is cached for that capture only: the graph that
    initializes it is the only one that uses it, and it re-initializes the
    buffer on every replay before using it. Buffers built eagerly are already
    initialized and are shared with captures as before.
    """
    cache = {}

    @functools.wraps(fn)
    def wrapper(*args):
        eager = cache.get((None, *args))
        if eager is not None:
            return eager
        key = (_capture_id(), *args)
        buf = cache.get(key)
        if buf is None:
            buf = cache[key] = fn(*args)
        return buf

    wrapper.eager_cached = lambda *args: cache.get((None, *args))
    wrapper.cache_clear = cache.clear
    return wrapper
