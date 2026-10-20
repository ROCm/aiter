# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cache policies: warm, rotate (input copies), flush (cache buster)."""

import copy
import math
from dataclasses import dataclass

import torch

MiB = 1024 * 1024

# Last-level cache per arch. CDNA3/CDNA4 put a 256 MB Infinity Cache (MALL)
# behind the per-XCD L2; torch only reports the 4 MB L2, so a buster sized from
# L2_cache_size would leave inputs resident in the MALL.
_LLC_BYTES = {
    "gfx942": 256 * MiB,
    "gfx950": 256 * MiB,
}

# Never spend more than this fraction of free device memory on input copies.
_ROTATE_FREE_MEM_FRACTION = 0.5


def gfx_of(device=None) -> str:
    props = torch.cuda.get_device_properties(device)
    return getattr(props, "gcnArchName", "").split(":", 1)[0]


def llc_bytes(device=None) -> int:
    """Size of the cache a cold measurement has to defeat."""
    props = torch.cuda.get_device_properties(device)
    return _LLC_BYTES.get(gfx_of(device), getattr(props, "L2_cache_size", 4 * MiB))


_flush_buffers: dict = {}


def flush_cache(device=None) -> None:
    """Evict the last-level cache by overwriting a buffer twice its size."""
    index = torch.cuda.current_device() if device is None else device
    buf = _flush_buffers.get(index)
    if buf is None:
        buf = torch.empty(2 * llc_bytes(index) // 4, dtype=torch.int32, device=index)
        _flush_buffers[index] = buf
    buf.zero_()


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _tensors(item)


def input_bytes(args, kwargs) -> int:
    """Bytes of distinct device tensors reachable from the call's arguments."""
    seen = set()
    total = 0
    for t in _tensors((args, kwargs)):
        if t.is_cuda and t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.untyped_storage().nbytes()
    return total


@dataclass(frozen=True)
class Rotation:
    copies: list  # [(args, kwargs)], the caller's originals first
    input_bytes: int
    llc_coverage: float  # bytes cycled through / LLC size
    cold: bool  # every timed call reads inputs that are not cache-resident


def make_rotation(args, kwargs, max_copies, timed_calls=None, device=None) -> Rotation:
    """Deep-copy the inputs so that each timed call reads cold inputs.

    Like run_perftest, but sized from the LLC instead of filling free memory.
    Inputs are cold when either every timed call gets its own copy
    (``timed_calls`` copies; None means calls repeat without bound, as in a
    graph replay) or cycling through the copies streams at least twice the
    LLC. The copy count is capped by ``max_copies`` and half the free memory.
    The cache is flushed after copying so no copy starts resident.
    """
    nbytes = input_bytes(args, kwargs)
    if nbytes == 0:
        raise ValueError(
            "cache='rotate' needs the inputs passed as arguments (time_fn(fn, "
            "args=...) or run_bench(fn, *args)); a closure has nothing to "
            "rotate. Use cache='flush' instead."
        )
    llc = llc_bytes(device)
    wanted = max(1, math.ceil(2 * llc / nbytes))
    if timed_calls is not None:
        wanted = min(wanted, timed_calls)
    free = torch.cuda.mem_get_info(device)[0]
    by_memory = max(1, int(free * _ROTATE_FREE_MEM_FRACTION // nbytes))
    n = min(wanted, max_copies, by_memory)
    copies = [(args, kwargs)] + [
        (copy.deepcopy(args), copy.deepcopy(kwargs)) for _ in range(n - 1)
    ]
    flush_cache(device)
    torch.cuda.synchronize()
    cold = n * nbytes >= 2 * llc or (timed_calls is not None and n >= timed_calls)
    return Rotation(
        copies=copies, input_bytes=nbytes, llc_coverage=n * nbytes / llc, cold=cold
    )
