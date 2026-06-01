# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""General utilities shared across all FlyDSL kernel families."""

import importlib.util
from functools import lru_cache

import torch

_FALLBACK_MAX_LDS_BYTES = 65536


def addressable_lds_bytes_for_gfx(gfx: str) -> int:
    g = (gfx or "").strip().lower().split(":")[0]
    if not g.startswith("gfx"):
        return _FALLBACK_MAX_LDS_BYTES
    if g.startswith("gfx950"):
        return 163840
    if g.startswith("gfx7") or g.startswith("gfx8"):
        return 32768
    return 65536


@lru_cache(maxsize=1)
def _default_cuda_device_index():
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return None


@lru_cache(maxsize=None)
def _get_shared_memory_per_block_cached(device_index: int, fallback_gfx: str) -> int:
    try:
        props = torch.cuda.get_device_properties(device_index)
        shared_memory_per_block = int(getattr(props, "shared_memory_per_block", 0) or 0)
        if shared_memory_per_block > 0:
            return shared_memory_per_block
        return addressable_lds_bytes_for_gfx(
            getattr(props, "gcnArchName", fallback_gfx)
        )
    except Exception:
        return addressable_lds_bytes_for_gfx(fallback_gfx)


def get_shared_memory_per_block(device=None, fallback_gfx: str = "") -> int:
    """Return per-block shared memory/LDS limit for the active device."""
    if device is None:
        device = _default_cuda_device_index()
    elif isinstance(device, torch.device):
        if device.type != "cuda":
            device = None
        elif device.index is None:
            device = _default_cuda_device_index()
        else:
            device = int(device.index)
    else:
        try:
            device = int(device)
        except Exception:
            device = None

    if device is None:
        return addressable_lds_bytes_for_gfx(fallback_gfx)
    return _get_shared_memory_per_block_cached(device, fallback_gfx)


def is_flydsl_available() -> bool:
    return importlib.util.find_spec("flydsl") is not None
