# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""General utilities shared across all FlyDSL kernel families."""

import importlib.util

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


def get_shared_memory_per_block(device=None, fallback_gfx: str = "") -> int:
    """Return per-block shared memory/LDS limit for the active device."""
    try:
        if device is None:
            device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        shared_memory_per_block = int(getattr(props, "shared_memory_per_block", 0) or 0)
        if shared_memory_per_block > 0:
            return shared_memory_per_block
        return addressable_lds_bytes_for_gfx(
            getattr(props, "gcnArchName", fallback_gfx)
        )
    except Exception:
        return addressable_lds_bytes_for_gfx(fallback_gfx)


def is_flydsl_available() -> bool:
    return importlib.util.find_spec("flydsl") is not None
