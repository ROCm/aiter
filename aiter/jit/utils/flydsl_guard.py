# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import functools


@functools.lru_cache(maxsize=1)
def is_flydsl_available() -> bool:
    """Whether the FlyDSL ops can be imported."""
    try:
        import aiter.ops.flydsl  # noqa: F401  availability probe
    except ImportError:
        return False
    return True
