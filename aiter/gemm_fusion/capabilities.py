# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Runtime capability probe. Every probe is guarded; none may raise."""

from __future__ import annotations

import importlib.util
from typing import FrozenSet


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def detect_arch() -> str:
    try:
        from aiter.jit.utils.chip_info import get_gfx

        return get_gfx()
    except Exception:
        return "unknown"


def detect_cu() -> int:
    try:
        from aiter.jit.utils.chip_info import get_cu_num

        return int(get_cu_num())
    except Exception:
        return 0


def detect_capabilities(group=None, arch: str = None) -> FrozenSet[str]:
    """Capabilities usable by variants' Contract.requires.

    mori_cco        : optional ``mori`` package importable (installed library only).
    sdma            : mori_cco and an SDMA-capable build (probed via mori attribute; False if unknown).
    gfx950_sc_bypass: arch is gfx950 (SC bypass loads are a gfx950 ISA feature).
    ``group`` is accepted for API symmetry; no collective is run here.
    """
    caps = set()
    arch = arch or detect_arch()
    if arch.startswith("gfx950"):
        caps.add("gfx950_sc_bypass")
    if _has_module("mori"):
        caps.add("mori_cco")
        try:
            import mori  # type: ignore

            if hasattr(mori, "cco") or hasattr(mori, "shmem"):
                caps.add("sdma")
        except Exception:
            pass
    return frozenset(caps)
