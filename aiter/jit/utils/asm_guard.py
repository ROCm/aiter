# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import functools
import logging

from .chip_info import get_asic_revision, get_gfx_runtime
from .torch_guard import torch_compile_guard

logger = logging.getLogger("aiter")

_A0_ALLOWLIST: frozenset[str] = frozenset()


def _probe_arch_is_gfx1250() -> bool:
    # Arch undeterminable -> don't block; the C++ gate is authoritative for A0.
    try:
        return get_gfx_runtime() == "gfx1250"
    except Exception:  # noqa: BLE001
        return False


# Module-level bool, not a cached call: the gate runs at the top of hot ops on
# every arch, and a constant lets torch.compile(fullgraph=True) trace through.
_ARCH_IS_GFX1250 = _probe_arch_is_gfx1250()


@functools.cache
def _gfx1250_stepping_ok() -> bool:
    try:
        return get_asic_revision() >= 1
    except Exception as e:  # noqa: BLE001
        # Arch is gfx1250 here, so an unknown stepping may be A0: fail closed.
        logger.warning(
            "gfx1250 asm gate: could not read ASIC revision (%s); "
            "treating device as unsupported (fail-closed).",
            e,
        )
        return False


@torch_compile_guard()
def is_gfx1250_asm_supported() -> bool:
    """False on gfx1250 A0 (shipped asm is B0+ only); True otherwise.

    Frameworks can call this at startup to select a backend before the hard gate.
    """
    return not _ARCH_IS_GFX1250 or _gfx1250_stepping_ok()


def require_gfx1250_asm(op_name: str) -> None:
    """Raise on gfx1250 A0 (shipped asm is B0+ only); no-op otherwise."""
    if not _ARCH_IS_GFX1250 or op_name in _A0_ALLOWLIST or _gfx1250_stepping_ok():
        return
    raise RuntimeError(
        f"{op_name} asm is only supported on gfx1250 B0+ "
        "(current device is gfx1250 A0)."
    )
