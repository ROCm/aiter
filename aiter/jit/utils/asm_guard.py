# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import functools
import logging

from .chip_info import _current_hip_device, get_asic_revision, get_gfx_runtime
from .torch_guard import torch_compile_guard

logger = logging.getLogger("aiter")


@functools.cache
def _is_gfx1250_asm_supported_cached(device_id: int) -> bool:
    # device_id keys the cache per current HIP device (mixed-stepping nodes);
    # the body reads that same current device via get_asic_revision().
    # Arch undeterminable -> don't block (C++ gate is authoritative for A0).
    try:
        arch = get_gfx_runtime()
    except Exception:  # noqa: BLE001
        return True
    if arch != "gfx1250":
        return True
    # gfx1250: B0+ (asicRevision >= 1) only; unreadable stepping -> fail closed.
    try:
        return get_asic_revision() >= 1
    except Exception as e:  # noqa: BLE001
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
    return _is_gfx1250_asm_supported_cached(_current_hip_device())


def require_gfx1250_asm(op_name: str) -> None:
    """Raise on gfx1250 A0 (shipped asm is B0+ only); no-op otherwise."""
    if is_gfx1250_asm_supported():
        return
    raise RuntimeError(
        f"{op_name} asm is only supported on gfx1250 B0+ "
        "(current device is gfx1250 A0)."
    )
