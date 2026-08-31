# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import functools
import logging

from .chip_info import get_asic_revision, get_gfx_runtime


@functools.lru_cache(maxsize=1)
def is_gfx1250_asm_supported() -> bool:
    """The py_itfs_cu asm kernels target gfx1250 B0 (asicRevision >= 1) only.

    Returns False on gfx1250 A0 so dispatch can fall back to another path or raise.
    """
    try:
        if get_gfx_runtime() != "gfx1250":
            return True
        return get_asic_revision() >= 1
    except Exception:  # noqa: BLE001
        return True


def require_gfx1250_asm(op_name: str) -> None:
    """Warn  and raise on gfx1250 A0 (asm is B0+ only); no-op otherwise."""
    if is_gfx1250_asm_supported():
        return
    logging.getLogger("aiter").warning(
        "\033[93m[SKIP] %s asm is gfx1250 B0-only supported "
        "(current device is gfx1250 A0)\033[0m",
        op_name,
    )
    raise RuntimeError(
        f"{op_name} asm is only supported on gfx1250 B0+ (current device is gfx1250 A0)"
    )
