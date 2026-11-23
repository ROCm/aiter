# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

_LDS_CAPACITY_BYTES = {
    "gfx90a": 64 * 1024,
    "gfx942": 64 * 1024,
    "gfx950": 160 * 1024,
    "gfx1100": 64 * 1024,
    "gfx1101": 64 * 1024,
    "gfx1102": 64 * 1024,
    "gfx1150": 64 * 1024,
    "gfx1151": 64 * 1024,
    "gfx1200": 64 * 1024,
    "gfx1201": 64 * 1024,
    "gfx1250": 320 * 1024,
    "gfx1250-strict": 320 * 1024,
}


def get_lds_capacity_bytes(gfx: str) -> int:
    """Return architectural workgroup LDS capacity for an explicit target."""
    arch = gfx.split(":", 1)[0].lower()
    try:
        return _LDS_CAPACITY_BYTES[arch]
    except KeyError as exc:
        raise ValueError(f"Unknown LDS capacity for architecture {arch!r}") from exc
