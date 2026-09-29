# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime arch allowlist for the FlyDSL QSA wrappers."""

_QSA_ARCHS = ("gfx942", "gfx950")


def qsa_device_arch(gcn_arch_name: str) -> str:
    """Return ``gfx942`` or ``gfx950`` from a device ``gcnArchName``.

    The ISA token is the text before the first colon, so
    ``gfx950:sramecc+:xnack-`` is gfx950. A longer token such as
    ``gfx9420`` is not gfx942. Anything else raises; callers must not
    treat it as the gfx942 tile.
    """
    name = gcn_arch_name.split(":", 1)[0]
    if name not in _QSA_ARCHS:
        raise ValueError(f"QSA supports gfx942 and gfx950, got {gcn_arch_name!r}")
    return name
