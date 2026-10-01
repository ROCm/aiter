# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime arch allowlist for the FlyDSL QSA wrappers."""

_QSA_ARCHS = ("gfx942", "gfx950")


def qsa_arch_is_supported(gcn_arch_name: str) -> bool:
    """True when the ISA token is gfx942 or gfx950.

    The token is the text before the first colon, so
    ``gfx950:sramecc+:xnack-`` is gfx950. A longer token such as
    ``gfx9420`` is not gfx942.
    """
    return gcn_arch_name.split(":", 1)[0] in _QSA_ARCHS


def qsa_device_arch(gcn_arch_name: str) -> str:
    """Return ``gfx942`` or ``gfx950`` from a device ``gcnArchName``.

    Anything else raises; callers must not treat it as the gfx942 tile.
    ``auto`` uses :func:`qsa_arch_is_supported` and stays on Triton instead.
    """
    name = gcn_arch_name.split(":", 1)[0]
    if not qsa_arch_is_supported(gcn_arch_name):
        raise ValueError(f"QSA supports gfx942 and gfx950, got {gcn_arch_name!r}")
    return name
