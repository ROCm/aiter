# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL -- high-performance GPU kernels implemented using FlyDSL.

Kernel compilation and public APIs are only available when a compatible
``flydsl`` package is installed. Use ``is_flydsl_available()`` to check
whether the optional dependency exists before relying on FlyDSL kernels.
"""

from importlib.metadata import PackageNotFoundError, version

from .utils import is_flydsl_available

_REQUIRED_FLYDSL_VERSION = "0.1.2"


def _flydsl_matches_pinned(installed_ver: str, pinned_ver: str) -> bool:
    """Same X.Y.Z as pin, allowing dev/rc/post/local segments (e.g. 0.1.2.dev462 == 0.1.2)."""
    try:
        from packaging.version import Version
    except ImportError:
        return installed_ver == pinned_ver
    return Version(installed_ver).release == Version(pinned_ver).release


__all__ = [
    "is_flydsl_available",
]

if is_flydsl_available():
    try:
        installed_flydsl_version = version("flydsl")
    except PackageNotFoundError as exc:
        raise ImportError(
            "`flydsl` is importable but package metadata is unavailable, "
            "so its version cannot be validated."
        ) from exc

    if not _flydsl_matches_pinned(installed_flydsl_version, _REQUIRED_FLYDSL_VERSION):
        raise ImportError(
            "Unsupported `flydsl` version: "
            f"expected `{_REQUIRED_FLYDSL_VERSION}` (same release, dev/rc/post allowed), "
            f"got `{installed_flydsl_version}`."
        )

    from .gemm_kernels import (
        flydsl_preshuffle_gemm_a8,
    )
    from .moe_kernels import (
        flydsl_moe_stage1,
        flydsl_moe_stage2,
    )

    from .gemm_kernels import flydsl_hgemm

    __all__ += [
        "flydsl_preshuffle_gemm_a8",
        "flydsl_moe_stage1",
        "flydsl_moe_stage2",
        "flydsl_hgemm",
    ]
