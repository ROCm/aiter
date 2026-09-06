# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL -- high-performance GPU kernels implemented using FlyDSL.

Kernel compilation and public APIs are only available when a compatible
``flydsl`` package is installed. Use ``is_flydsl_available()`` to check
whether the optional dependency exists before relying on FlyDSL kernels.
"""

from packaging.version import Version

from aiter.fused_moe_registry import register_fused_moe_impl

from .moe_common import GateMode
from .utils import is_flydsl_available

_MIN_FLYDSL_VERSION = Version("0.3.2")

__all__ = [
    "GateMode",
    "is_flydsl_available",
]

if is_flydsl_available():
    import flydsl as _flydsl

    installed_flydsl_version = getattr(_flydsl, "__version__", None)
    if installed_flydsl_version is None:
        raise ImportError(
            "`flydsl` is importable but its version cannot be determined."
        )

    _base_version = Version(installed_flydsl_version.split("+")[0])
    if _base_version < _MIN_FLYDSL_VERSION:
        raise ImportError(
            "Unsupported `flydsl` version: "
            f"expected >=`{_MIN_FLYDSL_VERSION}`, "
            f"got `{installed_flydsl_version}`."
        )

    register_fused_moe_impl(
        "flydsl_gfx942",
        "aiter.ops.flydsl.fused_moe_gfx942:run_flydsl_moe_gfx942_impl",
    )
