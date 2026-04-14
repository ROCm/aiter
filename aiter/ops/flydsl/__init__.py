# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL -- high-performance GPU kernels implemented using FlyDSL.

Kernel compilation and public APIs are only available when a compatible
``flydsl`` package is installed. Use ``is_flydsl_available()`` to check
whether the optional dependency exists before relying on FlyDSL kernels.
"""

from importlib.metadata import PackageNotFoundError, version

from .utils import is_flydsl_available

_REQUIRED_FLYDSL_VERSION = "0.1.3.1"

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

    if installed_flydsl_version != _REQUIRED_FLYDSL_VERSION:
        raise ImportError(
            "Unsupported `flydsl` version: "
            f"expected `{_REQUIRED_FLYDSL_VERSION}`, "
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

    from .linear_attention_kernels import flydsl_gdr_decode

    __all__ += [
        "flydsl_preshuffle_gemm_a8",
        "flydsl_moe_stage1",
        "flydsl_moe_stage2",
        "flydsl_hgemm",
        "flydsl_gdr_decode",
    ]

    try:
        from .rope_kernels import flydsl_fused_qk_rope_reshape_and_cache

        __all__ += ["flydsl_fused_qk_rope_reshape_and_cache"]
    except ImportError:
        pass
