# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL -- high-performance GPU kernels implemented using FlyDSL.

Kernel compilation and public APIs are only available when a compatible
``flydsl`` package is installed. Use ``is_flydsl_available()`` to check
whether the optional dependency exists before relying on FlyDSL kernels.
"""

from packaging.version import Version

from .utils import is_flydsl_available

_MIN_FLYDSL_VERSION = Version("0.1.3")

__all__ = [
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

    from .gemm_kernels import flydsl_hgemm, flydsl_preshuffle_gemm_a8
    from .moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    # from .linear_attention_kernels import flydsl_gdr_decode
    from .linear_attention_prefill_kernels import flydsl_chunk_gated_delta_rule_fwd_h

    __all__ += [
        "flydsl_preshuffle_gemm_a8",
        "flydsl_moe_stage1",
        "flydsl_moe_stage2",
        "flydsl_hgemm",
        # "flydsl_gdr_decode",
        "flydsl_chunk_gated_delta_rule_fwd_h",
    ]
