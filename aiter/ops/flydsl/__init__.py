# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL -- high-performance GPU kernels implemented using FlyDSL.

Kernel compilation and public APIs are only available when the ``flydsl``
package is installed.  Use ``is_flydsl_available()`` to check at runtime.
"""

from .utils import is_flydsl_available

__all__ = [
    "is_flydsl_available",
]

if is_flydsl_available():
    from .moe_kernels import (
        flydsl_moe_stage1,
        flydsl_moe_stage2,
    )

    __all__ += [
        "flydsl_moe_stage1",
        "flydsl_moe_stage2",
    ]
