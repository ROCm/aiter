# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton


def parse_triton_version(version: str) -> tuple[int, ...]:
    """Parse version string into comparable tuple format, handling possible development version suffixes."""
    # Remove potential suffixes like .dev, +git etc.
    version = version.split("+")[0].split("-")[0]

    # Split version number and convert to integers
    parts: list[int] = []
    for part in version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


TRITON_VERSION: tuple[int, ...] = parse_triton_version(triton.__version__)

# There are some AOT compiled kernels that were compiled with Triton 3.5.
# Triton 3.6 can't load them because kernel metadata now requires absent properties.
TRITON_VERSION_EQ_3_5: bool = TRITON_VERSION[0:2] == (3, 5)

# Before Triton 3.6.0 AMDMFMALayout instruction shape is 2D [M, N].
# From 3.6.0 onwards it's 3D [M, N, K].
TRITON_VERSION_GE_3_6_0: bool = TRITON_VERSION >= (3, 6, 0)
