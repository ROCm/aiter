# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Pieces shared by the conv3d kernel modules.

``conv3d_implicit.py`` and ``conv3d_transpose.py`` are two independent kernels
that happen to be launched back to back, so what they have in common is only
this: the element width they both assume and the compile hints they are both
built under, plus the stream coercion their launchers take. Keeping it here is
what lets the transpose stand on its own without either module importing the
other. Running a compiled launcher is a host concern and lives in
``../conv_kernels.py`` with the rest of the dispatch.
"""

import flydsl.expr as fx

BF16_BYTES = 2

# Compile hints applied to both conv3d kernels. Empty by default; a caller that
# needs to pass FlyDSL a hint sets it before the first compile.
CONV_COMPILE_HINTS = {}


def _as_stream(stream):
    return stream if hasattr(stream, "_is_stream_param") else fx.Stream(stream)
