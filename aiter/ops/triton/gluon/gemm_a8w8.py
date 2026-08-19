# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Deprecated alias for the relocated gluon kernel; see this package's __init__."""

import sys

from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a8w8 as _impl

sys.modules[__name__] = _impl
