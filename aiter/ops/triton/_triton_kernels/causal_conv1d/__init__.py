# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton utilities module for causal_conv1d
Provides a consistent way to import triton and triton.language
"""

from typing import TYPE_CHECKING

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

__all__ = ["HAS_TRITON", "triton", "tl"]

