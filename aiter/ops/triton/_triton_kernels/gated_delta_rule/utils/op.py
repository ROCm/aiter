# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Operation utilities for Triton kernels.

This module provides utility functions and wrappers for common operations
used in Triton kernels, including math functions and TMA descriptors.
"""

import os

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from ..gated_delta_rule_utils import IS_GATHER_SUPPORTED, IS_TMA_SUPPORTED

# Use fast math operations if enabled
if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
    exp2 = tldevice.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    exp = tl.exp
    exp2 = tl.math.exp2
    log = tl.log
    log2 = tl.log2


# Gather operation support
if not IS_GATHER_SUPPORTED:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Fallback gather operation when tl.gather is not supported.
        Returns None to make triton compiler happy.
        """
        return None

else:
    gather = tl.gather


# TMA descriptor support
if IS_TMA_SUPPORTED:
    if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
        # For Triton 3.3.x
        make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
    elif hasattr(triton.language, "make_tensor_descriptor"):
        # For Triton 3.4.x and later
        make_tensor_descriptor = triton.language.make_tensor_descriptor
    else:
        # Should not reach here if IS_TMA_SUPPORTED is True
        @triton.jit
        def make_tensor_descriptor(base, shape, strides, block_shape, _builder=None):
            return None

else:

    @triton.jit
    def make_tensor_descriptor(base, shape, strides, block_shape, _builder=None):
        """
        Fallback implementation when TMA is not supported.
        Returns None to indicate TMA descriptors are unavailable.
        """
        return None
