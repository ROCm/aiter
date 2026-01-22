# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Gated Delta Rule Operations (Forward Only).

This module provides optimized Triton kernels for gated delta rule computations.

Available operations:
- Fused recurrent forward: _fused_recurrent_gated_delta_rule_fwd_kernel
- Chunk-based forward: chunk_gated_delta_rule_fwd
- Hidden state computation: chunk_gated_delta_rule_fwd_h
- Output computation: chunk_fwd_o

Note: Only forward pass is implemented. Backward pass is not supported in aiter.
      For training with gradients, please use the flash-linear-attention library.
"""

from .decode.fused_recurrent import _fused_recurrent_gated_delta_rule_fwd_kernel
from .prefill.chunk import chunk_gated_delta_rule_fwd
from .prefill.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .prefill.chunk_o import chunk_fwd_o
from . import gated_delta_rule_utils

__all__ = [
    "_fused_recurrent_gated_delta_rule_fwd_kernel",
    "chunk_gated_delta_rule_fwd",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_fwd_o",
    "gated_delta_rule_utils",
]
