# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Gated Delta Rule Decode Operations (Forward Only).

This module provides optimized Triton kernels for decode/inference operations.
"""

from .fused_recurrent import _fused_recurrent_gated_delta_rule_fwd_kernel
from .fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update

__all__ = [
    "_fused_recurrent_gated_delta_rule_fwd_kernel",
    "fused_sigmoid_gating_delta_rule_update",
]
