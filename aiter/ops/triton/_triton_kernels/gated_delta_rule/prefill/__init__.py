# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Gated Delta Rule Prefill Operations (Forward Only).

This module provides optimized Triton kernels for prefill/training operations.
"""

from .chunk import chunk_gated_delta_rule_fwd
from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
from .fused_cumsum_kkt import fused_cumsum_kkt
from .fused_gdn_gating_prefill import fused_gdn_gating_and_sigmoid

__all__ = [
    "chunk_gated_delta_rule_fwd",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_fwd_o",
    "fused_cumsum_kkt",
    "fused_gdn_gating_and_sigmoid",
]
