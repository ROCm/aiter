# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Gated Delta Net Operations (Forward Only).

This module provides high-level Triton implementations for gated delta rule.
"""

from .fused_rearrange_sigmoid_gdr import fused_rearrange_sigmoid_gated_delta_rule
from .gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

__all__ = [
    "chunk_gated_delta_rule",
    "fused_rearrange_sigmoid_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
]
