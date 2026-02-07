# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Gated Delta Net Operations (Forward Only).

This module provides high-level Triton implementations for gated delta rule.
"""

from .gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

__all__ = [
    "fused_recurrent_gated_delta_rule",
    "chunk_gated_delta_rule",
]
