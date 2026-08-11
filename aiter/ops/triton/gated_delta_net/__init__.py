# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Gated Delta Net Operations (Forward Only).

This module provides high-level Triton implementations for gated delta rule.
"""

from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import (
    GatedDeltaRulePrefillMetadata,
    build_gated_delta_rule_prefill_metadata,
)

from .gated_delta_rule import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_opt_vk,
    fused_recurrent_gated_delta_rule,
)

__all__ = [
    "GatedDeltaRulePrefillMetadata",
    "build_gated_delta_rule_prefill_metadata",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_opt_vk",
    "fused_recurrent_gated_delta_rule",
]
