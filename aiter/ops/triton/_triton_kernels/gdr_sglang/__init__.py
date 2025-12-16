# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

"""
Gated Delta Rule (GDR) - SGLang Implementation

This module provides kernels and high-level operations for Gated Delta Rule
linear attention, adapted from SGLang.

Submodules:
    - gated_delta_rule: High-level GDR operations
    - chunk: Chunk-based parallel implementation
    - fused_recurrent: Fused recurrent implementation
    - fused_sigmoid_gating_recurrent: Fused sigmoid gating + recurrent
    - fused_gdn_gating: GDN gating computation
    - Other helper kernels

Author: AIter Team
License: Apache 2.0
"""

from .gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update,
    fused_gdn_gating,
    compute_gating_params,
    GatedDeltaRuleOp,
)

__all__ = [
    # High-level operations
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_update",
    "fused_sigmoid_gating_delta_rule_update",
    "fused_gdn_gating",
    "compute_gating_params",
    "GatedDeltaRuleOp",
]


