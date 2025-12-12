# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

# Gated Delta Rule (GDN) operations
from aiter.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update,
    fused_gdn_gating,
    compute_gating_params,
    GatedDeltaRuleOp,
)

__all__ = [
    # Gated Delta Rule
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_update",
    "fused_sigmoid_gating_delta_rule_update",
    "fused_gdn_gating",
    "compute_gating_params",
    "GatedDeltaRuleOp",
]
