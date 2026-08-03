# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports for shared prefill batch metadata."""

from aiter.ops.prefill_batch_metadata import (
    ChunkGrid,
    CausalConvPrefillMetadata,
    GatedDeltaRulePrefillMetadata,
    GatedDeltaRuleChunkSchedule,
    PrefillBatchLayout,
    build_causal_conv_prefill_metadata,
    build_gated_delta_rule_prefill_metadata,
)

__all__ = [
    "ChunkGrid",
    "CausalConvPrefillMetadata",
    "GatedDeltaRulePrefillMetadata",
    "GatedDeltaRuleChunkSchedule",
    "PrefillBatchLayout",
    "build_causal_conv_prefill_metadata",
    "build_gated_delta_rule_prefill_metadata",
]
