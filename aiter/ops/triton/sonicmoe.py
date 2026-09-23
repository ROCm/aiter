# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility re-export for the SonicMoE API."""

from aiter.ops.triton.moe.sonicmoe import (
    ActivationType,
    SonicMoEActivationType,
    is_glu,
    moe_general_routing_inputs,
    moe_pre_routed_inputs,
    moe_TC_softmax_topk_layer,
    sonicmoe_is_glu,
)

__all__ = [
    "ActivationType",
    "SonicMoEActivationType",
    "is_glu",
    "moe_TC_softmax_topk_layer",
    "moe_general_routing_inputs",
    "moe_pre_routed_inputs",
    "sonicmoe_is_glu",
]
