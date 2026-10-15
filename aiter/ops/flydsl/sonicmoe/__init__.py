# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public API for the gfx950 FlyDSL SonicMoE implementation."""

from .dynamic_policy import E16RoutePolicy, e16_route_policy_representative
from .op import (
    E16_ROUTE_POLICY_SIZES,
    SonicMoE,
    SonicMoEConfig,
    SonicMoEDynamicWorkspacePool,
    SonicMoEForwardState,
    SonicMoERoutesForwardState,
    SonicMoEWeights,
    SonicMoEWorkspace,
    canonical_e16_route_policy_size,
    prepare_sonic_bf16_weights,
    prepare_sonic_fp16_weights,
    prepare_sonic_mxfp4_weights,
    select_e16_route_policy,
    sonic_moe_backward,
    sonic_moe_backward_expert_major,
    sonic_moe_backward_routes,
    sonic_moe_mxfp4_reference,
    sonic_moe_reference,
    warmup_sonic_e16_training,
)

__all__ = [
    "E16_ROUTE_POLICY_SIZES",
    "E16RoutePolicy",
    "SonicMoE",
    "SonicMoEConfig",
    "SonicMoEDynamicWorkspacePool",
    "SonicMoEForwardState",
    "SonicMoERoutesForwardState",
    "SonicMoEWeights",
    "SonicMoEWorkspace",
    "canonical_e16_route_policy_size",
    "e16_route_policy_representative",
    "prepare_sonic_bf16_weights",
    "prepare_sonic_fp16_weights",
    "prepare_sonic_mxfp4_weights",
    "select_e16_route_policy",
    "sonic_moe_backward",
    "sonic_moe_backward_expert_major",
    "sonic_moe_backward_routes",
    "sonic_moe_mxfp4_reference",
    "sonic_moe_reference",
    "warmup_sonic_e16_training",
]
