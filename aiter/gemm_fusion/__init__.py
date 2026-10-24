# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
from .capabilities import detect_arch, detect_capabilities, detect_cu
from .catalog import Contract, VariantInfo, get_variant, register_variant
from .match import ContractError, MatchCtx, match_prefix
from .plan import Plan, WindowReq
from .prebuild import prebuild_profile
from .profile import Profile, ProfileError, Rule, load_profile, parse_profile
from .resources import (
    CcoBackend,
    FusionResources,
    LaunchGateError,
    ResetError,
    ResourceError,
    Window,
    round_up,
)
from .spec import (
    SCHEMA_VERSION,
    Field,
    FusionSpec,
    GemmSpec,
    HeadNormRopeCache,
    SPExchange,
    canonical,
    qkv_fields,
)

__all__ = [
    "SCHEMA_VERSION",
    "CcoBackend",
    "Contract",
    "ContractError",
    "Field",
    "FusionResources",
    "FusionSpec",
    "GemmSpec",
    "HeadNormRopeCache",
    "LaunchGateError",
    "MatchCtx",
    "Plan",
    "Profile",
    "ProfileError",
    "ResetError",
    "ResourceError",
    "Rule",
    "SPExchange",
    "VariantInfo",
    "Window",
    "WindowReq",
    "canonical",
    "detect_arch",
    "detect_capabilities",
    "detect_cu",
    "get_variant",
    "load_profile",
    "match_prefix",
    "parse_profile",
    "prebuild_profile",
    "qkv_fields",
    "register_variant",
    "round_up",
]
