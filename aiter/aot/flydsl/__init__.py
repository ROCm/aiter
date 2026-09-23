# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL ahead-of-time compilation support."""

from .spec import (
    DEFAULT_AOT_SPEC_REGISTRY,
    AotSpec,
    AotSpecRegistry,
    register_default_specs,
)

__all__ = [
    "DEFAULT_AOT_SPEC_REGISTRY",
    "AotSpec",
    "AotSpecRegistry",
    "register_default_specs",
]
