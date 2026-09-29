# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility export for the backend-neutral K5+K6 fusion option.

The type itself lives in ``aiter.ops.gated_delta_rule_fusion`` so the FlyDSL
backend can use it without importing Triton internals.
"""

from aiter.ops.gated_delta_rule_fusion import K5K6Fusion

__all__ = ["K5K6Fusion"]
