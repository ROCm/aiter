# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math

from .enum import QuantType

# FP8 quant types whose non-split CK 2-stage stage1 clamps SiLU at a runtime limit.
RUNTIME_SILU_LIMIT_QUANT_TYPES = (
    QuantType.per_1x128,
    QuantType.per_Token,
    QuantType.per_Tensor,
)


def normalize_swiglu_limit(limit):
    """CK SiLU override: None/0/+inf disable; positive finite values clamp.

    Gate is upper-clamped and up is symmetrically clamped before SiLU.
    """
    if limit is None:
        return None
    limit = float(limit)
    if math.isnan(limit) or limit < 0:
        raise ValueError("swiglu_limit must be non-negative and not NaN")
    return None if limit == 0 or math.isinf(limit) else limit
