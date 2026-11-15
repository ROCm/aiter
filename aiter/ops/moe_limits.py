# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math


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
