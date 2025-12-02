# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from . import quant
from . import comms

# Re-export communication primitives at this level for convenience
try:
    from .comms import (
        IrisCommContext,
        reduce_scatter,
        all_gather,
        reduce_scatter_rmsnorm_quant_all_gather,
    )

    _COMMS_AVAILABLE = True
except ImportError:
    _COMMS_AVAILABLE = False

__all__ = ["quant", "comms"]

if _COMMS_AVAILABLE:
    __all__.extend(
        [
            "IrisCommContext",
            "reduce_scatter",
            "all_gather",
            "reduce_scatter_rmsnorm_quant_all_gather",
        ]
    )
