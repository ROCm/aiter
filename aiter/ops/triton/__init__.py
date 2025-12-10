# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from . import quant

# Try to import comms module (requires iris)
try:
    from . import comms

    # Re-export communication primitives at this level for convenience
    from .comms import (
        IrisCommContext,
        reduce_scatter,
        all_gather,
        reduce_scatter_rmsnorm_quant_all_gather,
        IRIS_COMM_AVAILABLE,
    )

    _COMMS_AVAILABLE = True
except ImportError:
    # Iris not available - comms module won't be available
    _COMMS_AVAILABLE = False
    IRIS_COMM_AVAILABLE = False
    comms = None

__all__ = ["quant"]

if _COMMS_AVAILABLE:
    __all__.extend(
        [
            "comms",
            "IrisCommContext",
            "reduce_scatter",
            "all_gather",
            "reduce_scatter_rmsnorm_quant_all_gather",
            "IRIS_COMM_AVAILABLE",
        ]
    )
