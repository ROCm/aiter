# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from . import quant
from . import comms

# Re-export communication primitives at this level for convenience
try:
    from .comms import (
        IrisCommContext,
        all_reduce_iris,
        all_reduce_iris_atomic,
        reduce_scatter_iris,
        all_gather_iris,
        reduce_scatter_rmsnorm_quant_all_gather,
    )
    _COMMS_AVAILABLE = True
except ImportError:
    _COMMS_AVAILABLE = False

__all__ = ["quant", "comms"]

if _COMMS_AVAILABLE:
    __all__.extend([
        "IrisCommContext",
        "all_reduce_iris",
        "all_reduce_iris_atomic",
        "reduce_scatter_iris",
        "all_gather_iris",
        "reduce_scatter_rmsnorm_quant_all_gather",
    ])
