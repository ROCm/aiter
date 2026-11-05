# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton-based communication primitives for AITER.

This submodule contains communication operations implemented using Triton,
including Iris-based GPU-initiated communication.

The Iris-based primitives are exported only when Iris is installed.
IRIS_COMM_AVAILABLE tells if they are. The other modules in this package
do not need Iris.
"""

try:
    import iris  # noqa: F401
except ImportError:
    IRIS_COMM_AVAILABLE = False
    __all__ = ["IRIS_COMM_AVAILABLE"]
else:
    from aiter.ops.triton.comms.all_gather import all_gather
    from aiter.ops.triton.comms.fused import reduce_scatter_rmsnorm_quant_all_gather
    from aiter.ops.triton.comms.iris import IrisCommContext, calculate_heap_size
    from aiter.ops.triton.comms.reduce_scatter import reduce_scatter

    IRIS_COMM_AVAILABLE = True
    __all__ = [
        "IRIS_COMM_AVAILABLE",
        "IrisCommContext",
        "all_gather",
        "calculate_heap_size",
        "reduce_scatter",
        "reduce_scatter_rmsnorm_quant_all_gather",
    ]
