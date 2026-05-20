# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Re-export shim: GateMode is defined in mixed_moe_gemm_2stage when FlyDSL is available."""

try:
    from .kernels.mixed_moe_gemm_2stage import GateMode  # noqa: F401
except ImportError:
    from enum import Enum

    class GateMode(str, Enum):  # type: ignore[no-redef]
        SEPARATED = "separated"
        MOCK_GATE_ONLY = "mock_gate_only"
        GATE_ONLY = "gate_only"
        INTERLEAVE = "interleave"


__all__ = ["GateMode"]
