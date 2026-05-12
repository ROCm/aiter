# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Re-export shim: GateMode is now defined in mixed_moe_gemm_2stage."""

from .kernels.mixed_moe_gemm_2stage import GateMode  # noqa: F401

__all__ = ["GateMode"]
