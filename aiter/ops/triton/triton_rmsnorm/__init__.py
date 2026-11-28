# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Triton fused all-reduce + residual add + RMSNorm for gfx950 (Lamport, wave split)."""
from aiter.ops.triton.triton_rmsnorm.tl_lamport2 import TlLamport2RMSNorm

__all__ = ["TlLamport2RMSNorm"]
