# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Neutral sp_qkv_fused contract; tuned kernels live in an out-of-tree provider."""

from .sp_qkv.op import sp_qkv_fused

__all__ = ["sp_qkv_fused"]
