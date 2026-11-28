# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Triton Lamport all-reduce for gfx950 (wave split, after aiter CAR)."""
from aiter.ops.triton.triton_ar.triton_ar import TritonAllReduce

__all__ = ["TritonAllReduce"]
