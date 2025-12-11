# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Attention backend utilities for causal_conv1d
Provides common constants and utilities for attention backends
"""

# Padding slot ID used in cache indices to indicate padding
PAD_SLOT_ID = -1

__all__ = ["PAD_SLOT_ID"]

