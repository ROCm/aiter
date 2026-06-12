# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared attention helpers with no kernel dependencies."""

from __future__ import annotations


def map_dims(shape, indices):
    """Reorder a shape/stride tuple by ``indices`` (layout-to-bshd mapping)."""
    return [shape[i] for i in indices]
