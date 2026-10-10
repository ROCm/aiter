# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Error raised when the MonoKernel cannot take a model or runtime state."""

from __future__ import annotations


class MonoUnsupported(Exception):
    """The loaded model or runtime state cannot use the native path."""
