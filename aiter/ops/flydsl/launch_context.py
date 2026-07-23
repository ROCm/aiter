# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Explicit runtime launch state shared by FlyDSL host wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

StreamT = TypeVar("StreamT")

__all__ = ["LaunchContext"]


@dataclass(frozen=True)
class LaunchContext(Generic[StreamT]):
    """Immutable runtime stream passed through argument packing and launch."""

    stream: StreamT
