# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The M3 mono kernels' sources, in every build's JIT cache key
(``atom.mono.plan.build_key``): this package and the shared mono framework."""

from atom.mono.plan.build_key import source_digest

SOURCES = source_digest("mono", "models/minimax_m3/mono")
