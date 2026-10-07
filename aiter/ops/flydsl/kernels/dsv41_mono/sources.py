# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The mono layer's kernel sources, in every build's JIT cache key
(``common.plan.build_key``): an edit anywhere in this package rebuilds."""

from aiter.ops.flydsl.kernels.dsv41_mono.common.plan.build_key import source_digest

SOURCES = source_digest(".")
