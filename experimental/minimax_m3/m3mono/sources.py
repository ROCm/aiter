# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The M3 mono kernels' sources, in every build's JIT cache key
(``vllm.models.minimax_m3.amd.mono.fw.plan.build_key``): this package and the shared mono framework."""

from vllm.models.minimax_m3.amd.mono.fw.plan.build_key import source_digest

SOURCES = source_digest(".")  # the vendored package: kernels + framework
