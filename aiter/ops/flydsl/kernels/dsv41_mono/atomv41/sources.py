# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono kernels' sources, in every build's JIT cache key
(``atom.mono.plan.build_key``): this package, the shared mono framework, and the
decode router whose device helpers the MoE kernel calls."""

from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.build_key import source_digest

# vendored: the framework (atomfw), the V4.1 kernels and router (atomv41), and
# the mono layer built on them with dsv41_mega_attn's attention stages
SOURCES = source_digest("atomfw", "atomv41", ".")
