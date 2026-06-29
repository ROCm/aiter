<!--
SPDX-License-Identifier: MIT
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
-->

# Opus MoE

This directory contains Opus MoE stage2 infrastructure. Kernel implementations
are added by dtype-specific follow-up changes; the common layer keeps the JIT
generator and architecture routing separate from any single fused MoE
integration path.

## File Layout

- `gen_instances.py`: JIT-time codegen entry point.
- `include/opus_moe_common.cuh`: shared C++ constants, kargs, and device helpers.
- `include/opus_moe_arch.cuh`: runtime architecture probe wrapper.

Kernel sources under `include/gfx950/` remain private until a later commit wires
them into an explicit runtime surface.
