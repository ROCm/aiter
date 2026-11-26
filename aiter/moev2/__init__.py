# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoE v2: segment-based MoE dispatch and execution.

This package coexists with the original MoE code (aiter/fused_moe.py and friends) and
takes over one path at a time behind AITER_MOEV2. It never imports aiter.fused_moe,
aiter.fhmoe or aiter.fused_moe_bf16_asm; the legacy modules may import from it.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""
