# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoE v2: segment-based MoE dispatch and execution.

Design: docs/fused_moe_revamp_plan.md; execution order: docs/fused_moe_revamp_steps.md.

This package coexists with the original MoE code (aiter/fused_moe.py and friends) and
takes over one path at a time behind AITER_MOEV2. It never imports aiter.fused_moe,
aiter.fhmoe or aiter.fused_moe_bf16_asm; the legacy modules may import from it.
"""
