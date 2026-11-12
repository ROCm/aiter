# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused expert-parallel MoE as one persistent kernel (MegaMoE EP), gfx950.

One launch over a hipIpc symmetric arena with epoch flags: every CTA sends its routing
and first payload rounds, plans, gathers, then pulls GEMM1 / GEMM2 tiles from a per-XCD
queue while a few sender CTAs stream the remaining payload rounds; GEMM2 scatters
straight back to the source ranks, which reduce their top-k partials.

The public API lives in ``aiter/ops/flydsl/mega_moe_ep.py``.
"""
