# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused expert-parallel MoE as one persistent kernel (MegaMoE EP), gfx950.

Mirrors :mod:`mega_moe_tp`: one launch, warp-specialized roles, a hipIpc symmetric
arena and epoch flags -- here replacing the multi-launch dispatch / stage1 / stage2 /
combine EP pipeline (see ``mega_moe/mega_moe_v2.py``) with a single persistent kernel
whose waves run dispatch-send / compute / A-loader / combine concurrently.

The public API lives in ``aiter/ops/flydsl/mega_moe_ep.py``.
"""
