# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MegaMoEGfx942's large-input path (large_M): the gfx942 EP4 / EP8 MegaMoE mega kernel with A and W staged through
LDS, on FP8 128x128 block-scale experts or a8w4's fd2 MXFP4 experts; the entry point is host.LargeMRuntime.
"""
