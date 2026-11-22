# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 EP8 MegaMoE mega kernel with A and W staged through LDS, on FP8 128x128 block-scale experts or
a8w4's fd2 MXFP4 experts; the entry point is host.LdsMegaMoE."""
