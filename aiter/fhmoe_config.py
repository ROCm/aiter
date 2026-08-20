# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lightweight capabilities shared by FHMoE runtime and AOT setup."""

# Inclusive actual-M ceiling for the no-padding gfx950 DSV4
# H7168/I384/E385/topk7 interleaved FHMoE path.
DSV4_I384_FHMOE_MAX_TOKENS = 2048
