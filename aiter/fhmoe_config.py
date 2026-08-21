# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lightweight capabilities shared by FHMoE runtime and AOT setup."""

# Inclusive actual-M ceiling for the no-padding gfx950 DSV4
# H7168/I384/E385/topk7 interleaved FHMoE path.
DSV4_I384_FHMOE_MAX_TOKENS = 2048

# I384 FHMoE reuses compatible FlyDSL kernel metadata from tuned I512 rows.
# This value is only a config lookup key. Tensors and AOT jobs remain physical
# I384, without padding or additional activation memory.
DSV4_I384_FHMOE_METADATA_SOURCE_INTER_DIM = 512
