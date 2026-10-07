# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Blocks-per-CU of a compiled FlyDSL fp8_mqa_logits kernel.

Occupancy is a property of the kernel instance, not the device: on gfx942
``mfma_r2_w4`` holds 8 blocks/CU at num_heads=16 and 2 at num_heads=128.

FlyDSL JITs, so there is no ``hipFunction_t`` to query before first dispatch.
The artifact carries the AMDHSA metadata occupancy is derived from, so this
reads that and applies the hardware's allocation rules. Verified equal to
``hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`` on every buildable
config: 216/216 on gfx942, 308/308 on gfx950.

Resolution order is artifact metadata, then ``_MEASURED_OCCUPANCY``, then
``DEFAULT_OCCUPANCY``. The table is the first-launch path, not dead code --
nothing has an artifact until it has run once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import torch

__all__ = ["DEFAULT_OCCUPANCY", "kernel_occupancy"]

# Under-estimating costs some parallelism; over-estimating costs a whole wave.
DEFAULT_OCCUPANCY = 2


@dataclass(frozen=True)
class _ArchLimits:
    """Register-file and wave-slot limits behind the blocks-per-CU derivation.

    ``vgpr_granule`` is the allocation quantum: 86 VGPRs are charged as 88.

    ``unified_agpr`` says whether AGPRs share the VGPR file (CDNA3) or sit in
    a separate one (CDNA4), i.e. whether a wave's demand is the sum or the max
    of the two counts. Load-bearing on gfx950, whose bkv256 variants reach
    vgpr=452/agpr=200: the sum overflows the 512-VGPR file and predicts
    occupancy 1 where hardware reports 2. Every gfx942 kernel here has agpr=0,
    so the two rules coincide on that arch.
    """

    vgpr_per_simd: int
    vgpr_granule: int
    simds_per_cu: int
    max_waves_per_simd: int
    unified_agpr: bool


_ARCH_LIMITS = {
    "gfx942": _ArchLimits(
        vgpr_per_simd=512,
        vgpr_granule=8,
        simds_per_cu=4,
        max_waves_per_simd=8,
        unified_agpr=True,
    ),
    "gfx950": _ArchLimits(
        vgpr_per_simd=512,
        vgpr_granule=8,
        simds_per_cu=4,
        max_waves_per_simd=8,
        unified_agpr=False,
    ),
}

# Keyed by (variant, num_heads, head_size); the entry is the minimum over the
# convert_*_fn / clean_logits build flags, which can shift the VGPR count
# across an allocation boundary.
#
# Regenerate with op_tests/flydsl/gen_fp8_mqa_logits_occupancy.py.
_MEASURED_OCCUPANCY = {
    "gfx942": {
        ("mfma_r1_w1", 16, 64): 28,
        ("mfma_r1_w1", 16, 128): 16,
        ("mfma_r1_w1", 32, 64): 24,
        ("mfma_r1_w1", 32, 128): 16,
        ("mfma_r1_w1", 64, 64): 20,
        ("mfma_r1_w1", 64, 128): 12,
        ("mfma_r1_w1", 128, 64): 16,
        ("mfma_r1_w1", 128, 128): 8,
        ("mfma_r1_w2", 16, 64): 16,
        ("mfma_r1_w2", 16, 128): 14,
        ("mfma_r1_w2", 32, 64): 16,
        ("mfma_r1_w2", 32, 128): 12,
        ("mfma_r1_w2", 64, 64): 12,
        ("mfma_r1_w2", 64, 128): 8,
        ("mfma_r1_w2", 128, 64): 8,
        ("mfma_r1_w2", 128, 128): 6,
        ("mfma_r1_w4", 16, 64): 8,
        ("mfma_r1_w4", 16, 128): 8,
        ("mfma_r1_w4", 32, 64): 8,
        ("mfma_r1_w4", 32, 128): 8,
        ("mfma_r1_w4", 64, 64): 8,
        ("mfma_r1_w4", 64, 128): 5,
        ("mfma_r1_w4", 128, 64): 5,
        ("mfma_r1_w4", 128, 128): 3,
        ("mfma_r2_w1", 16, 64): 20,
        ("mfma_r2_w1", 16, 128): 12,
        ("mfma_r2_w1", 32, 64): 16,
        ("mfma_r2_w1", 32, 128): 8,
        ("mfma_r2_w1", 64, 64): 12,
        ("mfma_r2_w1", 64, 128): 8,
        ("mfma_r2_w1", 128, 64): 8,
        ("mfma_r2_w1", 128, 128): 8,
        ("mfma_r2_w2", 16, 64): 16,
        ("mfma_r2_w2", 16, 128): 10,
        ("mfma_r2_w2", 32, 64): 12,
        ("mfma_r2_w2", 32, 128): 8,
        ("mfma_r2_w2", 64, 64): 8,
        ("mfma_r2_w2", 64, 128): 6,
        ("mfma_r2_w2", 128, 64): 4,
        ("mfma_r2_w2", 128, 128): 4,
        ("mfma_r2_w4", 16, 64): 8,
        ("mfma_r2_w4", 16, 128): 8,
        ("mfma_r2_w4", 32, 64): 8,
        ("mfma_r2_w4", 32, 128): 5,
        ("mfma_r2_w4", 64, 64): 5,
        ("mfma_r2_w4", 64, 128): 3,
        ("mfma_r2_w4", 128, 64): 3,
        ("mfma_r2_w4", 128, 128): 2,
        ("mfma_r4_w1", 16, 64): 16,
        ("mfma_r4_w1", 16, 128): 8,
        ("mfma_r4_w1", 32, 64): 12,
        ("mfma_r4_w1", 32, 128): 8,
        ("mfma_r4_w1", 64, 64): 8,
        ("mfma_r4_w1", 64, 128): 8,
        ("mfma_r4_w1", 128, 64): 8,
        ("mfma_r4_w1", 128, 128): 8,
        ("mfma_r4_w2", 16, 64): 12,
        ("mfma_r4_w2", 16, 128): 8,
        ("mfma_r4_w2", 32, 64): 8,
        ("mfma_r4_w2", 32, 128): 6,
        ("mfma_r4_w2", 64, 64): 6,
        ("mfma_r4_w2", 64, 128): 4,
        ("mfma_r4_w2", 128, 64): 4,
        ("mfma_r4_w2", 128, 128): 4,
        ("mfma_r4_w4", 16, 64): 8,
        ("mfma_r4_w4", 16, 128): 5,
        ("mfma_r4_w4", 32, 64): 4,
        ("mfma_r4_w4", 32, 128): 3,
        ("mfma_r4_w4", 64, 64): 3,
        ("mfma_r4_w4", 64, 128): 2,
        ("mfma_r4_w4", 128, 64): 2,
        ("mfma_r4_w4", 128, 128): 2,
    },
    # Most gfx950 variants stage KV through LDS, so these are largely
    # LDS-bound rather than register-bound.
    "gfx950": {
        ("mfma16x16x128_bkv128_r1_w1", 16, 128): 16,
        ("mfma16x16x128_bkv128_r1_w1", 32, 128): 16,
        ("mfma16x16x128_bkv128_r1_w1", 64, 128): 12,
        ("mfma16x16x128_bkv128_r1_w1", 128, 128): 8,
        ("mfma16x16x128_bkv128_r1_w2", 16, 128): 14,
        ("mfma16x16x128_bkv128_r1_w2", 32, 128): 10,
        ("mfma16x16x128_bkv128_r1_w2", 64, 128): 8,
        ("mfma16x16x128_bkv128_r1_w2", 128, 128): 4,
        ("mfma16x16x128_bkv128_r1_w2_lds2", 16, 128): 5,
        ("mfma16x16x128_bkv128_r1_w2_lds2", 32, 128): 5,
        ("mfma16x16x128_bkv128_r1_w2_lds2", 64, 128): 5,
        ("mfma16x16x128_bkv128_r1_w2_lds2", 128, 128): 4,
        ("mfma16x16x128_bkv128_r2_w1", 16, 128): 16,
        ("mfma16x16x128_bkv128_r2_w1", 32, 128): 12,
        ("mfma16x16x128_bkv128_r2_w1", 64, 128): 8,
        ("mfma16x16x128_bkv128_r2_w1", 128, 128): 8,
        ("mfma16x16x128_bkv128_r2_w2", 16, 128): 12,
        ("mfma16x16x128_bkv128_r2_w2", 32, 128): 8,
        ("mfma16x16x128_bkv128_r2_w2", 64, 128): 6,
        ("mfma16x16x128_bkv128_r2_w2", 128, 128): 4,
        ("mfma16x16x128_bkv128_r2_w2_lds2", 16, 128): 5,
        ("mfma16x16x128_bkv128_r2_w2_lds2", 32, 128): 5,
        ("mfma16x16x128_bkv128_r2_w2_lds2", 64, 128): 4,
        ("mfma16x16x128_bkv128_r2_w2_lds2", 128, 128): 4,
        ("mfma16x16x128_bkv128_r2_w2_lds3", 16, 128): 3,
        ("mfma16x16x128_bkv128_r2_w2_lds3", 32, 128): 3,
        ("mfma16x16x128_bkv128_r2_w2_lds3", 64, 128): 3,
        ("mfma16x16x128_bkv128_r2_w2_lds3", 128, 128): 3,
        ("mfma16x16x128_bkv128_r2_w4_lds2", 16, 128): 4,
        ("mfma16x16x128_bkv128_r2_w4_lds2", 32, 128): 3,
        ("mfma16x16x128_bkv128_r2_w4_lds2", 64, 128): 2,
        ("mfma16x16x128_bkv128_r2_w4_lds2", 128, 128): 2,
        ("mfma16x16x128_bkv128_r4_w4_lds3", 16, 128): 3,
        ("mfma16x16x128_bkv128_r4_w4_lds3", 32, 128): 2,
        ("mfma16x16x128_bkv128_r4_w4_lds3", 64, 128): 2,
        ("mfma16x16x128_bkv128_r4_w4_lds3", 128, 128): 2,
        ("mfma16x16x128_bkv256_r2_w2_lds2", 16, 128): 2,
        ("mfma16x16x128_bkv256_r2_w2_lds2", 32, 128): 2,
        ("mfma16x16x128_bkv256_r2_w2_lds2", 64, 128): 2,
        ("mfma16x16x128_bkv256_r2_w2_lds2", 128, 128): 2,
        ("mfma16x16x128_bkv64_r1_w2_lds3", 16, 128): 6,
        ("mfma16x16x128_bkv64_r1_w2_lds3", 32, 128): 6,
        ("mfma16x16x128_bkv64_r1_w2_lds3", 64, 128): 6,
        ("mfma16x16x128_bkv64_r1_w2_lds3", 128, 128): 4,
        ("mfma16x16x128_bkv64_r2_w2_lds2", 16, 128): 10,
        ("mfma16x16x128_bkv64_r2_w2_lds2", 32, 128): 8,
        ("mfma16x16x128_bkv64_r2_w2_lds2", 64, 128): 6,
        ("mfma16x16x128_bkv64_r2_w2_lds2", 128, 128): 4,
        ("mfma16x16x128_bkv64_r2_w2_lds3", 16, 128): 6,
        ("mfma16x16x128_bkv64_r2_w2_lds3", 32, 128): 6,
        ("mfma16x16x128_bkv64_r2_w2_lds3", 64, 128): 6,
        ("mfma16x16x128_bkv64_r2_w2_lds3", 128, 128): 4,
        ("mfma32x32x64_bkv128_r1_w1", 32, 64): 24,
        ("mfma32x32x64_bkv128_r1_w1", 32, 128): 16,
        ("mfma32x32x64_bkv128_r1_w1", 64, 64): 16,
        ("mfma32x32x64_bkv128_r1_w1", 64, 128): 12,
        ("mfma32x32x64_bkv128_r1_w1", 128, 64): 12,
        ("mfma32x32x64_bkv128_r1_w1", 128, 128): 8,
        ("mfma32x32x64_bkv128_r1_w2", 32, 64): 16,
        ("mfma32x32x64_bkv128_r1_w2", 32, 128): 10,
        ("mfma32x32x64_bkv128_r1_w2", 64, 64): 10,
        ("mfma32x32x64_bkv128_r1_w2", 64, 128): 8,
        ("mfma32x32x64_bkv128_r1_w2", 128, 64): 6,
        ("mfma32x32x64_bkv128_r1_w2", 128, 128): 4,
        ("mfma32x32x64_bkv128_r1_w2_lds2", 32, 64): 10,
        ("mfma32x32x64_bkv128_r1_w2_lds2", 32, 128): 5,
        ("mfma32x32x64_bkv128_r1_w2_lds2", 64, 64): 8,
        ("mfma32x32x64_bkv128_r1_w2_lds2", 64, 128): 4,
        ("mfma32x32x64_bkv128_r1_w2_lds2", 128, 64): 6,
        ("mfma32x32x64_bkv128_r1_w2_lds2", 128, 128): 4,
        ("mfma32x32x64_bkv128_r1_w2_lds3", 32, 64): 6,
        ("mfma32x32x64_bkv128_r1_w2_lds3", 32, 128): 3,
        ("mfma32x32x64_bkv128_r1_w2_lds3", 64, 64): 6,
        ("mfma32x32x64_bkv128_r1_w2_lds3", 64, 128): 3,
        ("mfma32x32x64_bkv128_r1_w2_lds3", 128, 64): 6,
        ("mfma32x32x64_bkv128_r1_w2_lds3", 128, 128): 3,
        ("mfma32x32x64_bkv128_r2_w1", 32, 64): 16,
        ("mfma32x32x64_bkv128_r2_w1", 32, 128): 12,
        ("mfma32x32x64_bkv128_r2_w1", 64, 64): 12,
        ("mfma32x32x64_bkv128_r2_w1", 64, 128): 8,
        ("mfma32x32x64_bkv128_r2_w1", 128, 64): 8,
        ("mfma32x32x64_bkv128_r2_w1", 128, 128): 8,
        ("mfma32x32x64_bkv128_r2_w2", 32, 64): 10,
        ("mfma32x32x64_bkv128_r2_w2", 32, 128): 8,
        ("mfma32x32x64_bkv128_r2_w2", 64, 64): 6,
        ("mfma32x32x64_bkv128_r2_w2", 64, 128): 4,
        ("mfma32x32x64_bkv128_r2_w2", 128, 64): 4,
        ("mfma32x32x64_bkv128_r2_w2", 128, 128): 4,
        ("mfma32x32x64_bkv128_r2_w2_lds2", 32, 64): 8,
        ("mfma32x32x64_bkv128_r2_w2_lds2", 32, 128): 4,
        ("mfma32x32x64_bkv128_r2_w2_lds2", 64, 64): 4,
        ("mfma32x32x64_bkv128_r2_w2_lds2", 64, 128): 4,
        ("mfma32x32x64_bkv128_r2_w2_lds2", 128, 64): 4,
        ("mfma32x32x64_bkv128_r2_w2_lds2", 128, 128): 4,
        ("mfma32x32x64_bkv128_r2_w4_lds2", 32, 64): 4,
        ("mfma32x32x64_bkv128_r2_w4_lds2", 32, 128): 2,
        ("mfma32x32x64_bkv128_r2_w4_lds2", 64, 64): 3,
        ("mfma32x32x64_bkv128_r2_w4_lds2", 64, 128): 2,
        ("mfma32x32x64_bkv128_r2_w4_lds2", 128, 64): 2,
        ("mfma32x32x64_bkv128_r2_w4_lds2", 128, 128): 2,
        ("mfma32x32x64_bkv128_r2_w4_lds3", 32, 64): 4,
        ("mfma32x32x64_bkv128_r2_w4_lds3", 32, 128): 3,
        ("mfma32x32x64_bkv128_r2_w4_lds3", 64, 64): 3,
        ("mfma32x32x64_bkv128_r2_w4_lds3", 64, 128): 2,
        ("mfma32x32x64_bkv128_r2_w4_lds3", 128, 64): 2,
        ("mfma32x32x64_bkv128_r2_w4_lds3", 128, 128): 2,
        ("mfma32x32x64_bkv256_r1_w2_lds2", 32, 64): 5,
        ("mfma32x32x64_bkv256_r1_w2_lds2", 32, 128): 2,
        ("mfma32x32x64_bkv256_r1_w2_lds2", 64, 64): 5,
        ("mfma32x32x64_bkv256_r1_w2_lds2", 64, 128): 2,
        ("mfma32x32x64_bkv256_r1_w2_lds2", 128, 64): 4,
        ("mfma32x32x64_bkv256_r1_w2_lds2", 128, 128): 2,
        ("mfma32x32x64_bkv256_r2_w2_lds2", 32, 64): 4,
        ("mfma32x32x64_bkv256_r2_w2_lds2", 32, 128): 2,
        ("mfma32x32x64_bkv256_r2_w2_lds2", 64, 64): 4,
        ("mfma32x32x64_bkv256_r2_w2_lds2", 64, 128): 2,
        ("mfma32x32x64_bkv256_r2_w2_lds2", 128, 64): 4,
        ("mfma32x32x64_bkv256_r2_w2_lds2", 128, 128): 2,
        ("mfma32x32x64_bkv64_r1_w2_lds2", 32, 64): 16,
        ("mfma32x32x64_bkv64_r1_w2_lds2", 32, 128): 10,
        ("mfma32x32x64_bkv64_r1_w2_lds2", 64, 64): 10,
        ("mfma32x32x64_bkv64_r1_w2_lds2", 64, 128): 6,
        ("mfma32x32x64_bkv64_r1_w2_lds2", 128, 64): 6,
        ("mfma32x32x64_bkv64_r1_w2_lds2", 128, 128): 4,
        ("mfma32x32x64_bkv64_r1_w2_lds3", 32, 64): 13,
        ("mfma32x32x64_bkv64_r1_w2_lds3", 32, 128): 6,
        ("mfma32x32x64_bkv64_r1_w2_lds3", 64, 64): 10,
        ("mfma32x32x64_bkv64_r1_w2_lds3", 64, 128): 6,
        ("mfma32x32x64_bkv64_r1_w2_lds3", 128, 64): 6,
        ("mfma32x32x64_bkv64_r1_w2_lds3", 128, 128): 4,
        ("mfma32x32x64_bkv64_r2_w2_lds2", 32, 64): 10,
        ("mfma32x32x64_bkv64_r2_w2_lds2", 32, 128): 6,
        ("mfma32x32x64_bkv64_r2_w2_lds2", 64, 64): 6,
        ("mfma32x32x64_bkv64_r2_w2_lds2", 64, 128): 4,
        ("mfma32x32x64_bkv64_r2_w2_lds2", 128, 64): 4,
        ("mfma32x32x64_bkv64_r2_w2_lds2", 128, 128): 4,
        ("mfma32x32x64_bkv64_r2_w2_lds3", 32, 64): 10,
        ("mfma32x32x64_bkv64_r2_w2_lds3", 32, 128): 6,
        ("mfma32x32x64_bkv64_r2_w2_lds3", 64, 64): 6,
        ("mfma32x32x64_bkv64_r2_w2_lds3", 64, 128): 4,
        ("mfma32x32x64_bkv64_r2_w2_lds3", 128, 64): 4,
        ("mfma32x32x64_bkv64_r2_w2_lds3", 128, 128): 4,
        ("mfma32x32x64_bkv64_r2_w4_lds2", 32, 64): 5,
        ("mfma32x32x64_bkv64_r2_w4_lds2", 32, 128): 3,
        ("mfma32x32x64_bkv64_r2_w4_lds2", 64, 64): 3,
        ("mfma32x32x64_bkv64_r2_w4_lds2", 64, 128): 2,
        ("mfma32x32x64_bkv64_r2_w4_lds2", 128, 64): 2,
        ("mfma32x32x64_bkv64_r2_w4_lds2", 128, 128): 2,
        ("mfma32x32x64_bkv64_r2_w4_lds3", 32, 64): 5,
        ("mfma32x32x64_bkv64_r2_w4_lds3", 32, 128): 3,
        ("mfma32x32x64_bkv64_r2_w4_lds3", 64, 64): 3,
        ("mfma32x32x64_bkv64_r2_w4_lds3", 64, 128): 2,
        ("mfma32x32x64_bkv64_r2_w4_lds3", 128, 64): 2,
        ("mfma32x32x64_bkv64_r2_w4_lds3", 128, 128): 2,
        ("mfma32x32x64_bkv64_r2_w4_lds3_swp", 32, 64): 4,
        ("mfma32x32x64_bkv64_r2_w4_lds3_swp", 32, 128): 3,
        ("mfma32x32x64_bkv64_r2_w4_lds3_swp", 64, 64): 2,
        ("mfma32x32x64_bkv64_r2_w4_lds3_swp", 64, 128): 2,
        ("mfma32x32x64_bkv64_r2_w4_lds3_swp", 128, 64): 2,
        ("mfma32x32x64_bkv64_r2_w4_lds3_swp", 128, 128): 2,
    },
}

# AMDHSA metadata on the ``gpu.binary`` the FlyDSL backend emits, e.g.
#   metadata = {agpr_count = 0 : i64, ..., vgpr_count = 86 : i64}
_METADATA_RE = re.compile(r"metadata = \{([^}]*)\}")
_INT_FIELD_RE = re.compile(r"(\w+) = (\d+) : i64")
_WORKGROUP_RE = re.compile(r"reqd_workgroup_size = array<i32: (\d+)")


def _artifact_metadata(launcher) -> dict | None:
    """AMDHSA resource metadata of *launcher*'s compiled kernel, or None.

    None is normal, not an error: ``_run_compiled`` attaches ``_cf`` only on
    the first dispatch.
    """
    artifact = getattr(getattr(launcher, "_cf", None), "_keepalive", None)
    ir_text = getattr(artifact, "_ir_text", None)
    if not ir_text:
        return None
    body = _METADATA_RE.search(ir_text)
    workgroup = _WORKGROUP_RE.search(ir_text)
    if body is None or workgroup is None:
        return None
    fields = {k: int(v) for k, v in _INT_FIELD_RE.findall(body.group(1))}
    if "vgpr_count" not in fields:
        return None
    fields["threads_per_block"] = int(workgroup.group(1))
    return fields


def _occupancy_from_metadata(fields: dict, arch: str, device_index: int) -> int | None:
    """Blocks resident per CU implied by *fields*: the binding minimum of the
    register-file, wave-slot and LDS limits.

    No workgroups-per-CU cap: HIP reports none here, and imposing 16
    disagreed with it on the 13 smallest kernels.
    """
    limits = _ARCH_LIMITS.get(arch)
    threads = fields.get("threads_per_block", 0)
    if limits is None or threads <= 0:
        return None
    try:
        props = torch.cuda.get_device_properties(device_index)
    except Exception:  # noqa: BLE001
        return None

    waves_per_block = max(1, threads // props.warp_size)
    vgpr_count = fields.get("vgpr_count", 0)
    agpr_count = fields.get("agpr_count", 0)
    vgprs = (
        vgpr_count + agpr_count if limits.unified_agpr else max(vgpr_count, agpr_count)
    )
    granules = max(1, -(-vgprs // limits.vgpr_granule)) * limits.vgpr_granule
    waves_per_simd = min(limits.max_waves_per_simd, limits.vgpr_per_simd // granules)

    by_registers = (waves_per_simd * limits.simds_per_cu) // waves_per_block
    by_wave_slots = props.max_threads_per_multi_processor // threads
    lds_bytes = fields.get("group_segment_fixed_size", 0)
    if lds_bytes:
        shared_per_cu = getattr(props, "shared_memory_per_multiprocessor", None)
        if shared_per_cu is None:
            return None
        by_lds = shared_per_cu // lds_bytes
    else:
        by_lds = by_wave_slots
    return max(1, min(by_registers, by_wave_slots, by_lds))


@lru_cache(maxsize=64)
def _measured_occupancy(arch: str, variant: str, num_heads: int, head_size: int):
    return _MEASURED_OCCUPANCY.get(arch, {}).get((variant, num_heads, head_size))


def kernel_occupancy(
    launcher,
    *,
    arch: str,
    variant: str,
    num_heads: int,
    head_size: int,
    device_index: int,
) -> int:
    """Blocks of *launcher* that fit concurrently on one CU (>= 1).

    Never raises: a launch heuristic must not fail because a resource query
    did.
    """
    try:
        fields = _artifact_metadata(launcher)
        if fields is not None:
            occupancy = _occupancy_from_metadata(fields, arch, device_index)
            if occupancy is not None:
                return occupancy
    except Exception:  # noqa: BLE001 -- a launch heuristic must never fail here
        pass
    measured = _measured_occupancy(arch, variant, num_heads, head_size)
    return measured if measured is not None else DEFAULT_OCCUPANCY
