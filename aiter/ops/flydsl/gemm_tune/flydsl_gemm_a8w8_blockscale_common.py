# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune space for the gfx950 8-wave FP8 blockscale GEMM.

Keep the copied kernel's tile, PID swizzle and permlane epilogue fixed. Race its
full/half-M pipelines and raw/tiled DMA using the existing blockscale tuner.
The preshuffle bit is part of the name so a CSV cannot silently select the
wrong weight layout. No FlyDSL compiler imports are needed to read this table.
"""

from dataclasses import dataclass
from itertools import product

TILE_M = 256
TILE_N = 256
TILE_K = 128
KERNEL_PREFIX = "flydsl_blockscale_8w_"


@dataclass(frozen=True)
class kernelInstance:
    preshuffle_b: bool
    split_m: bool
    use_tile_dma: bool

    @property
    def name(self) -> str:
        return (
            f"{KERNEL_PREFIX}{TILE_M}x{TILE_N}x{TILE_K}_F8_F8_B16_"
            f"ps{int(self.preshuffle_b)}_sm{int(self.split_m)}_"
            f"tdma{int(self.use_tile_dma)}"
        )


# Stable IDs across architectures; --preshuffle selects the matching half.
kernels_list = {
    i: kernelInstance(*flags)
    for i, flags in enumerate(product((False, True), repeat=3))
}
kernels_by_name = {ki.name: ki for ki in kernels_list.values()}


def lds_bytes(K: int) -> int:
    # Eight padded 128x128 FP8 buffers, two 512-f32 A-scale buffers,
    # and all K/128 B scales for each of the two N quadrants.
    return 8 * 16896 + 2 * 512 * 4 + 2 * (K // 128) * 4


def kernel_fits_shape(ki: kernelInstance, M: int, N: int, K: int, gfx: str) -> bool:
    if gfx != "gfx950" or M <= 0 or N <= 0 or K < 256 or K % 256:
        return False
    if N % (16 if ki.preshuffle_b else 8):
        return False
    if lds_bytes(K) > 160 * 1024:
        return False
    # The original kernel uses signed-i32 byte offsets. Include tail tiles and
    # the two speculative K prefetches, not only the logical tensor extents.
    padded_m = (M + TILE_M - 1) // TILE_M * TILE_M
    padded_n = (N + TILE_N - 1) // TILE_N * TILE_N
    # Shuffled B packs 16 rows into its K stride; its final speculative DMA
    # can extend 4096 bytes past the last 256-row tile rather than 256 bytes.
    b_prefetch_bytes = 2 * TILE_K * (16 if ki.preshuffle_b else 1)
    return (
        max(
            padded_m * K + 2 * TILE_K,
            padded_n * K + b_prefetch_bytes,
            padded_m * padded_n * 2,
        )
        < 2**31
    )
