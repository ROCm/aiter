# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune space for the gfx950 8-wave FP8 blockscale GEMM.

The tile, half-M pipeline, raw DMA, PID swizzle and permlane epilogue are fixed.
The preshuffle bit distinguishes the two weight layouts. Surviving names/IDs
stay stable; removed full-M/tiled-DMA names are not aliases for another kernel.
No FlyDSL compiler imports are needed to read this table.
"""

from dataclasses import dataclass

TILE_M = 256
TILE_N = 256
TILE_K = 128
KERNEL_PREFIX = "flydsl_blockscale_8w_"


@dataclass(frozen=True)
class kernelInstance:
    preshuffle_b: bool

    @property
    def name(self) -> str:
        return (
            f"{KERNEL_PREFIX}{TILE_M}x{TILE_N}x{TILE_K}_F8_F8_B16_"
            f"ps{int(self.preshuffle_b)}_sm1_tdma0"
        )


# Preserve the original IDs of half-M/raw DMA; --preshuffle selects one layout.
kernels_list = {
    2: kernelInstance(False),
    6: kernelInstance(True),
}
kernels_by_name = {ki.name: ki for ki in kernels_list.values()}


def lds_bytes(K: int) -> int:
    # Eight padded 128x128 FP8 buffers and two 512-f32 A-scale buffers.
    # ScaleB uses scalar global loads; LDS no longer grows with K.
    return 8 * 16896 + 2 * 512 * 4


def kernel_fits_shape(ki: kernelInstance, M: int, N: int, K: int, gfx: str) -> bool:
    if gfx != "gfx950" or M <= 0 or N <= 0 or K < 256 or K % 256:
        return False
    if N % (16 if ki.preshuffle_b else 8):
        return False
    if lds_bytes(K) > 160 * 1024:
        return False
    # A/B/C bases are rebased in i64 per WG. Only local descriptor-relative
    # spans must fit signed i32, including the two speculative K prefetches.
    padded_m = (M + TILE_M - 1) // TILE_M * TILE_M
    padded_n = (N + TILE_N - 1) // TILE_N * TILE_N
    b_prefetch_bytes = 2 * TILE_K * (16 if ki.preshuffle_b else 1)
    # Scale bases were not rebased. Keep their full extents/prefetch offsets
    # and the launch-grid arithmetic in range; do not restore a matrix-size cap.
    kb = K // 128
    return (
        max(
            TILE_M * K + 2 * TILE_K,
            TILE_N * K + b_prefetch_bytes,
            TILE_M * N * 2,
            (kb + 2) * padded_m * 4,
            (padded_n // 128) * kb * 4,
            (padded_m // TILE_M) * (padded_n // TILE_N) + 7,
        )
        < 2**31
    )
