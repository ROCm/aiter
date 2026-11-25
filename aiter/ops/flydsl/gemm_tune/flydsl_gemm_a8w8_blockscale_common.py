# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune space for the FlyDSL FP8 blockscale GEMMs: gfx950 8-wave, gfx942 2x4-LDS.

IDs 0 and 1 are the gfx950 8-wave kernel. Its tile, half-M pipeline, raw DMA,
PID swizzle and permlane epilogue are fixed. The preshuffle bit distinguishes
the two weight layouts: ID 0 is plain B and ID 1 is preshuffled B. Kernel
names omit the fixed pipeline and DMA flags. The historical B16 names also
serve FP16 calls; the output tensor specializes the epilogue, with separate
runtime/AOT cache entries for the two dtypes.

IDs 2 and 3 are the gfx942 kernel ported unchanged from PyHIP's 2x4-LDS pf1
(kernels/gemm_a8w8_blockscale_gfx942_2x4_lds_pf1.py): a 256x256 tile, eight
waves on a 2x4 (M x N) grid, A and B staged through 64 KiB of LDS with a
one-stage load lead. ID 2 is plain B and ID 3 preshuffled B. It reads FP8
E4M3FNUZ and writes BF16 only, truncated as PyHIP does. It needs
N % 16 == 0, K % 128 == 0 and K >= 128, and takes any M >= 1: partial M/N
tiles are range checked.

kernelInstance(preshuffle_b) is the gfx950 kernel. No FlyDSL compiler imports
are needed to read this table.
"""

from dataclasses import dataclass

TILE_M = 256
TILE_N = 256
TILE_K = 128
KERNEL_PREFIX = "flydsl_blockscale_8w_"
GFX942_KERNEL_PREFIX = "flydsl_blockscale_gfx942_2x4_lds_pf1_"


@dataclass(frozen=True)
class kernelInstance:
    preshuffle_b: bool
    # The arch the kernel is built for, which a tuned row must run on.
    gfx: str = "gfx950"

    @property
    def name(self) -> str:
        prefix = GFX942_KERNEL_PREFIX if self.gfx == "gfx942" else KERNEL_PREFIX
        return (
            f"{prefix}{TILE_M}x{TILE_N}x{TILE_K}_F8_F8_B16_"
            f"ps{int(self.preshuffle_b)}"
        )

    @property
    def out_dtypes(self) -> tuple[str, ...]:
        """torch dtype names of the outputs the kernel writes."""
        if self.gfx == "gfx942":
            return ("bfloat16",)
        return ("bfloat16", "float16")


# Candidate IDs are consecutive; --preshuffle selects the B layout. IDs 0/1 are
# referenced by tuned CSVs: append new kernels, do not renumber.
kernels_list = {
    0: kernelInstance(False),
    1: kernelInstance(True),
    2: kernelInstance(False, "gfx942"),
    3: kernelInstance(True, "gfx942"),
}
kernels_by_name = {ki.name: ki for ki in kernels_list.values()}


def lds_bytes(K: int) -> int:
    # Eight padded 128x128 FP8 buffers and two 512-f32 A-scale buffers.
    # ScaleB uses scalar global loads; LDS no longer grows with K.
    return 8 * 16896 + 2 * 512 * 4


def _gfx942_fits_shape(M: int, N: int, K: int, gfx: str) -> bool:
    # LDS is a fixed 64 KiB (one K block of A and of B), gfx942's capacity.
    if gfx != "gfx942" or M <= 0 or N <= 0 or N % 16 or K < TILE_K or K % TILE_K:
        return False
    kb = K // TILE_K
    # The A/B/C descriptors start at their tile (i64 base), so their i32
    # offsets and record counts span one tile: TILE*K bytes of A or of B
    # (plain or preshuffled) and TILE_M rows of N*2 bytes of C. ScaleA[KB, M]
    # and ScaleB[ceil(N/128), KB] are addressed whole with i32 offsets.
    tiles = -(-M // TILE_M) * -(-N // TILE_N)
    return (
        max(TILE_M * K, TILE_N * K, TILE_M * N * 2, kb * M * 4, -(-N // 128) * kb * 4)
        < 2**31
        # One 512-thread workgroup per tile in grid x; the dispatch packet
        # counts grid work-items in 32 bits.
        and tiles * 512 < 2**32
    )


def kernel_fits_shape(ki: kernelInstance, M: int, N: int, K: int, gfx: str) -> bool:
    if ki.gfx == "gfx942":
        return _gfx942_fits_shape(M, N, K, gfx)
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
    # Scale bases were not rebased. Keep their full extents/prefetch offsets
    # and the launch-grid arithmetic in range; do not restore a matrix-size cap.
    kb = K // 128
    return (
        max(
            TILE_M * K,
            TILE_N * K,
            TILE_M * N * 2,
            kb * padded_m * 4,
            (padded_n // 128) * kb * 4,
        )
        < 2**31
    )
