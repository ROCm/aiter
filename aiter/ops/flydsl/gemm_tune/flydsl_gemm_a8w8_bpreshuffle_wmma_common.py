# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Candidate catalogue for tuning the gfx1250 (WMMA) a8w8 bpreshuffle GEMM.

gfx1250 has no MFMA preshuffle kernel; it runs the vendored gemm_fp8fp4_gfx1250
WMMA kernel (ptpc) via ``bpreshuffle_gemm_gfx1250``. This is the WMMA counterpart
of ``flydsl_gemm_a8w8_bpreshuffle_common`` (which serves gfx942/gfx950 MFMA), with
its own perf knobs (num_buffers and split_k) and kernelName format. Cluster is
kept in the candidate schema/name but fixed to (1, 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import wmma_kernel_name
from aiter.ops.flydsl.utils import get_shared_memory_per_block

WMMA = 16  # WMMA M/N tile granularity
WARP = 2  # default m_warp / n_warp for WmmaKernelInstance
LDS_BYTES = get_shared_memory_per_block(fallback_gfx="gfx1250")
_MAX_WARP_TILE = 128

# Mirror the ptpc fp8 LDS layout
_LDS_PAD_A_BYTES = 16
_LDS_PAD_D_BYTES = 16
_ELEM_BYTES_D = 2  # bf16 / f16 output

# Columns: (tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers), grouped by M regime
# fmt: off
_CURATED_INSTANCES = (
    # small M (decode / token-gen): thin tile_m, wide tile_n, deep tile_k
    ( 16,  64, 256, 1, 2, 4), ( 16,  64, 512, 1, 2, 4), ( 16,  96, 256, 1, 2, 4),
    ( 16, 128, 128, 1, 4, 4),
    ( 16, 128, 256, 1, 2, 4), ( 16, 128, 512, 1, 2, 4), ( 16, 192, 256, 1, 2, 4),
    ( 16, 256, 256, 1, 4, 4), ( 16, 512, 128, 1, 4, 4),
    # M=32
    ( 32,  64, 256, 2, 2, 4), ( 32,  64, 512, 2, 2, 4), ( 32,  64, 512, 1, 2, 4),
    ( 32, 128, 128, 1, 4, 4), ( 32, 128, 256, 2, 2, 4),
    ( 32, 192, 256, 2, 2, 4), ( 32, 256, 256, 2, 4, 4),
    # M=64
    ( 64,  64, 256, 2, 2, 4), ( 64,  64, 512, 1, 2, 4),
    ( 64, 128, 128, 2, 2, 4), ( 64, 128, 128, 1, 4, 4),
    ( 64, 192, 128, 2, 2, 4),
    ( 64, 256, 128, 1, 4, 4), ( 64, 256, 128, 2, 2, 4),
    ( 64, 256, 128, 2, 4, 4), ( 64, 512, 128, 1, 4, 3),
    # M=128
    (128, 128, 128, 2, 2, 4), (128, 192, 128, 2, 2, 4), (128, 256, 128, 2, 2, 4),
    (128, 256, 128, 2, 4, 4), (128, 512, 128, 2, 4, 3),
    # large M (compute bound): big square tiles, shallow tile_k
    (256,  64, 128, 2, 1, 4), (256, 128, 128, 2, 2, 4), (256, 192, 128, 2, 2, 4),
    (256, 256, 128, 2, 2, 4), (256, 256, 128, 4, 4, 3), (256, 512, 128, 2, 4, 3),
)
# fmt: on
_SPLIT_K = (1,)


@dataclass
class WmmaKernelInstance:
    tile_m: int
    tile_n: int
    tile_k: int
    num_buffers: int
    split_k: int = 1
    cluster_m: int = 1
    cluster_n: int = 1
    m_warp: int = WARP
    n_warp: int = WARP

    @property
    def name(self) -> str:
        return wmma_kernel_name(
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            num_buffers=self.num_buffers,
            split_k=self.split_k,
            cluster_m=self.cluster_m,
            cluster_n=self.cluster_n,
            m_warp=self.m_warp,
            n_warp=self.n_warp,
        )


def _tile_valid(tm: int, tn: int, tk: int, mw: int, nw: int) -> bool:
    # Each warp tile must be a multiple of WMMA (16) and <= _MAX_WARP_TILE (VGPR
    # budget); tk a multiple of 128; block_threads = mw*nw*32 <= 1024.
    return (
        tm % (mw * WMMA) == 0
        and tn % (nw * WMMA) == 0
        and tm // mw <= _MAX_WARP_TILE
        and tn // nw <= _MAX_WARP_TILE
        and tk % 128 == 0
        and mw * nw <= 32
    )


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def kernel_instance_estimated_lds_bytes(ki: WmmaKernelInstance) -> int:
    """LDS bytes the ptpc fp8 WMMA kernel allocates for ``ki`` (must not under-estimate:
    an overflowing tile would pass the filter and fault at launch).

    Per-stage arena: A pool (rows padded by LDS_PAD_A_BYTES) + 16-aligned B pool, the
    stage 128- then 1024-aligned, times num_buffers. The split_k==1 epilogue also needs
    a TDM-store D buffer that can exceed the arena for small tiles, so take the max.
    """
    lds_a_data = ki.tile_m * (ki.tile_k + _LDS_PAD_A_BYTES)
    lds_b_data = ki.tile_n * ki.tile_k
    stage_bytes = _align_up(lds_a_data, 16) + lds_b_data
    stage_pitch = _align_up(_align_up(stage_bytes, 128), 1024)
    arena_bytes = stage_pitch * ki.num_buffers

    if ki.split_k == 1:  # split_k>1 uses the buffer/atomic store, no LDS D buffer
        warp_tile_m = ki.tile_m // ki.m_warp
        warp_tile_n = ki.tile_n // ki.n_warp
        d_row_stride = warp_tile_n * _ELEM_BYTES_D + _LDS_PAD_D_BYTES
        total_d_bytes = (ki.m_warp * ki.n_warp) * warp_tile_m * d_row_stride
        return max(arena_bytes, total_d_bytes)
    return arena_bytes


def _build_kernels_list():
    kl = {}
    idx = 0
    for tm, tn, tk, mw, nw, nb in _CURATED_INSTANCES:
        assert _tile_valid(  # an invalid curated entry is a typo -- fail loudly
            tm, tn, tk, mw, nw
        ), f"invalid curated instance: tile=({tm},{tn},{tk}) warp=({mw},{nw})"
        for sk in _SPLIT_K:
            ki = WmmaKernelInstance(tm, tn, tk, nb, sk, 1, 1, mw, nw)
            if kernel_instance_estimated_lds_bytes(ki) > LDS_BYTES:
                continue
            kl[idx] = ki
            idx += 1
    return kl


kernels_list: dict[int, WmmaKernelInstance] = _build_kernels_list()


def kernel_fits_shape(ki: WmmaKernelInstance, M: int, N: int, K: int) -> bool:
    """N must divide tile_n; K must divide split_k*tile_k with >= num_buffers K-tiles
    per chunk. M may be ragged (kernel OOB-clips, no divisibility needed). A cluster
    also needs N divisible by cluster_n*tile_n.
    """
    if N % ki.tile_n != 0 or K % (ki.split_k * ki.tile_k) != 0:
        return False
    if (K // ki.split_k) // ki.tile_k < ki.num_buffers:
        return False
    if ki.cluster_m > 1 or ki.cluster_n > 1:
        if N % (ki.cluster_n * ki.tile_n) != 0:
            return False
    return True
