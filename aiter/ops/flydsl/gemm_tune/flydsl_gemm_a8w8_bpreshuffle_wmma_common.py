# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Candidate catalogue for tuning the gfx1250 (WMMA) a8w8 bpreshuffle GEMM.

gfx1250 has no MFMA preshuffle kernel; it runs the vendored gemm_fp8fp4_gfx1250
WMMA kernel (ptpc) via ``bpreshuffle_gemm_gfx1250``. This is the WMMA counterpart
of ``flydsl_gemm_a8w8_bpreshuffle_common`` (which serves gfx942/gfx950 MFMA), with
its own perf knobs — num_buffers, split_k, cluster — and kernelName format.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import wmma_kernel_name

WMMA = 16  # WMMA M/N tile granularity
WARP = 2  # default m_warp / n_warp for WmmaKernelInstance
LDS_BYTES = 320 * 1024  # gfx1250 shared-memory capacity (== device limit)
_MAX_WARP_TILE = 128

# Mirror the ptpc fp8 LDS layout in gemm_fp8fp4_gfx1250.compile_fp8fp4_gemm so the
# candidate filter matches the kernel's real allocation. fp8 packs 1 byte/elem
# (packed_tile_k == tile_k); ptpc keeps no scale pool in LDS. Keep in sync with
# that file's LDS_PAD_A_BYTES / LDS_PAD_D_BYTES / elem_bytes_d.
_LDS_PAD_A_BYTES = 16
_LDS_PAD_D_BYTES = 16
_ELEM_BYTES_D = 2  # bf16 / f16 output

_TILE_M = (16, 32, 64, 256)
_TILE_N = (32, 64, 128, 256)
_TILE_K = (128, 256)
_NUM_BUFFERS = (3, 4)
_SPLIT_K = (1, 2, 4, 8)
# (m_warp, n_warp): wave-specialized ptpc requires m_warp*n_warp >= 2, and
# block_threads = m_warp*n_warp*32 <= 1024 (m_warp*n_warp <= 32). m_warp=1 with a
# small tile_m wins for decode (small M); m_warp=2 serves larger M; n_warp=4 helps
# very wide N. Add (1, 4) / (4, 2) here if a shape benefits.
_WARP_COMBOS = ((1, 2), (2, 2), (2, 4))
_CLUSTER = ((1, 1),)  # cluster_m * cluster_n <= 16


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
    # Mirrors the kernel's warp_tile_m/n constraints: each must be a multiple of
    # WMMA (16) and at most _MAX_WARP_TILE (per-wave VGPR/accumulator budget; see
    # the _MAX_WARP_TILE note). block_threads = mw*nw*32 <= 1024 (mw*nw <= 32).
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
    """LDS bytes the ptpc fp8 WMMA kernel actually allocates for ``ki``.

    Replicates the per-stage arena layout: each stage holds the A data pool
    (rows padded by LDS_PAD_A_BYTES) followed by the 16-aligned B data pool (no
    ptpc scale pool), the stage is 128- then 1024-aligned, and the arena is that
    pitch times num_buffers. The split_k==1 epilogue also needs a TDM-store D
    buffer, which can exceed the arena for small tiles, so take the max. The
    estimate must be exact/conservative: an under-estimate would let an
    overflowing tile through the candidate filter and fault the GPU at launch.
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


def max_lds_bytes_for_tune() -> int:
    return LDS_BYTES


def _build_kernels_list():
    kl = {}
    idx = 0
    for nb, sk, (cm, cn), (mw, nw), tm, tn, tk in product(
        _NUM_BUFFERS, _SPLIT_K, _CLUSTER, _WARP_COMBOS, _TILE_M, _TILE_N, _TILE_K
    ):
        if not _tile_valid(tm, tn, tk, mw, nw):
            continue
        ki = WmmaKernelInstance(tm, tn, tk, nb, sk, cm, cn, mw, nw)
        if kernel_instance_estimated_lds_bytes(ki) > LDS_BYTES:
            continue
        kl[idx] = ki
        idx += 1
    return kl


kernels_list: dict[int, WmmaKernelInstance] = _build_kernels_list()

default_kernels_dict = {
    (-1): WmmaKernelInstance(128, 128, 128, 2),
    (-2): WmmaKernelInstance(32, 64, 128, 2),
    (-3): WmmaKernelInstance(64, 128, 128, 2),
    (-4): WmmaKernelInstance(128, 256, 128, 2),
}


def kernel_fits_shape(ki: WmmaKernelInstance, M: int, N: int, K: int) -> bool:
    """N must divide tile_n (N is never padded); K must divide split_k*tile_k, and
    each split-k chunk must hold >= num_buffers K-tiles to fill the pipeline. M may
    be ragged — the kernel clips loads/stores to M via hardware out-of-bounds, so no
    M divisibility is required (a cluster just rounds the M-grid up and OOB-clips the
    extra tiles). A cluster still needs N cluster-tile-divisible and only pays off
    for M, N >= 4096. (LDS is bounded at build time, so it is not re-checked here.)
    """
    if N % ki.tile_n != 0 or K % (ki.split_k * ki.tile_k) != 0:
        return False
    if (K // ki.split_k) // ki.tile_k < ki.num_buffers:
        return False
    if ki.cluster_m > 1 or ki.cluster_n > 1:
        if M < 4096 or N < 4096:
            return False
        if N % (ki.cluster_n * ki.tile_n) != 0:
            return False
    return True
