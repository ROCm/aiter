# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Full-style gfx1250 WMMA candidates for a8w8 blockscale bpreshuffle GEMM."""

from __future__ import annotations

from dataclasses import dataclass

from aiter.ops.flydsl.utils import get_shared_memory_per_block

NAME_PREFIX = "flydsl_blockscale_bpreshuffle_wmma"

WMMA = 16  # WMMA M/N tile granularity
LDS_BYTES = get_shared_memory_per_block(fallback_gfx="gfx1250")
_MAX_WARP_TILE = 256
_MAX_TUNE_WARPS = 4
_MAX_ACC_FRAGMENTS = 64
_MAX_TILE_ELEMENTS = 64 * 1024

_LDS_PAD_A_BYTES = 16
_LDS_PAD_D_BYTES = 16
_ELEM_BYTES_D = 2  # bf16 / f16 output

_BLOCK_N = 128
_BLOCK_K = 128
_SCALE_BYTES = 1  # gfx1250 blockscale scales are fp8/e8m0 storage
_SCALE_GUARD_BYTES = 16

_TILE_M_OPTS = (16, 32, 64, 128, 256)
_TILE_N_OPTS = (32, 64, 128, 256, 512, 1024)
_TILE_K_OPTS = (128, 256, 512, 1024)
_NUM_BUFFERS_OPTS = (2, 3, 4)
_WARP_OPTS = ((1, 2), (1, 4), (2, 1), (2, 2), (4, 1))
_CLUSTER_OPTS = ((1, 1),)
_SPLIT_K = (1,)

_CLUSTER_MIN_DIM = 8192
_CLUSTER_MIN_TILES = 32
_LARGE_M_TILE_MIN = 256
_MAX_M_BLOCKS = 16


@dataclass
class WmmaKernelInstance:
    tile_m: int
    tile_n: int
    tile_k: int
    num_buffers: int
    split_k: int = 1
    cluster_m: int = 1
    cluster_n: int = 1
    m_warp: int = 2
    n_warp: int = 2

    @property
    def name(self) -> str:
        return (
            f"{NAME_PREFIX}_t{self.tile_m}x{self.tile_n}x{self.tile_k}_"
            f"mw{self.m_warp}_nw{self.n_warp}_nb{self.num_buffers}_sk{self.split_k}_"
            f"cm{self.cluster_m}_cn{self.cluster_n}"
        )


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _tile_valid(tm: int, tn: int, tk: int, mw: int, nw: int) -> bool:
    if mw < 1 or nw < 1:
        return False
    if tm > tn and mw < nw:
        return False
    if tm < tn and mw > nw:
        return False
    if tm == tn and mw != nw:
        return False
    if tm % (mw * WMMA) != 0 or tn % (nw * WMMA) != 0:
        return False
    warp_tile_m = tm // mw
    warp_tile_n = tn // nw
    if warp_tile_m > _MAX_WARP_TILE or warp_tile_n > _MAX_WARP_TILE:
        return False
    if tk % _BLOCK_K != 0:
        return False
    tune_warps = mw * nw
    if tune_warps < 2 or tune_warps > _MAX_TUNE_WARPS:
        return False
    acc_fragments = (warp_tile_m // WMMA) * (warp_tile_n // WMMA)
    if acc_fragments > _MAX_ACC_FRAGMENTS:
        return False
    if tm * tn > _MAX_TILE_ELEMENTS:
        return False
    return True


def _cluster_valid(cm: int, cn: int) -> bool:
    return cm >= 1 and cn >= 1 and cm * cn <= 16


def _blockscale_stage_scale_bytes(ki: WmmaKernelInstance) -> tuple[int, int]:
    k_blocks = _ceil_div(ki.tile_k, _BLOCK_K)
    a_scale_bytes = ki.tile_m * k_blocks * _SCALE_BYTES + _SCALE_GUARD_BYTES
    n_blocks = _ceil_div(ki.tile_n, _BLOCK_N) + 1
    b_scale_bytes = n_blocks * k_blocks * _SCALE_BYTES + _SCALE_GUARD_BYTES
    return a_scale_bytes, b_scale_bytes


def kernel_instance_estimated_lds_bytes(ki: WmmaKernelInstance) -> int:
    """Conservative LDS upper bound for blockscale FP8 WMMA candidates."""

    a_scale_bytes, b_scale_bytes = _blockscale_stage_scale_bytes(ki)

    lds_a_data = ki.tile_m * (ki.tile_k + _LDS_PAD_A_BYTES)
    lds_b_data = ki.tile_n * ki.tile_k
    stage_bytes = (
        _align_up(lds_a_data, 16)
        + _align_up(lds_b_data, 16)
        + _align_up(a_scale_bytes, 16)
        + _align_up(b_scale_bytes, 16)
    )
    stage_pitch = _align_up(_align_up(stage_bytes, 128), 1024)
    arena_bytes = stage_pitch * ki.num_buffers

    if ki.split_k == 1:  # split_k>1 uses the buffer/atomic store, no LDS D buffer
        warp_tile_m = ki.tile_m // ki.m_warp
        warp_tile_n = ki.tile_n // ki.n_warp
        d_row_stride = warp_tile_n * _ELEM_BYTES_D + _LDS_PAD_D_BYTES
        total_d_bytes = (ki.m_warp * ki.n_warp) * warp_tile_m * d_row_stride
        return max(arena_bytes, total_d_bytes)
    return arena_bytes


def _build_kernels_list() -> dict[int, WmmaKernelInstance]:
    kl = {}
    idx = 0
    for tm in _TILE_M_OPTS:
        for tn in _TILE_N_OPTS:
            for tk in _TILE_K_OPTS:
                for mw, nw in _WARP_OPTS:
                    if not _tile_valid(tm, tn, tk, mw, nw):
                        continue
                    for nb in _NUM_BUFFERS_OPTS:
                        for sk in _SPLIT_K:
                            for cm, cn in _CLUSTER_OPTS:
                                if not _cluster_valid(cm, cn):
                                    continue
                                ki = WmmaKernelInstance(
                                    tm, tn, tk, nb, sk, cm, cn, mw, nw
                                )
                                if kernel_instance_estimated_lds_bytes(ki) > LDS_BYTES:
                                    continue
                                kl[idx] = ki
                                idx += 1
    return kl


kernels_list: dict[int, WmmaKernelInstance] = _build_kernels_list()


def kernel_fits_shape(ki: WmmaKernelInstance, M: int, N: int, K: int) -> bool:
    """Return True iff ``ki`` can be launched for runtime shape ``(M, N, K)``."""

    if N % _BLOCK_N != 0 or K % _BLOCK_K != 0:
        return False
    if not _tile_valid(ki.tile_m, ki.tile_n, ki.tile_k, ki.m_warp, ki.n_warp):
        return False
    if not _cluster_valid(ki.cluster_m, ki.cluster_n):
        return False
    if kernel_instance_estimated_lds_bytes(ki) > LDS_BYTES:
        return False
    if N % ki.tile_n != 0 or K % (ki.split_k * ki.tile_k) != 0:
        return False
    if (K // ki.split_k) // ki.tile_k < ki.num_buffers:
        return False
    if ki.tile_m > M:
        return False

    m_blocks = _ceil_div(M, ki.tile_m)
    n_blocks = N // ki.tile_n
    if ki.tile_m < _LARGE_M_TILE_MIN and m_blocks > _MAX_M_BLOCKS:
        return False
    if ki.cluster_m > 1:
        if M < _CLUSTER_MIN_DIM or m_blocks < _CLUSTER_MIN_TILES:
            return False
    if ki.cluster_n > 1:
        if N < _CLUSTER_MIN_DIM or n_blocks < _CLUSTER_MIN_TILES:
            return False
        if n_blocks % ki.cluster_n != 0:
            return False
    return True
