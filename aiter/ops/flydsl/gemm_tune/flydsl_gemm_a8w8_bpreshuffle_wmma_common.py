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

WMMA = 16
WARP = 2  # m_warp = n_warp = 2 -> tile_m/n must be multiples of 32
LDS_BYTES = 320 * 1024

_TILE_M = (32, 64, 256)
_TILE_N = (32, 64, 256)
_TILE_K = (128, 256)
_NUM_BUFFERS = (2, 3, 4)
_SPLIT_K = (1, 2, 4, 8, 16)
_CLUSTER = ((1, 1), (2, 2), (2, 4), (4, 2))  # cluster_m * cluster_n <= 16


@dataclass
class WmmaKernelInstance:
    tile_m: int
    tile_n: int
    tile_k: int
    num_buffers: int
    split_k: int = 1
    cluster_m: int = 1
    cluster_n: int = 1

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
        )


def _tile_valid(tm: int, tn: int, tk: int) -> bool:
    return tm % (WARP * WMMA) == 0 and tn % (WARP * WMMA) == 0 and tk % 128 == 0


def kernel_instance_estimated_lds_bytes(ki: WmmaKernelInstance) -> int:
    return (ki.tile_m * ki.tile_k + ki.tile_n * ki.tile_k) * ki.num_buffers


def max_lds_bytes_for_tune() -> int:
    return LDS_BYTES


def _build_kernels_list():
    kl = {}
    idx = 0
    for nb, sk, (cm, cn), tm, tn, tk in product(
        _NUM_BUFFERS, _SPLIT_K, _CLUSTER, _TILE_M, _TILE_N, _TILE_K
    ):
        if not _tile_valid(tm, tn, tk):
            continue
        ki = WmmaKernelInstance(tm, tn, tk, nb, sk, cm, cn)
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
    each split-k chunk must hold >= num_buffers K-tiles to fill the pipeline. M is
    padded to tile_m, so ragged M is fine without a cluster; a cluster needs an
    evenly divisible grid and only pays off for M, N >= 4096.
    (LDS is bounded at build time, so it is not re-checked here.)
    """
    if N % ki.tile_n != 0 or K % (ki.split_k * ki.tile_k) != 0:
        return False
    if (K // ki.split_k) // ki.tile_k < ki.num_buffers:
        return False
    if ki.cluster_m > 1 or ki.cluster_n > 1:
        if M < 4096 or N < 4096:
            return False
        if M % (ki.cluster_m * ki.tile_m) != 0 or N % (ki.cluster_n * ki.tile_n) != 0:
            return False
    return True
