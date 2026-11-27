# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 FlyDSL candidates for the MXFP4 (a4w4) bpreshuffle GEMM.

Two kernels, both reading the 32x4-shuffled e8m0 scales and a 16x16-preshuffled B:

* the compute-bound 256x256 kernel (``flydsl_a4w4_bpreshuffle_compute_wmma_*``), 2x2 waves,
  4 buffers, a real workgroup cluster, optional clustered fused split-K;
* the general kernel (``flydsl_a4w4_bpreshuffle_wmma_*``), 4 waves in a 1x4, 2x2 or 4x1 layout and
  power-of-two tiles, including the M below the compute tile.

Compiling a config does not depend on (M, N, K), so ``kernels_list`` is shape independent and
:func:`kernel_fits_shape` keeps the ones one shape can launch. The launch rules themselves live
in the wrapper (:func:`a4w4_launch_error`); this module adds only what bounds the search: LDS
capacity, an accumulator budget, and which tiles are worth timing for an M range.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiter.jit.utils.chip_info import get_lds_capacity_bytes
from aiter.ops.flydsl.a4w4_mxscale_bpreshuffle_gemm_gfx1250 import (
    a4w4_fit_cluster_m,
    a4w4_kernel_name,
    a4w4_launch_error,
    parse_a4w4_kernel_name,
)

LDS_BYTES = get_lds_capacity_bytes("gfx1250")  # the kernels exist on gfx1250 only

# the profiles FlyDSL's own tests run for the compute kernel: (tile_m, tile_n, tile_k), plus the
# 192-row tile on the scheduled pipeline (192 x 256 x 256), which tiles M = 1536 into 8 rows
_COMPUTE_TILES = tuple(
    (m, n, k) for m in (256, 128) for n in (256, 128) for k in (256, 128)
) + (
    (192, 256, 256),
)
_COMPUTE_CLUSTER_M = (1, 2, 3, 4, 6, 8)
_COMPUTE_CLUSTER_N = (1, 2, 3, 4, 6, 7, 8)

# General kernel tiles, powers of two from 16 to 512 (tile_k up to 1024). Which tile_m / tile_n a
# warp layout accepts, and the accumulator budget below, decide the combinations that survive;
# tile_k must be a power of two (the TDM padding interval of an A row, tile_k / 8 dwords, is).
_GENERAL_TILE_M = (16, 32, 64, 128, 256, 512)
_GENERAL_TILE_N = (16, 32, 64, 128, 256, 512)
_GENERAL_TILE_K = (128, 256, 512, 1024)
_GENERAL_WARPS = ((1, 4), (2, 2), (4, 1))  # 4 waves
_GENERAL_NUM_BUFFERS = (2, 3, 4)
_GENERAL_CLUSTERS = (
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (1, 8),
    (3, 4),
)
_SPLIT_K = (1, 2, 4, 8)
# FlyDSL's widest tested general tile (128 x 512, 2x2 waves) holds 4 x 8 accumulator fragments
_MAX_ACC_FRAGMENTS = 32
# a grid of at most this many output tiles under-fills the part, the only case split-K can help
_SPLIT_K_MAX_TILES = 512


@dataclass(frozen=True)
class A4W4KernelInstance:
    tile_m: int
    tile_n: int
    tile_k: int
    num_buffers: int
    split_k: int = 1
    cluster_m: int = 1
    cluster_n: int = 1
    m_warp: int = 2
    n_warp: int = 2
    compute_bound: bool = False

    def name_for(self) -> str:
        return a4w4_kernel_name(
            self.tile_m,
            self.tile_n,
            self.tile_k,
            m_warp=self.m_warp,
            n_warp=self.n_warp,
            num_buffers=self.num_buffers,
            split_k=self.split_k,
            cluster_m=self.cluster_m,
            cluster_n=self.cluster_n,
            compute_bound=self.compute_bound,
        )

    @property
    def name(self) -> str:
        return self.name_for()

    @property
    def cfg(self) -> dict:
        cfg = parse_a4w4_kernel_name(self.name)
        assert cfg is not None
        return cfg


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _align_up(value: int, align: int) -> int:
    return _ceil_div(value, align) * align


def general_lds_bytes(ki: A4W4KernelInstance) -> int:
    """LDS the general kernel allocates (the arena of ``launch_gemm_a4w4``)."""
    pack_k = ki.tile_k // 2
    a_row, b_row = pack_k + 16, pack_k * 16
    sa_rows, sb_rows = max(1, ki.tile_m // 32), ki.tile_n // 32
    a_bytes, b_bytes = ki.tile_m * a_row, (ki.tile_n // 16) * b_row
    sb_off = a_bytes + b_bytes + sa_rows * ki.tile_k
    pitch = _align_up(sb_off + sb_rows * ki.tile_k, 1024)
    c_lds_row = ki.tile_n + 8
    return max(
        ki.num_buffers * pitch, ki.tile_m * c_lds_row * (4 if ki.split_k > 1 else 2)
    )


def compute_lds_bytes(ki: A4W4KernelInstance) -> int:
    """LDS the compute kernel allocates (the planar arena of ``launch_gemm_a4w4_256x256``)."""
    nb = ki.num_buffers
    pack_tk = ki.tile_k // 2
    stage_a = ki.tile_m * (pack_tk + 16)
    stage_b = (ki.tile_n // 16) * (pack_tk * 16)
    stage_sa, stage_sb = (ki.tile_m // 32) * ki.tile_k, (ki.tile_n // 32) * ki.tile_k
    sb_base = nb * stage_sa + nb * stage_a
    b_base = _align_up(sb_base + nb * stage_sb, 65536)
    c_lds_row = ki.tile_n + 8
    return max(
        b_base + nb * stage_b, ki.tile_m * c_lds_row * (4 if ki.split_k > 1 else 2)
    )


def kernel_lds_bytes(ki: A4W4KernelInstance) -> int:
    return compute_lds_bytes(ki) if ki.compute_bound else general_lds_bytes(ki)


# A huge shape every config divides: N is a multiple of tile_n * cluster_n for every listed
# cluster_n (3 and 7 included -- a plain 2^15 silently dropped cluster_n 3, 6 and 7).
_ALIGNED_SHAPE = (1 << 20, (1 << 15) * 3 * 7, 1 << 14)


def _launchable_in_principle(ki: A4W4KernelInstance) -> bool:
    """The shape independent part of the launch rules, checked at a huge, aligned shape."""
    if a4w4_launch_error(ki.cfg, *_ALIGNED_SHAPE) is not None:
        return False
    return kernel_lds_bytes(ki) <= LDS_BYTES


def _build_kernels_list() -> dict[int, A4W4KernelInstance]:
    kl: dict[int, A4W4KernelInstance] = {}
    idx = 0
    for tm in _GENERAL_TILE_M:
        for tn in _GENERAL_TILE_N:
            for tk in _GENERAL_TILE_K:
                for mw, nw in _GENERAL_WARPS:
                    divisible = tm % (16 * mw) == 0 and tn % (32 * nw) == 0
                    acc = (tm // (16 * mw)) * (tn // (32 * nw)) if divisible else 0
                    if acc == 0 or acc > _MAX_ACC_FRAGMENTS:
                        continue
                    for nb in _GENERAL_NUM_BUFFERS:
                        for cm, cn in _GENERAL_CLUSTERS:
                            for sk in _SPLIT_K:
                                ki = A4W4KernelInstance(
                                    tm, tn, tk, nb, sk, cm, cn, mw, nw, False
                                )
                                if _launchable_in_principle(ki):
                                    kl[idx] = ki
                                    idx += 1
    for tm, tn, tk in _COMPUTE_TILES:
        for cm in _COMPUTE_CLUSTER_M:
            for cn in _COMPUTE_CLUSTER_N:
                for sk in _SPLIT_K:
                    ki = A4W4KernelInstance(tm, tn, tk, 4, sk, cm, cn, 2, 2, True)
                    if _launchable_in_principle(ki):
                        kl[idx] = ki
                        idx += 1
    return kl


kernels_list: dict[int, A4W4KernelInstance] = _build_kernels_list()


def _general_tile_m_bounds(M: int) -> tuple[int, int]:
    """The general tile heights worth timing for M, as (lowest, highest): a tile taller than M
    rounded up to a scale group wastes the part, one far shorter re-reads B for every few rows.
    """
    if M <= 128:
        low = 16
    elif M <= 512:
        low = 32
    else:
        low = 64
    return low, max(32, _align_up(M, 32))


def kernel_fits_shape(ki: A4W4KernelInstance, M: int, N: int, K: int) -> bool:
    """Whether ``ki`` can launch for (M, N, K) and is worth timing there."""
    cfg = ki.cfg
    if a4w4_launch_error(cfg, M, N, K) is not None:
        return False
    # A compute tile launches below tile_m (one partial M tile), but the search keeps
    # screening it only where it fills whole tiles of M.
    if ki.compute_bound and M < ki.tile_m:
        return False
    # Time exactly the config named: no cluster_m reduction, no padded cluster rows.
    m_tiles = _ceil_div(M, ki.tile_m)
    if a4w4_fit_cluster_m(cfg, M) != ki.cluster_m:
        return False
    if N % (ki.tile_n * ki.cluster_n) != 0:
        return False
    if K % (ki.tile_k * ki.split_k) != 0 or K // ki.split_k < ki.tile_k:
        return False
    if ki.split_k > 1 and m_tiles * (N // ki.tile_n) > _SPLIT_K_MAX_TILES:
        return False
    if ki.compute_bound:
        return True
    low, high = _general_tile_m_bounds(M)
    return low <= ki.tile_m <= high


def candidates_for_shape(
    M: int, N: int, K: int
) -> list[tuple[int, A4W4KernelInstance]]:
    return [(i, ki) for i, ki in kernels_list.items() if kernel_fits_shape(ki, M, N, K)]
