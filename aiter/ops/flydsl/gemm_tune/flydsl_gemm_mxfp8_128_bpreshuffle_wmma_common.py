# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Full-style gfx1250 WMMA candidates for mxfp8_128 bpreshuffle GEMM."""

from __future__ import annotations

from dataclasses import dataclass, replace

from aiter.jit.utils.chip_info import get_gfx, get_lds_capacity_bytes
from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
    COMPUTE_WMMA_NAME_PREFIX as COMPUTE_NAME_PREFIX,
)
from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
    WMMA_NAME_PREFIX as NAME_PREFIX,
)
from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
    cluster_m_grid_ok,
    compute_kernel_k_pair,
    fused_splitk_ok,
    rank0_splitk_ok,
)

WMMA = 16  # WMMA M/N tile granularity
LDS_BYTES = get_lds_capacity_bytes(get_gfx())
_MAX_WARP_TILE = 256
_MAX_TUNE_WARPS = 4
_MAX_ACC_FRAGMENTS = 64
_MAX_TILE_ELEMENTS = 64 * 1024

_LDS_PAD_A_BYTES = 16
_LDS_PAD_D_BYTES = 16
_ELEM_BYTES_D = 2  # bf16 / f16 output

_BLOCK_N = 128
_BLOCK_K = 128
_SCALE_BYTES = 1  # gfx1250 mxfp8_128 scales are fp8/e8m0 storage
_SCALE_GUARD_BYTES = 16

_TILE_M_OPTS = (16, 32, 64, 128, 256)
_TILE_N_OPTS = (32, 64, 128, 256, 512, 1024)
_TILE_K_OPTS = (128, 256, 512, 1024)
_NUM_BUFFERS_OPTS = (2, 3, 4)
_WARP_OPTS = ((1, 4), (2, 2), (4, 1))
_CLUSTER_OPTS = ((1, 1),)
_SPLIT_K = (1, 2, 4, 8)
# Fused split_k the row-stripe epilogue lacks: split 0 of a tile sums the splits.
_RANK0_SPLIT_K = (3, 5, 6, 7, 10, 12, 14, 16)

_COMPUTE_PROFILES = (
    (256, 256, 128, 2, 2, 4),
    (256, 256, 128, 2, 2, 2),
    (128, 128, 128, 2, 2, 4),
    (128, 256, 128, 2, 2, 4),
    (128, 256, 128, 2, 2, 3),
)
_COMPUTE_CLUSTERS = ((2, 2), (4, 2), (2, 4), (4, 4), (1, 4))
# Power-of-two clusters up to 16 workgroups (no 1x16 / 16x1): what a cluster_m
# fitted to an M-tile count can reach. Their candidates come after the original
# ones, so existing kernelIds are unchanged.
_CLUSTERS_POW2 = tuple(
    (cm, cn) for cm in (1, 2, 4, 8) for cn in (1, 2, 4, 8) if cm * cn <= 16
)
_GENERIC_CLUSTERS_EXTRA = tuple(c for c in _CLUSTERS_POW2 if c != (1, 1))
_COMPUTE_CLUSTERS_EXTRA = tuple(c for c in _CLUSTERS_POW2 if c not in _COMPUTE_CLUSTERS)
# Clusters of 3 or 5 M tiles share each weight read when the M-tile count is not a
# power of two; tile_m that are not powers of two fit decode M between them.
_CLUSTERS_NPOT = ((3, 1), (3, 2), (5, 1), (5, 2))
_COMPUTE_CLUSTERS_NPOT = ((3, 1), (3, 2), (3, 4), (5, 1), (5, 2))
_TILE_M_NPOT = (96, 160, 192)
_COMPUTE_CLUSTERS_ALL = (
    _COMPUTE_CLUSTERS + _COMPUTE_CLUSTERS_EXTRA + _COMPUTE_CLUSTERS_NPOT
)
_COMPUTE_MIN_K_PER_SPLIT = 512
_MAX_CLUSTER = 16  # workgroups of a gfx1250 cluster
_PERSISTENT_N_TILES = (2, 4, 8)
# Spare N tiles a bounded_n launch may add, as a fraction of the real ones.
_N_TAIL_MAX_PAD = 0.5

_LARGE_M_TILE_MIN = 256
_MAX_M_BLOCKS = 32


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
    name_prefix: str = NAME_PREFIX
    a_preshuffle: bool = False
    persistent_n_tiles: int = 1
    fused_splitk: bool = False
    # LDS row pad (elements) of the C staging tile; -1 = default.
    c_pad: int = -1
    # Opt-in schedules: _qe (quadrant early loads), _nlb (no in-loop cluster barrier).
    quad_early: bool = False
    no_loop_cbar: bool = False

    def name_for(self, a_preshuffle: bool = False) -> str:
        return (
            f"{self.name_prefix}_t{self.tile_m}x{self.tile_n}x{self.tile_k}_"
            f"mw{self.m_warp}_nw{self.n_warp}_nb{self.num_buffers}_sk{self.split_k}_"
            f"cm{self.cluster_m}_cn{self.cluster_n}"
            + (f"_cp{self.c_pad}" if self.c_pad >= 0 else "")
            + ("_qe" if self.quad_early else "")
            + ("_nlb" if self.no_loop_cbar else "")
            + ("_fsk" if self.fused_splitk else "")
            + ("_apre" if (a_preshuffle or self.a_preshuffle) else "")
            + (f"_ps{self.persistent_n_tiles}" if self.persistent_n_tiles > 1 else "")
        )

    @property
    def name(self) -> str:
        return self.name_for()


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
    return tm * tn <= _MAX_TILE_ELEMENTS


def _cluster_valid(cm: int, cn: int) -> bool:
    return cm >= 1 and cn >= 1 and cm * cn <= 16


def _mxfp8_128_stage_scale_bytes(ki: WmmaKernelInstance) -> tuple[int, int]:
    k_blocks = _ceil_div(ki.tile_k, _BLOCK_K)
    a_scale_bytes = ki.tile_m * k_blocks * _SCALE_BYTES + _SCALE_GUARD_BYTES
    n_blocks = _ceil_div(ki.tile_n, _BLOCK_N) + 1
    b_scale_bytes = n_blocks * k_blocks * _SCALE_BYTES + _SCALE_GUARD_BYTES
    return a_scale_bytes, b_scale_bytes


def kernel_instance_estimated_lds_bytes(ki: WmmaKernelInstance) -> int:
    """Conservative LDS upper bound for mxfp8_128 FP8 WMMA candidates."""

    a_scale_bytes, b_scale_bytes = _mxfp8_128_stage_scale_bytes(ki)

    a_pair = 2 if ki.a_preshuffle else 1
    lds_a_data = (ki.tile_m // a_pair) * (a_pair * ki.tile_k + _LDS_PAD_A_BYTES)
    lds_b_data = ki.tile_n * ki.tile_k
    stage_bytes = (
        _align_up(lds_a_data, 16)
        + _align_up(lds_b_data, 16)
        + _align_up(a_scale_bytes, 16)
        + _align_up(b_scale_bytes, 16)
    )
    stage_pitch = _align_up(_align_up(stage_bytes, 128), 1024)
    arena_bytes = stage_pitch * ki.num_buffers

    warp_tile_m = ki.tile_m // ki.m_warp
    warp_tile_n = ki.tile_n // ki.n_warp
    d_row_stride = warp_tile_n * _ELEM_BYTES_D + _LDS_PAD_D_BYTES
    total_d_bytes = (ki.m_warp * ki.n_warp) * warp_tile_m * d_row_stride
    return max(arena_bytes, total_d_bytes)


def _generic_instances(clusters, fused, split_ks=_SPLIT_K, tile_ms=_TILE_M_OPTS):
    for sk in split_ks:
        for tm in tile_ms:
            for tn in _TILE_N_OPTS:
                for tk in _TILE_K_OPTS:
                    for mw, nw in _WARP_OPTS:
                        if not _tile_valid(tm, tn, tk, mw, nw):
                            continue
                        for nb in _NUM_BUFFERS_OPTS:
                            for cm, cn in clusters:
                                if not _cluster_valid(cm, cn):
                                    continue
                                ki = WmmaKernelInstance(
                                    tm,
                                    tn,
                                    tk,
                                    nb,
                                    sk,
                                    cm,
                                    cn,
                                    mw,
                                    nw,
                                    fused_splitk=fused and sk > 1,
                                )
                                if kernel_instance_estimated_lds_bytes(ki) > LDS_BYTES:
                                    continue
                                yield ki


def _compute_instances(clusters, fused, split_ks=_SPLIT_K):
    for sk in split_ks:
        for tm, tn, tk, mw, nw, nb in _COMPUTE_PROFILES:
            for cm, cn in clusters:
                for ps in (1, *_PERSISTENT_N_TILES) if sk == 1 else (1,):
                    yield WmmaKernelInstance(
                        tile_m=tm,
                        tile_n=tn,
                        tile_k=tk,
                        num_buffers=nb,
                        split_k=sk,
                        cluster_m=cm,
                        cluster_n=cn,
                        m_warp=mw,
                        n_warp=nw,
                        name_prefix=COMPUTE_NAME_PREFIX,
                        persistent_n_tiles=ps,
                        fused_splitk=fused and sk > 1,
                    )


def _base_instances() -> list[WmmaKernelInstance]:
    return [
        *_generic_instances(_CLUSTER_OPTS, False),
        *_compute_instances(_COMPUTE_CLUSTERS, False),
    ]


def _build_kernels_list() -> dict[int, WmmaKernelInstance]:
    base = _base_instances()
    # Appended in this order so every existing kernelId keeps its kernel: fused
    # twins, then the wider clusters (their split_k > 1 candidates are fused only,
    # and a fused cluster holds at most 16 workgroups).
    extra = [replace(ki, fused_splitk=True) for ki in base if ki.split_k > 1]
    wide = [
        *_generic_instances(_GENERIC_CLUSTERS_EXTRA, True),
        *_compute_instances(_COMPUTE_CLUSTERS_EXTRA, True),
    ]
    extra += [
        ki for ki in wide if ki.cluster_m * ki.cluster_n * ki.split_k <= _MAX_CLUSTER
    ]
    # Single opt-in candidates used by K3 rows, appended last.
    extra.append(
        WmmaKernelInstance(
            tile_m=64,
            tile_n=64,
            tile_k=256,
            num_buffers=4,
            split_k=4,
            cluster_m=2,
            cluster_n=2,
            m_warp=2,
            n_warp=2,
            fused_splitk=True,
            c_pad=4,
        )
    )
    extra.append(
        WmmaKernelInstance(
            tile_m=64,
            tile_n=128,
            tile_k=512,
            num_buffers=3,
            split_k=1,
            cluster_m=1,
            cluster_n=1,
            m_warp=1,
            n_warp=4,
            quad_early=True,
        )
    )
    extra.append(
        WmmaKernelInstance(
            tile_m=128,
            tile_n=128,
            tile_k=256,
            num_buffers=4,
            split_k=1,
            cluster_m=1,
            cluster_n=2,
            m_warp=2,
            n_warp=2,
            quad_early=True,
        )
    )
    extra.append(
        WmmaKernelInstance(
            tile_m=64,
            tile_n=128,
            tile_k=256,
            num_buffers=4,
            split_k=4,
            cluster_m=1,
            cluster_n=1,
            m_warp=1,
            n_warp=4,
            fused_splitk=True,
            quad_early=True,
            no_loop_cbar=True,
        )
    )
    # Rank-0 fused split-K of the generic kernel, appended after the opt-in ones.
    extra += [
        ki
        for ki in _generic_instances(_CLUSTERS_POW2, True, _RANK0_SPLIT_K)
        if ki.cluster_m * ki.cluster_n * ki.split_k <= _MAX_CLUSTER
    ]
    # Clusters of 3 / 5 M tiles, then tile_m that are not powers of two.
    npot = [
        *_generic_instances(_CLUSTERS_NPOT, True, (1, 2, 4), (16, 32, 64, 128)),
        *_compute_instances(_COMPUTE_CLUSTERS_NPOT, True, (1, 2, 4)),
        *_generic_instances(
            ((1, 1), (1, 2), (1, 4), (2, 1), (2, 2)), True, (1, 2, 4), _TILE_M_NPOT
        ),
    ]
    extra += [
        ki for ki in npot if ki.cluster_m * ki.cluster_n * ki.split_k <= _MAX_CLUSTER
    ]
    return dict(enumerate([*base, *extra]))


kernels_list: dict[int, WmmaKernelInstance] = _build_kernels_list()


def kernel_fits_shape(
    ki: WmmaKernelInstance, M: int, N: int, K: int, n_tail: bool = False
) -> bool:
    """Return True iff ``ki`` can be launched for runtime shape ``(M, N, K)``.

    ``n_tail`` admits a split_k=1 compute kernel whose N ends inside its last group
    of ``tile_n * cluster_n * persistent_n_tiles`` columns (bounded_n); the tuner
    sets it, the untuned fallback does not."""

    # Row pairs need both rows, so an odd M has no valid pairing.
    if ki.a_preshuffle and M % 2 != 0:
        return False

    # Fused split-K: the row-stripe epilogue where it fits, else rank 0 (generic only).
    rank0 = False
    if ki.fused_splitk and not fused_splitk_ok(
        ki.tile_m,
        ki.tile_n,
        ki.m_warp,
        ki.n_warp,
        ki.cluster_m,
        ki.cluster_n,
        ki.split_k,
    ):
        rank0 = not is_compute_kernel(ki) and rank0_splitk_ok(
            M, N, ki.tile_m, ki.cluster_m, ki.cluster_n, ki.split_k
        )
        if not rank0:
            return False

    if is_compute_kernel(ki):
        profile = (
            ki.tile_m,
            ki.tile_n,
            ki.tile_k,
            ki.m_warp,
            ki.n_warp,
            ki.num_buffers,
        )
        if M <= 0 or N <= 0 or K <= 0:
            return False
        if profile not in _COMPUTE_PROFILES:
            return False
        if (ki.cluster_m, ki.cluster_n) not in _COMPUTE_CLUSTERS_ALL:
            return False
        # M below tile_m is one partial tile (M OOB), so it clusters no M rows.
        if not cluster_m_grid_ok(M, ki.tile_m, ki.cluster_m):
            return False
        if ki.persistent_n_tiles > 1 and ki.split_k != 1:
            return False
        if N % (ki.tile_n * ki.cluster_n * ki.persistent_n_tiles):
            if not n_tail or ki.split_k != 1 or N % _BLOCK_N:
                return False
            n_tiles = _ceil_div(N, ki.tile_n)
            if _padded_n_tiles(ki, N) - n_tiles > _N_TAIL_MAX_PAD * n_tiles:
                return False
        if K % ki.split_k != 0:
            return False
        k_per_split = K // ki.split_k
        k_pair = compute_kernel_k_pair(ki.num_buffers, ki.tile_n)
        return (
            k_per_split >= _COMPUTE_MIN_K_PER_SPLIT
            and k_per_split % (ki.tile_k * k_pair) == 0
        )

    if ki.persistent_n_tiles > 1:
        return False
    if N % _BLOCK_N != 0 or K % _BLOCK_K != 0:
        return False
    if not _tile_valid(ki.tile_m, ki.tile_n, ki.tile_k, ki.m_warp, ki.n_warp):
        return False
    if not _cluster_valid(ki.cluster_m, ki.cluster_n):
        return False
    if kernel_instance_estimated_lds_bytes(ki) > LDS_BYTES:
        return False
    if N % ki.tile_n != 0 or K % (ki.tile_k * (1 if rank0 else ki.split_k)) != 0:
        return False
    # Generic K=128 with two buffers needs no steady-loop iteration.
    min_k_tiles = 1 if ki.num_buffers == 2 and K == ki.tile_k == 128 else ki.num_buffers
    if (K // ki.tile_k) // ki.split_k < min_k_tiles:
        return False
    # M may be ragged; the kernel handles partial M tiles via OOB clamping,
    # so tile_m > M is legal (and needed for small-M decode shapes).

    m_blocks = _ceil_div(M, ki.tile_m)
    n_blocks = N // ki.tile_n
    if ki.tile_m < _LARGE_M_TILE_MIN and m_blocks > _MAX_M_BLOCKS:
        return False
    # A multicast cluster needs at least two tiles along each clustered axis (N below).
    if ki.cluster_m > 1 and M < 2 * ki.tile_m:
        return False
    if ki.cluster_m > 1 and not cluster_m_grid_ok(M, ki.tile_m, ki.cluster_m):
        return False
    return not (
        ki.cluster_n > 1 and (n_blocks < 2 * ki.cluster_n or n_blocks % ki.cluster_n)
    )


def _padded_n_tiles(ki: WmmaKernelInstance, N: int) -> int:
    """N tiles of the launch: the real ones, filled up to whole cluster / persistent groups."""
    group = ki.cluster_n * ki.persistent_n_tiles
    return _ceil_div(_ceil_div(N, ki.tile_n), group) * group


def compute_grid_ctas(ki: WmmaKernelInstance, M: int, N: int) -> int:
    gx = _ceil_div(M, ki.tile_m)
    return gx * (_padded_n_tiles(ki, N) // ki.persistent_n_tiles) * ki.split_k


def is_pow2_geometry(ki: WmmaKernelInstance) -> bool:
    """Power-of-two tile_m and cluster: the families the untuned heuristics pick from."""
    return all(v & (v - 1) == 0 for v in (ki.tile_m, ki.cluster_m, ki.cluster_n))


def is_compute_kernel(kernel: WmmaKernelInstance) -> bool:
    """Return whether a tuning instance targets the compute-bound kernel."""
    return kernel.name_prefix == COMPUTE_NAME_PREFIX
