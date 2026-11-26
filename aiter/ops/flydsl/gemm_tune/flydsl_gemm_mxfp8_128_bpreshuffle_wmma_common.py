# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Full-style gfx1250 WMMA candidates for mxfp8_128 bpreshuffle GEMM."""

from __future__ import annotations

from dataclasses import dataclass

from aiter.jit.utils.chip_info import get_gfx, get_lds_capacity_bytes
from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
    COMPUTE_WMMA_NAME_PREFIX as COMPUTE_NAME_PREFIX,
)
from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
    MX32_SMALL_WMMA_NAME_PREFIX as SMALL_NAME_PREFIX,
)
from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
    WMMA_NAME_PREFIX as NAME_PREFIX,
)
from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
    cluster_m_grid_ok,
    compute_kernel_k_pair,
    ragged_n_ok,
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

_COMPUTE_PROFILES = (
    (256, 256, 128, 2, 2, 4),
    (256, 256, 128, 2, 2, 2),
    (256, 192, 128, 2, 2, 2),
    (192, 256, 128, 2, 2, 4),
    (192, 256, 128, 2, 2, 2),
    (192, 192, 128, 2, 2, 2),
    (128, 128, 128, 2, 2, 4),
    (128, 128, 128, 2, 2, 2),
    (128, 256, 128, 2, 2, 4),
    (128, 256, 128, 2, 2, 3),
)
_COMPUTE_CLUSTERS = ((2, 2), (4, 2), (2, 4), (4, 4), (1, 4))
_COMPUTE_UNCLUSTERED = ((1, 1),)
_NARROW_N_COMPUTE_PROFILES = ((256, 128, 128, 2, 2, 2),)
_NARROW_N_COMPUTE_CLUSTERS = ((1, 1), (2, 1), (4, 1))
_COMPUTE_MIN_K_PER_SPLIT = 512
_SMALL_PROFILES = ((64, 128, 128, 2, 2, 2),)
_PERSISTENT_N_TILES = (2, 4, 8)

_CLUSTER_MIN_DIM = 8192
_CLUSTER_MIN_TILES = 32
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
    early_a: bool = False
    early_b: bool = False
    fused_splitk: bool = False
    l2_prefetch: int = 0

    def name_for(self, a_preshuffle: bool = False) -> str:
        return (
            f"{self.name_prefix}_t{self.tile_m}x{self.tile_n}x{self.tile_k}_"
            f"mw{self.m_warp}_nw{self.n_warp}_nb{self.num_buffers}_sk{self.split_k}_"
            f"cm{self.cluster_m}_cn{self.cluster_n}"
            + ("_fsk" if self.fused_splitk else "")
            + ("_apre" if (a_preshuffle or self.a_preshuffle) else "")
            + (f"_ps{self.persistent_n_tiles}" if self.persistent_n_tiles > 1 else "")
            + (
                "_eab"
                if self.early_a and self.early_b
                else ("_ea" if self.early_a else ("_eb" if self.early_b else ""))
            )
            + (f"_pf{self.l2_prefetch}" if self.l2_prefetch else "")
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


def mxfp8_32_kernel_fits_shape(ki: WmmaKernelInstance, M: int, N: int, K: int) -> bool:
    """Whether a shared WMMA instance fits the shape and mx32 LDS footprint."""
    if not kernel_fits_shape(ki, M, N, K, allow_ragged_n=True):
        return False
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx1250 import mx32_lds_bytes

    config = {
        "tile_m": ki.tile_m,
        "tile_n": ki.tile_n,
        "tile_k": ki.tile_k,
        "num_buffers": ki.num_buffers,
    }
    return mx32_lds_bytes(config) <= LDS_BYTES


def _build_kernels_list() -> dict[int, WmmaKernelInstance]:
    kl = {}
    idx = 0
    for sk in _SPLIT_K:
        for tm in _TILE_M_OPTS:
            for tn in _TILE_N_OPTS:
                for tk in _TILE_K_OPTS:
                    for mw, nw in _WARP_OPTS:
                        if not _tile_valid(tm, tn, tk, mw, nw):
                            continue
                        for nb in _NUM_BUFFERS_OPTS:
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
    for sk in _SPLIT_K:
        for tm, tn, tk, mw, nw, nb in _COMPUTE_PROFILES:
            for cm, cn in _COMPUTE_CLUSTERS:
                for ps in (1, *_PERSISTENT_N_TILES) if sk == 1 else (1,):
                    kl[idx] = WmmaKernelInstance(
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
                    )
                    idx += 1
    # Keep the established numeric IDs above stable.  The direct B=1 path does
    # not need multicast, and on medium/large M its cluster-free compute
    # pipeline fills the CUs better than forcing 4+ workgroups to co-reside.
    for sk in _SPLIT_K:
        for tm, tn, tk, mw, nw, nb in _COMPUTE_PROFILES:
            for cm, cn in _COMPUTE_UNCLUSTERED:
                for ps in (1, *_PERSISTENT_N_TILES) if sk == 1 else (1,):
                    kl[idx] = WmmaKernelInstance(
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
                    )
                    idx += 1
    # A/B TDM early merge is a strategy candidate, not the default: it helps
    # selected wide-N occupancy regimes and can regress narrower grids. Append
    # split_k=1, non-persistent variants so existing numeric IDs remain stable
    # and per-shape tuning decides whether to use it.
    for early_a, early_b in ((True, False), (False, True), (True, True)):
        for tm, tn, tk, mw, nw, nb in _COMPUTE_PROFILES:
            for cm, cn in (*_COMPUTE_CLUSTERS, *_COMPUTE_UNCLUSTERED):
                kl[idx] = WmmaKernelInstance(
                    tile_m=tm,
                    tile_n=tn,
                    tile_k=tk,
                    num_buffers=nb,
                    split_k=1,
                    cluster_m=cm,
                    cluster_n=cn,
                    m_warp=mw,
                    n_warp=nw,
                    name_prefix=COMPUTE_NAME_PREFIX,
                    early_a=early_a,
                    early_b=early_b,
                )
                idx += 1
    # Cluster-local split-K keeps all splits of an output tile in one hardware
    # cluster and reduces there, avoiding the global FP32 workspace reduction
    # launch. It is a separate strategy because the larger cluster can reduce
    # occupancy; tuning chooses it only where the locality wins.
    for sk in _SPLIT_K:
        if sk == 1:
            continue
        for tm, tn, tk, mw, nw, nb in _COMPUTE_PROFILES:
            for cm, cn in (*_COMPUTE_CLUSTERS, *_COMPUTE_UNCLUSTERED):
                if tm % sk or cm * cn * sk > 16:
                    continue
                kl[idx] = WmmaKernelInstance(
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
                    fused_splitk=True,
                )
                idx += 1
    # Native 1x32 small-M split-K uses the generic row-wise K schedule.  Its
    # per-wave WMMA geometry is only 2x4 (half_m/half_n 1x2), so it
    # deliberately does not enter the large compute kernel's four-quadrant
    # seed schedule.
    for sk in _SPLIT_K:
        if sk not in (2, 8):
            continue
        for tm, tn, tk, mw, nw, nb in _SMALL_PROFILES:
            if tm % sk:
                continue
            kl[idx] = WmmaKernelInstance(
                tile_m=tm,
                tile_n=tn,
                tile_k=tk,
                num_buffers=nb,
                split_k=sk,
                cluster_m=1,
                cluster_n=1,
                m_warp=mw,
                n_warp=nw,
                name_prefix=SMALL_NAME_PREFIX,
                fused_splitk=True,
            )
            idx += 1
    # Retained by a graph-replay screen.  The generic four-owner TDM
    # pipeline needs at least two waves; do not add one-wave variants.
    kl[idx] = WmmaKernelInstance(
        tile_m=16,
        tile_n=32,
        tile_k=512,
        num_buffers=2,
        split_k=4,
        cluster_m=1,
        cluster_n=1,
        m_warp=1,
        n_warp=2,
        name_prefix=NAME_PREFIX,
        fused_splitk=True,
    )
    idx += 1
    # Keep these IDs after the established catalog. Narrow-N shapes have too
    # few N tiles for the 192/256-column compute profiles; a 256x128 tile keeps
    # their low-overhead pipeline while M-only clusters multicast B without
    # padding the N grid. Early-load merging remains a per-shape strategy.
    for tm, tn, tk, mw, nw, nb in _NARROW_N_COMPUTE_PROFILES:
        for cm, cn in _NARROW_N_COMPUTE_CLUSTERS:
            for early_a, early_b in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                kl[idx] = WmmaKernelInstance(
                    tile_m=tm,
                    tile_n=tn,
                    tile_k=tk,
                    num_buffers=nb,
                    split_k=1,
                    cluster_m=cm,
                    cluster_n=cn,
                    m_warp=mw,
                    n_warp=nw,
                    name_prefix=COMPUTE_NAME_PREFIX,
                    early_a=early_a,
                    early_b=early_b,
                )
                idx += 1
    # After every established ID: two-slot clustered compute kernels that also
    # hint the A rows and B groups of the next-but-one K rev into L2 (_pf1). A
    # two-slot pipeline refills one stage ahead, too short for weights that are
    # not cache-resident; four slots hide that by themselves, so they get no
    # _pf variant.
    for tm, tn, tk, mw, nw, nb in _COMPUTE_PROFILES:
        if nb != 2:
            continue
        for cm, cn in _COMPUTE_CLUSTERS:
            for early_a, early_b in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                kl[idx] = WmmaKernelInstance(
                    tile_m=tm,
                    tile_n=tn,
                    tile_k=tk,
                    num_buffers=nb,
                    split_k=1,
                    cluster_m=cm,
                    cluster_n=cn,
                    m_warp=mw,
                    n_warp=nw,
                    name_prefix=COMPUTE_NAME_PREFIX,
                    early_a=early_a,
                    early_b=early_b,
                    l2_prefetch=1,
                )
                idx += 1
    return kl


kernels_list: dict[int, WmmaKernelInstance] = _build_kernels_list()


def kernel_fits_shape(
    ki: WmmaKernelInstance, M: int, N: int, K: int, allow_ragged_n: bool = False
) -> bool:
    """Return True iff ``ki`` can be launched for runtime shape ``(M, N, K)``.
    ``allow_ragged_n`` (mxfp8_32): a compute kernel may end N on a partial tile
    where its launcher supports it (``ragged_n_ok``)."""

    # Row pairs need both rows, so an odd M has no valid pairing.
    if ki.a_preshuffle and M % 2 != 0:
        return False

    if is_small_kernel(ki):
        profile = (
            ki.tile_m,
            ki.tile_n,
            ki.tile_k,
            ki.m_warp,
            ki.n_warp,
            ki.num_buffers,
        )
        if profile not in _SMALL_PROFILES or M < ki.tile_m:
            return False
        if (
            not ki.fused_splitk
            or ki.split_k <= 1
            or ki.tile_m % ki.split_k
            or ki.cluster_m != 1
            or ki.cluster_n != 1
            or N % ki.tile_n
            or K % (ki.split_k * ki.tile_k)
        ):
            return False
        return (K // ki.split_k) // ki.tile_k >= ki.num_buffers

    if is_compute_kernel(ki):
        if kernel_instance_estimated_lds_bytes(ki) > LDS_BYTES:
            return False
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
        if profile not in (*_COMPUTE_PROFILES, *_NARROW_N_COMPUTE_PROFILES):
            return False
        if (ki.cluster_m, ki.cluster_n) not in (
            *_COMPUTE_CLUSTERS,
            *_COMPUTE_UNCLUSTERED,
            *_NARROW_N_COMPUTE_CLUSTERS,
        ):
            return False
        if ki.fused_splitk and (
            ki.split_k <= 1
            or ki.tile_m % ki.split_k
            or ki.tile_m & (ki.tile_m - 1)
            or (ki.m_warp * ki.n_warp * 32) % (ki.tile_n // 4)
            or ki.cluster_m * ki.cluster_n * ki.split_k > 16
        ):
            return False
        # Compute-bound kernels allocate a full tile_m tile; unlike sync WMMA they
        # must not launch when M < tile_m (cluster barrier hang on decode M).
        if M < ki.tile_m:
            return False
        if N % (ki.tile_n * ki.cluster_n) != 0 and not (
            allow_ragged_n
            and ragged_n_ok(
                N,
                ki.tile_n,
                ki.cluster_m,
                ki.cluster_n,
                ki.split_k,
                ki.persistent_n_tiles,
                True,
            )
        ):
            return False
        if not cluster_m_grid_ok(M, ki.tile_m, ki.cluster_m):
            return False
        if ki.persistent_n_tiles > 1:
            if ki.split_k != 1:
                return False
            n_tiles = N // ki.tile_n
            if n_tiles % ki.persistent_n_tiles:
                return False
            if (n_tiles // ki.persistent_n_tiles) % ki.cluster_n:
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
    if ki.fused_splitk and (
        ki.split_k <= 1
        or ki.tile_m % ki.split_k
        or ki.tile_m & (ki.tile_m - 1)
        or ki.cluster_m != 1
        or ki.cluster_n != 1
        or ki.m_warp * ki.n_warp < 2
    ):
        return False
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
    # Generic K=128 with two buffers needs no steady-loop iteration.
    min_k_tiles = 1 if ki.num_buffers == 2 and K == ki.tile_k == 128 else ki.num_buffers
    if (K // ki.split_k) // ki.tile_k < min_k_tiles:
        return False
    # M may be ragged; the kernel handles partial M tiles via OOB clamping,
    # so tile_m > M is legal (and needed for small-M decode shapes).

    m_blocks = _ceil_div(M, ki.tile_m)
    n_blocks = N // ki.tile_n
    if ki.tile_m < _LARGE_M_TILE_MIN and m_blocks > _MAX_M_BLOCKS:
        return False
    if ki.cluster_m > 1:
        if M < _CLUSTER_MIN_DIM or m_blocks < _CLUSTER_MIN_TILES:
            return False
        if not cluster_m_grid_ok(M, ki.tile_m, ki.cluster_m):
            return False
    if ki.cluster_n > 1:
        if N < _CLUSTER_MIN_DIM or n_blocks < _CLUSTER_MIN_TILES:
            return False
        if n_blocks % ki.cluster_n != 0:
            return False
    return True


def compute_grid_ctas(ki: WmmaKernelInstance, M: int, N: int) -> int:
    gx = _ceil_div(M, ki.tile_m)
    gy = N // ki.tile_n
    return gx * (gy // ki.persistent_n_tiles) * ki.split_k


def is_compute_kernel(kernel: WmmaKernelInstance) -> bool:
    """Return whether a tuning instance targets the compute-bound kernel."""
    return kernel.name_prefix == COMPUTE_NAME_PREFIX


def is_small_kernel(kernel: WmmaKernelInstance) -> bool:
    """Whether an instance uses the native low-resource small-M schedule."""
    return kernel.name_prefix == SMALL_NAME_PREFIX
