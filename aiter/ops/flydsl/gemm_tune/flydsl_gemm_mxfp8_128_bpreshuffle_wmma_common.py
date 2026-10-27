# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Full-style gfx1250 WMMA candidates for mxfp8_128 bpreshuffle GEMM."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from aiter.jit.utils.chip_info import get_gfx, get_lds_capacity_bytes
from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
    COMPUTE_WMMA_NAME_PREFIX as COMPUTE_NAME_PREFIX,
)
from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
    WMMA_NAME_PREFIX as NAME_PREFIX,
)
from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
    _fused_splitk_ok,
    cluster_m_grid_ok,
    compute_kernel_k_pair,
    n_is_bounded,
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
    (128, 128, 128, 2, 2, 4),
    (128, 256, 128, 2, 2, 4),
    (128, 256, 128, 2, 2, 3),
)
_COMPUTE_CLUSTERS = ((2, 2), (4, 2), (2, 4), (4, 4), (1, 4))
_COMPUTE_MIN_K_PER_SPLIT = 512
_PERSISTENT_N_TILES = (2, 4, 8)
# A padded N tile (partial last tile, or a spare one filling a cluster / persistent group)
# still runs the whole loop: candidates may pad at most this fraction of the real tiles.
_N_TAIL_MAX_PAD = 0.5

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
    fused_splitk: bool = False

    def name_for(self, a_preshuffle: bool = False) -> str:
        return (
            f"{self.name_prefix}_t{self.tile_m}x{self.tile_n}x{self.tile_k}_"
            f"mw{self.m_warp}_nw{self.n_warp}_nb{self.num_buffers}_sk{self.split_k}_"
            f"cm{self.cluster_m}_cn{self.cluster_n}"
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


def _stage_scale_bytes(
    ki: WmmaKernelInstance, scale_block: int, w_scale_32x32: bool
) -> tuple[int, int]:
    if scale_block == 128:
        k_blocks = _ceil_div(ki.tile_k, _BLOCK_K)
        a_scale_bytes = ki.tile_m * k_blocks * _SCALE_BYTES + _SCALE_GUARD_BYTES
        n_blocks = _ceil_div(ki.tile_n, _BLOCK_N) + 1
        b_scale_bytes = n_blocks * k_blocks * _SCALE_BYTES + _SCALE_GUARD_BYTES
    elif scale_block == 32:
        a_scale_bytes = ki.tile_m * _ceil_div(ki.tile_k, 32) + _SCALE_GUARD_BYTES
        b_k_blocks = _ceil_div(ki.tile_k, 32)
        b_n_blocks = _ceil_div(ki.tile_n, 32)
        b_scale_bytes = (
            b_n_blocks * (b_k_blocks if w_scale_32x32 else ki.tile_k)
            + _SCALE_GUARD_BYTES
        )
    else:
        raise ValueError(f"unsupported scale block: {scale_block}")
    return a_scale_bytes, b_scale_bytes


def kernel_instance_estimated_lds_bytes(
    ki: WmmaKernelInstance,
    *,
    scale_block: int = 128,
    w_scale_32x32: bool = False,
) -> int:
    """Conservative LDS upper bound for the selected scale layout and WMMA tile."""

    a_scale_bytes, b_scale_bytes = _stage_scale_bytes(ki, scale_block, w_scale_32x32)

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
    return kl


kernels_list: dict[int, WmmaKernelInstance] = _build_kernels_list()


def uses_cluster(ki: WmmaKernelInstance) -> bool:
    """Whether the launch groups workgroups into a cluster: cm * cn, times the K splits
    of a fused split-K, which live in the cluster too."""
    k_splits = ki.split_k if ki.fused_splitk else 1
    return ki.cluster_m * ki.cluster_n * k_splits > 1


def serves_padded_m(ki: WmmaKernelInstance, M: int) -> bool:
    """Whether a table row tuned at ``M`` may hold ``ki``. The row also serves the M
    below its own (down to about half), and a clustered compute kernel must not
    launch below its tile_m, so it needs M >= 2 * tile_m."""
    return not (uses_cluster(ki) and M < 2 * ki.tile_m)


def kernel_fits_shape(
    ki: WmmaKernelInstance,
    M: int,
    N: int,
    K: int,
    *,
    scale_block: int = _BLOCK_K,
    w_scale_32x32: bool = False,
    allow_n_tail: bool = False,
) -> bool:
    """Return True iff ``ki`` can be launched for runtime shape ``(M, N, K)``.

    ``scale_block`` sets the required K-scale alignment. A partial final K tile
    is supported for a single split; split-K chunks remain tile-aligned.
    ``allow_n_tail`` admits a compute kernel whose N is not a whole number of
    ``cluster_n * persistent_n_tiles`` tiles (it runs with ``bounded_n``, single split
    only); the untuned fallback and the batched GEMM leave it off.
    """

    if w_scale_32x32 and scale_block != 32:
        return False
    if w_scale_32x32 and ki.tile_n % 32:
        return False
    k_align = scale_block

    if ki.fused_splitk and not (
        is_compute_kernel(ki)
        and _fused_splitk_ok(ki.tile_m, ki.cluster_m, ki.cluster_n, ki.split_k, True)
    ):
        return False

    # Row pairs need both rows, so an odd M has no valid pairing.
    if ki.a_preshuffle and M % 2 != 0:
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
        if not _cluster_valid(ki.cluster_m, ki.cluster_n):
            return False
        # Compute-bound kernels allocate a full tile_m tile; unlike sync WMMA a
        # clustered one must not launch when M < tile_m (cluster barrier hang on
        # decode M). A 1x1 launch has no cluster barrier.
        if M < ki.tile_m and uses_cluster(ki):
            return False
        if ki.persistent_n_tiles > 1 and ki.split_k != 1:
            return False
        if n_is_bounded(N, ki.tile_n, ki.cluster_n, ki.persistent_n_tiles, True):
            n_align = 32 if w_scale_32x32 else _BLOCK_N
            if not allow_n_tail or ki.split_k != 1 or N % n_align:
                return False
            n_tiles = _ceil_div(N, ki.tile_n)
            if _padded_n_tiles(ki, N) - n_tiles > _N_TAIL_MAX_PAD * n_tiles:
                return False
        if not cluster_m_grid_ok(M, ki.tile_m, ki.cluster_m):
            return False
        if K % ki.split_k != 0:
            return False
        k_per_split = K // ki.split_k
        k_pair = compute_kernel_k_pair(ki.num_buffers, ki.tile_n)
        return (
            k_per_split >= _COMPUTE_MIN_K_PER_SPLIT
            and K % k_align == 0
            and (k_per_split % (ki.tile_k * k_pair) == 0 or ki.split_k == 1)
        )

    if ki.persistent_n_tiles > 1:
        return False
    n_align = 32 if w_scale_32x32 else _BLOCK_N
    if N % n_align != 0 or K % k_align != 0:
        return False
    if not _tile_valid(ki.tile_m, ki.tile_n, ki.tile_k, ki.m_warp, ki.n_warp):
        return False
    if not _cluster_valid(ki.cluster_m, ki.cluster_n):
        return False
    if (
        kernel_instance_estimated_lds_bytes(
            ki, scale_block=scale_block, w_scale_32x32=w_scale_32x32
        )
        > LDS_BYTES
    ):
        return False
    if N % ki.tile_n != 0:
        return False
    # The sync kernel takes a partial last K tile only on one split with the
    # mxfp8_32 32x32 weight scale (launch_gemm_a8w8's k_tail).
    if K % (ki.split_k * ki.tile_k) != 0 and (ki.split_k != 1 or not w_scale_32x32):
        return False
    if _ceil_div(K // ki.split_k, ki.tile_k) < ki.num_buffers:
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


def _padded_n_tiles(ki: WmmaKernelInstance, N: int) -> int:
    """N tiles of the launch: the real ones, filled up to whole cluster / persistent groups."""
    group = ki.cluster_n * ki.persistent_n_tiles
    return _ceil_div(_ceil_div(N, ki.tile_n), group) * group


def compute_grid_ctas(ki: WmmaKernelInstance, M: int, N: int) -> int:
    gx = _ceil_div(M, ki.tile_m)
    return gx * (_padded_n_tiles(ki, N) // ki.persistent_n_tiles) * ki.split_k


def is_compute_kernel(kernel: WmmaKernelInstance) -> bool:
    """Return whether a tuning instance targets the compute-bound kernel."""
    return kernel.name_prefix == COMPUTE_NAME_PREFIX


# Fallback ranking constants, fitted on the tuned gfx1250 rows (the picks land
# within ~5% geomean of the tuned optimum).
_MAC_PER_NS = 12600.0  # per-CU WMMA rate at 256x256, ~sqrt(area) for smaller tiles
_BYTES_PER_NS = 85.0  # per-CTA global->LDS bandwidth
_LOAD_LATENCY_NS = 900.0  # global->LDS round trip, shared by num_buffers - 1 stages
_COMPUTE_TILE_NS = 1100.0  # compute-bound kernel's per-tile barrier / TDM setup


def fallback_rank(ki: WmmaKernelInstance, M: int, N: int, K: int, cu_num: int) -> tuple:
    """Sort key picking a kernel for an untuned shape (split_k == 1 candidates).

    Cost = waves of CTAs x per-tile time, so small shapes fill the CUs first and
    large ones take the biggest tile. A tile iteration is bound by the WMMA rate,
    by global->LDS bytes (cluster multicast splits them) or, for the sync kernel,
    by the load latency. Ties prefer the larger cluster, then more buffers for
    the sync kernel and fewer for the compute-bound one.
    """
    tm, tn, tk = ki.tile_m, ki.tile_n, ki.tile_k
    compute = is_compute_kernel(ki)
    mac = tm * tn * tk / (_MAC_PER_NS * min(1.0, math.sqrt(tm * tn / 65536)))
    mem = (tm / ki.cluster_n + tn / ki.cluster_m) * tk / _BYTES_PER_NS
    latency = 0.0 if compute else _LOAD_LATENCY_NS / (ki.num_buffers - 1)
    tile_ns = _ceil_div(K, tk) * max(mac, mem, latency)
    if compute:
        tile_ns += _COMPUTE_TILE_NS
    waves = _ceil_div(compute_grid_ctas(ki, M, N), cu_num)
    return (
        waves * ki.persistent_n_tiles * tile_ns,
        -ki.cluster_m * ki.cluster_n,
        ki.num_buffers if compute else -ki.num_buffers,
    )


# ---------------------------------------------------------------------------
# The 32x32 weight-scale tuner's space
# ---------------------------------------------------------------------------
# kernels_list is shape-agnostic and is also what the batched GEMM enumerates, so
# the wider space the 32x32 tuner searches is built per shape instead of growing it.
_TUNE_TILE_M_OPTS = (16, 32, 64, 96, 128, 192, 256)
_TUNE_TILE_N_MAX = 1024
_TUNE_WARP_OPTS = ((1, 2), (2, 1), (1, 4), (2, 2), (4, 1))
# Every cluster with cm, cn <= 4, 1x1 included. The runtime fits cluster_m to the M a
# table row serves by stepping down to cm = 1, which is always a valid launch. A fused
# split-K puts its sk splits into the cluster too: cm * cn * sk <= 16.
_TUNE_COMPUTE_CLUSTERS = tuple((cm, cn) for cm in (1, 2, 4) for cn in (1, 2, 4))


def _tuning_candidates(N: int):
    """Unfiltered candidates for a weight with ``N`` rows: tile_n is every multiple
    of 32 up to _TUNE_TILE_N_MAX that divides N, so a shape like N = 1152 gets
    tile_n = 96 / 192 / 288 / 576 and N = 5120 gets 160 / 320 / 640."""
    tile_ns = [tn for tn in range(32, min(N, _TUNE_TILE_N_MAX) + 1, 32) if N % tn == 0]
    # The sync kernel has no fused epilogue: its split-K runs the reduce kernel.
    for sk, tm, tn, tk, (mw, nw), nb in itertools.product(
        _SPLIT_K,
        _TUNE_TILE_M_OPTS,
        tile_ns,
        _TILE_K_OPTS,
        _TUNE_WARP_OPTS,
        _NUM_BUFFERS_OPTS,
    ):
        if _tile_valid(tm, tn, tk, mw, nw):
            yield WmmaKernelInstance(tm, tn, tk, nb, sk, 1, 1, mw, nw)
    # The compute kernel's split-K is always the fused one.
    for sk, (tm, tn, tk, mw, nw, nb), (cm, cn) in itertools.product(
        _SPLIT_K, _COMPUTE_PROFILES, _TUNE_COMPUTE_CLUSTERS
    ):
        for ps in (1, *_PERSISTENT_N_TILES) if sk == 1 else (1,):
            yield WmmaKernelInstance(
                tm,
                tn,
                tk,
                nb,
                sk,
                cm,
                cn,
                mw,
                nw,
                name_prefix=COMPUTE_NAME_PREFIX,
                persistent_n_tiles=ps,
                fused_splitk=sk > 1,
            )


def tuning_kernels(
    M: int, N: int, K: int, *, scale_block: int, w_scale_32x32: bool
) -> list[WmmaKernelInstance]:
    """Every kernel of the tuner's space that can run ``(M, N, K)`` and may hold a
    table row tuned at ``M`` (serves_padded_m)."""
    return [
        ki
        for ki in _tuning_candidates(N)
        if kernel_fits_shape(
            ki,
            M,
            N,
            K,
            scale_block=scale_block,
            w_scale_32x32=w_scale_32x32,
            allow_n_tail=True,
        )
        and serves_padded_m(ki, M)
    ]
