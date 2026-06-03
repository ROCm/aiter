# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Candidate kernel catalogue for tuning the gfx1250 MXScale (mxfp8 / a8w4) GEMM.

Each :class:`MxScaleKernelInstance` is one tunable point in the *safe* search
space: only the performance knobs that the tuner is allowed to vary
(``tile_m/n/k``, ``m_warp/n_warp``, ``num_buffers``, ``split_k``) are enumerated
here. Every experimental codegen flag (schedule, scale-load path, wave
specialization, ...) is held at the conservative default baked into
``flydsl_mxscale_kernel_name`` so tuned kernels never depend on unstable knobs.

``.name`` round-trips through ``parse_flydsl_mxscale_kernel_name`` and is what
gets written into the a8w8 bpreshuffle tuned CSV for MXScale rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from aiter.ops.flydsl.mxscale_gemm import flydsl_mxscale_kernel_name
from aiter.ops.flydsl.mxscale_layout import (
    SCALE_BLOCK,
    validate_mxscale_kernel_shape,
)

# gfx1250 WMMA tile dims — must match the compile_mxscale_gemm contract.
WMMA_M = 16  # == mxscale_layout.WMMA_DIM
WMMA_N = 16  # == mxscale_layout.WMMA_DIM
WMMA_K = 128

# gfx1250 addressable LDS per workgroup. Used only as a coarse filter;
# the kernel performs its own exact LDS validation at compile time.
_GFX1250_LDS_BYTES = 160 * 1024


@dataclass
class MxScaleKernelInstance:
    # Only the tuner-searched perf knobs. Experimental codegen flags are pinned
    # internally in mxscale_gemm._FIXED_CODEGEN and are never enumerated here.
    tile_m: int
    tile_n: int
    tile_k: int
    m_warp: int
    n_warp: int
    num_buffers: int
    split_k: int = 1
    cluster_m: int = 1
    cluster_n: int = 1
    data_format: str = "fp8"
    out_dtype: str = "bf16"

    @property
    def name(self) -> str:
        return flydsl_mxscale_kernel_name(
            data_format=self.data_format,
            out_dtype=self.out_dtype,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            m_warp=self.m_warp,
            n_warp=self.n_warp,
            num_buffers=self.num_buffers,
            split_k=self.split_k,
            cluster_m=self.cluster_m,
            cluster_n=self.cluster_n,
        )


# Tuner search grid. Kept deliberately small but representative; the per-shape
# filter in the tuner driver drops candidates that do not divide the problem.
_TILE_M_OPTIONS = (16, 64, 128, 256)
_TILE_N_OPTIONS = (32, 64, 128, 256)
_TILE_K_OPTIONS = (128, 256)
_WARP_OPTIONS = ((2, 2),)
_NUM_BUFFER_OPTIONS = (2, 3, 4)
_SPLIT_K_OPTIONS = (1, 2, 4, 8, 16)
# cluster_m * cluster_n <= 16
_CLUSTER_OPTIONS = ((1, 1),)


def _is_valid_static(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    data_format: str,
) -> bool:
    if tile_m % (m_warp * WMMA_M) != 0:
        return False
    if tile_n % (n_warp * WMMA_N) != 0:
        return False
    wmma_n_eff = 32 if data_format == "fp4" else 16
    if (tile_n // n_warp) % wmma_n_eff != 0:
        return False
    if tile_k % WMMA_K != 0:
        return False
    return True


def _build_kernels_list(
    data_format: str = "fp8", out_dtype: str = "bf16"
) -> dict[int, MxScaleKernelInstance]:
    kernels: dict[int, MxScaleKernelInstance] = {}
    idx = 0
    for (
        tile_m,
        tile_n,
        tile_k,
        (m_warp, n_warp),
        num_buffers,
        split_k,
        (
            cluster_m,
            cluster_n,
        ),
    ) in product(
        _TILE_M_OPTIONS,
        _TILE_N_OPTIONS,
        _TILE_K_OPTIONS,
        _WARP_OPTIONS,
        _NUM_BUFFER_OPTIONS,
        _SPLIT_K_OPTIONS,
        _CLUSTER_OPTIONS,
    ):
        if not _is_valid_static(tile_m, tile_n, tile_k, m_warp, n_warp, data_format):
            continue
        kernels[idx] = MxScaleKernelInstance(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            m_warp=m_warp,
            n_warp=n_warp,
            num_buffers=num_buffers,
            split_k=split_k,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            data_format=data_format,
            out_dtype=out_dtype,
        )
        idx += 1
    return kernels


# Default catalogue (mxfp8, bf16 output). The tuner can rebuild for other
# (data_format, out_dtype) pairs via ``_build_kernels_list``.
kernels_list: dict[int, MxScaleKernelInstance] = _build_kernels_list()


def kernel_instance_estimated_lds_bytes(ki: MxScaleKernelInstance) -> int:
    """Coarse LDS estimate (A + B tiles x pipeline depth) for pre-filtering."""
    pack_a = 1
    pack_b = 1 if ki.data_format == "fp8" else 2
    a_tile = ki.tile_m * (ki.tile_k // pack_a)
    b_tile = ki.tile_n * (ki.tile_k // pack_b)
    scale_tile = (ki.tile_m + ki.tile_n) * (ki.tile_k // SCALE_BLOCK)
    return (a_tile + b_tile + scale_tile) * ki.num_buffers


def max_lds_bytes_for_tune() -> int:
    return _GFX1250_LDS_BYTES


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def kernel_fits_shape(ki: MxScaleKernelInstance, M: int, N: int, K: int) -> bool:
    """Whether this candidate can run the given (M, N, K) without violating
    the kernel's structural constraints (divisibility + pipeline depth +
    workgroup-cluster grid evenness)."""
    if K % SCALE_BLOCK != 0:
        return False
    # K is never padded: the kernel must divide the logical (N, K) directly.
    try:
        validate_mxscale_kernel_shape(
            N=N,
            K=K,
            tile_n=ki.tile_n,
            tile_k=ki.tile_k,
            num_buffers=ki.num_buffers,
            split_k=ki.split_k,
        )
    except ValueError:
        return False
    if kernel_instance_estimated_lds_bytes(ki) > max_lds_bytes_for_tune():
        return False
    # Workgroup clusters need the tile grid to divide evenly by the cluster dims.
    num_m_tiles = _ceil_div(M, ki.tile_m)
    num_n_tiles = N // ki.tile_n
    if num_m_tiles % ki.cluster_m != 0 or num_n_tiles % ki.cluster_n != 0:
        return False
    return True
