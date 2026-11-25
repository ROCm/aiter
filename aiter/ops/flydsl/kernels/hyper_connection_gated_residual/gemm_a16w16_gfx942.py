# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Internal a16w16 GEMM supporting the gated-residual K1 path on gfx942.

The implementation uses gfx942's 32-bit asynchronous LDS loads, ``16x16x16``
bf16 MFMA shape, and 64 KiB LDS limit. It remains package-internal because its
interface and scheduling constraints are specific to K1.
"""

from dataclasses import dataclass
from typing import Any

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from aiter.jit.utils.chip_info import get_gfx_runtime, get_lds_capacity_bytes
from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950_utils import (
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    get_wave_lds_offset,
    make_lds_layout,
    make_transposed_lds_layout,
    swizzled_col_idx,
    transposed_contiguous_idx,
)

GEMM_A16W16_DTYPE_FP32 = 1
GEMM_A16W16_DTYPE_BF16 = 2
GEMM_A16W16_DTYPE_FP16 = 3


@fx.struct
class GemmA16W16Gfx950Param:
    in_dtype_id: fx.Constexpr[int]
    out_dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    is_split_k: fx.Constexpr[bool]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    k_waves: fx.Constexpr[int]
    group_m: fx.Constexpr[int]
    use_half_tile_interleaved: fx.Constexpr[bool]
    a_is_transposed: fx.Constexpr[bool]
    b_is_transposed: fx.Constexpr[bool]
    has_bias: fx.Constexpr[bool]
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    # Values derived once and reused by the generated kernel.
    async_load_bytes: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
    out_data_bytes: fx.Constexpr[int]
    cshuffle_r2g_vec_size: fx.Constexpr[int]
    ldg_x_threads: fx.Constexpr[int]
    block_threads: fx.Constexpr[int]
    ldg_a_iters: fx.Constexpr[int]
    ldg_b_iters: fx.Constexpr[int]


@dataclass(slots=True, kw_only=True, eq=False)
class GemmABLoadContext:
    wave_offset: Any
    tid: Any
    ks_begin: Any
    param: GemmA16W16Gfx950Param
    async_g2s_copy_atom: Any
    a_s2r_copy_atom: Any
    b_s2r_copy_atom: Any
    a_tiled_copy_atom: Any
    b_tiled_copy_atom: Any


@dataclass(slots=True, kw_only=True, eq=False)
class AsyncLoadTile:
    lds_base: Any
    src_base: Any
    lds_layout: Any
    outer_tile_size: Any
    outer_bound: Any
    global_outer_offset: Any
    leading_stride: Any
    k_tile: Any


def make_gemm_a16w16_gfx950_param(
    in_dtype_id: int,
    out_dtype_id: int,
    block_m: int = 256,
    block_n: int = 256,
    block_k: int = 64,
    stages: int = 2,
    split_k: int = 1,
    m_waves: int = 2,
    n_waves: int = 4,
    k_waves: int = 1,
    group_m: int = 0,
    use_half_tile_interleaved: bool = False,
    a_is_transposed: bool = False,
    b_is_transposed: bool = True,
    has_bias: bool = False,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 32,
) -> GemmA16W16Gfx950Param:
    if in_dtype_id not in (GEMM_A16W16_DTYPE_BF16, GEMM_A16W16_DTYPE_FP16):
        raise ValueError(f"unsupported in_dtype_id={in_dtype_id}")
    # K1 consumes only the single-pass, 16-bit-output load schedule. The gfx950
    # split-K, fp32-output and half-tile-interleaved modes were not adapted.
    if out_dtype_id != in_dtype_id:
        raise ValueError(
            f"unsupported out_dtype_id={out_dtype_id} for in_dtype_id={in_dtype_id}"
        )
    if split_k != 1 or use_half_tile_interleaved:
        raise ValueError("gfx942 K1 helpers support only split_k=1 without HTI")
    if block_m <= 0 or block_n <= 0 or block_k <= 0 or stages <= 0 or split_k <= 0:
        raise ValueError(
            "block_m, block_n, block_k, stages, and split_k must be positive"
        )
    if (mma_m, mma_n) != (16, 16) or mma_k not in (16, 32):
        raise ValueError(
            "the layout kernel requires mma=16x16x32 (gfx950/CDNA4) or "
            "16x16x16 (gfx942/CDNA3)"
        )
    if stages < 2:
        raise ValueError("stages must be at least 2 for the staged LDS pipeline")
    if m_waves <= 0 or n_waves <= 0 or k_waves <= 0:
        raise ValueError("m_waves, n_waves, and k_waves must be positive")
    if m_waves * n_waves * k_waves > 16:
        raise ValueError("the workgroup cannot contain more than 16 waves")
    if group_m < 0:
        raise ValueError("group_m must be non-negative")
    in_dbytes = 2  # Shared C remains in the 16-bit input dtype.
    out_dbytes = 2
    block_threads = m_waves * n_waves * k_waves * GFX950_WAVE_SIZE
    cshuffle_r2g_vec_size = 16 // out_dbytes
    assert block_n % cshuffle_r2g_vec_size == 0
    smem_bytes = stages * (block_m + block_n) * block_k * in_dbytes
    smem_bytes = max(smem_bytes, k_waves * block_m * block_n * in_dbytes)
    arch = get_gfx_runtime()
    smem_capacity = get_lds_capacity_bytes(arch)
    if smem_bytes > smem_capacity:
        raise ValueError(
            "staged LDS buffers exceed the device shared-memory capacity: "
            f"stages={stages}, block_m={block_m}, block_n={block_n}, "
            f"block_k={block_k}, smem_bytes={smem_bytes}, "
            f"capacity={smem_capacity} for arch={arch}"
        )
    # gfx942 uses 32-bit asynchronous LDS loads; gfx950 uses 128-bit loads.
    async_bytes = 4 if arch == "gfx942" else GFX950_DMA_BYTES
    async_load_vec_size = async_bytes // in_dbytes
    ldg_x_threads = block_k // async_load_vec_size
    if ldg_x_threads * async_load_vec_size != block_k:
        raise ValueError(
            "block_k must be divisible by the async load vector size: "
            f"block_k={block_k}, async_load_vec_size={async_load_vec_size}, "
            f"covered_k={ldg_x_threads * async_load_vec_size}"
        )
    ldg_y_threads = block_threads // ldg_x_threads
    if ldg_y_threads * ldg_x_threads != block_threads:
        raise ValueError(
            "ldg thread layout must exactly cover the workgroup: "
            f"ldg_y_threads={ldg_y_threads}, ldg_x_threads={ldg_x_threads}, "
            f"block_threads={block_threads}"
        )
    ldg_a_iters = (block_m * block_k) // (block_threads * async_load_vec_size)
    ldg_b_iters = (block_n * block_k) // (block_threads * async_load_vec_size)
    if ldg_a_iters * block_threads * async_load_vec_size != block_m * block_k:
        raise ValueError(
            "A async load tile must be exactly covered by whole-thread vector loads: "
            f"block_m={block_m}, block_k={block_k}, "
            f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}, "
            f"ldg_a_iters={ldg_a_iters}, "
            f"covered={ldg_a_iters * block_threads * async_load_vec_size}, "
            f"required={block_m * block_k}"
        )
    if ldg_b_iters * block_threads * async_load_vec_size != block_n * block_k:
        raise ValueError(
            "B async load tile must be exactly covered by whole-thread vector loads: "
            f"block_n={block_n}, block_k={block_k}, "
            f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}, "
            f"ldg_b_iters={ldg_b_iters}, "
            f"covered={ldg_b_iters * block_threads * async_load_vec_size}, "
            f"required={block_n * block_k}"
        )
    assert (stages - 2) * (ldg_a_iters + ldg_b_iters) < 63
    mma_m_repeat = block_m // m_waves // mma_m
    mma_n_repeat = block_n // n_waves // mma_n
    mma_k_repeat = block_k // k_waves // mma_k
    if mma_m_repeat * m_waves * mma_m != block_m:
        raise ValueError(
            "block_m must be divisible by m_waves * mma_m: "
            f"block_m={block_m}, m_waves={m_waves}, mma_m={mma_m}, "
            f"mma_m_repeat={mma_m_repeat}, covered_m={mma_m_repeat * m_waves * mma_m}"
        )
    if mma_n_repeat * n_waves * mma_n != block_n:
        raise ValueError(
            "block_n must be divisible by n_waves * mma_n: "
            f"block_n={block_n}, n_waves={n_waves}, mma_n={mma_n}, "
            f"mma_n_repeat={mma_n_repeat}, covered_n={mma_n_repeat * n_waves * mma_n}"
        )
    if mma_k_repeat * k_waves * mma_k != block_k:
        raise ValueError(
            "block_k must be divisible by k_waves * mma_k: "
            f"block_k={block_k}, k_waves={k_waves}, mma_k={mma_k}, "
            f"mma_k_repeat={mma_k_repeat}, "
            f"covered_k={mma_k_repeat * k_waves * mma_k}"
        )
    return GemmA16W16Gfx950Param(
        in_dtype_id=in_dtype_id,
        out_dtype_id=out_dtype_id,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        is_split_k=split_k > 1,
        m_waves=m_waves,
        n_waves=n_waves,
        k_waves=k_waves,
        group_m=group_m,
        use_half_tile_interleaved=use_half_tile_interleaved,
        a_is_transposed=a_is_transposed,
        b_is_transposed=b_is_transposed,
        has_bias=has_bias,
        async_load_bytes=async_bytes,
        in_data_bytes=in_dbytes,
        out_data_bytes=out_dbytes,
        cshuffle_r2g_vec_size=cshuffle_r2g_vec_size,
        ldg_x_threads=ldg_x_threads,
        block_threads=block_threads,
        ldg_a_iters=ldg_a_iters,
        ldg_b_iters=ldg_b_iters,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
    )


def make_gemm_ab_lds_layouts(rows_a, rows_b, block_k, a_is_transposed, b_is_transposed):
    a_lds_layout = (
        make_transposed_lds_layout(rows_a, block_k)
        if const_expr(a_is_transposed)
        else make_lds_layout(rows_a, block_k)
    )
    b_lds_layout = (
        make_transposed_lds_layout(rows_b, block_k)
        if const_expr(not b_is_transposed)
        else make_lds_layout(rows_b, block_k)
    )
    return a_lds_layout, b_lds_layout


def make_gemm_ab_load_context(
    elem_dtype,
    load_tid,
    ks_begin,
    param: GemmA16W16Gfx950Param,
):
    uni_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    if const_expr(param.async_load_bytes >= 16):
        async_g2s_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128
        )
    else:
        async_g2s_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.BufferLoadAsyncLDS32b(), 32
        )

    if const_expr(param.a_is_transposed):
        a_s2r_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans(16, 64), elem_dtype
        )
        a_tiled_copy_atom = a_s2r_copy_atom
    else:
        a_s2r_copy_atom = uni_copy_atom
        a_tiled_copy_atom = buffer_copy_atom
    if const_expr(not param.b_is_transposed):
        b_s2r_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans(16, 64), elem_dtype
        )
        b_tiled_copy_atom = b_s2r_copy_atom
    else:
        b_s2r_copy_atom = uni_copy_atom
        b_tiled_copy_atom = buffer_copy_atom

    return GemmABLoadContext(
        wave_offset=get_wave_lds_offset(load_tid, param.async_load_bytes),
        tid=load_tid,
        ks_begin=ks_begin,
        param=param,
        async_g2s_copy_atom=async_g2s_copy_atom,
        a_s2r_copy_atom=a_s2r_copy_atom,
        b_s2r_copy_atom=b_s2r_copy_atom,
        a_tiled_copy_atom=a_tiled_copy_atom,
        b_tiled_copy_atom=b_tiled_copy_atom,
    )


def async_load_to_lds(
    tile: AsyncLoadTile,
    context: GemmABLoadContext,
    load_iters,
    is_k_major,
):
    param = context.param
    tid = context.tid
    block_threads = param.block_threads
    async_load_vec_size = param.async_load_bytes // param.in_data_bytes
    ldg_x_threads = param.ldg_x_threads
    block_k = param.block_k
    elem_bytes = tile.src_base.dtype.width // 8
    lds_ptr = tile.lds_base + fx.Int32(context.wave_offset) // elem_bytes
    g2s_copy_layout = fx.make_layout(async_load_vec_size, 1)
    for i in range_constexpr(load_iters):
        global_tid = block_threads * i + tid
        if const_expr(is_k_major):
            outer_x_threads = tile.outer_tile_size // async_load_vec_size
            outer_lds_idx = global_tid % outer_x_threads * async_load_vec_size
            k_local_idx = global_tid // outer_x_threads
            outer_local_idx = transposed_contiguous_idx(
                outer_lds_idx,
                k_local_idx,
                tile.lds_layout,
                tile.outer_tile_size,
            )
            global_k_idx = context.ks_begin + tile.k_tile * block_k + k_local_idx
        else:
            outer_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            global_k_idx = (
                context.ks_begin
                + tile.k_tile * block_k
                + swizzled_col_idx(
                    outer_local_idx,
                    k_local_idx,
                    tile.lds_layout,
                    block_k,
                )
            )
        global_outer_idx = tile.global_outer_offset + outer_local_idx
        safe_global_outer_idx = (global_outer_idx < tile.outer_bound).select(
            global_outer_idx, 0
        )
        if const_expr(is_k_major):
            global_offset = global_k_idx * tile.leading_stride + safe_global_outer_idx
        else:
            global_offset = safe_global_outer_idx * tile.leading_stride + global_k_idx
        src = fx.make_view(tile.src_base + global_offset, g2s_copy_layout)
        dst = fx.make_view(lds_ptr, g2s_copy_layout)
        fx.copy_atom_call(context.async_g2s_copy_atom, src, dst)
        if i < load_iters - 1:
            lds_ptr = lds_ptr + block_threads * async_load_vec_size
