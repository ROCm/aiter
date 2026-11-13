# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
from dataclasses import dataclass
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch

from aiter.utility.graph_alloc import capture_scoped_cache, persistent_alloc

from .gemm_a16w16_gfx950_utils import (
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    SPLIT_K_SEMAPHORE_MAX_LEN,
    BlockSwizzle,
    SplitKProtocol,
    get_wave_lds_offset,
    make_lds_layout,
    make_transposed_lds_layout,
    swizzled_col_idx,
    transposed_contiguous_idx,
    wait_vmcnt_and_barrier,
)
from .kernels_common import run_cached, set_slp_vectorize_once_used
from .tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    _run_compiled,
    ptr_arg,
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
    k_stagger: fx.Constexpr[int]
    b_non_temporal: fx.Constexpr[bool]
    use_half_tile_interleaved: fx.Constexpr[bool]
    a_is_transposed: fx.Constexpr[bool]
    b_is_transposed: fx.Constexpr[bool]
    has_bias: fx.Constexpr[bool]
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    streamk_blocks: fx.Constexpr[int]
    split_init_ring: fx.Constexpr[bool]
    unroll_hti: fx.Constexpr[bool]
    n_grouped_loads: fx.Constexpr[bool]
    # derived params
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
    non_temporal: bool = False


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
    k_stagger: int = 0,
    b_non_temporal: bool = False,
    n_grouped_loads: bool = False,
    unroll_hti: bool = False,
    streamk_blocks: int = 0,
    split_init_ring: bool = False,
) -> GemmA16W16Gfx950Param:
    if in_dtype_id not in (GEMM_A16W16_DTYPE_BF16, GEMM_A16W16_DTYPE_FP16):
        raise ValueError(f"unsupported in_dtype_id={in_dtype_id}")
    if out_dtype_id not in (in_dtype_id, GEMM_A16W16_DTYPE_FP32):
        raise ValueError(
            f"unsupported out_dtype_id={out_dtype_id} for in_dtype_id={in_dtype_id}"
        )
    if block_m <= 0 or block_n <= 0 or block_k <= 0 or stages <= 0 or split_k <= 0:
        raise ValueError(
            "block_m, block_n, block_k, stages, and split_k must be positive"
        )
    if any(
        not isinstance(v, int) or isinstance(v, bool) for v in (mma_m, mma_n, mma_k)
    ):
        raise ValueError("MFMA dimensions must be integers")
    if (mma_m, mma_n, mma_k) not in ((16, 16, 32), (32, 32, 16)):
        raise ValueError("supported MFMA shapes are 16x16x32 and 32x32x16")
    if (mma_m, mma_n, mma_k) == (32, 32, 16) and (
        in_dtype_id != GEMM_A16W16_DTYPE_BF16
        or use_half_tile_interleaved
        or a_is_transposed
        or not b_is_transposed
        or k_waves != 1
    ):
        raise ValueError(
            "MFMA32 currently supports BF16 NT full-tile with k_waves=1 only"
        )
    if (
        not isinstance(streamk_blocks, int)
        or isinstance(streamk_blocks, bool)
        or not 0 <= streamk_blocks <= 1024
    ):
        raise ValueError("streamk_blocks must be an integer in [0, 1024]")
    if streamk_blocks and (
        in_dtype_id != GEMM_A16W16_DTYPE_BF16
        or out_dtype_id != GEMM_A16W16_DTYPE_BF16
        or use_half_tile_interleaved
        or a_is_transposed
        or not b_is_transposed
        or has_bias
        or split_k != 1
        or k_waves != 1
        or group_m
        or k_stagger
        or b_non_temporal
    ):
        raise ValueError(
            "Stream-K requires unscaled BF16 NT full-tile, split_k=k_waves=1, group_m=0, no bias/stagger/cache hint"
        )
    if (streamk_blocks or mma_m == 32) and block_k % 64:
        raise ValueError(
            "MFMA32/Stream-K require block_k divisible by 64 for the current LDS swizzle"
        )
    if stages < 2:
        raise ValueError("stages must be at least 2 for the staged LDS pipeline")
    if m_waves <= 0 or n_waves <= 0 or k_waves <= 0:
        raise ValueError("m_waves, n_waves, and k_waves must be positive")
    if m_waves * n_waves * k_waves > 16:
        raise ValueError("the workgroup cannot contain more than 16 waves")
    if not isinstance(k_stagger, int) or isinstance(k_stagger, bool) or k_stagger < 0:
        raise ValueError("k_stagger must be a non-negative integer")
    if not isinstance(b_non_temporal, bool):
        raise TypeError("b_non_temporal must be a bool")
    if group_m < 0:
        raise ValueError("group_m must be non-negative")
    in_dbytes = 2
    # Shared C uses the output dtype: fp32 output is never rounded to 16 bits.
    out_dbytes = 4 if out_dtype_id == GEMM_A16W16_DTYPE_FP32 else 2
    block_threads = m_waves * n_waves * k_waves * GFX950_WAVE_SIZE
    max_cshuffle_r2g_vec_size = 16 // out_dbytes
    if use_half_tile_interleaved:
        if k_waves != 1:
            raise ValueError("half-tile interleaved does not support slice-K")
        half_block_m = block_m // 2
        half_block_n = block_n // 2
        assert stages == 2
        assert m_waves == 2 and n_waves >= 2
        assert half_block_m * 2 == block_m
        assert half_block_n * 2 == block_n
        mma_m_half_repeat = half_block_m // m_waves // mma_m
        mma_n_half_repeat = half_block_n // n_waves // mma_n
        assert mma_m_half_repeat * m_waves * mma_m == half_block_m
        assert mma_n_half_repeat * n_waves * mma_n == half_block_n
        stg_size_per_m_step = m_waves * mma_m * half_block_n
        assert stg_size_per_m_step % block_threads == 0
        stg_work_size_per_m_step = stg_size_per_m_step // block_threads
        cshuffle_r2g_vec_size = min(max_cshuffle_r2g_vec_size, stg_work_size_per_m_step)
        assert cshuffle_r2g_vec_size in (4, 8)
        assert stg_work_size_per_m_step % cshuffle_r2g_vec_size == 0
        assert half_block_n % cshuffle_r2g_vec_size == 0
    else:
        cshuffle_r2g_vec_size = (
            min(max_cshuffle_r2g_vec_size, 4)
            if split_k > 1
            else max_cshuffle_r2g_vec_size
        )
        assert block_n % cshuffle_r2g_vec_size == 0
    smem_bytes = stages * (block_m + block_n) * block_k * in_dbytes
    smem_bytes = max(smem_bytes, k_waves * block_m * block_n * out_dbytes)
    arch = get_rocm_arch()
    SMEM_CAPACITY_MAP = {
        "gfx942": 65536,
        "gfx950": 163840,
    }
    if arch != "gfx950" and ((mma_m, mma_n, mma_k) != (16, 16, 32) or streamk_blocks):
        raise ValueError("MFMA32 and Stream-K extensions require gfx950")
    smem_capacity = SMEM_CAPACITY_MAP[arch]
    if smem_bytes > smem_capacity:
        raise ValueError(
            "staged LDS buffers exceed the device shared-memory capacity: "
            f"stages={stages}, block_m={block_m}, block_n={block_n}, "
            f"block_k={block_k}, smem_bytes={smem_bytes}, "
            f"capacity={smem_capacity} for arch={arch}"
        )
    # async load check
    async_load_vec_size = GFX950_DMA_BYTES // in_dbytes
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
    if use_half_tile_interleaved:
        half_ldg_a_iters = ((block_m // 2) * block_k) // (
            block_threads * async_load_vec_size
        )
        half_ldg_b_iters = ((block_n // 2) * block_k) // (
            block_threads * async_load_vec_size
        )
        if (
            half_ldg_a_iters * block_threads * async_load_vec_size
            != (block_m // 2) * block_k
        ):
            raise ValueError(
                "Half-tile A async load tile must be exactly covered by whole-thread vector loads: "
                f"half_block_m={block_m // 2}, block_k={block_k}, "
                f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}, "
                f"half_ldg_a_iters={half_ldg_a_iters}"
            )
        if (
            half_ldg_b_iters * block_threads * async_load_vec_size
            != (block_n // 2) * block_k
        ):
            raise ValueError(
                "Half-tile B async load tile must be exactly covered by whole-thread vector loads: "
                f"half_block_n={block_n // 2}, block_k={block_k}, "
                f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}, "
                f"half_ldg_b_iters={half_ldg_b_iters}"
            )
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
        split_init_ring=split_init_ring,
        streamk_blocks=streamk_blocks,
        n_grouped_loads=n_grouped_loads,
        unroll_hti=unroll_hti,
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
        k_stagger=k_stagger,
        b_non_temporal=b_non_temporal,
        use_half_tile_interleaved=use_half_tile_interleaved,
        a_is_transposed=a_is_transposed,
        b_is_transposed=b_is_transposed,
        has_bias=has_bias,
        async_load_bytes=GFX950_DMA_BYTES,
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


def make_gemm_a16w16_gfx950_kernel_name(param: GemmA16W16Gfx950Param):
    dtype_str = "fp16" if param.in_dtype_id == GEMM_A16W16_DTYPE_FP16 else "bf16"
    # "_fp32c" (fp32 C shuffle) changes FlyDSL's disk-cache key, which ignores
    # @flyc.kernel bodies, so old bf16-rounding fp32 binaries are never reused.
    out_suffix = "_fp32c" if param.out_dtype_id == GEMM_A16W16_DTYPE_FP32 else ""
    name = f"hgemm_{dtype_str}{out_suffix}_t{param.block_m}x{param.block_n}x{param.block_k}x{param.stages}"
    name += "_kst4" if param.is_split_k else "_ks1"
    name += f"_w{param.m_waves}x{param.n_waves}x{param.k_waves}"
    name += f"_gm{param.group_m}"
    name += f"_stg{param.k_stagger}" if param.k_stagger else ""
    name += "_bnt" if param.b_non_temporal else ""
    name += f"_bias{int(param.has_bias)}"
    a_layout = "t" if param.a_is_transposed else "n"
    b_layout = "t" if param.b_is_transposed else "n"
    name += f"_l{a_layout}{b_layout}"
    name += "_phti" if param.use_half_tile_interleaved else "_pft"
    # Kernel bodies do not participate in FlyDSL's disk-cache key. Distinguish
    # the specialized HTI implementations from their original schedules.
    name += "_ng4sk_u9lg" if param.n_grouped_loads else ""
    name += "_hti5sk" if param.unroll_hti else ""
    name += "_initring4" if param.split_init_ring else ""
    if (param.mma_m, param.mma_n, param.mma_k) != (16, 16, 32):
        name += f"_mi{param.mma_m}x{param.mma_n}x{param.mma_k}"
    if param.streamk_blocks:
        name += f"_skg{param.streamk_blocks}_v1"
    return name


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
    async_g2s_copy_atom = fx.make_copy_atom(
        fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128
    )

    if const_expr(param.a_is_transposed):
        a_s2r_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans16_64b(), elem_dtype
        )
        a_tiled_copy_atom = a_s2r_copy_atom
    else:
        a_s2r_copy_atom = uni_copy_atom
        a_tiled_copy_atom = buffer_copy_atom
    if const_expr(not param.b_is_transposed):
        b_s2r_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans16_64b(), elem_dtype
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
        rocdl.sched_barrier(0)
        if const_expr(param.n_grouped_loads or param.unroll_hti):
            # The NT specialization uses a lane-varying row/K-local offset and
            # a wave-uniform K-tile offset. The copy atom scales soffset from
            # elements to bytes; keeping K in SGPRs avoids repeated VALU address
            # updates before each DMA. Async groups and cache policy are unchanged.
            k_offset = context.ks_begin + tile.k_tile * block_k
            vector_offset = (
                safe_global_outer_idx * tile.leading_stride
                + swizzled_col_idx(
                    outer_local_idx, k_local_idx, tile.lds_layout, block_k
                )
            )
            src = fx.make_view(tile.src_base + vector_offset, g2s_copy_layout)
            fx.copy(context.async_g2s_copy_atom, src, dst, soffset=fx.Int32(k_offset))
        elif const_expr(tile.non_temporal):
            # The async copy atom has no cache-policy operand. Keep the raw
            # load async too, so the existing asyncmark/wait_asyncmark groups
            # still track its completion. src_base is the buffer origin;
            # lds_ptr is wave-uniform and hardware adds the lane offset.
            rocdl.raw_ptr_buffer_load_async_lds(
                fx.rocdl.get_buffer_rsrc(tile.src_base),
                fx.to_llvm_ptr(lds_ptr),
                fx.Int32(GFX950_DMA_BYTES).ir_value(),
                fx.Int32(global_offset * elem_bytes).ir_value(),
                fx.Int32(0).ir_value(),
                fx.Int32(0).ir_value(),
                aux=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 2),
            )
        else:
            fx.copy_atom_call(context.async_g2s_copy_atom, src, dst)
        rocdl.sched_barrier(0)
        if i < load_iters - 1:
            lds_ptr = lds_ptr + block_threads * async_load_vec_size


@flyc.jit
def write_cshuffle_vec_to_global(
    out,
    out_buf,
    global_offset,
    c_vec,
    is_split_k,
    is_fp32_output,
):
    vec_size = c_vec.numel
    elem_dtype = c_vec.dtype
    if const_expr(is_fp32_output):
        c_vec_global = c_vec.to(fx.Float32)
        if const_expr(is_split_k):
            atomic_atom = fx.make_copy_atom(
                fx.rocdl.BufferAtomicAdd(fx.Float32), fx.Float32
            )
            scalar_layout = fx.make_layout(1, 1)
            scalar_frag = fx.make_rmem_tensor(scalar_layout, fx.Float32)
            for elem_idx in range_constexpr(vec_size):
                scalar_frag.store(
                    fx.Vector.from_elements(
                        [c_vec_global[elem_idx]],
                        fx.Float32,
                    )
                )
                scalar_dst = fx.make_view(
                    fx.get_iter(out_buf) + global_offset + elem_idx,
                    scalar_layout,
                )
                fx.copy_atom_call(atomic_atom, scalar_frag, scalar_dst)
        else:
            fx.ptr_store(c_vec_global, fx.get_iter(out) + global_offset)
    elif const_expr(is_split_k):
        atomic_atom = fx.make_copy_atom(
            fx.rocdl.BufferAtomicPkAdd(elem_dtype), elem_dtype
        )
        pair_layout = fx.make_layout(2, 1)
        pair_frag = fx.make_rmem_tensor(pair_layout, elem_dtype)
        for pair_idx in range_constexpr(vec_size // 2):
            pair = fx.Vector.from_elements(
                [
                    c_vec[pair_idx * 2],
                    c_vec[pair_idx * 2 + 1],
                ],
                elem_dtype,
            )
            pair_frag.store(pair)
            pair_dst = fx.make_view(
                fx.get_iter(out_buf) + global_offset + pair_idx * 2,
                pair_layout,
            )
            fx.copy_atom_call(atomic_atom, pair_frag, pair_dst)
    else:
        fx.ptr_store(c_vec, fx.get_iter(out) + global_offset)


@flyc.jit
def _gemm_a16w16_ft_device(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    split_k: fx.Int32,
    working_k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmA16W16Gfx950Param,
    stream_segment: fx.Constexpr[bool],
    tile_id: fx.Int32,
    begin_k: fx.Int32,
    count_k: fx.Int32,
    first_worker: fx.Int32,
    last_worker: fx.Int32,
    slots: fx.Constexpr[int],
):
    is_split_k = param.is_split_k
    is_slice_k = param.k_waves > 1
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    k_waves = param.k_waves
    k_mma_iters_per_wave = block_k // (k_waves * param.mma_k)
    stages = param.stages
    block_threads = param.block_threads
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters
    cshuffle_r2g_vec_size = param.cshuffle_r2g_vec_size
    elem_dtype = (
        fx.Float16
        if const_expr(param.in_dtype_id == GEMM_A16W16_DTYPE_FP16)
        else fx.BFloat16
    )
    global_output_dtype = (
        fx.Float32
        if const_expr(param.out_dtype_id == GEMM_A16W16_DTYPE_FP32)
        else elem_dtype
    )
    if const_expr(is_split_k):
        splitk_protocol = SplitKProtocol(
            block_m,
            block_n,
            cshuffle_r2g_vec_size,
            param.out_data_bytes,
            block_threads,
            param.has_bias,
        )

    tid = fx.thread_idx.x
    threads_per_k_slice = param.m_waves * param.n_waves * GFX950_WAVE_SIZE
    tid_in_k_slice = tid % threads_per_k_slice
    k_wave_idx = tid // threads_per_k_slice
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    if const_expr(stream_segment):
        bid_m = tile_id // num_pid_n
        bid_n = tile_id % num_pid_n
        ks_idx = fx.Int32(0)
        ks_begin = begin_k * block_k
        k_tiles = count_k
    else:
        block_swizzle = BlockSwizzle(
            NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
        )
        bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
        ks_idx = fx.block_idx.y
        ks_begin = ks_idx * working_k
        ks_end = ks_begin + working_k
        ks_end = (ks_end < k).select(ks_end, k)
        k_tiles = (ks_end - ks_begin) // block_k
    if const_expr(param.k_stagger > 0):
        # Rotate only the global K loads, never the LDS ring index. The
        # actual partition length also handles a shorter final split.
        k_start = (fx.block_idx.x * param.k_stagger) % k_tiles
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[global_output_dtype, k_waves * block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
    if const_expr(param.has_bias):
        bias_buf = fx.rocdl.make_buffer_tensor(bias, max_size=True)
    else:
        bias_buf = None

    if const_expr(is_split_k):
        splitk_protocol.init(
            semaphore,
            signal,
            out,
            bias,
            tid,
            ks_idx,
            m,
            n,
            block_m_offset,
            block_n_offset,
            global_output_dtype,
            fx.block_idx.x,
            n,
        )

    ab_load_context = make_gemm_ab_load_context(
        elem_dtype,
        load_tid=tid,
        ks_begin=ks_begin,
        param=param,
    )
    a_s2r_copy_atom = ab_load_context.a_s2r_copy_atom
    b_s2r_copy_atom = ab_load_context.b_s2r_copy_atom
    a_tiled_copy_atom = ab_load_context.a_tiled_copy_atom
    b_tiled_copy_atom = ab_load_context.b_tiled_copy_atom

    gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]

    thr_mma = tiled_mma.thr_slice(tid_in_k_slice)
    thr_copy_A = fx.make_tiled_copy_A(a_tiled_copy_atom, tiled_mma).get_slice(
        tid_in_k_slice
    )
    thr_copy_B = fx.make_tiled_copy_B(b_tiled_copy_atom, tiled_mma).get_slice(
        tid_in_k_slice
    )

    a_lds_layout, b_lds_layout = make_gemm_ab_lds_layouts(
        block_m,
        block_n,
        block_k,
        param.a_is_transposed,
        param.b_is_transposed,
    )
    c_lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))

    sA = fx.make_view(smem_a, a_lds_layout)
    sB = fx.make_view(smem_b, b_lds_layout)
    sC_write = fx.make_view(smem_c + k_wave_idx * block_m * block_n, c_lds_layout)

    frag_A = thr_mma.make_fragment_A(sA)
    frag_B = thr_mma.make_fragment_B(sB)
    frag_C = thr_mma.make_fragment_C(gC)

    # `retile` does not allocate new data; it reinterprets the MMA register
    # fragments with the tiled-copy layout so LDS-to-register `fx.copy` can fill them.
    frag_A_retile = thr_copy_A.retile(frag_A)
    frag_B_retile = thr_copy_B.retile(frag_B)

    row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
    thr_mma_cRow = thr_mma.partition_C(row_coords)
    thr_mma_cCol = thr_mma.partition_C(col_coords)

    if const_expr(is_split_k):
        frag_C.fill(0.0)
    elif const_expr(param.has_bias):
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            col_idx = fx.get_scalar(thr_mma_cCol[i])
            global_n_idx = block_n_offset + col_idx
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            bias_val = bias_buf[safe_global_n_idx].to(fx.Float32)
            if const_expr(is_slice_k):
                is_first_k_slice = k_wave_idx == 0
                bias_val = is_first_k_slice.select(bias_val, fx.Float32(0.0))
            frag_C[i] = bias_val
    else:
        frag_C.fill(0.0)

    def async_load_a_to_lds(k_tile, stage):
        if const_expr(param.k_stagger > 0):
            shifted = k_tile + k_start
            k_tile = (shifted < k_tiles).select(shifted, shifted - k_tiles)
        async_load_to_lds(
            tile=AsyncLoadTile(
                lds_base=smem_a + stage * block_m * block_k,
                src_base=fx.get_iter(a_buf),
                lds_layout=a_lds_layout,
                outer_tile_size=block_m,
                outer_bound=m,
                global_outer_offset=block_m_offset,
                leading_stride=a_leading_stride,
                k_tile=k_tile,
            ),
            context=ab_load_context,
            load_iters=ldg_a_iters,
            is_k_major=param.a_is_transposed,
        )

    def async_load_b_to_lds(k_tile, stage):
        if const_expr(param.k_stagger > 0):
            shifted = k_tile + k_start
            k_tile = (shifted < k_tiles).select(shifted, shifted - k_tiles)
        async_load_to_lds(
            tile=AsyncLoadTile(
                lds_base=smem_b + stage * block_n * block_k,
                src_base=fx.get_iter(b_buf),
                non_temporal=param.b_non_temporal,
                lds_layout=b_lds_layout,
                outer_tile_size=block_n,
                outer_bound=n,
                global_outer_offset=block_n_offset,
                leading_stride=b_leading_stride,
                k_tile=k_tile,
            ),
            context=ab_load_context,
            load_iters=ldg_b_iters,
            is_k_major=not param.b_is_transposed,
        )

    def compute_stage(read_stage, k_tile):
        thr_sA_s2r = thr_copy_A.partition_S(
            fx.make_view(smem_a + read_stage * block_m * block_k, a_lds_layout)
        )
        thr_sB_s2r = thr_copy_B.partition_S(
            fx.make_view(smem_b + read_stage * block_n * block_k, b_lds_layout)
        )

        def compute_k_chunk(block_k_iter):
            frag_A_chunk = frag_A[None, None, block_k_iter]
            fx.copy(
                b_s2r_copy_atom,
                thr_sB_s2r[None, None, block_k_iter],
                frag_B_retile[None, None, block_k_iter],
            )
            fx.copy(
                a_s2r_copy_atom,
                thr_sA_s2r[None, None, block_k_iter],
                frag_A_retile[None, None, block_k_iter],
            )
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A_chunk,
                frag_B[None, None, block_k_iter],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

        for k_slice in range_constexpr(k_waves):
            if k_wave_idx == k_slice:
                for block_k_iter in range_constexpr(k_mma_iters_per_wave):
                    k_iter = k_slice * k_mma_iters_per_wave + block_k_iter
                    compute_k_chunk(k_iter)

    def run_pipeline():
        # Prime the staged LDS pipeline: preload the first `stages - 1` K tiles
        # before entering the main loop that overlaps async loads with compute.
        if const_expr(param.split_init_ring):
            # Retire output initialization before issuing A/B DMA. Otherwise
            # the owner's vmcnt(0) publication also drains its first prefetches.
            splitk_protocol.zero_c()
            expected_signal = splitk_protocol.take_ticket(split_k)
            splitk_protocol.publish(expected_signal)
        for stage in range_constexpr(stages - 1):
            async_load_b_to_lds(stage, stage)
            async_load_a_to_lds(stage, stage)
            rocdl.asyncmark()
        if const_expr(is_split_k and not param.split_init_ring):
            splitk_protocol.zero_c()
            expected_signal = splitk_protocol.take_ticket(split_k)
            splitk_protocol.publish(expected_signal)
        rocdl.sched_barrier(0)

        main_loop_end = k_tiles - (stages - 1)
        if const_expr(param.split_init_ring):
            # Static slots, same K order, async groups, waits and barriers.
            cycles_end = main_loop_end // stages * stages
            for cycle in range(0, cycles_end, stages):
                for phase in range_constexpr(stages):
                    rocdl.wait_asyncmark(stages - 2)
                    rocdl.s_barrier()
                    async_load_b_to_lds(
                        cycle + phase + (stages - 1), (phase + stages - 1) % stages
                    )
                    async_load_a_to_lds(
                        cycle + phase + (stages - 1), (phase + stages - 1) % stages
                    )
                    rocdl.asyncmark()
                    compute_stage(phase, cycle + phase)
                    rocdl.sched_barrier(0)

        else:
            cycles_end = 0
        for k_tile in range(cycles_end, main_loop_end, 1):
            current_stage = k_tile % stages
            write_stage = (current_stage + stages - 1) % stages
            rocdl.wait_asyncmark(stages - 2)
            rocdl.s_barrier()
            async_load_b_to_lds(k_tile + (stages - 1), write_stage)
            async_load_a_to_lds(k_tile + (stages - 1), write_stage)
            rocdl.asyncmark()
            compute_stage(current_stage, k_tile)
            rocdl.sched_barrier(0)

        current_stage = main_loop_end % stages
        for s in range_constexpr(0, stages - 1):
            rocdl.wait_asyncmark(stages - 2 - s)
            rocdl.s_barrier()
            compute_stage(current_stage, main_loop_end + s)
            current_stage = (current_stage + 1) % stages

        if const_expr(is_split_k):
            return expected_signal
        return fx.Int32(0)

    if const_expr(stream_segment):
        if k_tiles < stages - 1:
            # A Stream-K interval can end with fewer tiles than the prologue.
            # No speculative K load is permitted beyond that interval.
            for tile in range(k_tiles):
                async_load_b_to_lds(tile, 0)
                async_load_a_to_lds(tile, 0)
                rocdl.asyncmark()
                rocdl.wait_asyncmark(0)
                gpu.barrier()
                compute_stage(0, tile)
                gpu.barrier()
        else:
            run_pipeline()
    else:
        expected_signal = run_pipeline()

    if const_expr(stream_segment):
        # No spinning: each worker publishes once per intersected output tile.
        # Complete tiles bypass scratch, while the last publisher reduces all
        # FP32 partials in worker/K order and rounds to BF16 exactly once.
        participants = last_worker - first_worker + 1
        worker = fx.block_idx.x
        if participants == 1:
            for i in range_constexpr(fx.size(frag_C.shape).unpack()):
                row = fx.get_scalar(thr_mma_cRow[i]) + block_m_offset
                col = fx.get_scalar(thr_mma_cCol[i]) + block_n_offset
                if (row < m) and (col < n):
                    fx.ptr_store(
                        frag_C[i].to(fx.BFloat16), fx.get_iter(out) + row * n + col
                    )
        else:
            scratch = fx.get_iter(semaphore)
            tile_elems = block_m * block_n
            for i in range_constexpr(fx.size(frag_C.shape).unpack() // 4):
                row = fx.get_scalar(thr_mma_cRow[4 * i])
                col = fx.get_scalar(thr_mma_cCol[4 * i])
                off = (
                    (tile_id * slots + worker - first_worker) * tile_elems
                    + row // 4 * block_n * 4
                    + col * 4
                )
                vals = fx.Vector.from_elements(
                    [frag_C[4 * i + j] for j in range_constexpr(4)], fx.Float32
                )
                fx.ptr_store(vals, scratch + off)
            # Every lane releases its writes before the leader's publication.
            # Use standard agent-scope fences/RMW, not an architecture-specific
            # cache-policy shortcut. LDS/barriers transfer the ticket to peers.
            fx.memory_fence(
                ordering=fx.AtomicOrdering.Release, syncscope=fx.rocdl.SyncScope.Agent
            )
            gpu.barrier()
            ticket = fx.recast_iter(fx.Int32, smem_c)
            counter = fx.get_iter(signal) + tile_id
            if tid == 0:
                arrived = fx.atomic_add(
                    counter,
                    fx.Int32(1),
                    ordering=fx.AtomicOrdering.AcqRel,
                    syncscope=fx.rocdl.SyncScope.Agent,
                )
                fx.ptr_store(arrived, ticket)
            gpu.barrier()
            arrived = fx.ptr_load(ticket)
            if arrived == participants - 1:
                fx.memory_fence(
                    ordering=fx.AtomicOrdering.Acquire,
                    syncscope=fx.rocdl.SyncScope.Agent,
                )
                for i in range_constexpr(fx.size(frag_C.shape).unpack() // 4):
                    row = fx.get_scalar(thr_mma_cRow[4 * i])
                    col = fx.get_scalar(thr_mma_cCol[4 * i])
                    off = (
                        tile_id * slots * tile_elems + row // 4 * block_n * 4 + col * 4
                    )
                    acc = fx.ptr_load(
                        scratch + off, result_type=fx.Vector.make_type(4, fx.Float32)
                    )
                    for part in range(1, participants):
                        acc = acc + fx.ptr_load(
                            scratch + off + part * tile_elems,
                            result_type=fx.Vector.make_type(4, fx.Float32),
                        )
                    vals = acc.to(fx.BFloat16)
                    for j in range_constexpr(4):
                        gr = block_m_offset + row + j
                        gc = block_n_offset + col
                        if (gr < m) and (gc < n):
                            fx.ptr_store(vals[j], fx.get_iter(out) + gr * n + gc)
        # All lanes finish using the union before the next persistent tile.
        gpu.barrier()
    else:
        frag_C_out = fx.make_fragment_like(frag_C, global_output_dtype)
        frag_C_out.store(frag_C.load().to(global_output_dtype))

        gpu.barrier()
        for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
            row = fx.get_scalar(thr_mma_cRow[i])
            col = fx.get_scalar(thr_mma_cCol[i])
            sC_write[row, col] = frag_C_out[i]

        if const_expr(is_split_k):
            splitk_protocol.wait_until_initialized(expected_signal)
        else:
            gpu.barrier()

        cshuffle_r2g_x_threads = block_n // cshuffle_r2g_vec_size
        cshuffle_vectors = block_m * block_n // cshuffle_r2g_vec_size
        cshuffle_iters = (cshuffle_vectors + block_threads - 1) // block_threads
        lane_contig_atomics = is_split_k
        if const_expr(lane_contig_atomics):
            # Split-K: lanes walk consecutive output elements (one fp32, or one bf16/fp16
            # pair for the packed atomics), so each wave-level atomic covers whole 64 B
            # lines (16 lanes per line) and the memory-side read-modify-write runs once per
            # line per instruction. The vec-per-lane mapping below would touch 16 lines with
            # 4 lanes each and pay four passes per line.
            atom_elems = 1 if param.out_dtype_id == GEMM_A16W16_DTYPE_FP32 else 2
            wave_idx = tid // GFX950_WAVE_SIZE
            lane_idx = tid % GFX950_WAVE_SIZE
            tile_elems = block_m * block_n
            for i in range_constexpr(cshuffle_iters):
                chunk = (
                    block_threads * i + wave_idx * GFX950_WAVE_SIZE
                ) * cshuffle_r2g_vec_size
                for j in range_constexpr(cshuffle_r2g_vec_size // atom_elems):
                    elem_idx = chunk + (j * GFX950_WAVE_SIZE + lane_idx) * atom_elems
                    if elem_idx < tile_elems:
                        local_row = elem_idx // block_n
                        local_col = elem_idx % block_n
                        global_row = block_m_offset + local_row
                        global_col = block_n_offset + local_col
                        if (global_row < m) and (global_col < n):
                            c_val = fx.ptr_load(
                                smem_c + elem_idx,
                                result_type=fx.Vector.make_type(
                                    atom_elems, global_output_dtype
                                ),
                            )
                            for k_slice in range_constexpr(1, k_waves):
                                c_val = c_val + fx.ptr_load(
                                    smem_c + k_slice * tile_elems + elem_idx,
                                    result_type=fx.Vector.make_type(
                                        atom_elems, global_output_dtype
                                    ),
                                )
                            write_cshuffle_vec_to_global(
                                out,
                                out_buf,
                                global_row * n + global_col,
                                c_val,
                                is_split_k,
                                param.out_dtype_id == GEMM_A16W16_DTYPE_FP32,
                            )
        for i in range_constexpr(0 if lane_contig_atomics else cshuffle_iters):
            vector_idx = block_threads * i + tid
            if vector_idx < cshuffle_vectors:
                local_row = vector_idx // cshuffle_r2g_x_threads
                local_col = vector_idx % cshuffle_r2g_x_threads * cshuffle_r2g_vec_size
                global_row = block_m_offset + local_row
                global_col = block_n_offset + local_col
                if (global_row < m) and (global_col < n):
                    c_vec = fx.ptr_load(
                        smem_c + local_row * block_n + local_col,
                        result_type=fx.Vector.make_type(
                            cshuffle_r2g_vec_size, global_output_dtype
                        ),
                    )
                    for k_slice in range_constexpr(1, k_waves):
                        peer_c_vec = fx.ptr_load(
                            smem_c
                            + k_slice * block_m * block_n
                            + local_row * block_n
                            + local_col,
                            result_type=fx.Vector.make_type(
                                cshuffle_r2g_vec_size, global_output_dtype
                            ),
                        )
                        c_vec = c_vec + peer_c_vec
                    global_offset = global_row * n + global_col
                    write_cshuffle_vec_to_global(
                        out,
                        out_buf,
                        global_offset,
                        c_vec,
                        is_split_k,
                        param.out_dtype_id == GEMM_A16W16_DTYPE_FP32,
                    )


@flyc.kernel
def gemm_a16w16_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    split_k: fx.Int32,
    working_k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmA16W16Gfx950Param,
):
    _gemm_a16w16_ft_device(
        out,
        a,
        b,
        bias,
        semaphore,
        signal,
        m,
        n,
        k,
        split_k,
        working_k,
        a_leading_stride,
        b_leading_stride,
        tiled_mma,
        param,
        False,
        fx.Int32(0),
        fx.Int32(0),
        fx.Int32(0),
        fx.Int32(0),
        fx.Int32(0),
        1,
    )


@flyc.kernel
def gemm_a16w16_hti_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    split_k: fx.Int32,
    working_k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmA16W16Gfx950Param,
):
    is_split_k = param.is_split_k
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    half_block_m = block_m // 2
    half_block_n = block_n // 2
    stages = param.stages
    block_threads = param.block_threads
    n_waves = param.n_waves
    half_ldg_a_iters = param.ldg_a_iters // 2
    half_ldg_b_iters = param.ldg_b_iters // 2
    cshuffle_r2g_vec_size = param.cshuffle_r2g_vec_size
    elem_dtype = (
        fx.Float16
        if const_expr(param.in_dtype_id == GEMM_A16W16_DTYPE_FP16)
        else fx.BFloat16
    )
    global_output_dtype = (
        fx.Float32
        if const_expr(param.out_dtype_id == GEMM_A16W16_DTYPE_FP32)
        else elem_dtype
    )
    if const_expr(is_split_k):
        splitk_protocol = SplitKProtocol(
            block_m,
            block_n,
            cshuffle_r2g_vec_size,
            param.out_data_bytes,
            block_threads,
            param.has_bias,
        )

    tid = fx.thread_idx.x
    wid = tid // GFX950_WAVE_SIZE
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    )
    if const_expr(param.n_grouped_loads):
        # Reuse the existing XCD-aware bijection with transposed tile axes:
        # groups of four N tiles improve A reuse for this projection.
        bid_n, bid_m = block_swizzle.swizzle(num_pid_n, num_pid_m, fx.block_idx.x)
    else:
        bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    ks_idx = fx.block_idx.y
    ks_begin = ks_idx * working_k
    ks_end = ks_begin + working_k
    ks_end = (ks_end < k).select(ks_end, k)
    if const_expr(param.unroll_hti):
        # The validated specialization has K7168, block_k64 and no split-K.
        k_tiles = 112
    else:
        k_tiles = (ks_end - ks_begin) // block_k
    if const_expr(param.k_stagger > 0):
        # Rotate only the global K loads, never the LDS ring index. The
        # actual partition length also handles a shorter final split.
        k_start = (fx.block_idx.x * param.k_stagger) % k_tiles
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[global_output_dtype, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
    if const_expr(param.has_bias):
        bias_buf = fx.rocdl.make_buffer_tensor(bias, max_size=True)
    else:
        bias_buf = None

    if const_expr(is_split_k):
        splitk_protocol.init(
            semaphore,
            signal,
            out,
            bias,
            tid,
            ks_idx,
            m,
            n,
            block_m_offset,
            block_n_offset,
            global_output_dtype,
            fx.block_idx.x,
            n,
        )

    ab_load_context = make_gemm_ab_load_context(
        elem_dtype,
        load_tid=tid,
        ks_begin=ks_begin,
        param=param,
    )
    a_s2r_copy_atom = ab_load_context.a_s2r_copy_atom
    b_s2r_copy_atom = ab_load_context.b_s2r_copy_atom
    a_tiled_copy_atom = ab_load_context.a_tiled_copy_atom
    b_tiled_copy_atom = ab_load_context.b_tiled_copy_atom

    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(a_tiled_copy_atom, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(b_tiled_copy_atom, tiled_mma).get_slice(tid)

    a_lds_layout, b_lds_layout = make_gemm_ab_lds_layouts(
        half_block_m,
        half_block_n,
        block_k,
        param.a_is_transposed,
        param.b_is_transposed,
    )
    c_lds_layout = fx.make_layout((half_block_m, half_block_n), (half_block_n, 1))

    def half_a_base(stage, m_part):
        return smem_a + (stage * block_m + m_part * half_block_m) * block_k

    def half_b_base(stage, n_part):
        return smem_b + (stage * block_n + n_part * half_block_n) * block_k

    def async_load_a_to_lds(m_part, k_tile, stage):
        if const_expr(param.k_stagger > 0):
            shifted = k_tile + k_start
            k_tile = (shifted < k_tiles).select(shifted, shifted - k_tiles)
        async_load_to_lds(
            tile=AsyncLoadTile(
                lds_base=half_a_base(stage, m_part),
                src_base=fx.get_iter(a_buf),
                lds_layout=a_lds_layout,
                outer_tile_size=half_block_m,
                outer_bound=m,
                global_outer_offset=block_m_offset + m_part * half_block_m,
                leading_stride=a_leading_stride,
                k_tile=k_tile,
            ),
            context=ab_load_context,
            load_iters=half_ldg_a_iters,
            is_k_major=param.a_is_transposed,
        )

    def async_load_b_to_lds(n_part, k_tile, stage):
        if const_expr(param.k_stagger > 0):
            shifted = k_tile + k_start
            k_tile = (shifted < k_tiles).select(shifted, shifted - k_tiles)
        async_load_to_lds(
            tile=AsyncLoadTile(
                lds_base=half_b_base(stage, n_part),
                src_base=fx.get_iter(b_buf),
                non_temporal=param.b_non_temporal,
                lds_layout=b_lds_layout,
                outer_tile_size=half_block_n,
                outer_bound=n,
                global_outer_offset=block_n_offset + n_part * half_block_n,
                leading_stride=b_leading_stride,
                k_tile=k_tile,
            ),
            context=ab_load_context,
            load_iters=half_ldg_b_iters,
            is_k_major=not param.b_is_transposed,
        )

    def make_gC(m_part, n_part):
        return fx.flat_divide(out_buf, (half_block_m, half_block_n))[
            None, None, bid_m * 2 + m_part, bid_n * 2 + n_part
        ]

    row_coords = fx.make_view(0, fx.make_layout((half_block_m, half_block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((half_block_m, half_block_n), (0, 1)))
    thr_mma_cRow = thr_mma.partition_C(row_coords)
    thr_mma_cCol = thr_mma.partition_C(col_coords)

    def make_c_fragment(m_part, n_part):
        gC = make_gC(m_part, n_part)
        return thr_mma.make_fragment_C(gC)

    def load_a_fragment(m_part, read_stage):
        sA = fx.make_view(half_a_base(read_stage, m_part), a_lds_layout)
        frag_A = thr_mma.make_fragment_A(sA)
        frag_A_retile = thr_copy_A.retile(frag_A)
        thr_sA_s2r = thr_copy_A.partition_S(sA)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            fx.copy(
                a_s2r_copy_atom,
                thr_sA_s2r[None, None, block_k_iter],
                frag_A_retile[None, None, block_k_iter],
            )
        return frag_A

    def load_b_fragment(n_part, read_stage):
        sB = fx.make_view(half_b_base(read_stage, n_part), b_lds_layout)
        frag_B = thr_mma.make_fragment_B(sB)
        frag_B_retile = thr_copy_B.retile(frag_B)
        thr_sB_s2r = thr_copy_B.partition_S(sB)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            fx.copy(
                b_s2r_copy_atom,
                thr_sB_s2r[None, None, block_k_iter],
                frag_B_retile[None, None, block_k_iter],
            )
        return frag_B

    def consume(frag_C, frag_A, frag_B, emit_sched_barrier):
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A[None, None, block_k_iter],
                frag_B[None, None, block_k_iter],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)

    def half_c_base(m_part, n_part):
        tile_idx = m_part * 2 + n_part
        return smem_c + tile_idx * half_block_m * half_block_n

    def store_half_tile_to_lds(m_part, n_part, frag_C):
        sC = fx.make_view(half_c_base(m_part, n_part), c_lds_layout)
        frag_C_out = fx.make_fragment_like(frag_C, global_output_dtype)
        frag_C_out.store(frag_C.load().to(global_output_dtype))

        for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
            row = fx.get_scalar(thr_mma_cRow[i])
            col = fx.get_scalar(thr_mma_cCol[i])
            sC[row, col] = frag_C_out[i]

    def store_half_tile_to_global(m_part, n_part):
        sC_base = half_c_base(m_part, n_part)
        cshuffle_r2g_x_threads = half_block_n // cshuffle_r2g_vec_size
        cshuffle_vectors = half_block_m * half_block_n // cshuffle_r2g_vec_size
        cshuffle_iters = (cshuffle_vectors + block_threads - 1) // block_threads
        for i in range_constexpr(cshuffle_iters):
            vector_idx = block_threads * i + tid
            if vector_idx < cshuffle_vectors:
                local_row = vector_idx // cshuffle_r2g_x_threads
                local_col = vector_idx % cshuffle_r2g_x_threads * cshuffle_r2g_vec_size
                global_row = block_m_offset + m_part * half_block_m + local_row
                global_col = block_n_offset + n_part * half_block_n + local_col
                if (global_row < m) and (global_col < n):
                    c_vec = fx.ptr_load(
                        sC_base + local_row * half_block_n + local_col,
                        result_type=fx.Vector.make_type(
                            cshuffle_r2g_vec_size, global_output_dtype
                        ),
                    )
                    global_offset = global_row * n + global_col
                    write_cshuffle_vec_to_global(
                        out,
                        out_buf,
                        global_offset,
                        c_vec,
                        is_split_k,
                        param.out_dtype_id == GEMM_A16W16_DTYPE_FP32,
                    )

    c00 = make_c_fragment(0, 0)
    c01 = make_c_fragment(0, 1)
    c10 = make_c_fragment(1, 0)
    c11 = make_c_fragment(1, 1)

    if const_expr(is_split_k):
        c00.fill(0.0)
        c01.fill(0.0)
        c10.fill(0.0)
        c11.fill(0.0)
    elif const_expr(param.has_bias):
        for i in range_constexpr(fx.size(c00.shape).unpack()):
            col_idx = fx.get_scalar(thr_mma_cCol[i])
            global_n0_idx = block_n_offset + col_idx
            global_n1_idx = global_n0_idx + half_block_n
            safe_global_n0_idx = (global_n0_idx < n).select(global_n0_idx, 0)
            safe_global_n1_idx = (global_n1_idx < n).select(global_n1_idx, 0)
            bias0 = bias_buf[safe_global_n0_idx].to(fx.Float32)
            bias1 = bias_buf[safe_global_n1_idx].to(fx.Float32)
            c00[i] = bias0
            c01[i] = bias1
            c10[i] = bias0
            c11[i] = bias1
    else:
        c00.fill(0.0)
        c01.fill(0.0)
        c10.fill(0.0)
        c11.fill(0.0)

    async_load_b_to_lds(0, 0, 0)
    async_load_a_to_lds(0, 0, 0)
    async_load_b_to_lds(1, 0, 0)
    async_load_a_to_lds(1, 0, 0)
    if const_expr(is_split_k):
        splitk_protocol.zero_c()
        expected_signal = splitk_protocol.take_ticket(split_k)
        splitk_protocol.publish(expected_signal)
    rocdl.sched_barrier(0)
    if wid // n_waves == 1:
        rocdl.s_barrier()
        # One barrier ahead, this group's next barrier releases the first group to
        # read stage 0: retire its copies.
        rocdl.s_waitcnt(vmcnt=0)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    async_load_b_to_lds(0, 1, 1)
    async_load_a_to_lds(0, 1, 1)
    async_load_b_to_lds(1, 1, 1)
    rocdl.sched_barrier(0)
    wait_vmcnt_and_barrier(half_ldg_b_iters + half_ldg_a_iters)

    main_loop_end = k_tiles - 2
    # Exact down: 110 steady-state tiles / 10 = 11 iterations.
    # Exact up: 54 steady-state tiles / 18 = 3 iterations.
    # Both retain the same ordered MFMA/DMA/barrier sequence and two-tile
    # drain. These flags are derived from exact shape/policy, not user options.
    hti_periods = (
        9
        if const_expr(param.n_grouped_loads)
        else 5 if const_expr(param.unroll_hti) else 1
    )
    for cycle in range(0, main_loop_end, 2 * hti_periods):
        for period in range_constexpr(hti_periods):
            k_tile = cycle + period * 2
            next_k_tile = k_tile + 2
            # 0
            b0 = load_b_fragment(0, 0)
            a0 = load_a_fragment(0, 0)
            async_load_a_to_lds(1, k_tile + 1, 1)
            rocdl.s_barrier()
            consume(c00, a0, b0, True)
            rocdl.s_barrier()
            b1 = load_b_fragment(1, 0)
            async_load_b_to_lds(0, next_k_tile, 0)
            rocdl.s_barrier()
            consume(c01, a0, b1, True)
            rocdl.s_barrier()
            a1 = load_a_fragment(1, 0)
            async_load_a_to_lds(0, next_k_tile, 0)
            rocdl.s_barrier()
            consume(c10, a1, b0, True)
            rocdl.s_barrier()
            b0 = load_b_fragment(0, 1)
            async_load_b_to_lds(1, next_k_tile, 0)
            wait_vmcnt_and_barrier(2 * half_ldg_b_iters + half_ldg_a_iters)
            consume(c11, a1, b1, True)
            rocdl.s_barrier()
            # 1
            a0 = load_a_fragment(0, 1)
            async_load_a_to_lds(1, next_k_tile, 0)
            rocdl.s_barrier()
            consume(c00, a0, b0, True)
            rocdl.s_barrier()
            b1 = load_b_fragment(1, 1)
            async_load_b_to_lds(0, next_k_tile + 1, 1)
            rocdl.s_barrier()
            consume(c01, a0, b1, True)
            rocdl.s_barrier()
            a1 = load_a_fragment(1, 1)
            async_load_a_to_lds(0, next_k_tile + 1, 1)
            rocdl.s_barrier()
            consume(c10, a1, b0, True)
            rocdl.s_barrier()
            async_load_b_to_lds(1, next_k_tile + 1, 1)
            wait_vmcnt_and_barrier(half_ldg_b_iters + half_ldg_a_iters)
            consume(c11, a1, b1, True)
            rocdl.s_barrier()

    k_tile = main_loop_end
    # 0
    if const_expr(is_split_k):
        wait_vmcnt_and_barrier(0)
    b0 = load_b_fragment(0, 0)
    a0 = load_a_fragment(0, 0)
    async_load_a_to_lds(1, k_tile + 1, 1)
    rocdl.s_barrier()
    consume(c00, a0, b0, True)
    rocdl.s_barrier()
    b1 = load_b_fragment(1, 0)
    rocdl.s_barrier()
    consume(c01, a0, b1, True)
    rocdl.s_barrier()
    a1 = load_a_fragment(1, 0)
    rocdl.s_barrier()
    consume(c10, a1, b0, True)
    rocdl.s_barrier()
    b0 = load_b_fragment(0, 1)
    if wid // n_waves == 1:
        # As in the prologue: the next barrier releases the first group to read the
        # stage 1 A half 0. The newer B half 1 and A half 1 copies may stay in flight.
        rocdl.s_waitcnt(vmcnt=half_ldg_b_iters + half_ldg_a_iters)
    rocdl.s_barrier()
    consume(c11, a1, b1, True)
    wait_vmcnt_and_barrier(0)
    # 1
    a0 = load_a_fragment(0, 1)
    rocdl.s_barrier()
    consume(c00, a0, b0, True)
    rocdl.s_barrier()
    b1 = load_b_fragment(1, 1)
    rocdl.s_barrier()
    consume(c01, a0, b1, True)
    rocdl.s_barrier()
    a1 = load_a_fragment(1, 1)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    store_half_tile_to_lds(0, 0, c00)
    consume(c10, a1, b0, False)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    store_half_tile_to_lds(0, 1, c01)
    consume(c11, a1, b1, False)
    rocdl.sched_barrier(0)
    if const_expr(is_split_k):
        if wid // n_waves == 0:
            rocdl.s_barrier()
        rocdl.s_barrier()
        store_half_tile_to_lds(1, 0, c10)
        store_half_tile_to_lds(1, 1, c11)
        splitk_protocol.wait_until_initialized(expected_signal)
        store_half_tile_to_global(0, 0)
        store_half_tile_to_global(0, 1)
        store_half_tile_to_global(1, 0)
        store_half_tile_to_global(1, 1)
    else:

        def finish_c_lds():
            if const_expr(param.n_grouped_loads):
                # Exact large-up only. The drain has completed every A/B DMA
                # before the final K tile's MFMA. This C-only phase needs LDS
                # writes/reads to retire, not disjoint global output stores.
                # Retain the rendezvous and explicit LDS completion wait.
                rocdl.s_waitcnt(lgkmcnt=0)
                rocdl.s_barrier()
            else:
                wait_vmcnt_and_barrier(0)

        finish_c_lds()
        store_half_tile_to_global(0, 0)
        store_half_tile_to_global(0, 1)
        finish_c_lds()
        store_half_tile_to_lds(1, 0, c10)
        store_half_tile_to_lds(1, 1, c11)
        finish_c_lds()
        store_half_tile_to_global(1, 0)
        store_half_tile_to_global(1, 1)


@flyc.jit
def gemm_a16w16_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    split_k: fx.Int32,
    param: GemmA16W16Gfx950Param,
    stream: fx.Stream = fx.Stream(None),  # noqa: B008
):
    m = fx.Int32(fx.get_scalar(a.shape[0]))
    n = fx.Int32(fx.get_scalar(b.shape[1]))
    k = fx.Int32(fx.get_scalar(a.shape[1]))
    a_leading_stride = fx.Int32(
        fx.get_scalar(a.stride[1] if const_expr(param.a_is_transposed) else a.stride[0])
    )
    b_leading_stride = fx.Int32(
        fx.get_scalar(b.stride[1] if const_expr(param.b_is_transposed) else b.stride[0])
    )
    elem_dtype = (
        fx.Float16
        if const_expr(param.in_dtype_id == GEMM_A16W16_DTYPE_FP16)
        else fx.BFloat16
    )
    mma_atom = fx.make_mma_atom(
        fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype)
    )
    k_per_mfma_group = param.mma_k // 4
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout(
            (param.m_waves, param.n_waves, 1),
            (param.n_waves, 1, 0),
        ),
        fx.make_tile(
            None,
            None,
            fx.make_layout(
                (k_per_mfma_group, 4),
                (1, k_per_mfma_group),
            ),
        ),
    )
    split_alignment = GFX950_DMA_BYTES // param.in_data_bytes
    working_k = (k + split_k - 1) // split_k
    working_k = (working_k + split_alignment - 1) // split_alignment * split_alignment
    num_pid_m = (m + param.block_m - 1) // param.block_m
    num_pid_n = (n + param.block_n - 1) // param.block_n
    gemm_a16w16_kernel_impl = (
        gemm_a16w16_hti_gfx950_kernel
        if param.use_half_tile_interleaved
        else gemm_a16w16_gfx950_kernel
    )
    gemm_a16w16_kernel_impl._known_block_size = [param.block_threads, 1, 1]
    gemm_a16w16_kernel_impl._func.__name__ = make_gemm_a16w16_gfx950_kernel_name(param)
    gemm_a16w16_kernel_impl(
        out,
        a,
        b,
        bias,
        semaphore,
        signal,
        m,
        n,
        k,
        split_k,
        working_k,
        a_leading_stride,
        b_leading_stride,
        tiled_mma,
        param,
    ).launch(
        grid=(num_pid_m * num_pid_n, split_k, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


gemm_a16w16_gfx950.compile_hints = {
    "llvm_options": {
        "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
        "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    },
}
set_slp_vectorize_once_used(gemm_a16w16_gfx950, False)


@flyc.jit
def _split_init_ring_ptr(
    op: fx.Pointer,
    ap: fx.Pointer,
    bp: fx.Pointer,
    sp: fx.Pointer,
    fp: fx.Pointer,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    lda: fx.Int32,
    ldb: fx.Int32,
    split_k: fx.Int32,
    param: GemmA16W16Gfx950Param,
    stream: fx.Stream,
):
    out = fx.make_view(op, fx.make_layout((m, n), (n, 1)))
    a = fx.make_view(ap, fx.make_layout((m, k), (lda, 1)))
    b = fx.make_view(bp, fx.make_layout((k, n), (1, ldb)))
    sem = fx.make_view(sp, fx.make_layout(SPLIT_K_SEMAPHORE_MAX_LEN, 1))
    sig = fx.make_view(fp, fx.make_layout(SPLIT_K_SEMAPHORE_MAX_LEN, 1))
    mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16)),
        fx.make_layout((2, 2, 1), (2, 1, 0)),
        fx.make_tile(None, None, fx.make_layout((8, 4), (1, 8))),
    )
    working_k = (k + split_k - 1) // split_k
    working_k = (working_k + 7) // 8 * 8
    gemm_a16w16_gfx950_kernel._known_block_size = [256, 1, 1]
    gemm_a16w16_gfx950_kernel._func.__name__ = (
        make_gemm_a16w16_gfx950_kernel_name(param) + "_ptr"
    )
    gemm_a16w16_gfx950_kernel(
        out, a, b, a, sem, sig, m, n, k, split_k, working_k, lda, ldb, mma, param
    ).launch(
        grid=(((m + 63) // 64) * ((n + 63) // 64), split_k, 1),
        block=(256, 1, 1),
        stream=stream,
    )


_split_init_ring_ptr.compile_hints = gemm_a16w16_gfx950.compile_hints
set_slp_vectorize_once_used(_split_init_ring_ptr, False)


@flyc.kernel
def _streamk_gemm_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    partial: fx.Tensor,
    counters: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    lda: fx.Int32,
    ldb: fx.Int32,
    mma: fx.TiledMma,
    param: GemmA16W16Gfx950Param,
    slots: fx.Constexpr[int],
):
    nt = (n + param.block_n - 1) // param.block_n
    tiles = ((m + param.block_m - 1) // param.block_m) * nt
    kt = k // param.block_k
    total = tiles * kt
    workers = param.streamk_blocks
    wid = fx.block_idx.x
    begin = wid * total // workers
    end = (wid + 1) * total // workers
    cursor = begin
    while cursor < end:
        tile = cursor // kt
        stop = ((tile + 1) * kt < end).select((tile + 1) * kt, end)
        # Inverse of floor(wid*total/workers), including uneven partitions.
        first = ((tile * kt + 1) * workers - 1) // total
        last = ((tile + 1) * kt * workers - 1) // total
        _gemm_a16w16_ft_device(
            out,
            a,
            b,
            a,
            partial,
            counters,
            m,
            n,
            k,
            fx.Int32(1),
            k,
            lda,
            ldb,
            mma,
            param,
            True,
            tile,
            cursor - tile * kt,
            stop - cursor,
            first,
            last,
            slots,
        )
        cursor = stop


@flyc.jit
def _streamk_gemm(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    partial: fx.Tensor,
    counters: fx.Tensor,
    param: GemmA16W16Gfx950Param,
    slots: fx.Constexpr[int],
    stream: fx.Stream,
):
    m = fx.Int32(fx.get_scalar(a.shape[0]))
    k = fx.Int32(fx.get_scalar(a.shape[1]))
    n = fx.Int32(fx.get_scalar(b.shape[1]))
    lda = fx.Int32(fx.get_scalar(a.stride[0]))
    ldb = fx.Int32(fx.get_scalar(b.stride[1]))
    group_k = param.mma_k // 4
    mma = fx.make_tiled_mma(
        fx.make_mma_atom(
            fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, fx.BFloat16)
        ),
        fx.make_layout((param.m_waves, param.n_waves, 1), (param.n_waves, 1, 0)),
        fx.make_tile(None, None, fx.make_layout((group_k, 4), (1, group_k))),
    )
    _streamk_gemm_kernel._known_block_size = [param.block_threads, 1, 1]
    _streamk_gemm_kernel._func.__name__ = make_gemm_a16w16_gfx950_kernel_name(param)
    _streamk_gemm_kernel(
        out, a, b, partial, counters, m, n, k, lda, ldb, mma, param, slots
    ).launch(
        grid=(param.streamk_blocks, 1, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


_streamk_gemm.compile_hints = gemm_a16w16_gfx950.compile_hints
set_slp_vectorize_once_used(_streamk_gemm, False)


@functools.cache
def _gemm_a16w16_gfx950_param_for(items, split_init_shape=False):
    # Building the param struct coerces every field through FlyDSL's typing
    # layer, which costs more host time than the launch itself; it depends
    # only on the config and selected schedule, not on the exact M value.
    config = dict(items)
    if split_init_shape:
        config["split_init_ring"] = _use_split_init_ring(128, 3584, 7168, config)
    return make_gemm_a16w16_gfx950_param(**config)


def _is_latent_hti_policy(kwargs):
    return (
        kwargs.get("streamk_blocks", 0) == 0
        and (kwargs.get("mma_m", 16), kwargs.get("mma_n", 16), kwargs.get("mma_k", 32))
        == (16, 16, 32)
        and kwargs.get("in_dtype_id") == GEMM_A16W16_DTYPE_BF16
        and kwargs.get("out_dtype_id") == GEMM_A16W16_DTYPE_BF16
        and not kwargs.get("has_bias", False)
        and not kwargs.get("a_is_transposed", False)
        and kwargs.get("b_is_transposed", True)
        and kwargs.get("use_half_tile_interleaved", False)
        and not kwargs.get("b_non_temporal", False)
        and kwargs.get("k_stagger", 0) == 0
        and kwargs.get("group_m", 0) == 4
        and kwargs.get("split_k", 1) == 1
        and kwargs.get("block_m") == 256
        and kwargs.get("block_n") == 256
        and kwargs.get("block_k") == 64
        and kwargs.get("stages") == 2
        and kwargs.get("m_waves") == 2
        and kwargs.get("n_waves") == 4
        and kwargs.get("k_waves", 1) == 1
    )


def _use_n_grouped_loads(m, n, k, kwargs):
    # Only the measured full-M up projection; padded-M consumers keep old code.
    return (m, n, k) == (16384, 7168, 3584) and _is_latent_hti_policy(kwargs)


def _use_unrolled_hti(m, n, k, kwargs):
    # This exact down projection has 112 K tiles, including a two-tile drain.
    return (m, n, k) == (16384, 3584, 7168) and _is_latent_hti_policy(kwargs)


def _use_split_init_ring(m, n, k, kwargs):
    # Preserve all padded-M consumers and all other explicit configurations.
    return (
        (m, n, k) == (128, 3584, 7168)
        and kwargs.get("in_dtype_id") == GEMM_A16W16_DTYPE_BF16
        and kwargs.get("out_dtype_id") == GEMM_A16W16_DTYPE_BF16
        and not kwargs.get("has_bias", False)
        and not kwargs.get("a_is_transposed", False)
        and kwargs.get("b_is_transposed", True)
        and not kwargs.get("use_half_tile_interleaved", False)
        and not kwargs.get("b_non_temporal", False)
        and kwargs.get("k_stagger", 0) == 0
        and kwargs.get("streamk_blocks", 0) == 0
        and (kwargs.get("mma_m", 16), kwargs.get("mma_n", 16), kwargs.get("mma_k", 32))
        == (16, 16, 32)
        and (kwargs.get("block_m"), kwargs.get("block_n"), kwargs.get("block_k"))
        == (64, 64, 64)
        and (kwargs.get("stages"), kwargs.get("split_k")) == (4, 4)
        and (kwargs.get("m_waves"), kwargs.get("n_waves"), kwargs.get("k_waves", 1))
        == (2, 2, 1)
        and kwargs.get("group_m", 0) == 0
    )


def make_gemm_a16w16_param_and_validate(m, n, k, kwargs):
    # Python caches compare True == 1 and 16.0 == 16. Validate the new
    # fields before looking up a previously built constexpr parameter.
    if type(kwargs.get("streamk_blocks", 0)) is not int or any(
        type(kwargs.get(key, default)) is not int
        for key, default in zip(("mma_m", "mma_n", "mma_k"), (16, 16, 32))
    ):
        return None
    result = None
    # Include the shape-dependent schedule in the constexpr parameter/cache key.
    # Do not mutate caller policy dictionaries or expose an unchecked override.
    kwargs = {
        **kwargs,
        "n_grouped_loads": _use_n_grouped_loads(m, n, k, kwargs),
        "unroll_hti": _use_unrolled_hti(m, n, k, kwargs),
    }
    # Never accept the internal schedule flag from a caller. Only the
    # exact-shape cache entry can derive it from a validated configuration.
    kwargs.pop("split_init_ring", None)
    try:
        if (m, n, k) == (128, 3584, 7168):
            result = _gemm_a16w16_gfx950_param_for(
                tuple(sorted(kwargs.items())), split_init_shape=True
            )
        else:
            result = _gemm_a16w16_gfx950_param_for(tuple(sorted(kwargs.items())))
    except Exception:  # noqa: BLE001 - invalid policies are filtered here
        return None
    split_k = kwargs.get("split_k", 1)
    if result.streamk_blocks:
        tiles = ((m + result.block_m - 1) // result.block_m) * (
            (n + result.block_n - 1) // result.block_n
        )
        kt = k // result.block_k
        slots = (result.streamk_blocks + tiles - 1) // tiles + 1 if tiles else 0
        if (
            min(m, n, k) <= 0
            or k % result.block_k
            or result.streamk_blocks > tiles * kt
            or tiles * kt * result.streamk_blocks >= (1 << 31)
            or tiles * slots * result.block_m * result.block_n * 4 >= (1 << 31)
        ):
            return None
    try:
        if not result.streamk_blocks:
            assert_no_k_tail(k, kwargs)
    except AssertionError:
        return None
    cshuffle_r2g_vec_size = result.cshuffle_r2g_vec_size
    if n % cshuffle_r2g_vec_size != 0:
        return None
    async_load_vec_size = GFX950_DMA_BYTES // result.in_data_bytes
    if result.a_is_transposed and m % async_load_vec_size != 0:
        return None
    if not result.b_is_transposed and n % async_load_vec_size != 0:
        return None
    if result.b_is_transposed and k % async_load_vec_size != 0:
        return None
    num_pid_m = (m + result.block_m - 1) // result.block_m
    num_pid_n = (n + result.block_n - 1) // result.block_n
    if split_k > 1:
        c_elements_per_iteration = result.block_threads * cshuffle_r2g_vec_size
        if (
            num_pid_m * num_pid_n > SPLIT_K_SEMAPHORE_MAX_LEN
            or result.block_m * result.block_n % c_elements_per_iteration != 0
        ):
            return None
    return result


def assert_no_k_tail(k: int, kwargs: dict):
    split_k = kwargs["split_k"]
    block_k = kwargs["block_k"]
    stages = kwargs["stages"]
    use_half_tile_interleaved = kwargs["use_half_tile_interleaved"]
    async_load_vec_size = GFX950_DMA_BYTES // 2
    working_k = (k + split_k - 1) // split_k
    working_k = (
        (working_k + async_load_vec_size - 1)
        // async_load_vec_size
        * async_load_vec_size
    )
    last_working_k = k - (split_k - 1) * working_k
    assert working_k % block_k == 0, (
        "K-tail is unsupported: aligned split-K partition size "
        f"{working_k} is not divisible by block_k={block_k}"
    )
    assert last_working_k > 0 and last_working_k % block_k == 0, (
        "K-tail is unsupported: final split-K partition size "
        f"{last_working_k} must be positive and divisible by block_k={block_k}"
    )
    working_k_tiles = working_k // block_k
    last_working_k_tiles = last_working_k // block_k
    min_k_tiles = stages - 1
    assert working_k_tiles >= min_k_tiles, (
        f"split-K partitions require at least {min_k_tiles} K tiles, "
        f"got {working_k_tiles}"
    )
    assert last_working_k_tiles >= min_k_tiles, (
        f"the final split-K partition requires at least {min_k_tiles} K tiles, "
        f"got {last_working_k_tiles}"
    )
    if use_half_tile_interleaved:
        assert working_k_tiles >= 2 and working_k_tiles % 2 == 0, (
            "HTI requires at least two and an even number of K tiles "
            "per split-K partition, "
            f"got {working_k_tiles}"
        )
        assert last_working_k_tiles >= 2 and last_working_k_tiles % 2 == 0, (
            "HTI requires at least two and an even number of K tiles "
            "in the final split-K partition, "
            f"got {last_working_k_tiles}"
        )


@capture_scoped_cache
def get_split_k_buffers(stream, device, split_k):
    # Keyed by split_k: a semaphore word only ever advances by split_k per launch,
    # which is what lets the kernel derive the launch index from its ticket.
    with persistent_alloc(torch.device(device)):
        semaphore = torch.zeros(
            (SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int64, device=device
        )
        signal = torch.zeros(
            (SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=device
        )
    return semaphore, signal


def _dynamic_tensor_arg(tensor, leading_dim):
    return flyc.from_dlpack(tensor.detach()).mark_layout_dynamic(
        leading_dim=leading_dim
    )


def gemm_a16w16(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    user_kwargs: dict | None = None,
    stream: torch.cuda.Stream | None = None,
    layout: str = "nt",
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute C[M, N] = A[M, K] @ B[K, N].

    Each layout character controls only the corresponding tensor stride:
    N is row-major and T is column-major. Logical tensor shapes never change.
    Inputs that violate the selected layout or DMA alignment are rejected.
    Set ``user_kwargs["split_k"]`` above 1 to atomically reduce K partitions.
    Set ``user_kwargs["k_waves"]`` above 1 for full-tile workgroup-local slice-K.
    ``out_dtype`` may be the input dtype or ``torch.float32``.
    """
    if stream is None:
        stream = torch.cuda.current_stream()
    layout = layout.lower()
    if layout not in ("nn", "nt", "tn", "tt"):
        raise ValueError(
            f"unsupported GEMM layout: {layout!r}; expected 'nn', 'nt', 'tn', or 'tt'"
        )
    a_is_transposed = layout[0] == "t"
    b_is_transposed = layout[1] == "t"
    device = a.device
    assert a.device == b.device
    assert a.ndim == 2 and b.ndim == 2
    m, k = a.shape
    assert b.shape[0] == k
    n = b.shape[1]
    assert a.dtype == b.dtype
    assert a.dtype in (torch.float16, torch.bfloat16)
    if a_is_transposed:
        a_vec_size = GFX950_DMA_BYTES // a.element_size()
        if (
            a.stride(0) != 1
            or a.data_ptr() % GFX950_DMA_BYTES != 0
            or a.stride(1) * a.element_size() % GFX950_DMA_BYTES != 0
            or m % a_vec_size != 0
        ):
            raise ValueError(
                "A does not satisfy the GFX950 DMA requirements for a "
                "column-major input: expected stride(0) == 1, a "
                f"{GFX950_DMA_BYTES}-byte-aligned data pointer and leading "
                f"stride, and M divisible by {a_vec_size}; got "
                f"shape={tuple(a.shape)} and stride={a.stride()}"
            )
    else:
        if (
            a.stride(1) != 1
            or a.data_ptr() % GFX950_DMA_BYTES != 0
            or a.stride(0) * a.element_size() % GFX950_DMA_BYTES != 0
        ):
            raise ValueError(
                "A does not satisfy the GFX950 DMA requirements for a "
                "row-major input: expected stride(1) == 1 and a "
                f"{GFX950_DMA_BYTES}-byte-aligned data pointer and leading "
                f"stride; got shape={tuple(a.shape)} and stride={a.stride()}"
            )
    if b_is_transposed:
        if (
            b.stride(0) != 1
            or b.data_ptr() % GFX950_DMA_BYTES != 0
            or b.stride(1) * b.element_size() % GFX950_DMA_BYTES != 0
        ):
            raise ValueError(
                "B does not satisfy the GFX950 DMA requirements for a "
                "column-major input: expected stride(0) == 1 and a "
                f"{GFX950_DMA_BYTES}-byte-aligned data pointer and leading "
                f"stride; got shape={tuple(b.shape)} and stride={b.stride()}"
            )
    else:
        if (
            b.stride(1) != 1
            or b.data_ptr() % GFX950_DMA_BYTES != 0
            or b.stride(0) * b.element_size() % GFX950_DMA_BYTES != 0
        ):
            raise ValueError(
                "B does not satisfy the GFX950 DMA requirements for a "
                "row-major input: expected stride(1) == 1 and a "
                f"{GFX950_DMA_BYTES}-byte-aligned data pointer and leading "
                f"stride; got shape={tuple(b.shape)} and stride={b.stride()}"
            )
    if out_dtype is None:
        out_dtype = a.dtype if out is None else out.dtype
    if out_dtype not in (a.dtype, torch.float32):
        raise ValueError(
            f"unsupported output dtype {out_dtype}; expected {a.dtype} or torch.float32"
        )
    if out is None:
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    else:
        assert out.dtype == out_dtype
        assert out.device == device
        assert out.is_contiguous()
    out = out.view(-1, n)
    assert out.shape[0] == m
    assert out.dtype == out_dtype

    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    kwargs = {
        "block_m": 256,
        "block_n": 256,
        "block_k": 64,
        "stages": 2,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 4,
        "k_waves": 1,
        "group_m": 0,
        "use_half_tile_interleaved": True,
    }

    kwargs.update(user_kwargs or {})
    kwargs["a_is_transposed"] = a_is_transposed
    kwargs["b_is_transposed"] = b_is_transposed
    kwargs["in_dtype_id"] = (
        GEMM_A16W16_DTYPE_FP16 if a.dtype is torch.float16 else GEMM_A16W16_DTYPE_BF16
    )
    kwargs["out_dtype_id"] = (
        GEMM_A16W16_DTYPE_FP32 if out.dtype is torch.float32 else kwargs["in_dtype_id"]
    )
    kwargs["has_bias"] = bias is not None
    split_k = kwargs["split_k"]
    if not kwargs.get("streamk_blocks", 0):
        assert_no_k_tail(k, kwargs)

    if bias is not None:
        assert bias.shape[0] == n
        assert bias.dtype == a.dtype

    param = make_gemm_a16w16_param_and_validate(m, n, k, kwargs)
    assert param is not None, "unsupported gemm_a16w16_gfx950 shape/config"
    if param.streamk_blocks:
        if device.type != "cuda" or stream.device != device:
            raise ValueError("Stream-K tensors and stream must share a CUDA device")
        a_span = ((m - 1) * a.stride(0) + k) * a.element_size()
        b_span = ((n - 1) * b.stride(1) + k) * b.element_size()
        if max(a_span, b_span, m * n * out.element_size()) >= (1 << 31):
            raise ValueError("Stream-K addressed spans must fit signed i32 bytes")
        tiles = ((m + param.block_m - 1) // param.block_m) * (
            (n + param.block_n - 1) // param.block_n
        )
        slots = (param.streamk_blocks + tiles - 1) // tiles + 1
        # Per-invocation storage on the launch stream. Captured invocations
        # own independent allocations. Counters are initialized on every call,
        # including graph replay; the initialization kernel is part of timing.
        with torch.cuda.stream(stream):
            partial = torch.empty(
                (tiles * slots * param.block_m * param.block_n,),
                device=device,
                dtype=torch.float32,
            )
            counters = torch.zeros((tiles,), device=device, dtype=torch.int32)
            args = (
                _dynamic_tensor_arg(out, 1),
                _dynamic_tensor_arg(a, 1),
                _dynamic_tensor_arg(b, 0),
                flyc.from_dlpack(partial),
                flyc.from_dlpack(counters),
                param,
                slots,
                stream,
            )
            _run_compiled(
                _streamk_gemm,
                *args,
                specialization_key=(
                    device.index,
                    a.dtype,
                    out.dtype,
                    tuple(a.shape),
                    tuple(a.stride()),
                    tuple(b.shape),
                    tuple(b.stride()),
                    param.__cache_signature__(),
                    slots,
                ),
            )
        return out
    if param.split_init_ring:
        # Exact no-bias BF16 NT policy only; ranks, shapes, layouts and DMA
        # alignment were checked above. Scalars/strides stay runtime inputs
        # so this launcher preserves the validated device schedule.
        if device.type != "cuda" or stream.device != device:
            raise ValueError("split GEMM tensors and stream must share a CUDA device")
        a_span = ((m - 1) * a.stride(0) + k) * a.element_size()
        b_span = ((n - 1) * b.stride(1) + k) * b.element_size()
        # Large spans keep the legacy tensor launcher rather than narrowing
        # its domain. The fast pointer ABI requires signed-i32 offsets.
        if max(a_span, b_span) <= (1 << 31) - 1:
            # A first eager use after cold capture may allocate new counters.
            # Initialize those on the launch stream, not the ambient stream.
            if stream == torch.cuda.current_stream(device):
                semaphore, signal = get_split_k_buffers(stream, device, split_k)
            else:
                with torch.cuda.stream(stream):
                    semaphore, signal = get_split_k_buffers(stream, device, split_k)
            _run_compiled(
                _split_init_ring_ptr,
                ptr_arg(out, fx.BFloat16),
                ptr_arg(a, fx.BFloat16),
                ptr_arg(b, fx.BFloat16),
                ptr_arg(semaphore, fx.Int64),
                ptr_arg(signal, fx.Int32),
                m,
                n,
                k,
                int(a.stride(0)),
                int(b.stride(1)),
                split_k,
                param,
                fx.Stream(stream),
                specialization_key=(
                    device.index,
                    a.dtype,
                    out.dtype,
                    param.__cache_signature__(),
                ),
            )
            return out
    semaphore, signal = get_split_k_buffers(stream, device, split_k)
    a_arg = _dynamic_tensor_arg(a, 0 if a_is_transposed else 1)
    b_arg = _dynamic_tensor_arg(b, 0 if b_is_transposed else 1)
    out_arg = _dynamic_tensor_arg(out, 1)
    bias_arg = a_arg if bias is None else _dynamic_tensor_arg(bias, 0)
    dispatch_args = (
        out_arg,
        a_arg,
        b_arg,
        bias_arg,
        semaphore,
        signal,
        split_k,
        param,
        stream,
    )
    run_cached(
        gemm_a16w16_gfx950,
        *dispatch_args,
        constexpr_param=param,
        compiler=flyc.compile,
        dispatch_args=dispatch_args,
    )
    return out
