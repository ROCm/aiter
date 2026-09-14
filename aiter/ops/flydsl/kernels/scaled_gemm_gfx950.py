# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch

from .gemm_a16w16_gfx950 import (
    _dynamic_tensor_arg,
    get_split_k_buffers,
    write_cshuffle_vec_to_global,
)
from .gemm_a16w16_gfx950_utils import (
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    SPLIT_K_SEMAPHORE_MAX_LEN,
    BlockSwizzle,
    SplitKProtocol,
    get_wave_lds_offset,
    wait_vmcnt_and_barrier,
)
from .kernels_common import run_cached
from .scaled_gemm_gfx950_utils import async_load_operand, make_fp8_lds_layout

SCALED_GEMM_DTYPE_FP32 = 1
SCALED_GEMM_DTYPE_BF16 = 2
SCALED_GEMM_DTYPE_FP8 = 4
SCALED_GEMM_DTYPE_MXFP8 = 5
MXFP8_BLOCK_SIZE = 32
# Two four-tile buffers occupy the same LDS as one eight-tile buffer.
MXFP8_HTI_SCALE_CHUNK_TILES = 4
MXFP8_HTI_SCALE_BUFFERS = 2


@fx.struct
class ScaledGemmGfx950Param:
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
    bpreshuffle: fx.Constexpr[bool]
    scale_block: fx.Constexpr[int]
    scale_a_transposed: fx.Constexpr[bool]
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
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
    param: ScaledGemmGfx950Param
    async_g2s_copy_atom: Any
    a_s2r_copy_atom: Any
    b_s2r_copy_atom: Any
    thr_copy_a: Any
    thr_copy_b: Any


@dataclass(slots=True, kw_only=True, eq=False)
class AsyncLoadOperand:
    context: GemmABLoadContext
    src_base: Any
    lds_layout: Any
    outer_tile_size: Any
    outer_bound: Any
    leading_stride: Any
    load_iters: Any
    is_k_major: Any
    is_preshuffled: bool = False


def mxfp8_scale_stage_bytes(rows, block_k, block_threads):
    # A complete 32-bit DMA per thread, padding the last workgroup-sized
    # chunk. Extra lanes duplicate valid rows rather than reading out of bounds.
    workgroup_bytes = block_threads * 4
    return (
        (rows * (block_k // MXFP8_BLOCK_SIZE) + workgroup_bytes - 1)
        // workgroup_bytes
        * workgroup_bytes
    )


def uses_lds_mxfp8_scales(param):
    # Four E8M0 bytes cover 128 K elements. K=64 tiles keep scalar loads:
    # their two-byte rows/tile offsets need not satisfy dword DMA alignment.
    return (
        param.in_dtype_id == SCALED_GEMM_DTYPE_MXFP8
        and param.scale_block == 32
        and param.block_k % 128 == 0
    )


def async_load_mxfp8_scales(
    scale, lds_base, tid, outer_offset, outer_bound, k_begin, rows, param
):
    """Stage four E8M0 bytes/lane with the same completion protocol as A/B."""
    scale_k = param.block_k // MXFP8_BLOCK_SIZE
    words_per_row = scale_k // 4
    stage_bytes = mxfp8_scale_stage_bytes(rows, param.block_k, param.block_threads)
    atom = fx.make_copy_atom(fx.rocdl.cdna4.BufferLoadAsyncLDS32b(), 32)
    layout = fx.make_layout(4, 1)
    wave_offset = fx.Int32(get_wave_lds_offset(tid, 4))
    stride = fx.Int32(fx.get_scalar(scale.stride[0]))
    for i in range_constexpr(stage_bytes // (param.block_threads * 4)):
        word = i * param.block_threads + tid
        row = outer_offset + (word // words_per_row) % rows
        safe_row = (row < outer_bound).select(row, 0)
        offset = (
            safe_row * stride + k_begin // MXFP8_BLOCK_SIZE + word % words_per_row * 4
        )
        src = fx.make_view(fx.get_iter(scale) + offset, layout)
        dst = fx.make_view(lds_base + wave_offset + i * param.block_threads * 4, layout)
        rocdl.sched_barrier(0)
        fx.copy_atom_call(atom, src, dst)
        rocdl.sched_barrier(0)


def async_load_mxfp8_scale_chunk(
    scale,
    lds_base,
    tid,
    outer_offset,
    outer_bound,
    k_begin,
    k_end,
    rows,
    chunk_k,
    param,
):
    """Coalesce scale traffic across K tiles; the caller fences the whole chunk.

    A short final chunk may extend past this split's K end. Its unused dwords
    duplicate a valid address, never issue an out-of-bounds global read.
    k_begin/k_end are 128-element aligned, so each valid dword is wholly valid.
    Unlike per-tile DMA, no per-instruction scheduler fences are needed here.
    """
    scale_k = chunk_k // MXFP8_BLOCK_SIZE
    words_per_row = scale_k // 4
    stage_bytes = mxfp8_scale_stage_bytes(rows, chunk_k, param.block_threads)
    atom = fx.make_copy_atom(fx.rocdl.cdna4.BufferLoadAsyncLDS32b(), 32)
    layout = fx.make_layout(4, 1)
    wave_offset = fx.Int32(get_wave_lds_offset(tid, 4))
    stride = fx.Int32(fx.get_scalar(scale.stride[0]))
    for i in range_constexpr(stage_bytes // (param.block_threads * 4)):
        word = i * param.block_threads + tid
        row = outer_offset + (word // words_per_row) % rows
        safe_row = (row < outer_bound).select(row, 0)
        sk = k_begin // MXFP8_BLOCK_SIZE + word % words_per_row * 4
        safe_sk = (sk < k_end // MXFP8_BLOCK_SIZE).select(sk, 0)
        src = fx.make_view(fx.get_iter(scale) + safe_row * stride + safe_sk, layout)
        dst = fx.make_view(lds_base + wave_offset + i * param.block_threads * 4, layout)
        fx.copy_atom_call(atom, src, dst)


def make_scaled_gemm_gfx950_param(
    in_dtype_id: int = SCALED_GEMM_DTYPE_FP8,
    out_dtype_id: int = SCALED_GEMM_DTYPE_BF16,
    block_m: int = 256,
    block_n: int = 256,
    block_k: int = 128,
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
    bpreshuffle: bool = False,
    scale_block: int = 32,
    scale_a_transposed: bool = False,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 128,
) -> ScaledGemmGfx950Param:
    if scale_block not in (32, 128):
        raise ValueError("scale_block must be 32 (MXFP8) or 128 (blockscale)")
    if scale_block == 32 and scale_a_transposed:
        raise ValueError("1x32 scales must be row-major")
    if bpreshuffle and (a_is_transposed or not b_is_transposed):
        raise ValueError("bpreshuffle requires the NT layout")
    if in_dtype_id not in (SCALED_GEMM_DTYPE_FP8, SCALED_GEMM_DTYPE_MXFP8):
        raise ValueError(f"unsupported in_dtype_id={in_dtype_id}")
    if out_dtype_id not in (SCALED_GEMM_DTYPE_BF16, SCALED_GEMM_DTYPE_FP32):
        raise ValueError(
            f"unsupported out_dtype_id={out_dtype_id} for in_dtype_id={in_dtype_id}"
        )
    if block_m <= 0 or block_n <= 0 or block_k <= 0 or stages <= 0 or split_k <= 0:
        raise ValueError(
            "block_m, block_n, block_k, stages, and split_k must be positive"
        )
    if (mma_m, mma_n, mma_k) not in ((16, 16, 128), (32, 32, 64)):
        raise ValueError("the gfx950 layout kernel requires mma=16x16x128 or 32x32x64")
    if stages < 2:
        raise ValueError("stages must be at least 2 for the staged LDS pipeline")
    if m_waves <= 0 or n_waves <= 0 or k_waves <= 0:
        raise ValueError("m_waves, n_waves, and k_waves must be positive")
    if m_waves * n_waves * k_waves > 16:
        raise ValueError("the workgroup cannot contain more than 16 waves")
    if group_m < 0:
        raise ValueError("group_m must be non-negative")
    in_dbytes = 1
    out_dbytes = 4 if out_dtype_id == SCALED_GEMM_DTYPE_FP32 else 2
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
    # Keep FP32 outputs and local slice-K reductions in FP32 until the
    # final global store; otherwise cancellation loses BF16 bits per slice.
    shuffle_bytes = 4 if out_dbytes == 4 or k_waves > 1 else 2
    smem_bytes = max(smem_bytes, k_waves * block_m * block_n * shuffle_bytes)
    if (
        in_dtype_id == SCALED_GEMM_DTYPE_MXFP8
        and scale_block == 32
        and block_k % 128 == 0
    ):
        parts = 2 if use_half_tile_interleaved else 1
        if use_half_tile_interleaved and block_k == 128:
            smem_bytes += (
                2
                * MXFP8_HTI_SCALE_BUFFERS
                * (
                    mxfp8_scale_stage_bytes(
                        block_m // 2,
                        block_k * MXFP8_HTI_SCALE_CHUNK_TILES,
                        block_threads,
                    )
                    + mxfp8_scale_stage_bytes(
                        block_n // 2,
                        block_k * MXFP8_HTI_SCALE_CHUNK_TILES,
                        block_threads,
                    )
                )
            )
        else:
            smem_bytes += (
                stages
                * parts
                * (
                    mxfp8_scale_stage_bytes(block_m // parts, block_k, block_threads)
                    + mxfp8_scale_stage_bytes(block_n // parts, block_k, block_threads)
                )
            )
    arch = get_rocm_arch()
    SMEM_CAPACITY_MAP = {
        "gfx942": 65536,
        "gfx950": 163840,
    }
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
    scale_load_iters = 0
    if (
        in_dtype_id == SCALED_GEMM_DTYPE_MXFP8
        and scale_block == 32
        and block_k % 128 == 0
    ):
        parts = 2 if use_half_tile_interleaved else 1
        scale_load_iters = (
            parts
            * (
                mxfp8_scale_stage_bytes(block_m // parts, block_k, block_threads)
                + mxfp8_scale_stage_bytes(block_n // parts, block_k, block_threads)
            )
            // (block_threads * 4)
        )
    assert (stages - 2) * (ldg_a_iters + ldg_b_iters + scale_load_iters) < 63
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
    return ScaledGemmGfx950Param(
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
        bpreshuffle=bpreshuffle,
        scale_block=scale_block,
        scale_a_transposed=scale_a_transposed,
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


def make_scaled_gemm_gfx950_kernel_name(param: ScaledGemmGfx950Param):
    dtype_str = "mxfp8" if param.in_dtype_id == SCALED_GEMM_DTYPE_MXFP8 else "fp8"
    out_suffix = "_fp32" if param.out_dtype_id == SCALED_GEMM_DTYPE_FP32 else ""
    name = f"hgemm_{dtype_str}{out_suffix}_t{param.block_m}x{param.block_n}x{param.block_k}x{param.stages}"
    name += "_ksd" if param.is_split_k else "_ks1"
    name += f"_w{param.m_waves}x{param.n_waves}x{param.k_waves}"
    name += f"_gm{param.group_m}"
    name += f"_bias{int(param.has_bias)}"
    a_layout = "t" if param.a_is_transposed else "n"
    b_layout = "t" if param.b_is_transposed else "n"
    name += f"_l{a_layout}{b_layout}"
    name += "_phti" if param.use_half_tile_interleaved else "_pft"
    return (
        name
        + f"_bp{int(param.bpreshuffle)}_sb{param.scale_block}_sat{int(param.scale_a_transposed)}"
    )


def make_gemm_ab_lds_layouts(rows_a, rows_b, block_k, a_is_transposed, b_is_transposed):
    return (
        make_fp8_lds_layout(rows_a, block_k, a_is_transposed),
        make_fp8_lds_layout(rows_b, block_k, not b_is_transposed),
    )


def make_gemm_ab_load_context(
    elem_dtype,
    tiled_mma,
    copy_tid,
    load_tid,
    ks_begin,
    param: ScaledGemmGfx950Param,
):
    uni_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    async_g2s_copy_atom = fx.make_copy_atom(
        fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128
    )

    if const_expr(param.a_is_transposed):
        a_s2r_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans8_64b(), elem_dtype
        )
        a_tiled_copy_atom = a_s2r_copy_atom
    else:
        a_s2r_copy_atom = uni_copy_atom
        a_tiled_copy_atom = buffer_copy_atom
    if const_expr(not param.b_is_transposed):
        b_s2r_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans8_64b(), elem_dtype
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
        thr_copy_a=fx.make_tiled_copy_A(a_tiled_copy_atom, tiled_mma).get_slice(
            copy_tid
        ),
        thr_copy_b=fx.make_tiled_copy_B(b_tiled_copy_atom, tiled_mma).get_slice(
            copy_tid
        ),
    )


def make_scaled_tiled_mma(param: ScaledGemmGfx950Param):
    mma_atom = fx.make_mma_atom(
        fx.rocdl.cdna4.MFMA_Scale(
            param.mma_m, param.mma_n, param.mma_k, fx.Float8E4M3FN
        )
    )
    if const_expr(param.mma_k == 128):
        k_perm = fx.make_layout((16, 2, 4), (1, 64, 16))
    elif const_expr(param.mma_k == 64):
        if const_expr(param.in_dtype_id == SCALED_GEMM_DTYPE_MXFP8):
            # f8f6f4 packs two 16-byte strips per lane. Preserve logical
            # 32-element scale groups (the PTPC path can permute K freely).
            k_perm = fx.make_layout((16, 2, 2), (1, 32, 16))
        else:
            k_perm = fx.make_layout((32, 2), (1, 32))
    else:
        assert (
            False
        ), f"unsupported MFMA_Scale MNK {(param.mma_m, param.mma_n, param.mma_k)}"
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout(
            (param.m_waves, param.n_waves, 1),
            (param.n_waves, 1, 0),
        ),
        fx.make_tile(None, None, k_perm),
    )
    return tiled_mma


def mxfp8_gemm(
    frag_C,
    frag_A,
    frag_B,
    scale_a,
    scale_b,
    tid,
    m_offset,
    n_offset,
    k_offset,
    m,
    n,
    param: ScaledGemmGfx950Param,
    scales_are_fragments=False,
):
    """Apply one MMA K step with unshuffled [outer, K // 32] E8M0 scales.

    For both supported MFMA shapes, lane % mma_m selects the row/column
    and lane // mma_m selects its 32-element K scale group. This scale
    mapping is independent of the FP8 operand register K permutation.
    opsel=0 consumes the low byte of each lane's i32 scale operand.
    HTI passes register scale fragments so each A/B scale can be reused
    after the corresponding LDS stage is overwritten by the next tile.
    """
    mma_atom = fx.make_mma_atom(
        fx.rocdl.cdna4.MFMA_Scale(
            param.mma_m, param.mma_n, param.mma_k, fx.Float8E4M3FN
        )
    )
    lane = tid % GFX950_WAVE_SIZE
    wave = tid // GFX950_WAVE_SIZE
    wave_m = wave // param.n_waves
    wave_n = wave % param.n_waves
    scale_lane = lane % param.mma_m
    scale_group = lane // param.mma_m
    scale_k = k_offset // MXFP8_BLOCK_SIZE + scale_group

    a_scales = []
    for mi in range_constexpr(fx.size(frag_A.shape[1]).unpack()):
        if const_expr(scales_are_fragments):
            a_scales.append(scale_a[mi])
        else:
            row = m_offset + (mi * param.m_waves + wave_m) * param.mma_m + scale_lane
            safe_row = (row < m).select(row, 0)
            a_scales.append(
                scale_a[safe_row, scale_k // (param.scale_block // 32)].to(fx.Int32)
            )
    b_scales = []
    for ni in range_constexpr(fx.size(frag_B.shape[1]).unpack()):
        if const_expr(scales_are_fragments):
            b_scales.append(scale_b[ni])
        else:
            col = n_offset + (ni * param.n_waves + wave_n) * param.mma_n + scale_lane
            safe_col = (col < n).select(col, 0)
            b_scales.append(
                scale_b[
                    safe_col // (128 if const_expr(param.scale_block == 128) else 1),
                    scale_k // (param.scale_block // 32),
                ].to(fx.Int32)
            )

    # Atom calls need rank-1 value vectors; coalesce only changes the view.
    for ni in range_constexpr(fx.size(frag_B.shape[1]).unpack()):
        for mi in range_constexpr(fx.size(frag_A.shape[1]).unpack()):
            fx.gemm(
                mma_atom,
                fx.coalesce(frag_C[None, mi, ni]),
                fx.coalesce(frag_A[None, mi]),
                fx.coalesce(frag_B[None, ni]),
                fx.coalesce(frag_C[None, mi, ni]),
                scale_a=a_scales[mi],
                scale_b=b_scales[ni],
            )


@flyc.kernel
def scaled_gemm_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    scale_a: fx.Tensor,
    scale_b: fx.Tensor,
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
    param: ScaledGemmGfx950Param,
):
    tiled_mma = make_scaled_tiled_mma(param)
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
    elem_dtype = fx.Float8E4M3FN
    global_output_dtype = (
        fx.Float32
        if const_expr(param.out_dtype_id == SCALED_GEMM_DTYPE_FP32)
        else fx.BFloat16
    )
    shuffle_dtype = (
        fx.Float32
        if const_expr(param.out_dtype_id == SCALED_GEMM_DTYPE_FP32 or param.k_waves > 1)
        else fx.BFloat16
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
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    )
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    ks_idx = fx.block_idx.y
    ks_begin = ks_idx * working_k
    ks_end = ks_begin + working_k
    ks_end = (ks_end < k).select(ks_end, k)
    k_tiles = (ks_end - ks_begin) // block_k
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[shuffle_dtype, k_waves * block_m * block_n, 16]

    allocator = fx.SharedAllocator()
    storage = allocator.allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
    scale_a_buf = fx.rocdl.make_buffer_tensor(scale_a, max_size=True)
    scale_b_buf = fx.rocdl.make_buffer_tensor(scale_b, max_size=True)
    use_lds_scale = uses_lds_mxfp8_scales(param)
    if const_expr(use_lds_scale):
        # Separate from the AB/C union: HTI's last MMA groups overlap C-shuffle.
        scale_k = block_k // MXFP8_BLOCK_SIZE
        scale_a_stage_bytes = mxfp8_scale_stage_bytes(block_m, block_k, block_threads)
        scale_b_stage_bytes = mxfp8_scale_stage_bytes(block_n, block_k, block_threads)
        smem_sa = (
            allocator.allocate(fx.Array[fx.Uint8, stages * scale_a_stage_bytes, 16])
            .peek()
            .ptr
        )
        smem_sb = (
            allocator.allocate(fx.Array[fx.Uint8, stages * scale_b_stage_bytes, 16])
            .peek()
            .ptr
        )
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
        tiled_mma,
        copy_tid=tid_in_k_slice,
        load_tid=tid,
        ks_begin=ks_begin,
        param=param,
    )
    a_s2r_copy_atom = ab_load_context.a_s2r_copy_atom
    b_s2r_copy_atom = ab_load_context.b_s2r_copy_atom
    thr_copy_A = ab_load_context.thr_copy_a
    thr_copy_B = ab_load_context.thr_copy_b

    gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]

    thr_mma = tiled_mma.thr_slice(tid_in_k_slice)

    a_lds_layout, b_lds_layout = make_gemm_ab_lds_layouts(
        block_m,
        block_n,
        block_k,
        param.a_is_transposed,
        param.b_is_transposed,
    )
    a_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(a_buf),
        lds_layout=a_lds_layout,
        outer_tile_size=block_m,
        outer_bound=m,
        leading_stride=a_leading_stride,
        load_iters=ldg_a_iters,
        is_k_major=param.a_is_transposed,
    )
    b_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(b_buf),
        lds_layout=b_lds_layout,
        outer_tile_size=block_n,
        outer_bound=n,
        leading_stride=b_leading_stride,
        load_iters=ldg_b_iters,
        is_k_major=not param.b_is_transposed,
        is_preshuffled=param.bpreshuffle,
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

    # A16W16 can seed frag_C with bias (C = A@B + bias). PTPC is
    # C = (A@B) * scale_a * scale_b + bias, so bias must not enter acc.
    # Split-K: zero_c() writes bias (or 0) into global C; partials atomic-add.
    # Non-split: add bias after the epilogue scale, same as A16W16's elif.
    frag_C.fill(0.0)
    if const_expr(is_split_k):
        splitk_protocol.zero_c()

    def async_load_a_to_lds(k_tile, stage):
        async_load_operand(
            a_load_operand,
            lds_base=smem_a + stage * block_m * block_k,
            global_outer_offset=block_m_offset,
            k_tile=k_tile,
        )
        if const_expr(use_lds_scale):
            async_load_mxfp8_scales(
                scale_a_buf,
                smem_sa + stage * scale_a_stage_bytes,
                tid,
                block_m_offset,
                m,
                ks_begin + k_tile * block_k,
                block_m,
                param,
            )

    def async_load_b_to_lds(k_tile, stage):
        async_load_operand(
            b_load_operand,
            lds_base=smem_b + stage * block_n * block_k,
            global_outer_offset=block_n_offset,
            k_tile=k_tile,
        )
        if const_expr(use_lds_scale):
            async_load_mxfp8_scales(
                scale_b_buf,
                smem_sb + stage * scale_b_stage_bytes,
                tid,
                block_n_offset,
                n,
                ks_begin + k_tile * block_k,
                block_n,
                param,
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
            if const_expr(use_lds_scale):
                mxfp8_gemm(
                    frag_C,
                    frag_A_chunk,
                    frag_B[None, None, block_k_iter],
                    fx.make_view(
                        smem_sa + read_stage * scale_a_stage_bytes,
                        fx.make_layout((block_m, scale_k), (scale_k, 1)),
                    ),
                    fx.make_view(
                        smem_sb + read_stage * scale_b_stage_bytes,
                        fx.make_layout((block_n, scale_k), (scale_k, 1)),
                    ),
                    tid_in_k_slice,
                    0,
                    0,
                    block_k_iter * param.mma_k,
                    block_m,
                    block_n,
                    param,
                )
            elif const_expr(param.in_dtype_id == SCALED_GEMM_DTYPE_MXFP8):
                mxfp8_gemm(
                    frag_C,
                    frag_A_chunk,
                    frag_B[None, None, block_k_iter],
                    scale_a_buf,
                    scale_b_buf,
                    tid_in_k_slice,
                    block_m_offset,
                    block_n_offset,
                    ks_begin + k_tile * block_k + block_k_iter * param.mma_k,
                    m,
                    n,
                    param,
                )
            else:
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

    for stage in range_constexpr(stages - 1):
        async_load_b_to_lds(stage, stage)
        async_load_a_to_lds(stage, stage)
        rocdl.asyncmark()
    rocdl.sched_barrier(0)

    main_loop_end = k_tiles - (stages - 1)
    for k_tile in range(0, main_loop_end, 1):
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

    frag_C_out = fx.make_fragment_like(frag_C, shuffle_dtype)
    for i in range_constexpr(fx.size(frag_C.shape).unpack()):
        row = fx.get_scalar(thr_mma_cRow[i])
        col = fx.get_scalar(thr_mma_cCol[i])
        global_row = block_m_offset + row
        global_col = block_n_offset + col
        safe_m = (global_row < m).select(global_row, 0)
        safe_n = (global_col < n).select(global_col, 0)
        acc = frag_C[i]
        if const_expr(param.in_dtype_id == SCALED_GEMM_DTYPE_FP8):
            acc = acc * scale_a_buf[safe_m] * scale_b_buf[safe_n]
        if const_expr(param.has_bias):  # noqa: SIM102 - staged compiler branch
            if const_expr(not is_split_k):
                bias_val = bias_buf[safe_n].to(fx.Float32)
                if const_expr(is_slice_k):
                    bias_val = (k_wave_idx == 0).select(bias_val, fx.Float32(0.0))
                acc = acc + bias_val
        frag_C_out[i] = acc.to(shuffle_dtype)

    gpu.barrier()
    for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
        row = fx.get_scalar(thr_mma_cRow[i])
        col = fx.get_scalar(thr_mma_cCol[i])
        sC_write[row, col] = frag_C_out[i]

    if const_expr(is_split_k):
        splitk_protocol.wait_until_initialized()
    else:
        gpu.barrier()

    cshuffle_r2g_x_threads = block_n // cshuffle_r2g_vec_size
    cshuffle_vectors = block_m * block_n // cshuffle_r2g_vec_size
    cshuffle_iters = (cshuffle_vectors + block_threads - 1) // block_threads
    for i in range_constexpr(cshuffle_iters):
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
                        cshuffle_r2g_vec_size, shuffle_dtype
                    ),
                )
                for k_slice in range_constexpr(1, k_waves):
                    peer_c_vec = fx.ptr_load(
                        smem_c
                        + k_slice * block_m * block_n
                        + local_row * block_n
                        + local_col,
                        result_type=fx.Vector.make_type(
                            cshuffle_r2g_vec_size, shuffle_dtype
                        ),
                    )
                    c_vec = c_vec + peer_c_vec
                write_cshuffle_vec_to_global(
                    out,
                    out_buf,
                    global_row * n + global_col,
                    c_vec.to(global_output_dtype),
                    is_split_k,
                    param.out_dtype_id == SCALED_GEMM_DTYPE_FP32,
                )
    if const_expr(is_split_k):
        splitk_protocol.finish_split(split_k)


@flyc.kernel
def scaled_gemm_hti_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    scale_a: fx.Tensor,
    scale_b: fx.Tensor,
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
    param: ScaledGemmGfx950Param,
):
    tiled_mma = make_scaled_tiled_mma(param)
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
    elem_dtype = fx.Float8E4M3FN
    global_output_dtype = (
        fx.Float32
        if const_expr(param.out_dtype_id == SCALED_GEMM_DTYPE_FP32)
        else fx.BFloat16
    )
    shuffle_dtype = (
        fx.Float32
        if const_expr(param.out_dtype_id == SCALED_GEMM_DTYPE_FP32 or param.k_waves > 1)
        else fx.BFloat16
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
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    ks_idx = fx.block_idx.y
    ks_begin = ks_idx * working_k
    ks_end = ks_begin + working_k
    ks_end = (ks_end < k).select(ks_end, k)
    k_tiles = (ks_end - ks_begin) // block_k
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[shuffle_dtype, block_m * block_n, 16]

    allocator = fx.SharedAllocator()
    storage = allocator.allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
    scale_a_buf = fx.rocdl.make_buffer_tensor(scale_a, max_size=True)
    scale_b_buf = fx.rocdl.make_buffer_tensor(scale_b, max_size=True)
    use_lds_scale = uses_lds_mxfp8_scales(param)
    use_scale_chunk = use_lds_scale and block_k == 128
    scale_chunk_tiles = (
        MXFP8_HTI_SCALE_CHUNK_TILES if const_expr(use_scale_chunk) else 1
    )
    if const_expr(use_lds_scale):
        # Separate from the AB/C union: HTI's last MMA groups overlap C-shuffle.
        scale_k = block_k * scale_chunk_tiles // MXFP8_BLOCK_SIZE
        scale_stages = (
            MXFP8_HTI_SCALE_BUFFERS if const_expr(use_scale_chunk) else stages
        )
        scale_a_stage_bytes = mxfp8_scale_stage_bytes(
            block_m // 2, block_k * scale_chunk_tiles, block_threads
        )
        scale_b_stage_bytes = mxfp8_scale_stage_bytes(
            block_n // 2, block_k * scale_chunk_tiles, block_threads
        )
        smem_sa = (
            allocator.allocate(
                fx.Array[fx.Uint8, scale_stages * 2 * scale_a_stage_bytes, 16]
            )
            .peek()
            .ptr
        )
        smem_sb = (
            allocator.allocate(
                fx.Array[fx.Uint8, scale_stages * 2 * scale_b_stage_bytes, 16]
            )
            .peek()
            .ptr
        )
        # Per-tile scale DMA shares the AB wait counts. Chunk prefetch is
        # separately fenced, so it must NOT inflate these counts.
        if const_expr(not use_scale_chunk):
            half_ldg_a_iters += scale_a_stage_bytes // (block_threads * 4)
            half_ldg_b_iters += scale_b_stage_bytes // (block_threads * 4)
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
        tiled_mma,
        copy_tid=tid,
        load_tid=tid,
        ks_begin=ks_begin,
        param=param,
    )
    a_s2r_copy_atom = ab_load_context.a_s2r_copy_atom
    b_s2r_copy_atom = ab_load_context.b_s2r_copy_atom
    thr_copy_A = ab_load_context.thr_copy_a
    thr_copy_B = ab_load_context.thr_copy_b
    thr_mma = tiled_mma.thr_slice(tid)
    a_lds_layout, b_lds_layout = make_gemm_ab_lds_layouts(
        half_block_m,
        half_block_n,
        block_k,
        param.a_is_transposed,
        param.b_is_transposed,
    )
    a_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(a_buf),
        lds_layout=a_lds_layout,
        outer_tile_size=half_block_m,
        outer_bound=m,
        leading_stride=a_leading_stride,
        load_iters=param.ldg_a_iters // 2,
        is_k_major=param.a_is_transposed,
    )
    b_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(b_buf),
        lds_layout=b_lds_layout,
        outer_tile_size=half_block_n,
        outer_bound=n,
        leading_stride=b_leading_stride,
        load_iters=param.ldg_b_iters // 2,
        is_k_major=not param.b_is_transposed,
        is_preshuffled=param.bpreshuffle,
    )
    c_lds_layout = fx.make_layout((half_block_m, half_block_n), (half_block_n, 1))

    def half_a_base(stage, m_part):
        return smem_a + (stage * block_m + m_part * half_block_m) * block_k

    def half_b_base(stage, n_part):
        return smem_b + (stage * block_n + n_part * half_block_n) * block_k

    def async_load_a_to_lds(m_part, k_tile, stage):
        async_load_operand(
            a_load_operand,
            lds_base=half_a_base(stage, m_part),
            global_outer_offset=block_m_offset + m_part * half_block_m,
            k_tile=k_tile,
        )
        if const_expr(use_lds_scale and not use_scale_chunk):
            async_load_mxfp8_scales(
                scale_a_buf,
                smem_sa + (stage * 2 + m_part) * scale_a_stage_bytes,
                tid,
                block_m_offset + m_part * half_block_m,
                m,
                ks_begin + k_tile * block_k,
                half_block_m,
                param,
            )

    def async_load_b_to_lds(n_part, k_tile, stage):
        async_load_operand(
            b_load_operand,
            lds_base=half_b_base(stage, n_part),
            global_outer_offset=block_n_offset + n_part * half_block_n,
            k_tile=k_tile,
        )
        if const_expr(use_lds_scale and not use_scale_chunk):
            async_load_mxfp8_scales(
                scale_b_buf,
                smem_sb + (stage * 2 + n_part) * scale_b_stage_bytes,
                tid,
                block_n_offset + n_part * half_block_n,
                n,
                ks_begin + k_tile * block_k,
                half_block_n,
                param,
            )

    def issue_scale_chunk(k_tile):
        # A chunk's scales are shared by all M/N waves. Alternate slots so
        # the next chunk's DMA can overlap computation on the current one.
        for part in range_constexpr(2):
            slot = (k_tile // scale_chunk_tiles) % MXFP8_HTI_SCALE_BUFFERS
            async_load_mxfp8_scale_chunk(
                scale_a_buf,
                smem_sa + (slot * 2 + part) * scale_a_stage_bytes,
                tid,
                block_m_offset + part * half_block_m,
                m,
                ks_begin + k_tile * block_k,
                ks_end,
                half_block_m,
                block_k * scale_chunk_tiles,
                param,
            )
            async_load_mxfp8_scale_chunk(
                scale_b_buf,
                smem_sb + (slot * 2 + part) * scale_b_stage_bytes,
                tid,
                block_n_offset + part * half_block_n,
                n,
                ks_begin + k_tile * block_k,
                ks_end,
                half_block_n,
                block_k * scale_chunk_tiles,
                param,
            )

    def prefetch_scale_chunk(k_tile):
        if const_expr(use_scale_chunk):  # noqa: SIM102 - static then dynamic
            if k_tile % scale_chunk_tiles == 0:
                # Balance HTI's staggered M-wave groups before reusing the
                # other slot, and wait for this chunk's earlier DMA to finish.
                rocdl.sched_barrier(0)
                if wid // n_waves == 0:
                    rocdl.s_barrier()
                wait_vmcnt_and_barrier(0)
                issue_scale_chunk(k_tile + scale_chunk_tiles)
                # No VMEM wait here: upcoming AB waits also retire these
                # older loads; the next chunk boundary fences them explicitly.
                # Restore the stagger needed by the original AB pipeline.
                if wid // n_waves == 1:
                    rocdl.s_barrier()
                rocdl.sched_barrier(0)

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

    def load_scale_fragment(
        smem, rows, wave_idx, waves, stage_bytes, part, stage, k_tile
    ):
        # Load alongside the FP8 fragment, BEFORE any DMA can reuse this
        # half-stage. Both consumers of the A/B fragment reuse these scales.
        scale_layout = fx.make_layout((rows, scale_k), (scale_k, 1))
        if const_expr(use_scale_chunk):
            slot = k_tile // scale_chunk_tiles % MXFP8_HTI_SCALE_BUFFERS
            src = fx.make_view(smem + (slot * 2 + part) * stage_bytes, scale_layout)
            scale_offset = k_tile % scale_chunk_tiles * (block_k // MXFP8_BLOCK_SIZE)
        else:
            src = fx.make_view(smem + (stage * 2 + part) * stage_bytes, scale_layout)
            scale_offset = 0
        frag = fx.make_rmem_tensor(
            (rows // waves // param.mma_m, block_k // param.mma_k), fx.Int32
        )
        for ki in range_constexpr(block_k // param.mma_k):
            for ri in range_constexpr(rows // waves // param.mma_m):
                row = (
                    ri * waves + wave_idx
                ) * param.mma_m + tid % GFX950_WAVE_SIZE % param.mma_m
                sk = (
                    ki * (param.mma_k // MXFP8_BLOCK_SIZE)
                    + tid % GFX950_WAVE_SIZE // param.mma_m
                )
                frag[ri, ki] = src[row, scale_offset + sk].to(fx.Int32)
        return frag

    def load_a_fragment(m_part, read_stage, k_tile):
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
        if const_expr(use_lds_scale):
            return frag_A, load_scale_fragment(
                smem_sa,
                half_block_m,
                wid // n_waves,
                param.m_waves,
                scale_a_stage_bytes,
                m_part,
                read_stage,
                k_tile,
            )
        return frag_A

    def load_b_fragment(n_part, read_stage, k_tile):
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
        if const_expr(use_lds_scale):
            return frag_B, load_scale_fragment(
                smem_sb,
                half_block_n,
                wid % n_waves,
                n_waves,
                scale_b_stage_bytes,
                n_part,
                read_stage,
                k_tile,
            )
        return frag_B

    def consume(frag_C, frag_A, frag_B, m_part, n_part, k_tile, emit_sched_barrier):
        if const_expr(use_lds_scale):
            frag_A, frag_SA = frag_A
            frag_B, frag_SB = frag_B
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(use_lds_scale):
                mxfp8_gemm(
                    frag_C,
                    frag_A[None, None, block_k_iter],
                    frag_B[None, None, block_k_iter],
                    frag_SA[None, block_k_iter],
                    frag_SB[None, block_k_iter],
                    tid,
                    0,
                    0,
                    0,
                    half_block_m,
                    half_block_n,
                    param,
                    scales_are_fragments=True,
                )
            elif const_expr(param.in_dtype_id == SCALED_GEMM_DTYPE_MXFP8):
                mxfp8_gemm(
                    frag_C,
                    frag_A[None, None, block_k_iter],
                    frag_B[None, None, block_k_iter],
                    scale_a_buf,
                    scale_b_buf,
                    tid,
                    block_m_offset + m_part * half_block_m,
                    block_n_offset + n_part * half_block_n,
                    ks_begin + k_tile * block_k + block_k_iter * param.mma_k,
                    m,
                    n,
                    param,
                )
            else:
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
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            row = fx.get_scalar(thr_mma_cRow[i])
            col = fx.get_scalar(thr_mma_cCol[i])
            global_row = block_m_offset + m_part * half_block_m + row
            global_col = block_n_offset + n_part * half_block_n + col
            safe_m = (global_row < m).select(global_row, 0)
            safe_n = (global_col < n).select(global_col, 0)
            acc = frag_C[i]
            if const_expr(param.in_dtype_id == SCALED_GEMM_DTYPE_FP8):
                acc = acc * scale_a_buf[safe_m] * scale_b_buf[safe_n]
            if const_expr(param.has_bias):  # noqa: SIM102 - staged compiler branch
                if const_expr(not is_split_k):
                    acc = acc + bias_buf[safe_n].to(fx.Float32)
            sC[row, col] = acc.to(shuffle_dtype)

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
                            cshuffle_r2g_vec_size, shuffle_dtype
                        ),
                    )
                    write_cshuffle_vec_to_global(
                        out,
                        out_buf,
                        global_row * n + global_col,
                        c_vec.to(global_output_dtype),
                        is_split_k,
                        param.out_dtype_id == SCALED_GEMM_DTYPE_FP32,
                    )

    c00 = make_c_fragment(0, 0)
    c01 = make_c_fragment(0, 1)
    c10 = make_c_fragment(1, 0)
    c11 = make_c_fragment(1, 1)

    # PTPC must not seed acc with bias; scale happens in store_half_tile_to_lds.
    c00.fill(0.0)
    c01.fill(0.0)
    c10.fill(0.0)
    c11.fill(0.0)
    if const_expr(is_split_k):
        splitk_protocol.zero_c()

    if const_expr(use_scale_chunk):
        # Prime the first slot while all waves are aligned, before the
        # AB prologue establishes the staggered barrier schedule.
        issue_scale_chunk(0)
        wait_vmcnt_and_barrier(0)

    async_load_b_to_lds(0, 0, 0)
    async_load_a_to_lds(0, 0, 0)
    async_load_b_to_lds(1, 0, 0)
    async_load_a_to_lds(1, 0, 0)
    rocdl.sched_barrier(0)
    if wid // n_waves == 1:
        rocdl.s_barrier()
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    async_load_b_to_lds(0, 1, 1)
    async_load_a_to_lds(0, 1, 1)
    async_load_b_to_lds(1, 1, 1)
    rocdl.sched_barrier(0)
    wait_vmcnt_and_barrier(half_ldg_b_iters + half_ldg_a_iters)

    main_loop_end = k_tiles - 2
    for k_tile in range(0, main_loop_end, 2):
        prefetch_scale_chunk(k_tile)
        next_k_tile = k_tile + 2
        # 0
        b0 = load_b_fragment(0, 0, k_tile)
        a0 = load_a_fragment(0, 0, k_tile)
        async_load_a_to_lds(1, k_tile + 1, 1)
        rocdl.s_barrier()
        consume(c00, a0, b0, 0, 0, k_tile, True)
        rocdl.s_barrier()
        b1 = load_b_fragment(1, 0, k_tile)
        async_load_b_to_lds(0, next_k_tile, 0)
        rocdl.s_barrier()
        consume(c01, a0, b1, 0, 1, k_tile, True)
        rocdl.s_barrier()
        a1 = load_a_fragment(1, 0, k_tile)
        async_load_a_to_lds(0, next_k_tile, 0)
        rocdl.s_barrier()
        consume(c10, a1, b0, 1, 0, k_tile, True)
        rocdl.s_barrier()
        b0 = load_b_fragment(0, 1, k_tile + 1)
        async_load_b_to_lds(1, next_k_tile, 0)
        wait_vmcnt_and_barrier(2 * half_ldg_b_iters + half_ldg_a_iters)
        consume(c11, a1, b1, 1, 1, k_tile, True)
        rocdl.s_barrier()
        # 1
        a0 = load_a_fragment(0, 1, k_tile + 1)
        async_load_a_to_lds(1, next_k_tile, 0)
        rocdl.s_barrier()
        consume(c00, a0, b0, 0, 0, k_tile + 1, True)
        rocdl.s_barrier()
        b1 = load_b_fragment(1, 1, k_tile + 1)
        async_load_b_to_lds(0, next_k_tile + 1, 1)
        rocdl.s_barrier()
        consume(c01, a0, b1, 0, 1, k_tile + 1, True)
        rocdl.s_barrier()
        a1 = load_a_fragment(1, 1, k_tile + 1)
        async_load_a_to_lds(0, next_k_tile + 1, 1)
        rocdl.s_barrier()
        consume(c10, a1, b0, 1, 0, k_tile + 1, True)
        rocdl.s_barrier()
        async_load_b_to_lds(1, next_k_tile + 1, 1)
        wait_vmcnt_and_barrier(half_ldg_b_iters + half_ldg_a_iters)
        consume(c11, a1, b1, 1, 1, k_tile + 1, True)
        rocdl.s_barrier()

    k_tile = main_loop_end
    prefetch_scale_chunk(k_tile)
    # 0
    if const_expr(is_split_k):
        wait_vmcnt_and_barrier(0)
    b0 = load_b_fragment(0, 0, k_tile)
    a0 = load_a_fragment(0, 0, k_tile)
    async_load_a_to_lds(1, k_tile + 1, 1)
    rocdl.s_barrier()
    consume(c00, a0, b0, 0, 0, k_tile, True)
    rocdl.s_barrier()
    b1 = load_b_fragment(1, 0, k_tile)
    rocdl.s_barrier()
    consume(c01, a0, b1, 0, 1, k_tile, True)
    rocdl.s_barrier()
    a1 = load_a_fragment(1, 0, k_tile)
    rocdl.s_barrier()
    consume(c10, a1, b0, 1, 0, k_tile, True)
    rocdl.s_barrier()
    b0 = load_b_fragment(0, 1, k_tile + 1)
    rocdl.s_barrier()
    consume(c11, a1, b1, 1, 1, k_tile, True)
    wait_vmcnt_and_barrier(0)
    # 1
    a0 = load_a_fragment(0, 1, k_tile + 1)
    rocdl.s_barrier()
    consume(c00, a0, b0, 0, 0, k_tile + 1, True)
    rocdl.s_barrier()
    b1 = load_b_fragment(1, 1, k_tile + 1)
    rocdl.s_barrier()
    consume(c01, a0, b1, 0, 1, k_tile + 1, True)
    rocdl.s_barrier()
    a1 = load_a_fragment(1, 1, k_tile + 1)
    rocdl.s_barrier()
    # Balance the prologue's staggered M-wave groups before reusing AB
    # storage. Unlike the reference's wave-local C-shuffle, tiled partition_C
    # and the linear global-store mapping exchange data across both M groups.
    rocdl.sched_barrier(0)
    if wid // n_waves == 0:
        rocdl.s_barrier()
    wait_vmcnt_and_barrier(0)

    # Overlap the last two MMA groups with the PTPC epilogue, as in the
    # reference: C00/C01 -> global, then C10, then C11.
    store_half_tile_to_lds(0, 0, c00)
    store_half_tile_to_lds(0, 1, c01)
    consume(c10, a1, b0, 1, 0, k_tile + 1, False)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    if const_expr(is_split_k):
        splitk_protocol.wait_until_initialized()
    store_half_tile_to_global(0, 0)
    store_half_tile_to_global(0, 1)
    store_half_tile_to_lds(1, 0, c10)
    consume(c11, a1, b1, 1, 1, k_tile + 1, False)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    store_half_tile_to_global(1, 0)
    store_half_tile_to_lds(1, 1, c11)
    rocdl.s_barrier()
    store_half_tile_to_global(1, 1)
    if const_expr(is_split_k):
        splitk_protocol.finish_split(split_k)


@flyc.jit
def scaled_gemm_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    scale_a: fx.Tensor,
    scale_b: fx.Tensor,
    bias: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    split_k: fx.Int32,
    param: ScaledGemmGfx950Param,
    stream: fx.Stream = fx.Stream(None),  # noqa: B008 - FlyDSL signature
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
    split_alignment = GFX950_DMA_BYTES // param.in_data_bytes
    working_k = (k + split_k - 1) // split_k
    working_k = (working_k + split_alignment - 1) // split_alignment * split_alignment
    num_pid_m = (m + param.block_m - 1) // param.block_m
    num_pid_n = (n + param.block_n - 1) // param.block_n
    use_hti = param.use_half_tile_interleaved
    if const_expr(use_hti):
        kernel_impl = scaled_gemm_hti_gfx950_kernel
    else:
        kernel_impl = scaled_gemm_gfx950_kernel
    kernel_impl._known_block_size = [param.block_threads, 1, 1]
    kernel_impl._func.__name__ = make_scaled_gemm_gfx950_kernel_name(param)
    kernel_impl(
        out,
        a,
        b,
        scale_a,
        scale_b,
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
        param,
    ).launch(
        grid=(num_pid_m * num_pid_n, split_k, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


def make_scaled_gemm_param_and_validate(m, n, k, kwargs):
    if min(m, n, k) <= 0:
        return None
    result = None
    try:
        result = make_scaled_gemm_gfx950_param(**kwargs)
    except (ValueError, AssertionError, TypeError):
        return None
    split_k = kwargs.get("split_k", 1)
    try:
        assert_no_k_tail(k, kwargs)
    except AssertionError:
        return None
    if k % result.scale_block != 0:
        return None
    if result.bpreshuffle and n % 16 != 0:
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
    use_half_tile_interleaved = kwargs.get("use_half_tile_interleaved", False)
    async_load_vec_size = GFX950_DMA_BYTES
    working_k = (k + split_k - 1) // split_k
    working_k = (
        (working_k + async_load_vec_size - 1)
        // async_load_vec_size
        * async_load_vec_size
    )
    last_working_k = k - (split_k - 1) * working_k
    assert (
        working_k % block_k == 0
    ), f"K-tail is unsupported: aligned split-K partition size {working_k} is not divisible by block_k={block_k}"
    assert last_working_k > 0 and last_working_k % block_k == 0, (
        "K-tail is unsupported: final split-K partition size "
        f"{last_working_k} must be positive and divisible by block_k={block_k}"
    )
    working_k_tiles = working_k // block_k
    last_working_k_tiles = last_working_k // block_k
    min_k_tiles = stages - 1
    assert (
        working_k_tiles >= min_k_tiles
    ), f"split-K partitions require at least {min_k_tiles} K tiles, got {working_k_tiles}"
    assert (
        last_working_k_tiles >= min_k_tiles
    ), f"the final split-K partition requires at least {min_k_tiles} K tiles, got {last_working_k_tiles}"
    if use_half_tile_interleaved:
        assert (
            working_k_tiles >= 2 and working_k_tiles % 2 == 0
        ), f"HTI requires at least two and an even number of K tiles per split-K partition, got {working_k_tiles}"
        assert last_working_k_tiles >= 2 and last_working_k_tiles % 2 == 0, (
            "HTI requires at least two and an even number of K tiles "
            "in the final split-K partition, "
            f"got {last_working_k_tiles}"
        )


def scaled_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    user_kwargs: dict | None = None,
    stream: torch.cuda.Stream | None = None,
    layout: str = "nt",
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute a per-row/per-column FP8 or block-scaled MXFP8 GEMM.

    FP8 uses float32 scales of shapes [M] and [N] (per-row/per-column):
    C = (A @ B) * scale_a[:, None] * scale_b[None, :] [+ bias].
    MXFP8 is selected by ``torch.float8_e8m0fnu`` or raw ``torch.uint8``
    scales of shapes [M, K // 32] and [N, K // 32]. Each E8M0 byte scales
    32 consecutive K elements: C = dequant(A) @ dequant(B) [+ bias].
    Raw scale byte e represents 2**(e - 127); 255 represents NaN.
    Scale shapes are independent of ``layout``; no preshuffling is required.
    Both modes take ``torch.float8_e4m3fn`` data and support the same pipelines.
    MXFP8 HTI with block_k=128 double-buffers scales in four-K-tile chunks,
    overlapping next-chunk prefetch with current-chunk computation. This
    requires no input preshuffle or extra K alignment from the caller.

    Each layout character controls only the corresponding tensor stride:
    N is row-major and T is column-major. Logical tensor shapes never change.
    Inputs that violate the selected layout or DMA alignment are rejected.
    Set ``user_kwargs["split_k"]`` above 1 to atomically reduce K partitions.
    Set ``user_kwargs["k_waves"]`` above 1 for full-tile workgroup-local slice-K.
    Set ``user_kwargs["use_half_tile_interleaved"]`` for the A16W16-style HTI
    pipeline (no slice-K; even K-tile count).
    ``out_dtype`` may be ``torch.bfloat16`` or ``torch.float32``.
    """
    user_kwargs = {} if user_kwargs is None else user_kwargs
    if stream is None:
        stream = torch.cuda.current_stream(device=a.device)
    if stream.device != a.device:
        raise ValueError("stream must be on the input device")
    if min(*a.shape, *b.shape) <= 0:
        raise ValueError("GEMM dimensions must be positive")
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
    assert a.dtype == torch.float8_e4m3fn
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
    mx_scale_dtypes = (torch.uint8, torch.float8_e8m0fnu)
    is_mxfp8 = scale_a.dtype in mx_scale_dtypes or scale_b.dtype in mx_scale_dtypes
    assert scale_a.device == device and scale_b.device == device
    if is_mxfp8:
        if scale_a.dtype not in mx_scale_dtypes or scale_b.dtype not in mx_scale_dtypes:
            raise ValueError("MXFP8 requires both scales to have E8M0 or uint8 dtype")
        scale_block = user_kwargs.get("scale_block", 32)
        if k % scale_block != 0:
            raise ValueError(f"MXFP8 requires K divisible by {MXFP8_BLOCK_SIZE}")
        if tuple(scale_a.shape) != (m, k // scale_block):
            raise ValueError("MXFP8 scale_a has an invalid shape for scale_block")
        scale_rows = n if scale_block == 32 else (n + 127) // 128
        if tuple(scale_b.shape) != (scale_rows, k // scale_block):
            raise ValueError("MXFP8 scale_b has an invalid shape for scale_block")
        # Preserve E8M0 encodings, not a numeric conversion to integers.
        # Materialize on the launch stream so strided scale inputs are safe.
        with torch.cuda.stream(stream):
            scale_a = scale_a.view(torch.uint8)
            if user_kwargs.get("scale_a_transposed", False):
                # contiguous() alone preserves non-unit singleton strides.
                # Rebuild the compact view so FlyDSL's unit-stride ABI also
                # holds for decode M=1 and K/scale_block=1.
                scale_a = (
                    scale_a.t().contiguous().view(-1).view(k // scale_block, m).t()
                )
            else:
                scale_a = scale_a.contiguous().view(-1).view(m, k // scale_block)
            scale_b = (
                scale_b.view(torch.uint8)
                .contiguous()
                .view(-1)
                .view(scale_rows, k // scale_block)
            )
            # contiguous() is a no-op on an offset contiguous view. Dword
            # direct-to-LDS requires 4-byte alignment, including leading stride.
            if scale_block == 32 and user_kwargs.get("block_k", 128) % 128 == 0:
                if scale_a.data_ptr() % 4 != 0 or scale_a.stride(0) % 4 != 0:
                    scale_a = scale_a.clone(memory_format=torch.contiguous_format)
                if scale_b.data_ptr() % 4 != 0 or scale_b.stride(0) % 4 != 0:
                    scale_b = scale_b.clone(memory_format=torch.contiguous_format)
    else:
        assert scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32
        assert scale_a.ndim == 1 and scale_a.shape[0] == m
        assert scale_b.ndim == 1 and scale_b.shape[0] == n
        if not scale_a.is_contiguous():
            scale_a = scale_a.contiguous()
        if not scale_b.is_contiguous():
            scale_b = scale_b.contiguous()
    if out_dtype is None:
        out_dtype = torch.bfloat16 if out is None else out.dtype
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"unsupported output dtype {out_dtype}; expected torch.bfloat16 or torch.float32"
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
        "block_k": 128,
        "stages": 2,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 4,
        "k_waves": 1,
        "group_m": 0,
        "use_half_tile_interleaved": True,
    }

    kwargs.update(user_kwargs)
    kwargs["a_is_transposed"] = a_is_transposed
    kwargs["b_is_transposed"] = b_is_transposed
    kwargs["in_dtype_id"] = (
        SCALED_GEMM_DTYPE_MXFP8 if is_mxfp8 else SCALED_GEMM_DTYPE_FP8
    )
    kwargs["out_dtype_id"] = (
        SCALED_GEMM_DTYPE_FP32 if out.dtype is torch.float32 else SCALED_GEMM_DTYPE_BF16
    )
    kwargs["has_bias"] = bias is not None
    split_k = kwargs["split_k"]
    assert_no_k_tail(k, kwargs)

    if bias is not None:
        assert bias.shape[0] == n
        assert bias.dtype == out_dtype
        assert bias.ndim == 1 and bias.device == device

    param = make_scaled_gemm_param_and_validate(m, n, k, kwargs)
    assert param is not None, "unsupported scaled_gemm_gfx950 shape/config"
    semaphore, signal = (
        get_split_k_buffers(stream, device) if param.is_split_k else (None, None)
    )
    dispatch_args = scaled_gemm_dispatch_args(
        out, a, b, scale_a, scale_b, bias, semaphore, signal, split_k, param, stream
    )
    run_cached(
        scaled_gemm_gfx950,
        *dispatch_args,
        constexpr_param=param,
        compiler=flyc.compile,
        dispatch_args=dispatch_args,
    )
    return out


def scaled_gemm_dispatch_args(
    out, a, b, scale_a, scale_b, bias, semaphore, signal, split_k, param, stream
):
    """Identical layout-dynamic ABI for runtime and CPU-only AOT."""
    a_arg = _dynamic_tensor_arg(a, 0 if param.a_is_transposed else 1)
    b_arg = _dynamic_tensor_arg(b, 0 if param.b_is_transposed else 1)
    out_arg = _dynamic_tensor_arg(out, 1)
    is_mxfp8 = param.in_dtype_id == SCALED_GEMM_DTYPE_MXFP8
    scale_a_arg = _dynamic_tensor_arg(
        scale_a, 0 if param.scale_a_transposed or not is_mxfp8 else 1
    )
    scale_b_arg = _dynamic_tensor_arg(scale_b, 1 if is_mxfp8 else 0)
    bias_arg = scale_a_arg if bias is None else _dynamic_tensor_arg(bias, 0)
    if not param.is_split_k:
        semaphore = signal = scale_a_arg
    return (
        out_arg,
        a_arg,
        b_arg,
        scale_a_arg,
        scale_b_arg,
        bias_arg,
        semaphore,
        signal,
        split_k,
        param,
        stream,
    )
