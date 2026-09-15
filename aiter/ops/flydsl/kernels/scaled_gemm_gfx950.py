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
    write_cshuffle_vec_to_global,
)
from .gemm_a16w16_gfx950_utils import (
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    BlockSwizzle,
    get_wave_lds_offset,
    wait_vmcnt_and_barrier,
)
from .kernels_common import run_cached
from .scaled_gemm_gfx950_utils import async_load_operand, make_fp8_lds_layout

SCALED_GEMM_DTYPE_FP32 = 1
SCALED_GEMM_DTYPE_BF16 = 2
MXFP8_BLOCK_SIZE = 32
# Two four-tile buffers occupy the same LDS as one eight-tile buffer.
MXFP8_HTI_SCALE_CHUNK_TILES = 4
MXFP8_HTI_SCALE_BUFFERS = 2


@fx.struct
class ScaledGemmGfx950Param:
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
    direct_b: fx.Constexpr[bool]
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    # derived params
    async_load_bytes: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
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
    return param.block_k % 128 == 0


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


def uses_direct_b(bpreshuffle, use_hti, mma_k):
    return bpreshuffle and not use_hti and mma_k == 128


def make_scaled_gemm_gfx950_param(
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
    direct_b: bool = False,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 128,
) -> ScaledGemmGfx950Param:
    if not isinstance(direct_b, bool):
        raise TypeError("direct_b must be bool")
    if direct_b and not uses_direct_b(bpreshuffle, use_half_tile_interleaved, mma_k):
        raise ValueError("direct_b requires preshuffled B, full-tile and MMA16x16x128")
    if bpreshuffle and (a_is_transposed or not b_is_transposed):
        raise ValueError("bpreshuffle requires NT layout")
    if out_dtype_id not in (SCALED_GEMM_DTYPE_BF16, SCALED_GEMM_DTYPE_FP32):
        raise ValueError(f"unsupported out_dtype_id={out_dtype_id}")
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
    smem_bytes = stages * block_m * block_k * in_dbytes + (
        16 if direct_b else stages * block_n * block_k * in_dbytes
    )
    # Keep FP32 outputs and local slice-K reductions in FP32 until the
    # final global store; otherwise cancellation loses BF16 bits per slice.
    shuffle_bytes = 4 if out_dbytes == 4 or k_waves > 1 or (split_k > 1) else 2
    smem_bytes = max(smem_bytes, k_waves * block_m * block_n * shuffle_bytes)
    if block_k % 128 == 0:
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
    if block_k % 128 == 0:
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
        direct_b=direct_b,
        async_load_bytes=GFX950_DMA_BYTES,
        in_data_bytes=in_dbytes,
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
    dtype_str = "mxfp8"
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
    return name + f"_bp{int(param.bpreshuffle)}_bd{int(param.direct_b)}"


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
        k_perm = fx.make_layout((16, 2, 2), (1, 32, 16))
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
            a_scales.append(scale_a[safe_row, scale_k].to(fx.Int32))
    b_scales = []
    for ni in range_constexpr(fx.size(frag_B.shape[1]).unpack()):
        if const_expr(scales_are_fragments):
            b_scales.append(scale_b[ni])
        else:
            col = n_offset + (ni * param.n_waves + wave_n) * param.mma_n + scale_lane
            safe_col = (col < n).select(col, 0)
            b_scales.append(scale_b[safe_col, scale_k].to(fx.Int32))

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


def load_preshuffled_b(frag, b_buf, tid, n_offset, k_offset, n, leading_stride, param):
    """Two contiguous 16B strips per lane in shuffle_weight's (16,16) layout."""
    lane = tid % GFX950_WAVE_SIZE
    wave_n = tid // GFX950_WAVE_SIZE % param.n_waves
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FN)
    strip = fx.make_layout(16, 1)
    for ni in range_constexpr(fx.size(frag.shape[1]).unpack()):
        col = n_offset + (ni * param.n_waves + wave_n) * 16 + lane % 16
        safe_col = (col < n).select(col, 0)
        kk = k_offset + lane // 16 * 16
        base = (
            safe_col // 16 * leading_stride * 16 + kk // 16 * 256 + safe_col % 16 * 16
        )
        lo = fx.make_rmem_tensor(strip, fx.Float8E4M3FN)
        hi = fx.make_rmem_tensor(strip, fx.Float8E4M3FN)
        fx.copy_atom_call(atom, fx.make_view(fx.get_iter(b_buf) + base, strip), lo)
        fx.copy_atom_call(
            atom, fx.make_view(fx.get_iter(b_buf) + base + 1024, strip), hi
        )
        fx.coalesce(frag[None, ni]).store(lo.load().shuffle(hi.load(), list(range(32))))


@flyc.kernel
def scaled_gemm_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    scale_a: fx.Tensor,
    scale_b: fx.Tensor,
    bias: fx.Tensor,
    workspace: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    working_k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    param: ScaledGemmGfx950Param,
):
    tiled_mma = make_scaled_tiled_mma(param)
    direct_b = param.direct_b
    partial_split = param.is_split_k
    if const_expr(partial_split):
        out = fx.make_view(
            fx.get_iter(workspace)
            + fx.Int64(fx.block_idx.y) * fx.Int64(m) * fx.Int64(n),
            fx.make_layout((m, n), (n, 1)),
        )
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
        if const_expr(param.out_dtype_id == SCALED_GEMM_DTYPE_FP32 or partial_split)
        else fx.BFloat16
    )
    shuffle_dtype = (
        fx.Float32
        if const_expr(
            param.out_dtype_id == SCALED_GEMM_DTYPE_FP32
            or param.k_waves > 1
            or partial_split
        )
        else fx.BFloat16
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
        b: fx.Array[elem_dtype, 16 if direct_b else stages * block_n * block_k, 16]

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
    if const_expr(param.has_bias and not partial_split):
        bias_buf = fx.rocdl.make_buffer_tensor(bias, max_size=True)
    else:
        bias_buf = None

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

    # Accumulate in FP32. With split-K, bias is added once by the reducer.
    frag_C.fill(0.0)

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
        if const_expr(not direct_b):
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
            frag_A_chunk = frag_A[None, None, 0]
            if const_expr(direct_b):
                load_preshuffled_b(
                    frag_B[None, None, 0],
                    b_buf,
                    tid_in_k_slice,
                    block_n_offset,
                    ks_begin + k_tile * block_k + block_k_iter * param.mma_k,
                    n,
                    b_leading_stride,
                    param,
                )
            else:
                fx.copy(
                    b_s2r_copy_atom,
                    thr_sB_s2r[None, None, block_k_iter],
                    frag_B_retile[None, None, 0],
                )
            fx.copy(
                a_s2r_copy_atom,
                thr_sA_s2r[None, None, block_k_iter],
                frag_A_retile[None, None, 0],
            )
            if const_expr(use_lds_scale):
                mxfp8_gemm(
                    frag_C,
                    frag_A_chunk,
                    frag_B[None, None, 0],
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
            else:
                mxfp8_gemm(
                    frag_C,
                    frag_A_chunk,
                    frag_B[None, None, 0],
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

        # Each K-wave loads only its own slice into the same small register
        # fragment. Avoid divergent per-slice branches and loop-carried
        # partially initialized fragments (LLVM otherwise spills/aborts).
        for ki in range_constexpr(k_mma_iters_per_wave):
            compute_k_chunk(k_wave_idx * k_mma_iters_per_wave + ki)

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
        col = fx.get_scalar(thr_mma_cCol[i])
        global_col = block_n_offset + col
        safe_n = (global_col < n).select(global_col, 0)
        acc = frag_C[i]
        if const_expr(param.has_bias and not partial_split):
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
                    False,
                    param.out_dtype_id == SCALED_GEMM_DTYPE_FP32 or partial_split,
                )


@flyc.kernel
def scaled_gemm_hti_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    scale_a: fx.Tensor,
    scale_b: fx.Tensor,
    bias: fx.Tensor,
    workspace: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    working_k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    param: ScaledGemmGfx950Param,
):
    tiled_mma = make_scaled_tiled_mma(param)
    partial_split = param.is_split_k
    if const_expr(partial_split):
        out = fx.make_view(
            fx.get_iter(workspace)
            + fx.Int64(fx.block_idx.y) * fx.Int64(m) * fx.Int64(n),
            fx.make_layout((m, n), (n, 1)),
        )
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
        if const_expr(param.out_dtype_id == SCALED_GEMM_DTYPE_FP32 or partial_split)
        else fx.BFloat16
    )
    shuffle_dtype = (
        fx.Float32
        if const_expr(
            param.out_dtype_id == SCALED_GEMM_DTYPE_FP32
            or param.k_waves > 1
            or partial_split
        )
        else fx.BFloat16
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
    if const_expr(param.has_bias and not partial_split):
        bias_buf = fx.rocdl.make_buffer_tensor(bias, max_size=True)
    else:
        bias_buf = None

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
            else:
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
            global_col = block_n_offset + n_part * half_block_n + col
            safe_n = (global_col < n).select(global_col, 0)
            acc = frag_C[i]
            if const_expr(param.has_bias and not partial_split):
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
                        False,
                        param.out_dtype_id == SCALED_GEMM_DTYPE_FP32 or partial_split,
                    )

    c00 = make_c_fragment(0, 0)
    c01 = make_c_fragment(0, 1)
    c10 = make_c_fragment(1, 0)
    c11 = make_c_fragment(1, 1)

    # Bias is added in the epilogue or, for split-K, once in the reducer.
    c00.fill(0.0)
    c01.fill(0.0)
    c10.fill(0.0)
    c11.fill(0.0)

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

    # Overlap the last two MMA groups with the output epilogue, as in the
    # reference: C00/C01 -> global, then C10, then C11.
    store_half_tile_to_lds(0, 0, c00)
    store_half_tile_to_lds(0, 1, c01)
    consume(c10, a1, b0, 1, 0, k_tile + 1, False)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
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


@flyc.kernel
def scaled_gemm_splitk_reduce(
    partials: fx.Tensor,
    out: fx.Tensor,
    bias: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    split_k: fx.Int32,
    param: ScaledGemmGfx950Param,
):
    # One vector per thread; sum in a fixed order and cast only once.
    offset = (fx.block_idx.x * 256 + fx.thread_idx.x) * 4
    if offset < m * n:
        acc = fx.Vector.from_elements([fx.Float32(0.0)] * 4, fx.Float32)
        for split in range(split_k):
            value = fx.ptr_load(
                fx.get_iter(partials)
                + fx.Int64(split) * fx.Int64(m) * fx.Int64(n)
                + offset,
                result_type=fx.Vector.make_type(4, fx.Float32),
            )
            acc = acc + value
        if const_expr(param.has_bias):
            acc = acc + fx.ptr_load(
                fx.get_iter(bias) + offset % n,
                result_type=fx.Vector.make_type(4, out.dtype),
            ).to(fx.Float32)
        fx.ptr_store(acc.to(out.dtype), fx.get_iter(out) + offset)


@flyc.jit
def scaled_gemm_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    scale_a: fx.Tensor,
    scale_b: fx.Tensor,
    bias: fx.Tensor,
    workspace: fx.Tensor,
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
        workspace,
        m,
        n,
        k,
        working_k,
        a_leading_stride,
        b_leading_stride,
        param,
    ).launch(
        grid=(num_pid_m * num_pid_n, split_k, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )

    if const_expr(param.is_split_k):
        scaled_gemm_splitk_reduce._known_block_size = [256, 1, 1]
        scaled_gemm_splitk_reduce(workspace, out, bias, m, n, split_k, param).launch(
            grid=((m * n + 1023) // 1024, 1, 1), block=(256, 1, 1), stream=stream
        )


def make_scaled_gemm_param_and_validate(m, n, k, kwargs):
    if min(m, n, k) <= 0:
        return None
    try:
        result = make_scaled_gemm_gfx950_param(**kwargs)
    except (ValueError, AssertionError, TypeError):
        return None
    split_k = kwargs.get("split_k", 1)
    try:
        assert_no_k_tail(k, kwargs)
    except AssertionError:
        return None
    if k % MXFP8_BLOCK_SIZE != 0:
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
    if split_k > 1 and split_k * m * n * 4 > 1 << 32:
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
    """Standard MXFP8: E4M3 data, unshuffled E8M0 [outer,K/32] scales.

    Logical A[M,K], B[K,N]; layout describes data strides only. Optional
    bias has the output dtype. Split-K writes FP32 partials and reduces once.
    """
    user_kwargs = {} if user_kwargs is None else user_kwargs
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected logical A[M,K] and B[K,N]")
    if not a.is_cuda or b.device != a.device:
        raise ValueError("MXFP8 data must be on the same GPU")
    if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
        raise ValueError("MXFP8 data must be torch.float8_e4m3fn")
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
    m, k = a.shape
    n = b.shape[1]
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
    scale_dtypes = (torch.uint8, torch.float8_e8m0fnu)
    if scale_a.dtype not in scale_dtypes or scale_b.dtype not in scale_dtypes:
        raise ValueError("MXFP8 requires both scales to be E8M0 or uint8")
    if scale_a.device != device or scale_b.device != device:
        raise ValueError("MXFP8 scales must be on the input device")
    if k % MXFP8_BLOCK_SIZE:
        raise ValueError("MXFP8 requires K divisible by 32")
    if scale_a.shape != (m, k // 32) or scale_b.shape != (n, k // 32):
        raise ValueError("MXFP8 scales must be [M,K/32] and [N,K/32]")
    # Preserve scale encodings; canonicalize singleton strides for DLPack.
    with torch.cuda.stream(stream):
        scale_a = scale_a.view(torch.uint8).contiguous().view(-1).view(m, k // 32)
        scale_b = scale_b.view(torch.uint8).contiguous().view(-1).view(n, k // 32)
        if user_kwargs.get("block_k", 128) % 128 == 0:
            if scale_a.data_ptr() % 4:
                scale_a = scale_a.clone()
            if scale_b.data_ptr() % 4:
                scale_b = scale_b.clone()
    if out_dtype is None:
        out_dtype = torch.bfloat16 if out is None else out.dtype
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"unsupported output dtype {out_dtype}; expected torch.bfloat16 or torch.float32"
        )
    if out is None:
        out = torch.empty((m, n), dtype=out_dtype, device=device)
    elif (
        out.shape != (m, n)
        or out.dtype != out_dtype
        or out.device != device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous [M,N] on the input device with out_dtype"
        )
    if bias is not None:
        if bias.shape != (n,) or bias.dtype != out_dtype or bias.device != device:
            raise ValueError("bias must be [N] on the input device with out_dtype")
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
    kwargs["out_dtype_id"] = (
        SCALED_GEMM_DTYPE_FP32 if out.dtype is torch.float32 else SCALED_GEMM_DTYPE_BF16
    )
    kwargs["has_bias"] = bias is not None
    split_k = kwargs["split_k"]

    param = make_scaled_gemm_param_and_validate(m, n, k, kwargs)
    if param is None:
        raise ValueError("unsupported scaled_gemm_gfx950 shape/config")
    dispatch_args = scaled_gemm_dispatch_args(
        out, a, b, scale_a, scale_b, bias, split_k, param, stream
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
    out, a, b, scale_a, scale_b, bias, split_k, param, stream
):
    """Shared layout-dynamic ABI for runtime and tiny CPU-only AOT inputs."""
    a_arg = _dynamic_tensor_arg(a, 0 if param.a_is_transposed else 1)
    b_arg = _dynamic_tensor_arg(b, 0 if param.b_is_transposed else 1)
    out_arg = _dynamic_tensor_arg(out, 1)
    sa_arg = _dynamic_tensor_arg(scale_a, 1)
    sb_arg = _dynamic_tensor_arg(scale_b, 1)
    bias_arg = sa_arg if bias is None else _dynamic_tensor_arg(bias, 0)
    workspace = out_arg
    if param.is_split_k:
        partials = torch.empty(
            (split_k * out.shape[0], out.shape[1]),
            device=out.device,
            dtype=torch.float32,
        )
        workspace = _dynamic_tensor_arg(partials, 1)
    return (
        out_arg,
        a_arg,
        b_arg,
        sa_arg,
        sb_arg,
        bias_arg,
        workspace,
        split_k,
        param,
        stream,
    )
