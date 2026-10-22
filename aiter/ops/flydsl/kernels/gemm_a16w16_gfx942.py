# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native bf16 (a16w16, f32-accumulate) GEMM for gfx942 / CDNA3 (MI300X).

C[M, N] = A[M, K] @ B[K, N], nn.Linear ``nt`` convention (A row-major [M,K],
B column-major [K,N] i.e. the ``w.t()`` view of a [N,K] weight).

This is a ground-up CDNA3 kernel, *not* a de-tuned fork of the gfx950 kernel:
- bf16 ``16x16x16`` MFMA (gfx950 is ``16x16x32``); the tiled-MMA K layout uses
  ``ab_k_perm`` so a 128-bit LDS->reg read pairs with two MFMA K-groups.
- 64 KB LDS budget.
- Async global->LDS DMA width is a *tunable knob* (``async_load_bytes``): 4 selects
  the 32-bit ``BufferLoadAsyncLDS32b`` (CDNA3 native), 16 selects the 128-bit atom.
  The default is 32-bit per the CDNA3 ISA; the knob lets us settle the width
  empirically on-device (see PLAN_gfx942_gemm.md §7).

Phase 0 scope: staged (non-split-K, k_waves=1) pipeline with a direct
register->global tiled-copy epilogue. Split-K / cshuffle / bias / fp16 land later.

Reuses the arch-generic data-movement + layout helpers from the gfx950 module and
its utils (they honour ``param.async_load_bytes`` and are not gfx950-locked); the
param builder, load-context / async-atom selection, kernel body, default configs,
and torch API are written natively for gfx942.
"""

import csv
import functools
import os
from dataclasses import dataclass
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.utility.graph_alloc import persistent_alloc

from . import buffer_ops
from .gemm_a16w16_gfx950 import (
    AsyncLoadTile,
    async_load_to_lds,
    get_split_k_buffers,
    make_gemm_ab_lds_layouts,
    write_cshuffle_vec_to_global,
)
from .gemm_a16w16_gfx950_utils import (
    GFX950_WAVE_SIZE,
    SPLIT_K_SEMAPHORE_MAX_LEN,
    BlockSwizzle,
    SplitKProtocol,
    get_llvm_ptr,
    get_wave_lds_offset,
)
from .kernels_common import run_cached

WAVE_SIZE = GFX950_WAVE_SIZE  # 64 lanes on CDNA3
SMEM_CAPACITY_GFX942 = 65536  # 64 KB LDS / CU
GFX942_NUM_CUS = 304  # MI300X compute units (grid-fill threshold for stage picking)

GEMM_A16W16_DTYPE_BF16 = 2
GEMM_A16W16_DTYPE_FP32 = 1
# gfx940 buffer cache policy: sc0 | sc1 (system coherent, bypasses the per-XCD L2).
CPOL_SC0_SC1 = 1 | 16


def ab_k_perm(mma_k: int, *, dtype_width: int = 16, copy_bits: int = 128):
    """K-axis permutation pairing a ``copy_bits`` A/B load with the MFMA K fragment.

    One MFMA consumes ``k_group = mma_k // 4`` elements per lane. With bf16
    ``mma_k=16`` a 128-bit load spans ``num_frgv = 2`` K-groups; the extra axis
    interleaves them so the tiled-copy partition derived from the tiled MMA
    matches the wide load. Collapses to ``(k_group, 4):(1, k_group)`` when a load
    is a single K-group.
    """
    k_group = mma_k // 4
    num_frgv = copy_bits // (k_group * dtype_width)
    if num_frgv <= 1:
        return fx.make_layout((k_group, 4), (1, k_group))
    num_elems = copy_bits // dtype_width
    return fx.make_layout((k_group, 4, num_frgv), (1, num_elems, k_group))


@fx.struct
class GemmA16W16Gfx942Param:
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
    a_is_transposed: fx.Constexpr[bool]
    b_is_transposed: fx.Constexpr[bool]
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    buffered_load: fx.Constexpr[bool]
    direct_store: fx.Constexpr[bool]
    pingpong: fx.Constexpr[bool]
    split_k_reduce: fx.Constexpr[bool]
    prefetch_depth: fx.Constexpr[int]
    stream_operands: fx.Constexpr[bool]
    # derived
    async_load_bytes: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
    out_data_bytes: fx.Constexpr[int]
    cshuffle_r2g_vec_size: fx.Constexpr[int]
    c_chunks: fx.Constexpr[int]
    ldg_x_threads: fx.Constexpr[int]
    block_threads: fx.Constexpr[int]
    ldg_a_iters: fx.Constexpr[int]
    ldg_b_iters: fx.Constexpr[int]


@dataclass(slots=True, kw_only=True, eq=False)
class Gfx942LoadContext:
    wave_offset: Any
    tid: Any
    ks_begin: Any
    param: GemmA16W16Gfx942Param
    async_g2s_copy_atom: Any
    a_s2r_copy_atom: Any
    b_s2r_copy_atom: Any
    a_tiled_copy_atom: Any
    b_tiled_copy_atom: Any
    buffer_copy_atom: Any
    uni_copy_atom: Any


def make_gemm_a16w16_gfx942_param(
    in_dtype_id: int = GEMM_A16W16_DTYPE_BF16,
    out_dtype_id: int = GEMM_A16W16_DTYPE_BF16,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    stages: int = 2,
    split_k: int = 1,
    m_waves: int = 2,
    n_waves: int = 2,
    k_waves: int = 1,
    group_m: int = 0,
    a_is_transposed: bool = False,
    b_is_transposed: bool = True,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 16,
    async_load_bytes: int = 4,
    buffered_load: bool = False,
    direct_store: bool = False,
    pingpong: bool = False,
    split_k_reduce: bool = False,
    prefetch_depth: int = 1,
    stream_operands: bool = False,
) -> GemmA16W16Gfx942Param:
    if in_dtype_id != GEMM_A16W16_DTYPE_BF16:
        raise ValueError(f"gfx942 kernel is bf16-only for now (got in_dtype_id={in_dtype_id})")
    if out_dtype_id not in (in_dtype_id, GEMM_A16W16_DTYPE_FP32):
        raise ValueError(f"unsupported out_dtype_id={out_dtype_id}")
    if (mma_m, mma_n, mma_k) != (16, 16, 16):
        raise ValueError("gfx942 bf16 MFMA is 16x16x16")
    if stages < 2 and not buffered_load:
        raise ValueError("stages must be >= 2 for the staged LDS pipeline")
    if buffered_load:
        if async_load_bytes != 16:
            raise ValueError("buffered_load requires async_load_bytes=16 (128-bit buffer_load)")
        if stages not in (1, 2):
            raise ValueError("buffered_load supports stages=1 (single LDS buffer) or 2")
    if pingpong and not (buffered_load and stages == 1):
        raise ValueError("pingpong requires buffered_load with stages=1")
    if prefetch_depth not in (1, 2):
        raise ValueError("prefetch_depth must be 1 or 2")
    if prefetch_depth == 2 and not pingpong:
        raise ValueError("prefetch_depth=2 requires the pingpong pipeline")
    if stream_operands:
        if not pingpong or prefetch_depth != 1:
            raise ValueError("stream_operands requires pingpong with prefetch_depth=1")
        # Two equal halves of whole 128-bit (32-element) K chunks per K wave.
        if block_k % (k_waves * 64) != 0:
            raise ValueError("stream_operands needs block_k // k_waves to be a multiple of 64")
    if k_waves < 1:
        raise ValueError("k_waves must be positive")
    if k_waves > 1:
        if k_waves & (k_waves - 1):
            raise ValueError("k_waves must be a power of two")
        if not pingpong:
            raise ValueError("k_waves > 1 requires the pingpong pipeline")
        if a_is_transposed or not b_is_transposed:
            raise ValueError("k_waves > 1 supports the nt layout only")
        if block_k % (k_waves * 32) != 0:
            raise ValueError("each K wave needs whole 32-element K chunks")
    if split_k < 1:
        raise ValueError("split_k must be >= 1")
    if split_k_reduce and split_k == 1:
        raise ValueError("split_k_reduce requires split_k > 1")
    if direct_store and split_k != 1 and not split_k_reduce:
        raise ValueError("direct_store with split_k > 1 requires split_k_reduce")
    if m_waves <= 0 or n_waves <= 0:
        raise ValueError("m_waves and n_waves must be positive")
    if m_waves * n_waves * k_waves > 16:
        raise ValueError("the workgroup cannot contain more than 16 waves")

    in_dbytes = 2
    out_dbytes = 4 if out_dtype_id == GEMM_A16W16_DTYPE_FP32 else 2
    block_threads = m_waves * n_waves * k_waves * WAVE_SIZE

    # Register->global store vector width for the cshuffle epilogue. Atomic
    # split-K caps it at 4 so the atomic-add granularity stays modest.
    max_cshuffle_vec = 16 // out_dbytes
    atomic_split_k = split_k > 1 and not split_k_reduce
    cshuffle_r2g_vec_size = min(max_cshuffle_vec, 4) if atomic_split_k else max_cshuffle_vec
    if block_n % cshuffle_r2g_vec_size != 0:
        raise ValueError("block_n must be divisible by the cshuffle store vector size")

    # LDS budget: A tile + B tile, per stage.
    smem_bytes = stages * (block_m + block_n) * block_k * in_dbytes
    if smem_bytes > SMEM_CAPACITY_GFX942:
        raise ValueError(
            f"staged LDS buffers exceed 64 KB: stages={stages}, block_m={block_m}, "
            f"block_n={block_n}, block_k={block_k}, smem_bytes={smem_bytes}"
        )
    red_smem_bytes = (k_waves // 2) * block_m * block_n * 4
    if red_smem_bytes > SMEM_CAPACITY_GFX942:
        raise ValueError(
            f"k_waves reduction buffer exceeds 64 KB: k_waves={k_waves}, "
            f"block_m={block_m}, block_n={block_n}"
        )
    def c_chunk_bytes(chunks):
        return block_m // chunks * block_n * in_dbytes

    # A C tile over 64 KB goes through LDS in row chunks of whole MMA M
    # repeats, sized to fit the A/B buffers when possible so LDS does not grow.
    c_smem_bytes = c_chunk_bytes(1)
    c_chunks = 1
    if not direct_store and c_smem_bytes > SMEM_CAPACITY_GFX942:
        reps_m = block_m // (mma_m * m_waves)
        fits = [d for d in range(2, reps_m + 1)
                if reps_m % d == 0 and c_chunk_bytes(d) <= SMEM_CAPACITY_GFX942]
        if not fits:
            raise ValueError(
                f"cshuffle tile exceeds 64 KB: block_m={block_m}, block_n={block_n}, "
                f"smem_bytes={c_smem_bytes}; use direct_store for aligned output tiles"
            )
        c_chunks = next((d for d in fits if c_chunk_bytes(d) <= smem_bytes), fits[0])

    # Async global->LDS DMA: async_load_bytes per lane (4 => 32-bit, 16 => 128-bit).
    if async_load_bytes not in (4, 16):
        raise ValueError("async_load_bytes must be 4 (32-bit) or 16 (128-bit)")
    async_load_vec_size = async_load_bytes // in_dbytes
    ldg_x_threads = block_k // async_load_vec_size
    if ldg_x_threads * async_load_vec_size != block_k:
        raise ValueError(
            f"block_k must be divisible by async_load_vec_size={async_load_vec_size}"
        )
    ldg_y_threads = block_threads // ldg_x_threads
    if ldg_y_threads * ldg_x_threads != block_threads:
        raise ValueError("ldg thread layout must exactly cover the workgroup")
    ldg_a_iters = (block_m * block_k) // (block_threads * async_load_vec_size)
    ldg_b_iters = (block_n * block_k) // (block_threads * async_load_vec_size)
    if ldg_a_iters * block_threads * async_load_vec_size != block_m * block_k:
        raise ValueError("A tile not covered by whole-thread vector loads")
    if ldg_b_iters * block_threads * async_load_vec_size != block_n * block_k:
        raise ValueError("B tile not covered by whole-thread vector loads")
    # vmcnt budget for in-flight async loads (stages-2 groups queued ahead).
    if (stages - 2) * (ldg_a_iters + ldg_b_iters) >= 63:
        raise ValueError("too many in-flight async loads for the vmcnt budget")

    mma_m_repeat = block_m // m_waves // mma_m
    mma_n_repeat = block_n // n_waves // mma_n
    mma_k_repeat = block_k // k_waves // mma_k
    if mma_m_repeat * m_waves * mma_m != block_m:
        raise ValueError("block_m must be divisible by m_waves * mma_m")
    if mma_n_repeat * n_waves * mma_n != block_n:
        raise ValueError("block_n must be divisible by n_waves * mma_n")
    if mma_k_repeat * k_waves * mma_k != block_k:
        raise ValueError("block_k must be divisible by k_waves * mma_k")

    return GemmA16W16Gfx942Param(
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
        a_is_transposed=a_is_transposed,
        b_is_transposed=b_is_transposed,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
        buffered_load=buffered_load,
        direct_store=direct_store,
        pingpong=pingpong,
        split_k_reduce=split_k_reduce,
        prefetch_depth=prefetch_depth,
        stream_operands=stream_operands,
        async_load_bytes=async_load_bytes,
        in_data_bytes=in_dbytes,
        out_data_bytes=out_dbytes,
        cshuffle_r2g_vec_size=cshuffle_r2g_vec_size,
        c_chunks=c_chunks,
        ldg_x_threads=ldg_x_threads,
        block_threads=block_threads,
        ldg_a_iters=ldg_a_iters,
        ldg_b_iters=ldg_b_iters,
    )


def make_gemm_a16w16_gfx942_kernel_name(param: GemmA16W16Gfx942Param) -> str:
    out_suffix = "_fp32" if param.out_dtype_id == GEMM_A16W16_DTYPE_FP32 else ""
    sk = ("_ksr" if param.split_k_reduce else "_ksd") if param.is_split_k else "_ks1"
    dma = f"buf{param.async_load_bytes}" if param.buffered_load else str(param.async_load_bytes)
    store = "_r2g" if param.direct_store else (f"_c{param.c_chunks}" if param.c_chunks > 1 else "")
    if param.pingpong:
        pipe = "_pps" if param.stream_operands else ("_pp2" if param.prefetch_depth == 2 else "_pp")
    else:
        pipe = ""
    return (
        f"hgemm_bf16{out_suffix}_gfx942_"
        f"t{param.block_m}x{param.block_n}x{param.block_k}x{param.stages}{sk}_"
        f"w{param.m_waves}x{param.n_waves}x{param.k_waves}_"
        f"gm{param.group_m}_dma{dma}{store}{pipe}"
    )


def make_k_sliced_lds_layout(rows, block_k, k_waves):
    """Row-major ``(rows, block_k)`` LDS tile stored as ``k_waves`` K sub-tiles.

    Each ``(rows, block_k // k_waves)`` sub-tile is contiguous and carries the
    same swizzle as ``make_lds_layout``, so a K wave addresses its slice through
    a base-pointer offset and a static layout. The 3-bit swizzle never touches
    the sub-tile index bits because every sub-tile spans a multiple of 512
    elements.
    """
    block_k_w = block_k // k_waves
    assert (rows * block_k_w) % 512 == 0
    swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))
    return fx.make_composed_layout(
        swizzle,
        fx.make_layout(
            (rows, (block_k_w, k_waves)), (block_k_w, (1, rows * block_k_w))
        ),
    )


def make_gfx942_load_context(elem_dtype, load_tid, ks_begin, param: GemmA16W16Gfx942Param):
    """Build the AB load context, selecting the async-to-LDS atom by DMA width."""
    uni_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    if const_expr(param.buffered_load):
        # gfx942 has no 128-bit async-to-LDS; the buffered path stages the
        # 128-bit load through registers (buffer_load) then ds_write to LDS.
        async_g2s_copy_atom = None
    elif const_expr(param.async_load_bytes >= 16):
        async_g2s_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128
        )
    else:
        async_g2s_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.BufferLoadAsyncLDS32b(), 32
        )

    if const_expr(param.a_is_transposed):
        a_s2r_copy_atom = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans16_64b(), elem_dtype)
        a_tiled_copy_atom = a_s2r_copy_atom
    else:
        a_s2r_copy_atom = uni_copy_atom
        a_tiled_copy_atom = buffer_copy_atom
    if const_expr(not param.b_is_transposed):
        b_s2r_copy_atom = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans16_64b(), elem_dtype)
        b_tiled_copy_atom = b_s2r_copy_atom
    else:
        b_s2r_copy_atom = uni_copy_atom
        b_tiled_copy_atom = buffer_copy_atom

    return Gfx942LoadContext(
        wave_offset=get_wave_lds_offset(load_tid, param.async_load_bytes),
        tid=load_tid,
        ks_begin=ks_begin,
        param=param,
        async_g2s_copy_atom=async_g2s_copy_atom,
        a_s2r_copy_atom=a_s2r_copy_atom,
        b_s2r_copy_atom=b_s2r_copy_atom,
        a_tiled_copy_atom=a_tiled_copy_atom,
        b_tiled_copy_atom=b_tiled_copy_atom,
        buffer_copy_atom=buffer_copy_atom,
        uni_copy_atom=uni_copy_atom,
    )


# ---------------------------------------------------------------------------
# gfx950-compatible building-block interface.
#
# The gated-residual fused kernels (K1/K2) reuse the GEMM *primitives* (param,
# LDS layouts, load context, async_load_to_lds) and inline their own compute
# loop, folding the RMSNorm/epilogue in. They import the gfx950-named symbols
# and pick the arch module by ``mma_k`` (16 => gfx942, 32 => gfx950). These
# aliases make this module a drop-in replacement for ``gemm_a16w16_gfx950``'s
# building blocks on gfx942, backed by the native mma_k=16 implementation.
# ``AsyncLoadTile``, ``async_load_to_lds``, and ``make_gemm_ab_lds_layouts`` are
# arch-generic and already re-exported above via the gfx950 import.
# ---------------------------------------------------------------------------
GEMM_A16W16_DTYPE_FP16 = 3  # unsupported on this bf16-only path; name parity only
GemmA16W16Gfx950Param = GemmA16W16Gfx942Param
make_gemm_a16w16_gfx950_param = make_gemm_a16w16_gfx942_param
make_gemm_ab_load_context = make_gfx942_load_context
GemmABLoadContext = Gfx942LoadContext


@flyc.kernel
def gemm_a16w16_gfx942_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    workspace: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    split_k: fx.Int32,
    working_k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmA16W16Gfx942Param,
):
    # split_k_reduce: partials go to a workspace and the last split of each
    # tile reduces them; otherwise split-K accumulates with global atomics.
    split_k_reduce = param.split_k_reduce
    is_split_k = param.is_split_k and not split_k_reduce
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    stages = param.stages
    block_threads = param.block_threads
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters
    cshuffle_r2g_vec_size = param.cshuffle_r2g_vec_size
    direct_store = param.direct_store
    c_chunks = param.c_chunks
    narrow_down_schedule = const_expr(
        block_m == 32
        and block_n == 64
        and block_k == 256
        and param.m_waves == 1
        and param.n_waves == 4
    )
    fine_schedule = const_expr(
        (
            block_m == 128
            and block_n == 96
            and block_k == 128
            and param.n_waves == 2
        )
        or (block_m == 128 and block_n == 256 and block_k == 64)
        or narrow_down_schedule
    )
    elem_dtype = fx.BFloat16
    global_output_dtype = (
        fx.Float32
        if const_expr(param.out_dtype_id == GEMM_A16W16_DTYPE_FP32)
        else fx.BFloat16
    )
    is_fp32_output = param.out_dtype_id == GEMM_A16W16_DTYPE_FP32

    if const_expr(is_split_k):
        splitk_protocol = SplitKProtocol(
            block_m,
            block_n,
            cshuffle_r2g_vec_size,
            param.out_data_bytes,
            block_threads,
            False,
        )

    tid = fx.thread_idx.x
    # Workgroup-local K split: k_waves groups of m_waves x n_waves waves share
    # the cooperative global->LDS copies, but each group reads and multiplies
    # only its own block_k // k_waves slice of every LDS tile.
    k_waves = param.k_waves
    block_k_w = block_k // k_waves
    threads_per_k_slice = block_threads // k_waves
    if const_expr(k_waves > 1):
        tid_mn = tid % threads_per_k_slice
        k_wave_idx = fx.Int32(
            rocdl.readfirstlane(
                fx.Int32.ir_type, fx.Int32(tid // threads_per_k_slice)
            )
        )
    else:
        tid_mn = tid
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    # NUM_PIDS_THRESHOLD=0: only gate the XCD-aware swizzle on grid/XCD
    # divisibility (required for the remap bijection), not on a block-count
    # minimum. The old 256 floor silently disabled the swizzle for narrow-N
    # "hc-down" grids (e.g. M2048: 32x6=192 blocks < 256), which left us L2-bound
    # (44% hit vs hipBLASLt 81%). Swizzle only activates when group_m > 0.
    block_swizzle = BlockSwizzle(NUM_XCDS=8, NUM_PIDS_THRESHOLD=0, GROUP_M=param.group_m)
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
        c: fx.Array[elem_dtype, 1 if direct_store else block_m // c_chunks * block_n, 16]
        red: fx.Array[fx.Float32, max(1, (k_waves // 2) * block_m * block_n), 16]
        flag: fx.Array[fx.Int32, 4, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr
    smem_red = storage.red.peek().ptr
    smem_flag = storage.flag.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    # Direct stores are only range-checked by the descriptor: rows past M fall
    # beyond the buffer end, which is why direct_store needs an exact N.
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=not direct_store)

    if const_expr(is_split_k):
        splitk_protocol.init(
            semaphore,
            signal,
            out,
            out,  # dummy bias slot (HAS_BIAS=False)
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

    ctx = make_gfx942_load_context(elem_dtype, load_tid=tid, ks_begin=ks_begin, param=param)
    a_s2r_copy_atom = ctx.a_s2r_copy_atom
    b_s2r_copy_atom = ctx.b_s2r_copy_atom

    # *_lds_layout spans the whole (rows, block_k) tile and is used by the LDS
    # writers; *_rd_layout is one K wave's (rows, block_k_w) slice.
    if const_expr(k_waves > 1):
        a_lds_layout = make_k_sliced_lds_layout(block_m, block_k, k_waves)
        b_lds_layout = make_k_sliced_lds_layout(block_n, block_k, k_waves)
        a_rd_layout, b_rd_layout = make_gemm_ab_lds_layouts(
            block_m, block_n, block_k_w, False, True
        )
        a_rd_offset = k_wave_idx * (block_m * block_k_w)
        b_rd_offset = k_wave_idx * (block_n * block_k_w)
    else:
        a_lds_layout, b_lds_layout = make_gemm_ab_lds_layouts(
            block_m, block_n, block_k, param.a_is_transposed, param.b_is_transposed
        )
        a_rd_layout, b_rd_layout = a_lds_layout, b_lds_layout
        a_rd_offset = 0
        b_rd_offset = 0
    c_lds_layout = fx.make_layout((block_m // c_chunks, block_n), (block_n, 1))

    thr_mma = tiled_mma.thr_slice(tid_mn)
    thr_copy_A = fx.make_tiled_copy_A(ctx.a_tiled_copy_atom, tiled_mma).get_slice(tid_mn)
    thr_copy_B = fx.make_tiled_copy_B(ctx.b_tiled_copy_atom, tiled_mma).get_slice(tid_mn)

    gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]
    sC_write = fx.make_view(smem_c, c_lds_layout)

    sA0 = fx.make_view(smem_a, a_rd_layout)
    sB0 = fx.make_view(smem_b, b_rd_layout)
    frag_A = thr_mma.make_fragment_A(sA0)
    frag_B = thr_mma.make_fragment_B(sB0)
    frag_C = thr_mma.make_fragment_C(gC)
    frag_A_ret = thr_copy_A.retile(frag_A)
    frag_B_ret = thr_copy_B.retile(frag_B)
    if const_expr(direct_store):
        if const_expr(is_fp32_output):
            c_r2g_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        else:
            c_r2g_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), elem_dtype)
        thr_copy_C_direct = fx.make_tiled_copy_C(c_r2g_atom, tiled_mma).get_slice(tid_mn)
    # Register double-buffer (stages>=3): a second A/B fragment pair holds the
    # next K-tile's LDS->reg read while frag_A/frag_B feed the current MFMAs.
    if const_expr(stages >= 3 or param.pingpong):
        frag_pf_A = thr_mma.make_fragment_A(sA0)
        frag_pf_B = thr_mma.make_fragment_B(sB0)
        frag_pf_A_ret = thr_copy_A.retile(frag_pf_A)
        frag_pf_B_ret = thr_copy_B.retile(frag_pf_B)

    # Buffered-load tiled copies (128-bit buffer_load global->reg, then ds_write
    # reg->LDS). gfx942 has no 128-bit async-to-LDS; staging through registers
    # with a layout-aware r2s copy writes the *swizzled* LDS tile correctly at any
    # vector width (the manual contiguous-DMA trick only matched at vec=2).
    if const_expr(param.buffered_load):
        _vpt = param.async_load_bytes // param.in_data_bytes  # 128-bit = 8 bf16
        _buf_atom = ctx.buffer_copy_atom
        _uni_atom = ctx.uni_copy_atom

        def _make_g2s_tv(rows):
            thrs_k = block_k // _vpt
            thrs_r = block_threads // thrs_k
            tv = fx.make_layout(
                ((thrs_k, thrs_r), (1, _vpt)),
                ((thrs_r * _vpt, 1), (1, thrs_r)),
            )
            return tv, thrs_r

        # The descriptors span each operand's real extent, so edge tiles never
        # read past the allocation; rows past M/N only feed unstored outputs.
        # A: tile (block_m, block_k), global row-block bid_m, all K-tiles.
        _a_tv, _a_thrs_r = _make_g2s_tv(block_m)
        _a_bounded = fx.rocdl.make_buffer_tensor(a, max_size=False)
        _gA_all = fx.flat_divide(_a_bounded, (block_m, block_k))[None, None, bid_m, None]
        _a_g2r = fx.make_tiled_copy(
            _buf_atom, _a_tv, fx.make_tile(_a_thrs_r, block_k)
        ).get_slice(tid)
        _a_r2s = fx.make_tiled_copy(
            _uni_atom, _a_tv, fx.make_tile(_a_thrs_r, block_k)
        ).get_slice(tid)
        _gA_thr = _a_g2r.partition_S(_gA_all)
        a_cp_frag = fx.make_fragment_like(_gA_thr[None, None, None, 0])
        a_cp_frag_ret = _a_r2s.retile(a_cp_frag)

        # B: logical (N, K) view of the [K, N] weight (strides (b_leading_stride, 1)),
        # tile (block_n, block_k), global row-block bid_n, all K-tiles.
        _b_nk_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.get_iter(b), fx.make_layout((n, k), (b_leading_stride, 1))
            ),
            max_size=False,
        )
        _b_tv, _b_thrs_r = _make_g2s_tv(block_n)
        _gB_all = fx.flat_divide(_b_nk_buf, (block_n, block_k))[None, None, bid_n, None]
        _b_g2r = fx.make_tiled_copy(
            _buf_atom, _b_tv, fx.make_tile(_b_thrs_r, block_k)
        ).get_slice(tid)
        _b_r2s = fx.make_tiled_copy(
            _uni_atom, _b_tv, fx.make_tile(_b_thrs_r, block_k)
        ).get_slice(tid)
        _gB_thr = _b_g2r.partition_S(_gB_all)
        b_cp_frag = fx.make_fragment_like(_gB_thr[None, None, None, 0])
        b_cp_frag_ret = _b_r2s.retile(b_cp_frag)
        cp_frags = [(a_cp_frag, b_cp_frag, a_cp_frag_ret, b_cp_frag_ret)]
        if const_expr(param.pingpong):
            # Second staging set: the single-buffer pipeline loads tile k+3
            # while tile k+2 is still being written to LDS from the first.
            a_cp_frag1 = fx.make_fragment_like(_gA_thr[None, None, None, 0])
            b_cp_frag1 = fx.make_fragment_like(_gB_thr[None, None, None, 0])
            cp_frags.append(
                (
                    a_cp_frag1,
                    b_cp_frag1,
                    _a_r2s.retile(a_cp_frag1),
                    _b_r2s.retile(b_cp_frag1),
                )
            )
        _buffered_k_base = ks_begin // block_k

        def load_ab_to_reg(k_tile, buf=0):
            a_dst, b_dst, _, _ = cp_frags[buf]
            global_k_tile = _buffered_k_base + k_tile
            fx.copy(_buf_atom, _gA_thr[None, None, None, global_k_tile], a_dst)
            fx.copy(_buf_atom, _gB_thr[None, None, None, global_k_tile], b_dst)

        def commit_ab_to_lds(lds_stage, buf=0):
            _, _, a_src, b_src = cp_frags[buf]
            sA_w = _a_r2s.partition_D(
                fx.make_view(smem_a + lds_stage * block_m * block_k, a_lds_layout)
            )
            sB_w = _b_r2s.partition_D(
                fx.make_view(smem_b + lds_stage * block_n * block_k, b_lds_layout)
            )
            fx.copy(_uni_atom, a_src, sA_w)
            fx.copy(_uni_atom, b_src, sB_w)

    row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
    thr_mma_cRow = thr_mma.partition_C(row_coords)
    thr_mma_cCol = thr_mma.partition_C(col_coords)

    def async_load_a_to_lds(k_tile, stage):
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
            context=ctx,
            load_iters=ldg_a_iters,
            is_k_major=param.a_is_transposed,
        )

    def async_load_b_to_lds(k_tile, stage):
        async_load_to_lds(
            tile=AsyncLoadTile(
                lds_base=smem_b + stage * block_n * block_k,
                src_base=fx.get_iter(b_buf),
                lds_layout=b_lds_layout,
                outer_tile_size=block_n,
                outer_bound=n,
                global_outer_offset=block_n_offset,
                leading_stride=b_leading_stride,
                k_tile=k_tile,
            ),
            context=ctx,
            load_iters=ldg_b_iters,
            is_k_major=not param.b_is_transposed,
        )

    # K-chunking for the software-pipelined compute. One 128-bit LDS->reg read
    # (``ab_k_perm``) feeds ``num_frgv`` MFMA K-groups, so we interleave copy and
    # MFMA at the 128-bit-chunk granularity: issuing chunk i+1's ds_read between
    # chunk i's MFMAs lets the scheduler hide the LDS->reg (lgkmcnt) latency.
    mma_k = param.mma_k
    k_group = mma_k // 4
    num_frgv = const_expr(128 // (k_group * 16))
    mma_k_iters = const_expr(block_k_w // mma_k)
    copy_chunks = const_expr(mma_k_iters // num_frgv)

    def read_tile_into(read_stage, a_ret, b_ret, c_begin=0, c_end=copy_chunks):
        """ds_read this K wave's slice of LDS stage ``read_stage`` into registers.

        Only 128-bit K chunks ``[c_begin, c_end)`` are read.
        """
        thr_sA = thr_copy_A.partition_S(
            fx.make_view(
                smem_a + read_stage * block_m * block_k + a_rd_offset, a_rd_layout
            )
        )
        thr_sB = thr_copy_B.partition_S(
            fx.make_view(
                smem_b + read_stage * block_n * block_k + b_rd_offset, b_rd_layout
            )
        )
        for c in range_constexpr(c_begin, c_end):
            if const_expr(
                param.buffered_load and fine_schedule and stages == 1
            ):
                # Emit the first A/B pair together so the first MFMA waits on
                # two LDS reads rather than A0/A1/B0. Remaining repeats are
                # free to move under the MFMA cluster via the schedule below.
                m_repeat = const_expr(block_m // param.m_waves // param.mma_m)
                n_repeat = const_expr(block_n // param.n_waves // param.mma_n)
                for r in range_constexpr(max(m_repeat, n_repeat)):
                    if const_expr(r < m_repeat):
                        fx.copy(
                            a_s2r_copy_atom,
                            thr_sA[None, r, c],
                            a_ret[None, r, c],
                        )
                    if const_expr(r < n_repeat):
                        fx.copy(
                            b_s2r_copy_atom,
                            thr_sB[None, r, c],
                            b_ret[None, r, c],
                        )
            else:
                fx.copy(a_s2r_copy_atom, thr_sA[None, None, c], a_ret[None, None, c])
                fx.copy(b_s2r_copy_atom, thr_sB[None, None, c], b_ret[None, None, c])

    def mma_tile(a_frag, b_frag):
        for g in range_constexpr(mma_k_iters):
            fx.gemm(
                tiled_mma,
                frag_C,
                a_frag[None, None, g],
                b_frag[None, None, g],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

    def compute_stage(read_stage):
        """Single-buffer compute: chunk-pipeline the ds_read vs the MFMAs."""
        thr_sA = thr_copy_A.partition_S(
            fx.make_view(smem_a + read_stage * block_m * block_k, a_lds_layout)
        )
        thr_sB = thr_copy_B.partition_S(
            fx.make_view(smem_b + read_stage * block_n * block_k, b_lds_layout)
        )

        def rd(c):
            fx.copy(a_s2r_copy_atom, thr_sA[None, None, c], frag_A_ret[None, None, c])
            fx.copy(b_s2r_copy_atom, thr_sB[None, None, c], frag_B_ret[None, None, c])

        rd(0)
        for c in range_constexpr(copy_chunks):
            if const_expr(c + 1 < copy_chunks):
                rd(c + 1)
            for j in range_constexpr(num_frgv):
                g = c * num_frgv + j
                fx.gemm(
                    tiled_mma,
                    frag_C,
                    frag_A[None, None, g],
                    frag_B[None, None, g],
                    frag_C,
                    traversal_order=fx.GemmTraversalOrder.KNM,
                )

    frag_C.fill(0.0)
    if const_expr(is_split_k):
        splitk_protocol.zero_c()

    # Prime the pipeline: preload the first stages-1 K tiles (async path only).
    if const_expr(not param.buffered_load):
        for stage in range_constexpr(stages - 1):
            async_load_b_to_lds(stage, stage)
            async_load_a_to_lds(stage, stage)
            rocdl.asyncmark()
        rocdl.sched_barrier(0)

    main_loop_end = k_tiles - (stages - 1)

    if const_expr(param.pingpong):
        # Single LDS buffer, three tiles in flight. At the top of iteration k the
        # registers hold tile k (frag_A/B), LDS holds tile k+1 and tile k+2 is in
        # flight from global memory. Each iteration has two phases, each with
        # half of tile k's MFMAs to hide its memory traffic behind:
        #   A: ds_read tile k+1 into frag_pf             | barrier (LDS free)
        #   B: ds_write tile k+2, buffer_load tile k+3   | barrier (LDS full)
        # so no wait or barrier is ever reached without MFMA work before it.
        # prefetch_depth=2 keeps tiles k+2 and k+3 in flight instead: each
        # staging set is reloaded (tile k+4) right after its ds_write, giving
        # every global load two iterations instead of one to land, with no
        # extra registers. It needs k_tiles >= 4.
        deep = const_expr(param.prefetch_depth == 2)
        stream = const_expr(param.stream_operands)
        m_rep = const_expr(block_m // param.m_waves // param.mma_m)
        n_rep = const_expr(block_n // param.n_waves // param.mma_n)
        n_dsrd = const_expr(copy_chunks * (m_rep + n_rep))
        n_dswr = const_expr(ldg_a_iters + ldg_b_iters)
        split_g = const_expr(max(1, mma_k_iters // 2))

        # Two MMA operand sets ping-pong between iterations (the loop is unrolled
        # by two), so the next tile is read straight into its own registers.
        mma_frags = [
            (frag_A, frag_B, frag_A_ret, frag_B_ret),
            (frag_pf_A, frag_pf_B, frag_pf_A_ret, frag_pf_B_ret),
        ]

        def mma_groups(p, g_begin, g_end):
            a_frag, b_frag, _, _ = mma_frags[p]
            for g in range_constexpr(g_begin, g_end):
                fx.gemm(
                    tiled_mma,
                    frag_C,
                    a_frag[None, None, g],
                    b_frag[None, None, g],
                    frag_C,
                    traversal_order=fx.GemmTraversalOrder.KNM,
                )

        def schedule_phase(n_mfma, n_rd, n_wr, n_ld):
            # Spread the memory ops of one phase evenly across its MFMAs. With
            # two staging sets each next-tile load goes to the other set and is
            # issued before the LDS write of the same slot; with deep prefetch
            # (or streamed operands) it reuses the set being written, so it
            # follows that write.
            reload_in_place = const_expr(deep or stream)
            n_mem = const_expr(n_rd + n_wr + n_ld)
            used_mfma = 0
            rd = 0
            wr = 0
            ld = 0
            for i in range_constexpr(n_mem):
                if const_expr(rd < n_rd):
                    rocdl.sched_dsrd(1)
                    rd += 1
                elif const_expr((ld < wr if reload_in_place else ld <= wr) and ld < n_ld):
                    rocdl.sched_vmem(1)
                    ld += 1
                elif const_expr(wr < n_wr):
                    rocdl.sched_dswr(1)
                    wr += 1
                else:
                    rocdl.sched_vmem(1)
                    ld += 1
                target = const_expr((i + 1) * n_mfma // n_mem)
                if const_expr(target > used_mfma):
                    rocdl.sched_mfma(target - used_mfma)
                    used_mfma = target
            if const_expr(used_mfma < n_mfma):
                rocdl.sched_mfma(n_mfma - used_mfma)

        def pipelined_iter(p, k_tile, commit, prefetch, newer_in_flight=False):
            # Parity p: mma_frags[p] holds tile k and cp_frags[p] holds tile k+2;
            # tile k+1 is read into mma_frags[1-p], tile k+3 into cp_frags[1-p]
            # (deep: tile k+4 into cp_frags[p], while cp_frags[1-p] holds k+3).
            _, _, next_a_ret, next_b_ret = mma_frags[1 - p]
            # Phase A: read tile k+1 while the first half of tile k computes.
            read_tile_into(0, next_a_ret, next_b_ret)
            mma_groups(p, 0, split_g)
            schedule_phase(split_g * m_rep * n_rep, n_dsrd, 0, 0)
            rocdl.sched_barrier(0)
            if const_expr(commit):
                rocdl.s_waitcnt(lgkmcnt=0)  # every wave is done reading LDS
                rocdl.s_barrier()
                # Phase B: refill LDS while the second half computes. Wait for
                # tile k+2 only; a deep prefetch leaves tile k+3 in flight.
                rocdl.s_waitcnt(vmcnt=n_dswr if newer_in_flight else 0)
                commit_ab_to_lds(0, p)
                if const_expr(prefetch):
                    if const_expr(deep):
                        load_ab_to_reg(k_tile + 4, p)
                    else:
                        load_ab_to_reg(k_tile + 3, 1 - p)
            mma_groups(p, split_g, mma_k_iters)
            schedule_phase(
                (mma_k_iters - split_g) * m_rep * n_rep,
                0,
                n_dswr if commit else 0,
                n_dswr if prefetch else 0,
            )
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(lgkmcnt=0)
            if const_expr(commit):
                rocdl.s_barrier()           # tile k+2 visible before next read
            rocdl.sched_barrier(0)

        def drain(p, k_tile):
            if const_expr(deep):
                pipelined_iter(p, k_tile, True, False, newer_in_flight=True)
                pipelined_iter(1 - p, k_tile + 1, True, False)
                pipelined_iter(p, k_tile + 2, False, False)
                mma_groups(1 - p, 0, mma_k_iters)
            else:
                pipelined_iter(p, k_tile, True, False)
                pipelined_iter(1 - p, k_tile + 1, False, False)
                mma_groups(p, 0, mma_k_iters)

        # Streamed operands: one MMA operand set split into two K halves and
        # one staging set, so a large tile fits without spilling. At the top
        # of iteration k the registers hold half 0 of tile k, LDS holds tile
        # k and tile k+1 is in flight from global memory:
        #   1: ds_read half 1 of tile k         | MFMAs on half 0   | barrier
        #   2: ds_write tile k+1, reload the     | MFMAs on both     | barrier
        #      staging set with tile k+2        |   halves          |
        #   3: ds_read half 0 of tile k+1       | MFMAs on half 1
        # Needs k_tiles >= 2 (checked on the host).
        half_c = const_expr(copy_chunks // 2)
        half_g = const_expr(half_c * num_frgv)
        p1_end = const_expr(half_g // 2)
        p2_end = const_expr(half_g + (mma_k_iters - half_g) // 2)
        mfma_per_g = const_expr(m_rep * n_rep)
        n_dsrd_half = const_expr(half_c * (m_rep + n_rep))

        def stream_iter(k_tile, commit, prefetch):
            read_tile_into(0, frag_A_ret, frag_B_ret, half_c, copy_chunks)
            mma_groups(0, 0, p1_end)
            schedule_phase(p1_end * mfma_per_g, n_dsrd_half, 0, 0)
            rocdl.sched_barrier(0)
            if const_expr(commit):
                rocdl.s_waitcnt(lgkmcnt=0)  # every wave is done reading tile k
                rocdl.s_barrier()
                commit_ab_to_lds(0, 0)
                if const_expr(prefetch):
                    load_ab_to_reg(k_tile + 2, 0)
            mma_groups(0, p1_end, p2_end)
            schedule_phase(
                (p2_end - p1_end) * mfma_per_g,
                0,
                n_dswr if commit else 0,
                n_dswr if prefetch else 0,
            )
            rocdl.sched_barrier(0)
            if const_expr(commit):
                rocdl.s_waitcnt(lgkmcnt=0)
                rocdl.s_barrier()           # tile k+1 visible before reading it
                read_tile_into(0, frag_A_ret, frag_B_ret, 0, half_c)
            mma_groups(0, p2_end, mma_k_iters)
            schedule_phase(
                (mma_k_iters - p2_end) * mfma_per_g,
                n_dsrd_half if commit else 0,
                0,
                0,
            )
            rocdl.sched_barrier(0)

        if const_expr(stream):
            load_ab_to_reg(fx.Int32(0))
            rocdl.s_waitcnt(vmcnt=0)
            commit_ab_to_lds(0)
            load_ab_to_reg(fx.Int32(1))
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
            read_tile_into(0, frag_A_ret, frag_B_ret, 0, half_c)
            rocdl.sched_barrier(0)
            for k_tile in range(0, k_tiles - 2, 1):
                stream_iter(k_tile, True, True)
            stream_iter(k_tiles - 2, True, False)
            stream_iter(k_tiles - 1, False, False)

        else:
            # Prologue (requires k_tiles >= 3, or 4 when deep, checked on the
            # host): tile 0 into registers, tile 1 into LDS, tile 2 (and tile
            # 3) in flight.
            load_ab_to_reg(fx.Int32(0))
            rocdl.s_waitcnt(vmcnt=0)
            commit_ab_to_lds(0)
            load_ab_to_reg(fx.Int32(1))
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
            read_tile_into(0, frag_A_ret, frag_B_ret)
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
            rocdl.s_waitcnt(vmcnt=0)
            commit_ab_to_lds(0)
            load_ab_to_reg(fx.Int32(2))
            if const_expr(deep):
                load_ab_to_reg(fx.Int32(3), 1)
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            steady_iters = k_tiles - (4 if deep else 3)
            steady_pairs = steady_iters // 2
            for pair in range(0, steady_pairs, 1):
                pipelined_iter(0, 2 * pair, True, True, newer_in_flight=deep)
                pipelined_iter(1, 2 * pair + 1, True, True, newer_in_flight=deep)
            if steady_iters % 2 == 1:
                pipelined_iter(0, steady_iters - 1, True, True, newer_in_flight=deep)
                drain(1, steady_iters)
            else:
                drain(0, steady_iters)
    elif const_expr(param.buffered_load):
        # 128-bit buffer_load (global->reg) + ds_write (reg->LDS), depth-1 double
        # buffer. The next tile's vmem loads are in flight during the current
        # tile's MFMAs; the ds_write commits them once vmcnt drains. This moves
        # the DMA off the (bandwidth-starved) 32-bit async-to-LDS path.
        load_ab_to_reg(fx.Int32(0))
        rocdl.s_waitcnt(vmcnt=0)
        commit_ab_to_lds(0)
        rocdl.s_waitcnt(lgkmcnt=0)
        rocdl.s_barrier()
        for k_tile in range(0, k_tiles - 1, 1):
            write_stage = (k_tile + 1) % stages
            # 128-bit buffer_load of tile k+1 (vmem, overlaps the MFMAs below)
            load_ab_to_reg(k_tile + 1)
            read_tile_into(k_tile % stages, frag_A_ret, frag_B_ret)
            if const_expr(stages == 1):
                # Single LDS buffer, but tile k's operands are already live
                # in registers. Once those ds_reads land, overwrite LDS with
                # tile k+1 while mma_tile consumes only registers.
                rocdl.s_waitcnt(lgkmcnt=0)
                rocdl.s_barrier()
                rocdl.s_waitcnt(vmcnt=0)
                commit_ab_to_lds(write_stage)
                mma_tile(frag_A, frag_B)
                # Force each ds_write into the tail of the MFMA run.
                n_mfma = const_expr(
                    (block_m // param.m_waves // param.mma_m)
                    * (block_n // param.n_waves // param.mma_n)
                    * (block_k // param.mma_k)
                )
                n_dswr = const_expr(ldg_a_iters + ldg_b_iters)
                if const_expr(narrow_down_schedule):
                    # This four-wave 32x64x256 specialization has fewer MFMAs
                    # than the large-tile EM3 schedule expects. Keep the
                    # scheduler budget exact: preload two DSRDs, place the
                    # remaining reads and next-tile VMEM at the front, and
                    # overlap all single-buffer LDS writes with the MFMA tail.
                    n_dsrd = const_expr(
                        copy_chunks
                        * (
                            block_m // param.m_waves // param.mma_m
                            + block_n // param.n_waves // param.mma_n
                        )
                    )
                    dsrd_preload = const_expr(min(2, n_dsrd))
                    rocdl.sched_dsrd(dsrd_preload)
                    for sched_i in range_constexpr(n_mfma):
                        rocdl.sched_mfma(1)
                        if const_expr(sched_i < n_dsrd - dsrd_preload):
                            rocdl.sched_dsrd(1)
                        if const_expr(sched_i < n_dswr):
                            rocdl.sched_vmem(1)
                        if const_expr(sched_i >= n_mfma - n_dswr):
                            rocdl.sched_dswr(1)
                elif const_expr(fine_schedule):
                    # EM3EN4-style hot-loop schedule for buffered tiles.
                    # The operations above are deliberately emitted in
                    # dependency order (next-tile VMEM, current-tile DSRD,
                    # next-tile DSWR, current-tile MFMA); these directives make
                    # LLVM interleaves a small DSRD preload, paired DSRD/VMEM
                    # groups under MFMAs, any residual reads/loads, then DSWR
                    # under the MFMA tail. This mirrors the gfx942
                    # Opus/Tensile EM3EN4 schedule while deriving all counts
                    # from the selected tile.
                    n_dsrd = const_expr(
                        copy_chunks
                        * (
                            block_m // param.m_waves // param.mma_m
                            + block_n // param.n_waves // param.mma_n
                        )
                    )
                    dsrd_preload = const_expr(2)
                    paired_iters = const_expr(
                        min(n_dswr, (n_dsrd - dsrd_preload) // 2)
                    )
                    rocdl.sched_dsrd(dsrd_preload)
                    for _ in range_constexpr(paired_iters):
                        rocdl.sched_dsrd(1)
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(2)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(2)
                    for _ in range_constexpr(n_dswr - paired_iters):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(2)
                    used_mfma = const_expr(
                        4 * paired_iters + 2 * (n_dswr - paired_iters)
                    )
                    used_dsrd = const_expr(dsrd_preload + 2 * paired_iters)
                    for _ in range_constexpr(n_dsrd - used_dsrd):
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                    used_mfma += n_dsrd - used_dsrd
                    if const_expr(n_mfma - used_mfma - 2 * n_dswr > 0):
                        rocdl.sched_mfma(n_mfma - used_mfma - 2 * n_dswr)
                    for _ in range_constexpr(n_dswr):
                        rocdl.sched_dswr(1)
                        rocdl.sched_mfma(2)
                elif const_expr(n_mfma > n_dswr):
                    rocdl.sched_mfma(n_mfma - n_dswr)
                    for _ in range_constexpr(n_dswr):
                        rocdl.sched_dswr(1)
                        rocdl.sched_mfma(1)
                rocdl.s_waitcnt(lgkmcnt=0)   # tile k+1 writes landed
                rocdl.s_barrier()             # writes visible before next read
            else:
                mma_tile(frag_A, frag_B)
                rocdl.s_waitcnt(vmcnt=0)
                # ds_write tile k+1 into the other buffer (double-buffered)
                commit_ab_to_lds(write_stage)
                rocdl.s_waitcnt(lgkmcnt=0)
                rocdl.s_barrier()
            rocdl.sched_barrier(0)
        read_tile_into((k_tiles - 1) % stages, frag_A_ret, frag_B_ret)
        mma_tile(frag_A, frag_B)
    elif const_expr(stages < 3):
        # Single-buffered pipeline (shallow prefetch): read current tile inline.
        for k_tile in range(0, main_loop_end, 1):
            current_stage = k_tile % stages
            write_stage = (current_stage + stages - 1) % stages
            rocdl.wait_asyncmark(stages - 2)
            rocdl.s_barrier()
            async_load_b_to_lds(k_tile + (stages - 1), write_stage)
            async_load_a_to_lds(k_tile + (stages - 1), write_stage)
            rocdl.asyncmark()
            compute_stage(current_stage)
            rocdl.sched_barrier(0)

        current_stage = main_loop_end % stages
        for s in range_constexpr(0, stages - 1):
            rocdl.wait_asyncmark(stages - 2 - s)
            rocdl.s_barrier()
            compute_stage(current_stage)
            current_stage = (current_stage + 1) % stages
    else:
        # Register double-buffer: prefetch tile k+1's LDS->reg read into frag_pf
        # *before* tile k's MFMAs, so the next read (incl. chunk 0) stays in
        # flight during the MFMAs and the lgkmcnt wait lands on the following
        # iteration's work. The current tile's operands are carried as SSA loop
        # state and restored into frag_A/frag_B each trip. Requires stages>=3 so
        # the stage being read (k+1) differs from the one being async-written
        # (k+stages-1).
        rocdl.wait_asyncmark(stages - 2)
        rocdl.s_barrier()
        read_tile_into(0, frag_A_ret, frag_B_ret)
        for k_tile in range(0, main_loop_end, 1):
            write_stage = (k_tile + stages - 1) % stages
            async_load_b_to_lds(k_tile + (stages - 1), write_stage)
            async_load_a_to_lds(k_tile + (stages - 1), write_stage)
            rocdl.asyncmark()
            rocdl.wait_asyncmark(stages - 2)
            rocdl.s_barrier()
            read_tile_into((k_tile + 1) % stages, frag_pf_A_ret, frag_pf_B_ret)
            mma_tile(frag_A, frag_B)
            # Rotate the just-read next tile into the MFMA fragments. The ds_read
            # above is only consumed here (after the MFMAs), so its lgkmcnt wait
            # is hidden behind the current tile's MFMAs.
            frag_A.store(frag_pf_A.load())
            frag_B.store(frag_pf_B.load())
            rocdl.sched_barrier(0)

        # Drain: compute the final stages-1 resident tiles, still prefetching one
        # tile ahead (no new async loads; progressively relax the wait count).
        for s in range_constexpr(0, stages - 1):
            if const_expr(s < stages - 2):
                rocdl.wait_asyncmark(stages - 2 - (s + 1))
                rocdl.s_barrier()
                read_tile_into(
                    (main_loop_end + (s + 1)) % stages,
                    frag_pf_A_ret,
                    frag_pf_B_ret,
                )
            mma_tile(frag_A, frag_B)
            if const_expr(s < stages - 2):
                frag_A.store(frag_pf_A.load())
                frag_B.store(frag_pf_B.load())
            rocdl.sched_barrier(0)

    if const_expr(k_waves > 1):
        # Tree-reduce the K waves' fp32 partial tiles into K wave 0 through
        # k_waves // 2 LDS slots. Peer waves with the same tid_mn hold
        # identically laid out fragments, so the exchange is fragment-ordered
        # in 128-bit lane-interleaved chunks (ds_write_b128 / ds_read_b128).
        # Waves that are already folded keep adding a peer slot; their sums are
        # never read again.
        acc = frag_C.load()
        n_vec4 = const_expr(acc.numel // 4)
        vec4_ty = fx.Vector.make_type(4, fx.Float32)

        def red_ptr(slot, j):
            return smem_red + ((slot * n_vec4 + j) * threads_per_k_slice + tid_mn) * 4

        for level in range_constexpr(k_waves.bit_length() - 1):
            active = const_expr(k_waves >> level)
            half = const_expr(active // 2)
            gpu.barrier()
            if (k_wave_idx >= half) & (k_wave_idx < active):
                for j in range_constexpr(n_vec4):
                    fx.ptr_store(
                        acc.shuffle(acc, [4 * j + t for t in range(4)]),
                        red_ptr(k_wave_idx - half, j),
                    )
            gpu.barrier()
            peer = [fx.ptr_load(red_ptr(k_wave_idx % half, j), result_type=vec4_ty)
                    for j in range_constexpr(n_vec4)]
            acc = acc + fx.Vector.from_elements(
                [p[t] for p in peer for t in range(4)], fx.Float32
            ).reshape(acc.shape)
        frag_C.store(acc)

    # Lanes that own the final tile: K wave 0, and with split_k_reduce only in
    # the split that arrives last for this tile.
    gated_epilogue = const_expr(k_waves > 1 or split_k_reduce)
    if const_expr(k_waves > 1):
        owns_tile = k_wave_idx == 0
    else:
        owns_tile = fx.Int32(1) == 1
    is_last = fx.Int32(1) == 1
    if const_expr(split_k_reduce):
        # Each split stores its fp32 partial tile in fragment order (128-bit,
        # lane-interleaved, coalesced), bumps the tile's arrival counter, and
        # the last split to arrive sums every partial and runs the epilogue.
        # Partials are written and read with sc0 sc1 so they are coherent
        # across XCDs without agent-scope fences: those lower to a full L2
        # writeback/invalidate that would evict every workgroup's A/B reuse.
        part = frag_C.load()
        n_vec4 = const_expr(part.numel // 4)
        tile_id = fx.block_idx.x
        ptr1_ty = ir.Type.parse("!llvm.ptr<1>")
        ws_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)

        def ws_offset(split, j):
            return (
                ((tile_id * split_k + split) * n_vec4 + j) * threads_per_k_slice
                + tid_mn
            ) * 4

        if owns_tile:
            for j in range_constexpr(n_vec4):
                buffer_ops.buffer_store(
                    part.shuffle(part, [4 * j + t for t in range(4)]).ir_value(),
                    ws_rsrc,
                    ws_offset(ks_idx, j),
                    cache_modifier=CPOL_SC0_SC1,
                )
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid == 0:
            sem_ptr = get_llvm_ptr(semaphore, tile_id, 4, ptr1_ty)
            arrived = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                sem_ptr,
                fx.Int32(1).ir_value(),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result
            fx.ptr_store(fx.Int32(arrived), smem_flag)
            if fx.Int32(arrived) == split_k - 1:
                llvm.StoreOp(
                    fx.Int32(0).ir_value(),
                    sem_ptr,
                    alignment=4,
                    ordering=llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                )
        gpu.barrier()
        is_last = fx.Int32(fx.ptr_load(smem_flag)) == split_k - 1
        owns_tile = owns_tile & is_last
        if is_last:
            frag_C.fill(0.0)
            for split in range(0, split_k, 1):
                peer = [
                    fx.Vector(
                        buffer_ops.buffer_load(
                            ws_rsrc,
                            ws_offset(split, j),
                            vec_width=4,
                            cache_modifier=CPOL_SC0_SC1,
                        )
                    )
                    for j in range_constexpr(n_vec4)
                ]
                frag_C.store(
                    frag_C.load()
                    + fx.Vector.from_elements(
                        [p[t] for p in peer for t in range(4)], fx.Float32
                    ).reshape(part.shape)
                )

    if const_expr(direct_store):
        # Aligned, non-split tiles can store the MMA fragment directly. This
        # removes the full-tile LDS cshuffle/barrier and, more importantly,
        # allows macro-tiles whose C footprint exceeds the 64 KB LDS capacity.
        if const_expr(is_fp32_output):
            frag_C_out = frag_C
        else:
            frag_C_out = fx.make_fragment_like(frag_C, elem_dtype)
            frag_C_out.store(frag_C.load().to(elem_dtype))
        if const_expr(gated_epilogue):
            c_src = thr_copy_C_direct.retile(frag_C_out)
            c_dst = thr_copy_C_direct.partition_S(gC)
            if owns_tile:
                fx.copy(c_r2g_atom, c_src, c_dst)
        else:
            fx.copy(
                c_r2g_atom,
                thr_copy_C_direct.retile(frag_C_out),
                thr_copy_C_direct.partition_S(gC),
            )
    else:
        # General epilogue: stage through LDS for coalescing, boundary masking,
        # split-K atomics, and optional f32 conversion.
        frag_C_out = fx.make_fragment_like(frag_C, elem_dtype)
        frag_C_out.store(frag_C.load().to(elem_dtype))
        # Fragment element i sits in MMA M repeat (i // mma_vals) % reps_m, so
        # chunk c owns the contiguous rows [c * chunk_rows, (c + 1) * chunk_rows).
        mma_vals = param.mma_m * param.mma_n // WAVE_SIZE
        reps_m = block_m // (param.mma_m * param.m_waves)
        chunk_reps = reps_m // c_chunks
        chunk_rows = block_m // c_chunks
        frag_size = fx.size(frag_C_out.shape).unpack()
        cshuffle_r2g_x_threads = block_n // cshuffle_r2g_vec_size
        cshuffle_vectors = chunk_rows * block_n // cshuffle_r2g_vec_size
        cshuffle_iters = (cshuffle_vectors + block_threads - 1) // block_threads
        for chunk in range_constexpr(c_chunks):
            gpu.barrier()
            chunk_elems = [
                i for i in range(frag_size)
                if (i // mma_vals) % reps_m // chunk_reps == chunk
            ]
            if const_expr(gated_epilogue):
                if owns_tile:
                    for i in range_constexpr(len(chunk_elems)):
                        e = chunk_elems[i]
                        row = fx.get_scalar(thr_mma_cRow[e])
                        if const_expr(chunk > 0):
                            row = row - chunk * chunk_rows
                        col = fx.get_scalar(thr_mma_cCol[e])
                        sC_write[row, col] = frag_C_out[e]
            else:
                for i in range_constexpr(len(chunk_elems)):
                    e = chunk_elems[i]
                    row = fx.get_scalar(thr_mma_cRow[e])
                    if const_expr(chunk > 0):
                        row = row - chunk * chunk_rows
                    col = fx.get_scalar(thr_mma_cCol[e])
                    sC_write[row, col] = frag_C_out[e]

            if const_expr(is_split_k and chunk == 0):
                splitk_protocol.wait_until_initialized()
            else:
                gpu.barrier()

            for i in range_constexpr(cshuffle_iters):
                vector_idx = block_threads * i + tid
                if vector_idx < cshuffle_vectors:
                    local_row = vector_idx // cshuffle_r2g_x_threads
                    local_col = vector_idx % cshuffle_r2g_x_threads * cshuffle_r2g_vec_size
                    global_row = block_m_offset + chunk * chunk_rows + local_row
                    global_col = block_n_offset + local_col
                    if (global_row < m) and (global_col < n) and is_last:
                        c_vec = fx.ptr_load(
                            smem_c + local_row * block_n + local_col,
                            result_type=fx.Vector.make_type(cshuffle_r2g_vec_size, elem_dtype),
                        )
                        global_offset = global_row * n + global_col
                        write_cshuffle_vec_to_global(
                            out,
                            out_buf,
                            global_offset,
                            c_vec,
                            is_split_k,
                            is_fp32_output,
                        )
        if const_expr(is_split_k):
            splitk_protocol.finish_split(split_k)


@flyc.jit
def gemm_a16w16_gfx942(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    workspace: fx.Tensor,
    split_k: fx.Int32,
    param: GemmA16W16Gfx942Param,
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
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, fx.BFloat16))
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((param.m_waves, param.n_waves, 1), (param.n_waves, 1, 0)),
        fx.make_tile(None, None, ab_k_perm(param.mma_k)),
    )
    # Whole block_k tiles per split-K partition; the last one may be shorter.
    split_alignment = param.block_k
    working_k = (k + split_k - 1) // split_k
    working_k = (working_k + split_alignment - 1) // split_alignment * split_alignment
    num_pid_m = (m + param.block_m - 1) // param.block_m
    num_pid_n = (n + param.block_n - 1) // param.block_n
    gemm_a16w16_gfx942_kernel._known_block_size = [param.block_threads, 1, 1]
    gemm_a16w16_gfx942_kernel._func.__name__ = make_gemm_a16w16_gfx942_kernel_name(param)
    gemm_a16w16_gfx942_kernel(
        out,
        a,
        b,
        semaphore,
        signal,
        workspace,
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


def _dynamic_tensor_arg(tensor, leading_dim):
    return flyc.from_dlpack(tensor.detach()).mark_layout_dynamic(leading_dim=leading_dim)


_SPLIT_K_WORKSPACE: dict = {}


def get_split_k_workspace(device, numel: int) -> torch.Tensor:
    """Grow-only fp32 scratch for split_k_reduce partial tiles (never needs zeroing)."""
    key = torch.device(device)
    ws = _SPLIT_K_WORKSPACE.get(key)
    if ws is None or ws.numel() < numel:
        with persistent_alloc(key):
            ws = torch.empty((max(numel, 1 << 20),), dtype=torch.float32, device=key)
        _SPLIT_K_WORKSPACE[key] = ws
    return ws


# Pingpong tiles with their (m, n, k) wave layouts, earlier entries winning
# cost ties. The k_waves=2 layouts give each wave a 64-wide output side over
# half of block_k, which halves the LDS reads per MFMA of a 32-wide wave tile;
# they win unsplit, but their LDS reduction costs ~10% per tile under split-K.
# 128x128 is left out: it loses to the legacy 128x128 4x4-wave path
# (2048^3 52.4 vs 50.1us).
PINGPONG_FILL_TILES = (
    (64, 128, ((1, 2, 2), (2, 2, 1))),
    (128, 64, ((2, 1, 2), (2, 2, 1))),
    (64, 96, ((1, 2, 2), (2, 2, 1))),
    (128, 96, ((2, 1, 2), (2, 2, 1))),
    (64, 64, ((2, 1, 2), (2, 2, 1))),
    (32, 64, ((1, 2, 2), (1, 4, 1))),
    # Only for N a multiple of 112 (three N tiles at N=336): 4096x336x10240
    # 88.9->81.7us, 3072x336 72.9->66.6, 2048x336 51.8->50.0.
    (64, 112, ((2, 1, 2),)),
    (128, 112, ((2, 1, 2),)),
)


def pingpong_fill_cost_us(bm: int, bn: int, k_waves: int, split_k: int, k_tiles: int) -> float:
    """Modelled runtime of one round of single-CU pingpong workgroups.

    Least-squares fit (14% rms) to ~1500 graph-timed pingpong configs on GPU6
    (N=256..2048, K=2048..10240): one bk128 tile costs 0.0205us per MFMA per
    SIMD plus 0.23us, and the split-K workspace reduction adds 2.86us plus
    54us per MB of fp32 partials.
    """
    t_tile = 0.0205 * (bm * bn / 128) + 0.23
    per_split = -(-k_tiles // split_k)
    if k_waves > 1 and split_k > 1 and bm * bn <= 64 * 64:
        # Small k_waves=2 tiles save ~7% per K tile against a fixed LDS
        # reduction of ~1.7 tiles, so they win on long splits (kw2/kw1: 1.10
        # at 10 tiles per split, 0.97 at 27; 1024x336x10240 sk3 36.0->32.4us).
        cost = 1.8 + (0.93 * per_split + 1.7) * t_tile
    else:
        if k_waves > 1 and split_k > 1:
            t_tile *= 1.107
        cost = 1.8 + per_split * t_tile
    if split_k > 1:
        cost += 2.86 + 54.25e-6 * split_k * bm * bn * 4
    return cost


def pingpong_fill_config(m: int, n: int, k: int, max_split_k: int = 8) -> dict | None:
    """Cheapest pingpong tile + split-K that runs in one round of workgroups.

    These kernels run one workgroup per CU, so candidates are limited to at
    most one workgroup per CU and ranked by ``pingpong_fill_cost_us``; bigger
    tiles amortize the per-tile overhead, split-K shortens each workgroup's K
    loop. Graph-timed on GPU6 (us, this rule vs the previous fill-count rule vs
    hipBLASLt): N=336 K=10240 M768 29.5/40.7/25.7, M1536 46.2/56.4/41.6,
    M3072 73.5/81.4/62.8; picks average 2% off the best swept config.
    """
    if k % 128:
        return None
    k_tiles = k // 128
    if k_tiles < 3:
        return None
    best = None
    if 320 <= n <= 384 and k >= 8192 and 128 < m <= 640:
        # Narrow-N long-K at small M is bound by re-reading A once per N tile,
        # which the cost model ignores; wider-N tiles win (N=336 K=10240:
        # M256 32x96 sk8 15.2us vs 32x64 16.3; M512 32x128 sk6 22.6 vs 23.8).
        bm, bn, waves, sk = (32, 96, (2, 2, 1), 8) if m <= 384 else (32, 128, (1, 4, 1), 6)
        tiles = (-(-m // bm)) * (-(-n // bn))
        sk = min(sk, GFX942_NUM_CUS // tiles)
        if k_tiles - (sk - 1) * -(-k_tiles // sk) >= 3:
            best = (None, bm, bn, waves, sk, tiles * sk)
    for order, (bm, bn, layouts) in enumerate(() if best else PINGPONG_FILL_TILES):
        if bn == 112 and n % 112:
            continue
        tiles = (-(-m // bm)) * (-(-n // bn))
        for sk in range(1, max_split_k + 1):
            per_split = -(-k_tiles // sk)
            if k_tiles - (sk - 1) * per_split < 3 or tiles * sk > GFX942_NUM_CUS:
                break
            for waves in layouts:
                cost = pingpong_fill_cost_us(bm, bn, waves[2], sk, k_tiles)
                if best is None or (cost, order) < best[0]:
                    best = ((cost, order), bm, bn, waves, sk, tiles * sk)
    if best is None or best[5] < 128:
        return None
    _, bm, bn, (mw, nw, kw), sk, _ = best
    # Streamed operands halve the operand and staging registers, which stops
    # the 128x96 spill (4096x336x10240 96.9->86.8us, 8192x384x10240
    # 165.4->153.6us) and is neutral to better on the small tiles.
    return dict(block_m=bm, block_n=bn, block_k=128, split_k=sk,
                m_waves=mw, n_waves=nw, k_waves=kw, group_m=8,
                direct_store=n % bn == 0, split_k_reduce=sk > 1,
                stream_operands=True, prefetch_depth=1,
                stages=1, buffered_load=True, async_load_bytes=16, pingpong=True)


def gfx942_auto_config(m: int, n: int, k: int) -> dict:
    """Best-known tile/wave/split-K config for a shape (from Phase-3 rocprofv3 tuning).

    Picks per regime: tiny-M decode/GEMV leans on split-K along K (narrow-N) or
    small tiles (wide-N); narrow-N tall-skinny ("hc-down") grows the tile with M
    and uses modest split-K at small M; wide-N short-K ("hc-up") uses big tiles;
    square compute-bound uses a 128x128 tile with 4x4 waves to hide the 32-bit
    async-DMA latency. The returned dict is a ``user_kwargs`` for ``gemm_a16w16``.
    """
    bk = 64
    # Two-phase single-LDS-buffer pipeline.
    pingpong = dict(stages=1, buffered_load=True, async_load_bytes=16, pingpong=True,
                    prefetch_depth=1)

    def stream_tile_config() -> dict | None:
        # 4-wave streamed-operand pingpong tile. Its runtime tracks the work
        # of the busiest CU, rounds x tile area, so take the exact tile that
        # minimizes it (smaller tile on ties; ragged tiles are slow). vs the
        # 16/8-wave b40 tiles: 4096x8192x512 107.4->91.5us, 8192x768x4096
        # 140.9->113.1 (128x192: one round instead of two), 2048x4096x1024
        # 49.3->44.8, 4096x1024x4096 90.4->78.7, 2048^3 51.1->47.6.
        if k % 64 or k < 2 * 64:
            return None
        best = None
        for order, (bm, bn) in enumerate(((128, 128), (128, 192), (256, 128))):
            if m % bm or n % bn:
                continue
            tiles = (m // bm) * (n // bn)
            if tiles < 128:
                continue
            key = (-(-tiles // GFX942_NUM_CUS) * bm * bn, order)
            if best is None or key < best[0]:
                best = (key, bm, bn)
        if best is None:
            return None
        _, bm, bn = best
        tiles = (m // bm) * (n // bn)
        # Multi-round grids want wider M groups for L2 reuse (2048x4096x1024
        # 45.7->43.3us, 2048x4096x512 26.3->25.6); one-round grids lose a
        # little with them (2048x2048x1024 23.9->24.4).
        group_m = 8 if tiles > GFX942_NUM_CUS else 4
        # A full single round with long K halves the barriers with bk128
        # (2048^3 48.0->45.2us); neutral at 2048x2048x4096 and 4096x1024x4096,
        # a loss at K=1024 and on the 128-tile 2048x1024x4096.
        bk_s = 64
        if (bm, bn) == (128, 128) and k % 128 == 0 and k >= 2048 \
                and 256 <= tiles <= GFX942_NUM_CUS:
            bk_s = 128
        return dict(block_m=bm, block_n=bn, block_k=bk_s, split_k=1,
                    m_waves=2, n_waves=2, k_waves=1, group_m=group_m,
                    stream_operands=True, direct_store=False, **pingpong)

    def split_k_for(bm: int, bn: int, cap: int = 16) -> int:
        # one semaphore per output tile (independent of split_k)
        num_tiles = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
        if num_tiles > SPLIT_K_SEMAPHORE_MAX_LEN:
            return 1
        for cand in (16, 8, 4, 2):
            if cand > cap:
                continue
            if k % (cand * bk):
                continue
            return cand
        return 1

    narrow_n = n <= 1024
    short_k = k <= 1024
    large_rect_blocks = ((m + 255) // 256) * ((n + 127) // 128)
    use_large_rect_tile = (
        k >= 1024
        and m % 256 == 0
        and n % 128 == 0
        and large_rect_blocks >= 256
    )

    # Small square problems are a grid-fill regime, not the long-K narrow-N
    # regime below.  In particular, 512^3 already has 256 independent 32x32
    # output tiles; split-K only adds redundant loads/atomics and regresses
    # 5.5us -> 16.6us.  At 1024^3, 64x64 st4 is the best balance of fill and
    # per-tile efficiency (14.3us -> 10.7us on GPU6).
    if short_k and 256 <= m == n:
        if m <= 512:
            return dict(block_m=32, block_n=32, block_k=bk, stages=2,
                        split_k=1, m_waves=2, n_waves=2, group_m=0)
        if m <= 1024:
            if m % 64 == 0 and k % 128 == 0 and k >= 256:
                # The streamed-operand pingpong runs the same 64x64 grid
                # faster: 1024^3 11.4->9.5us (hipBLASLt 9.8us).
                return dict(block_m=64, block_n=64, block_k=128, split_k=1,
                            m_waves=2, n_waves=2, k_waves=1, group_m=0,
                            stream_operands=True, direct_store=False, **pingpong)
            return dict(block_m=64, block_n=64, block_k=bk, stages=4,
                        split_k=1, m_waves=2, n_waves=2, group_m=0)

    if m <= 8:  # decode / GEMV
        if narrow_n:  # long-K, narrow-N: split-K is decisive
            bm = 64 if m <= 2 else 16
            bn = 32
            return dict(block_m=bm, block_n=bn, block_k=bk, stages=2,
                        split_k=split_k_for(bm, bn),
                        m_waves=2 if bm >= 32 else 1, n_waves=2)
        bm = 32 if m <= 2 else 16  # wide-N, short-K: no split-K
        return dict(block_m=bm, block_n=64, block_k=bk, stages=2, split_k=1,
                    m_waves=2 if bm >= 32 else 1, n_waves=2)

    # Long-K: fill the CUs with pingpong tiles (+ workspace
    # split-K). N=1024 at M>=2048 keeps the 128x128 b40 rule below, which the
    # fill rule's 128x128 pingpong only ties (4096x1024x4096 93.7 vs 92.0us).
    if not short_k and n <= 2048 and not (n == 1024 and m >= 2048):
        fill = pingpong_fill_config(m, n, k)
        if fill is not None:
            return fill

    if narrow_n:  # tall-skinny "hc-down", larger M
        # Narrow-N grids have few N-tiles, so without XCD-aware block swizzling
        # the workgroups sharing an A row-block scatter across XCDs and miss in
        # L2 (measured 44% hit vs hipBLASLt 81%). group_m groups M-tiles onto the
        # same XCD, lifting L2 hit to ~77% and cutting HBM traffic ~2.5x.
        if n == 1024 and m >= 2048 and k >= 2048 and k % 128 == 0:
            # The upper edge of "narrow" already has enough N tiles to fill the
            # machine with a 128x128 macro-tile. Use the barrier-halving b40
            # path instead of the 64x64 heuristic meant for N=256..512:
            # 4096x1024x4096 120.5->99.2us; 2048x1024x4096 ->64.6us.
            pp = stream_tile_config()
            if pp is not None:
                return pp
            return dict(block_m=128, block_n=128, block_k=128, stages=1,
                        split_k=1, m_waves=4, n_waves=4, group_m=8,
                        buffered_load=True, async_load_bytes=16)
        if m <= 1024:
            # 32x32 fills the machine via split-K along the long K while the grid
            # underfills (<=304 tiles, M<=~680). Once the 32x32 grid already
            # overfills the CUs (M~1024: 384 tiles), split-K only oversubscribes;
            # a wider-N tile with deep block_k=128 instead amortizes the 32x32
            # tile's ~3.3x VALU/MFMA addressing overhead (b34: M1024 61->53us,
            # 1.90x->1.62x of hipBLASLt).
            tiles32 = ((m + 31) // 32) * ((n + 31) // 32)
            if (
                m >= 256
                and n <= 512
                and 96 <= tiles32 <= 144
                and k % (4 * 128) == 0
            ):
                # A 128-bit buffered load with bk128 halves the K-loop/barrier
                # count while split-K keeps this medium-small output grid full.
                # Unlike the old async bk64 path, each split starts its buffered
                # loads at its own K offset. GPU6 rocprof A/B: 256x384x10240
                # 24.2->21.9us, 384x384x10240 29.5->27.9us, and
                # 256x512x4096 18.8->13.4us; a 512x256x4096 control is neutral.
                return dict(block_m=32, block_n=32, block_k=128, stages=1,
                            split_k=4, m_waves=2, n_waves=2, group_m=8,
                            buffered_load=True, async_load_bytes=16)
            if tiles32 > GFX942_NUM_CUS:
                return dict(block_m=32, block_n=64, block_k=128, stages=2,
                            split_k=1, m_waves=2, n_waves=2, group_m=8)
            return dict(block_m=32, block_n=32, block_k=bk, stages=2,
                        split_k=split_k_for(32, 32, cap=4), m_waves=2, n_waves=2,
                        group_m=8)
        if m <= 4096:
            # 64x64 tile. stages picks on grid-fill: st4 fills the full 64 KB
            # LDS (2*(64+64)*64*2 per stage x4 = 64 KB) -> 1 workgroup/CU and
            # also takes the stages>=3 register-double-buffer path (~1.75x more
            # VALU). That deep prefetch only pays off while the grid *underfills*
            # the machine (blocks <= #CUs), where latency hiding beats occupancy.
            # Once blocks exceed the CU count, st2 (32 KB -> 2 WG/CU, leaner
            # single-buffer loop) wins decisively (measured M4096 down, 384
            # blocks on 304 CUs: st4 184us -> st2 134us, -27%; M<=3072 st4 wins).
            bm = bn = 64
            blocks = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
            stages = 2 if blocks > GFX942_NUM_CUS else 4
            return dict(block_m=bm, block_n=bn, block_k=bk, stages=stages,
                        split_k=1, m_waves=2, n_waves=2, group_m=8)
        # 128x128 tile at large M: raise to 4x4 waves (1024 threads = 16
        # wavefronts/block) to hide the long-K (K=10240) global-load latency.
        # The narrow-N grid launches few blocks (M8192: 64*3=192 on 304 CUs), so
        # per-block occupancy -- not block count -- is the lever here: more blocks
        # (64x64, split-K) and deeper K/wider N were all pinned at ~257us, but
        # 4x4 waves drop it to ~220us (b32: -14%, gm8 marginally > gm16).
        pp = stream_tile_config()
        if pp is not None:
            return pp
        if k % 128 == 0:
            # block_k=128 halves the per-tile s_barrier count (K/128 vs K/64),
            # the dominant stall here. It fits 64 KB only single-buffered
            # (1*(block_m+block_n)*128*2), so take the buffered (global->reg->LDS)
            # path, which prefetches one tile ahead. Reverse-engineered from
            # hipBLASLt's MT*x128 down kernel; MfmaUtil 28.8->36%, M8192 221->211us,
            # M6144 204->188us (b37). Needs the stages==1 lgkmcnt drain for the
            # single-buffer WAR hazard at large block_k.
            if n % 96 == 0:
                # block_n=96 (mw4 nw2, 512 threads) exactly tiles narrow N=384
                # (4 n-blocks vs 3 for bn128 -> 256 vs 192 blocks, better fill)
                # and matches hipBLASLt's MT128x96x128 down kernel. Reproducible
                # M8192 206->200us, M6144 191->183us vs bn128 (b39, idle GPU6).
                # Below a full 256-block grid, direct R2G also wins by removing
                # the cshuffle barrier (M6144 175.8->169.7us under rocprof).
                use_direct_store = m % 128 == 0 and m < 8192
                return dict(block_m=128, block_n=96, block_k=128, stages=1,
                            split_k=1, m_waves=4, n_waves=2, group_m=8,
                            buffered_load=True, async_load_bytes=16,
                            direct_store=use_direct_store)
            return dict(block_m=128, block_n=128, block_k=128, stages=1,
                        split_k=1, m_waves=4, n_waves=4, group_m=8,
                        buffered_load=True, async_load_bytes=16)
        return dict(block_m=128, block_n=128, block_k=bk, stages=2, split_k=1,
                    m_waves=4, n_waves=4, group_m=8)

    if short_k:  # wide-N "hc-up"
        # 128x128 + 4x4 waves (16 wavefronts/block, b33: hides the global-load
        # latency) wins from prefill all the way down to M256 (b34: M256
        # 13.5->9.5us -- now < hipBLASLt; M512 20.8->18.9us; M1024 36->31.6us).
        # Below M256 the padded 128-row tile wastes work, so fall back to 64x128
        # 2x2 (4x4 over-subdivides the smaller tile, as on the 64x64 down branch).
        bm = 128 if m >= 256 else 64
        waves = 4 if bm == 128 else 2
        # 128x128 runs two workgroups per CU; past 608 tiles its last round is
        # nearly empty, so take hipBLASLt's one-round 256x160 tile instead
        # (1024x10240x320 26.2->24.8us). Its C tile goes through LDS in chunks.
        if m % 256 == 0 and n % 160 == 0 and k % 64 == 0 and k >= 128 \
                and (m // 128) * (n // 128) > 2 * GFX942_NUM_CUS \
                and (m // 256) * (n // 160) <= GFX942_NUM_CUS:
            return dict(block_m=256, block_n=160, block_k=64, split_k=1,
                        m_waves=4, n_waves=1, k_waves=1, group_m=0,
                        stream_operands=True, direct_store=False, **pingpong)
        pp = stream_tile_config() if m >= 1024 else None
        if pp is not None:
            return pp
        if use_large_rect_tile:
            # Four-wave hipBLASLt-class tiles reduce LDS instructions but leave
            # too little latency hiding in FlyDSL. Eight waves (64x64 per wave)
            # retain the reuse win: 2048x4096x1024 57.1->50.7us and its
            # transpose 59.1->54.5us. group_m=4 lifts the L2 hit rate (63%
            # vs hipBLASLt's 79% without it): 4096x2048x1024 55.0->48.8us.
            return dict(block_m=256, block_n=128, block_k=64, stages=1,
                        split_k=1, m_waves=4, n_waves=2, group_m=4,
                        buffered_load=True, async_load_bytes=16)
        if m >= 1024 and k % 128 == 0 and m % 128 == 0 and n % 128 == 0:
            # With enough output tiles to fill the machine, bk128 halves the
            # K-loop barriers and b40's write-under-compute schedule hides the
            # single-buffer ds_writes.  Broad-shape sweep: 1024x4096x512
            # 17.4->16.3us; 4096x8192x512 113.9->107.8us.
            return dict(block_m=128, block_n=128, block_k=128, stages=1,
                        split_k=1, m_waves=4, n_waves=4,
                        buffered_load=True, async_load_bytes=16)
        if k % bk == 0 and k >= 3 * bk:
            # Pingpong pipeline on 4-wave (or narrower) tiles. Up 128..2048 x
            # 10240 x 320: 7.8->6.5, 10.6->9.8, 29.8->28.1, 51.8->50.0us; it
            # also beats the 128x128 st2 path at M4096..8192 (96.2->89.5,
            # 142.0->135.1, 189.2->181.7us). With only a few K tiles the
            # epilogue matters: the MFMA fragment's direct store is 16-bit per
            # element, so stage through LDS for 128-bit stores from M1024
            # (M1024 28.5->26.7us, M2048 50.1->48.2us).
            if m <= 128:
                tile = dict(block_m=32, block_n=64, m_waves=2, n_waves=2)
            elif m <= 256:
                tile = dict(block_m=64, block_n=64, m_waves=1, n_waves=2)
            else:
                tile = dict(block_m=64, block_n=128, m_waves=1, n_waves=4)
            return dict(block_k=bk, split_k=1, group_m=0,
                        direct_store=m < 1024 and n % tile["block_n"] == 0,
                        stream_operands=True, **tile, **pingpong)
        return dict(block_m=bm, block_n=128, block_k=bk, stages=2, split_k=1,
                    m_waves=waves, n_waves=waves)

    # square / general compute-bound
    if use_large_rect_tile:
        # Large output grids amortize a 256x128 macro-tile while eight waves
        # preserve FlyDSL's required latency hiding. On square 4096 the direct
        # A full 256x256 tile with 16 waves amortizes cooperative loads across
        # twice the output area and gives enough resident waves to hide the
        # single-buffer barriers. On GPU6 it improves 4096^3 357->304us and
        # 8192^3 2437->2119us. Rectangles retain 256x128 because the square
        # tile's smaller grid under-fills there. Both choices require exact
        # M/N tiles. group_m=4 keeps A/B row blocks L2-resident: 4096^3
        # 295.5->279.0us.
        if m == n and m >= 4096:
            return dict(block_m=256, block_n=256, block_k=64, stages=1,
                        split_k=1, m_waves=4, n_waves=4, group_m=4,
                        buffered_load=True, async_load_bytes=16,
                        direct_store=True)
        pp = stream_tile_config()
        if pp is not None:
            return pp
        return dict(block_m=256, block_n=128, block_k=64, stages=1,
                    split_k=1, m_waves=4, n_waves=2, group_m=4,
                    buffered_load=True, async_load_bytes=16)
    if k % 128 == 0 and m % 128 == 0 and n % 128 == 0:
        # Same barrier-halving/write-overlap win as large-M down, now that the
        # grid is naturally full.  It improves square 2048/4096 by 6/4% and
        # moderate rectangles by 3-11% in the GPU6 broad-regime sweep;
        # group_m=4 adds 2048^3 52.1->50.6us.
        pp = stream_tile_config()
        if pp is not None:
            return pp
        return dict(block_m=128, block_n=128, block_k=128, stages=1,
                    split_k=1, m_waves=4, n_waves=4, group_m=4,
                    buffered_load=True, async_load_bytes=16)
    return dict(block_m=128, block_n=128, block_k=bk, stages=2, split_k=1,
                m_waves=4, n_waves=4)


_STRUCTURAL_KEYS = frozenset(
    ("block_m", "block_n", "block_k", "stages", "m_waves", "n_waves", "buffered_load")
)
_AUTO_ONLY_PIPELINE_KEYS = (
    "pingpong", "k_waves", "split_k_reduce", "direct_store", "prefetch_depth",
    "stream_operands",
)

# Config columns of the tuned table (aiter/configs/flydsl_gemm_a16w16_gfx942_tuned.csv);
# a row holds a complete config, so it never inherits heuristic flags.
TUNED_CONFIG_COLUMNS = (
    "block_m", "block_n", "block_k", "stages", "split_k", "m_waves", "n_waves",
    "k_waves", "group_m", "async_load_bytes", "buffered_load", "direct_store",
    "pingpong", "split_k_reduce", "prefetch_depth", "stream_operands",
)
TUNED_BOOL_COLUMNS = frozenset(
    ("buffered_load", "direct_store", "pingpong", "split_k_reduce", "stream_operands")
)
TUNED_KEY_COLUMNS = ("gfx", "cu_num", "M", "N", "K", "out_dtype")


def tuned_out_dtype_name(out_dtype: torch.dtype) -> str:
    return "fp32" if out_dtype == torch.float32 else "bf16"


def parse_tuned_config(row: dict) -> dict:
    """Config dict of a tuned-table row; blank cells keep the kernel default."""
    cfg = {}
    for name in TUNED_CONFIG_COLUMNS:
        value = (row.get(name) or "").strip()
        if value:
            cfg[name] = value.lower() == "true" if name in TUNED_BOOL_COLUMNS else int(value)
    return cfg


@functools.lru_cache(maxsize=None)
def _load_tuned_table(path: str) -> dict:
    table = {}
    if not os.path.exists(path):
        return table
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["gfx"], int(row["cu_num"]), int(row["M"]), int(row["N"]),
                   int(row["K"]), row["out_dtype"])
            table[key] = parse_tuned_config(row)
    return table


@functools.lru_cache(maxsize=None)
def tuned_table_path() -> str:
    """Resolved table path (env override + model_configs merge), cached per process."""
    from aiter.jit.core import AITER_CONFIGS

    return AITER_CONFIGS.AITER_CONFIG_FLYDSL_GEMM_A16W16_GFX942_FILE


def reload_tuned_table() -> None:
    """Re-read the table, e.g. after the tuner rewrote it or the env var changed."""
    tuned_table_path.cache_clear()
    _load_tuned_table.cache_clear()


@functools.lru_cache(maxsize=None)
def _device_key(device_index: int) -> tuple[str, int]:
    props = torch.cuda.get_device_properties(device_index)
    return props.gcnArchName.split(":")[0], props.multi_processor_count


def gfx942_tuned_config(m: int, n: int, k: int, out_dtype: torch.dtype = torch.bfloat16,
                        device: torch.device | None = None) -> dict | None:
    """The tuned-table config for this exact shape and device, or None."""
    if os.getenv("AITER_FLYDSL_GEMM_GFX942_HEURISTIC", "0") == "1":
        return None
    index = torch.cuda.current_device() if device is None or device.index is None else device.index
    gfx, cu_num = _device_key(index)
    cfg = _load_tuned_table(tuned_table_path()).get(
        (gfx, cu_num, m, n, k, tuned_out_dtype_name(out_dtype))
    )
    return dict(cfg) if cfg is not None else None


def gfx942_default_config(m: int, n: int, k: int, out_dtype: torch.dtype = torch.bfloat16,
                          device: torch.device | None = None) -> dict:
    """What ``gemm_a16w16`` runs without overrides: the tuned row, else the heuristic.

    ``AITER_FLYDSL_GEMM_GFX942_HEURISTIC=1`` skips the table.
    """
    cfg = gfx942_tuned_config(m, n, k, out_dtype, device)
    return cfg if cfg is not None else gfx942_auto_config(m, n, k)


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
    """Compute C[M, N] = A[M, K] @ B[K, N] on gfx942 (bf16 in, f32 accum).

    ``layout`` follows the a16w16 convention: N = row-major, T = column-major.
    ``nt`` is the nn.Linear call (A [M,K] row-major, B = w.t() column-major).
    """
    if bias is not None:
        raise NotImplementedError("gfx942 GEMM bias fusion is not implemented yet")
    if stream is None:
        stream = torch.cuda.current_stream()
    layout = layout.lower()
    if layout not in ("nn", "nt", "tn", "tt"):
        raise ValueError(f"unsupported GEMM layout: {layout!r}")
    a_is_transposed = layout[0] == "t"
    b_is_transposed = layout[1] == "t"

    assert a.ndim == 2 and b.ndim == 2
    assert a.device == b.device
    m, k = a.shape
    assert b.shape[0] == k
    n = b.shape[1]
    assert a.dtype == b.dtype == torch.bfloat16, "gfx942 kernel is bf16-only"

    if out_dtype is None:
        out_dtype = a.dtype if out is None else out.dtype
    if out_dtype not in (a.dtype, torch.float32):
        raise ValueError(f"unsupported output dtype {out_dtype}")
    if out is None:
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    else:
        assert out.dtype == out_dtype and out.device == a.device
        assert out.is_contiguous()
    out = out.view(-1, n)
    assert out.shape[0] == m

    # Base defaults, overlaid by the shape's tuned-table row (else the
    # gfx942_auto_config heuristic), then by any caller overrides.
    kwargs = {
        "block_m": 128,
        "block_n": 128,
        "block_k": 64,
        "stages": 2,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 2,
        "k_waves": 1,
        "group_m": 0,
        "async_load_bytes": 4,
    }
    auto = gfx942_default_config(m, n, k, out_dtype, a.device)
    user_kwargs = dict(user_kwargs or {})
    if _STRUCTURAL_KEYS & user_kwargs.keys():
        # A caller-chosen tile/pipeline must not inherit flags that were only
        # valid for the auto-config's own tile/pipeline.
        for key in _AUTO_ONLY_PIPELINE_KEYS:
            auto.pop(key, None)
    kwargs.update(auto)
    kwargs.update(user_kwargs)
    kwargs["a_is_transposed"] = a_is_transposed
    kwargs["b_is_transposed"] = b_is_transposed
    kwargs["in_dtype_id"] = GEMM_A16W16_DTYPE_BF16
    split_k = kwargs["split_k"]
    if split_k == 1:
        kwargs["split_k_reduce"] = False
    split_k_reduce = kwargs.get("split_k_reduce", False)

    # gfx942/CDNA3 has no packed-bf16 buffer atomic add, so bf16-output atomic
    # split-K accumulates through an f32 scratch buffer and casts to bf16 at the
    # end. f32-output split-K accumulates in place with f32 atomics.
    # split_k_reduce writes the final dtype directly.
    f32_accum_for_bf16 = (
        split_k > 1 and not split_k_reduce and out.dtype is torch.bfloat16
    )
    if f32_accum_for_bf16:
        kernel_out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    else:
        kernel_out = out
    kwargs["out_dtype_id"] = (
        GEMM_A16W16_DTYPE_FP32
        if kernel_out.dtype is torch.float32
        else GEMM_A16W16_DTYPE_BF16
    )

    if k % kwargs["block_k"] != 0:
        raise ValueError(
            f"K={k} must be divisible by block_k={kwargs['block_k']} (K-tail unsupported)"
        )
    if kwargs.get("direct_store", False) and n % kwargs["block_n"] != 0:
        raise ValueError("direct_store requires N to be a block_n multiple")
    k_tiles = k // kwargs["block_k"]
    split_tiles = (k_tiles + split_k - 1) // split_k
    last_split_tiles = k_tiles - (split_k - 1) * split_tiles
    if split_k > 1:
        # Every split-K partition is a whole number of block_k tiles; the last
        # one takes the remainder and must not be empty.
        if last_split_tiles < 1:
            raise ValueError(
                f"split_k={split_k} leaves an empty K partition "
                f"(K={k}, block_k={kwargs['block_k']})"
            )
        num_pid_m = (m + kwargs["block_m"] - 1) // kwargs["block_m"]
        num_pid_n = (n + kwargs["block_n"] - 1) // kwargs["block_n"]
        if num_pid_m * num_pid_n > SPLIT_K_SEMAPHORE_MAX_LEN:
            raise ValueError(
                f"split-K needs {num_pid_m * num_pid_n} tile semaphores, "
                f"more than {SPLIT_K_SEMAPHORE_MAX_LEN}"
            )

    if kwargs.get("pingpong", False):
        if kwargs.get("stream_operands", False):
            min_tiles = 2
        else:
            min_tiles = 4 if kwargs.get("prefetch_depth", 1) == 2 else 3
        if last_split_tiles < min_tiles:
            raise ValueError(
                f"pingpong needs at least {min_tiles} K tiles per split "
                f"(K={k}, split_k={split_k}, block_k={kwargs['block_k']})"
            )

    param = make_gemm_a16w16_gfx942_param(**kwargs)

    semaphore, signal = get_split_k_buffers(stream, a.device)
    if split_k_reduce:
        num_tiles = ((m + param.block_m - 1) // param.block_m) * (
            (n + param.block_n - 1) // param.block_n
        )
        workspace = get_split_k_workspace(
            a.device, num_tiles * split_k * param.block_m * param.block_n
        )
    else:
        workspace = signal
    a_arg = _dynamic_tensor_arg(a, 0 if a_is_transposed else 1)
    b_arg = _dynamic_tensor_arg(b, 0 if b_is_transposed else 1)
    out_arg = _dynamic_tensor_arg(kernel_out, 1)
    dispatch_args = (
        out_arg, a_arg, b_arg, semaphore, signal, workspace, split_k, param, stream
    )
    run_cached(
        gemm_a16w16_gfx942,
        *dispatch_args,
        constexpr_param=param,
        compiler=flyc.compile,
        dispatch_args=dispatch_args,
    )
    if f32_accum_for_bf16:
        out.copy_(kernel_out)
    return out
