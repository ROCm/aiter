# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native bf16 (a16w16, f32-accumulate) GEMM for gfx942 / CDNA3 (MI300X).

C[M, N] = A[M, K] @ B[K, N], nn.Linear ``nt`` convention (A row-major [M,K],
B column-major [K,N] i.e. the ``w.t()`` view of a [N,K] weight). Only ``nt`` is
supported: the transposed-operand LDS reads need CDNA4's ``ds_read_tr``.

CDNA3 specifics:
- bf16 ``16x16x16`` MFMA (gfx950 is ``16x16x32``); the tiled-MMA K layout uses
  ``ab_k_perm`` so a 128-bit LDS->reg read pairs with two MFMA K-groups.
- 64 KB LDS budget.
- Global->LDS movement is either 32-bit async DMA (``async_load_bytes=4``) or
  ``buffered_load``: a 128-bit ``buffer_load`` to registers then ``ds_write``.

Main-loop pipelines (selected by the param, one per compiled kernel):
- ``pingpong``: one LDS buffer with the MMA operand set streamed in two K
  halves; the default for most shapes.
- ``buffered``: buffered_load with one or two LDS buffers.
- ``async``: async DMA with ``stages`` LDS buffers; ``stages >= 3`` adds a
  register double buffer.
Epilogues: optional K-wave reduction, split-K (global atomics or a workspace
reduced by the last split), then a direct register->global store or an LDS
cshuffle store.

Configs come from the tuned table (``aiter/configs/flydsl_gemm_a16w16_gfx942_tuned.csv``)
and fall back to a cost-model search for shapes that are not in it.

Entry points are ``gemm_a16w16`` and ``flydsl_hgemm``; the table is tuned by
``op_tests/flydsl_tests/tune_gfx942_csv.py``. TunedGemm's FlyDSL catalog
(``gemm_a16w16_policy``) and AOT compilation do not cover gfx942: their
``flydsl_hgemm_*`` kernel names cannot encode the pipeline flags
(``pingpong``, ``buffered_load``, ``direct_store``, ``split_k_reduce``).
"""

import contextlib
import csv
import functools
import inspect
import itertools
import os
import types
from dataclasses import dataclass
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

from aiter.utility.graph_alloc import persistent_alloc

from .gemm_a16w16_gfx950 import (
    AsyncLoadTile,
    async_load_to_lds,
    write_cshuffle_vec_to_global,
)
from .gemm_a16w16_gfx950_utils import (
    GFX950_WAVE_SIZE,
    BlockSwizzle,
    SplitKProtocol,
    get_wave_lds_offset,
)
from .kernels_common import run_cached

# Unstable rocdl calls kept on purpose (no stable equivalent in FlyDSL 0.4):
# - rocdl.s_barrier: gpu.barrier adds workgroup fences, which drain the
#   in-flight global loads the pipelines overlap with the barrier.
# - rocdl.sched_barrier(0): pins the hand-scheduled MFMA/DS/VMEM interleave;
#   the stable sched_* helpers only emit sched_group_barrier.
# - rocdl.readfirstlane: makes k_wave_idx wave-uniform (SGPR), so the
#   per-K-wave LDS offsets and branches stay scalar.

WAVE_SIZE = GFX950_WAVE_SIZE  # 64 lanes on CDNA3
SMEM_CAPACITY_GFX942 = 65536  # 64 KB LDS / CU
GFX942_NUM_CUS = 304  # MI300X CUs: workgroups per round in the cost model

GEMM_A16W16_DTYPE_BF16 = 2
GEMM_A16W16_DTYPE_FP32 = 1
# CDNA3 buffer cache policy sc0 | sc1: system coherent, bypasses the per-XCD L2.
CPOL_SC0_SC1 = 1 | 16
# Split-K semaphore and signal slots per launch: one per output tile.
SPLIT_K_MAX_TILES = 1024


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
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    buffered_load: fx.Constexpr[bool]
    direct_store: fx.Constexpr[bool]
    pingpong: fx.Constexpr[bool]
    split_k_reduce: fx.Constexpr[bool]
    async_load_bytes: fx.Constexpr[int]
    # derived
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
    buffer_copy_atom: Any
    uni_copy_atom: Any


# Cached: building the frozen param struct costs ~90 us of host time per launch.
@functools.cache
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
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 16,
    async_load_bytes: int = 4,
    buffered_load: bool = False,
    direct_store: bool = False,
    pingpong: bool = False,
    split_k_reduce: bool = False,
) -> GemmA16W16Gfx942Param:
    if in_dtype_id != GEMM_A16W16_DTYPE_BF16:
        raise ValueError(f"gfx942 kernel is bf16-only (got in_dtype_id={in_dtype_id})")
    if out_dtype_id not in (in_dtype_id, GEMM_A16W16_DTYPE_FP32):
        raise ValueError(f"unsupported out_dtype_id={out_dtype_id}")
    if (mma_m, mma_n, mma_k) != (16, 16, 16):
        raise ValueError("gfx942 bf16 MFMA is 16x16x16")
    if not buffered_load:
        if stages < 2:
            raise ValueError(
                "the async pipeline needs stages >= 2; stages=1 needs buffered_load"
            )
        if async_load_bytes != 4:
            raise ValueError(
                "gfx942 async global->LDS loads are 32-bit only (async_load_bytes=4); "
                "128-bit loads need buffered_load"
            )
    if buffered_load:
        if async_load_bytes != 16:
            raise ValueError(
                "buffered_load requires async_load_bytes=16 (128-bit buffer_load)"
            )
        if stages not in (1, 2):
            raise ValueError("buffered_load supports stages=1 (single LDS buffer) or 2")
    if pingpong:
        if not (buffered_load and stages == 1):
            raise ValueError("pingpong requires buffered_load with stages=1")
        # Two equal halves of whole 128-bit (32-element) K chunks per K wave.
        if block_k % (k_waves * 64) != 0:
            raise ValueError("pingpong needs block_k // k_waves to be a multiple of 64")
    if k_waves < 1:
        raise ValueError("k_waves must be positive")
    if k_waves > 1:
        if k_waves & (k_waves - 1):
            raise ValueError("k_waves must be a power of two")
        if not pingpong:
            raise ValueError("k_waves > 1 requires the pingpong pipeline")
    # Each K wave reads LDS in 128-bit (32-element) K chunks of two MFMA K steps.
    if block_k % (k_waves * 32) != 0:
        raise ValueError("block_k // k_waves must be a multiple of 32")
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

    # LDS->global vector width of the cshuffle epilogue: 128 bits. Atomic
    # split-K always outputs fp32 (gemm_a16w16 gives bf16 an fp32 scratch),
    # so its vector is 4 wide.
    max_cshuffle_vec = 16 // out_dbytes
    atomic_split_k = split_k > 1 and not split_k_reduce
    cshuffle_r2g_vec_size = (
        min(max_cshuffle_vec, 4) if atomic_split_k else max_cshuffle_vec
    )
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
        return block_m // chunks * block_n * out_dbytes

    # A C tile over 64 KB goes through LDS in row chunks of whole MMA M
    # repeats, sized to fit the A/B buffers when possible so LDS does not grow.
    c_smem_bytes = c_chunk_bytes(1)
    c_chunks = 1
    if not direct_store and c_smem_bytes > SMEM_CAPACITY_GFX942:
        reps_m = block_m // (mma_m * m_waves)
        fits = [
            d
            for d in range(2, reps_m + 1)
            if reps_m % d == 0 and c_chunk_bytes(d) <= SMEM_CAPACITY_GFX942
        ]
        if not fits:
            raise ValueError(
                f"cshuffle tile exceeds 64 KB: block_m={block_m}, block_n={block_n}, "
                f"smem_bytes={c_smem_bytes}; use direct_store for aligned output tiles"
            )
        c_chunks = next((d for d in fits if c_chunk_bytes(d) <= smem_bytes), fits[0])

    # Global->LDS bytes per lane: 4 for async DMA, 16 for buffered_load.
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
    # vmcnt tracks at most 63 outstanding loads; the async pipelines keep
    # stages-2 tiles of loads in flight.
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
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
        buffered_load=buffered_load,
        direct_store=direct_store,
        pingpong=pingpong,
        split_k_reduce=split_k_reduce,
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


# Config keys: the make_gemm_a16w16_gfx942_param arguments except the dtypes
# (set by gemm_a16w16) and the MFMA shape (fixed on gfx942), with their
# signature defaults. Unset keys of a config take these values.
CONFIG_DEFAULTS = {
    name: arg.default
    for name, arg in inspect.signature(make_gemm_a16w16_gfx942_param).parameters.items()
    if name not in ("in_dtype_id", "out_dtype_id", "mma_m", "mma_n", "mma_k")
}


def make_gemm_a16w16_gfx942_kernel_name(param: GemmA16W16Gfx942Param) -> str:
    out_suffix = "_fp32" if param.out_dtype_id == GEMM_A16W16_DTYPE_FP32 else ""
    sk = ("_ksr" if param.split_k_reduce else "_ksd") if param.is_split_k else "_ks1"
    dma = (
        f"buf{param.async_load_bytes}"
        if param.buffered_load
        else str(param.async_load_bytes)
    )
    store = (
        "_r2g"
        if param.direct_store
        else (f"_c{param.c_chunks}" if param.c_chunks > 1 else "")
    )
    pipe = "_pp" if param.pingpong else ""
    return (
        f"hgemm_bf16{out_suffix}_gfx942_"
        f"t{param.block_m}x{param.block_n}x{param.block_k}x{param.stages}{sk}_"
        f"w{param.m_waves}x{param.n_waves}x{param.k_waves}_"
        f"gm{param.group_m}_dma{dma}{store}{pipe}"
    )


def make_k_sliced_lds_layout(rows, block_k, k_waves):
    """Row-major ``(rows, block_k)`` LDS tile stored as ``k_waves`` K sub-tiles.

    Each ``(rows, block_k // k_waves)`` sub-tile is contiguous and carries the
    swizzle of ``make_lds_layout`` (gemm_a16w16_gfx950_utils); a K wave reads
    its sub-tile by slicing the ``k_waves`` mode. The 3-bit swizzle never
    touches the sub-tile index bits because every sub-tile spans a multiple of
    512 elements. With ``k_waves=1`` this is ``make_lds_layout``.
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


def make_gfx942_load_context(
    elem_dtype, load_tid, ks_begin, param: GemmA16W16Gfx942Param
):
    """Copy atoms and per-thread offsets of the A/B global->LDS loads."""
    uni_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    if const_expr(param.buffered_load):
        # gfx942 has no 128-bit async-to-LDS; the buffered path stages the
        # 128-bit load through registers (buffer_load) then ds_write to LDS.
        async_g2s_copy_atom = None
    else:
        # The 32-bit form is CDNA3's buffer_load_dword ... lds. Its async atom
        # lives under cdna4 but runs on gfx942; the synchronous
        # fx.rocdl.BufferCopyLDS32b would make the compiler wait on vmcnt
        # before every LDS read and serialize the asyncmark-tracked stages.
        async_g2s_copy_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.BufferLoadAsyncLDS32b(), 32
        )

    return Gfx942LoadContext(
        wave_offset=get_wave_lds_offset(load_tid, param.async_load_bytes),
        tid=load_tid,
        ks_begin=ks_begin,
        param=param,
        async_g2s_copy_atom=async_g2s_copy_atom,
        buffer_copy_atom=buffer_copy_atom,
        uni_copy_atom=uni_copy_atom,
    )


@dataclass(slots=True, kw_only=True, eq=False)
class _KernelState:
    """Values and helpers of one compiled kernel shared by its pipeline stages."""

    param: GemmA16W16Gfx942Param
    tiled_mma: Any
    k_tiles: Any
    # Compile-time K chunking: 128-bit LDS reads per K wave tile, MFMA K steps
    # per tile, and MFMA K steps fed by one 128-bit read.
    copy_chunks: int
    mma_k_iters: int
    num_frgv: int
    frag_A: Any
    frag_B: Any
    frag_A_ret: Any
    frag_B_ret: Any
    frag_pf_A: Any
    frag_pf_B: Any
    frag_pf_A_ret: Any
    frag_pf_B_ret: Any
    frag_C: Any
    read_tile_into: Any
    mma_tile: Any
    compute_stage: Any
    load_ab_to_reg: Any
    commit_ab_to_lds: Any
    async_load_a_to_lds: Any
    async_load_b_to_lds: Any


def _mma_groups(s: _KernelState, a_frag, b_frag, g_begin, g_end):
    for g in range_constexpr(g_begin, g_end):
        fx.gemm(
            s.tiled_mma,
            s.frag_C,
            a_frag[None, None, g],
            b_frag[None, None, g],
            s.frag_C,
            traversal_order=fx.GemmTraversalOrder.KNM,
        )


def _schedule_phase(n_mfma, n_rd, n_wr, n_ld):
    """Spread one phase's memory ops evenly across its MFMAs.

    A next-tile load refills the staging registers being written to LDS, so
    each load follows the LDS write of the same slot.
    """
    n_mem = n_rd + n_wr + n_ld
    used_mfma = 0
    rd = 0
    wr = 0
    ld = 0
    for i in range(n_mem):
        if rd < n_rd:
            rocdl.sched_dsrd(1)
            rd += 1
        elif ld < wr and ld < n_ld:
            rocdl.sched_vmem(1)
            ld += 1
        elif wr < n_wr:
            rocdl.sched_dswr(1)
            wr += 1
        else:
            rocdl.sched_vmem(1)
            ld += 1
        target = (i + 1) * n_mfma // n_mem
        if target > used_mfma:
            rocdl.sched_mfma(target - used_mfma)
            used_mfma = target
    if used_mfma < n_mfma:
        rocdl.sched_mfma(n_mfma - used_mfma)


@flyc.jit
def _main_loop_pingpong(s: _KernelState):
    """One LDS buffer, with the MMA operand set streamed in two K halves.

    One operand set and one staging set, so a large tile fits without
    spilling. At the top of iteration k the registers hold half 0 of tile k,
    LDS holds tile k and tile k+1 is in flight from global memory:
      1: ds_read half 1 of tile k         | MFMAs on half 0   | barrier
      2: ds_write tile k+1, reload the    | MFMAs on both     | barrier
         staging set with tile k+2        |   halves          |
      3: ds_read half 0 of tile k+1       | MFMAs on half 1
    Needs k_tiles >= 2 (checked on the host).
    """
    param = s.param
    k_tiles = s.k_tiles
    frag_A = s.frag_A
    frag_B = s.frag_B
    frag_A_ret = s.frag_A_ret
    frag_B_ret = s.frag_B_ret
    read_tile_into = s.read_tile_into
    load_ab_to_reg = s.load_ab_to_reg
    commit_ab_to_lds = s.commit_ab_to_lds
    copy_chunks = s.copy_chunks
    mma_k_iters = s.mma_k_iters
    m_rep = const_expr(param.block_m // param.m_waves // param.mma_m)
    n_rep = const_expr(param.block_n // param.n_waves // param.mma_n)
    n_dswr = const_expr(param.ldg_a_iters + param.ldg_b_iters)
    half_c = const_expr(copy_chunks // 2)
    half_g = const_expr(half_c * s.num_frgv)
    p1_end = const_expr(half_g // 2)
    p2_end = const_expr(half_g + (mma_k_iters - half_g) // 2)
    mfma_per_g = const_expr(m_rep * n_rep)
    n_dsrd_half = const_expr(half_c * (m_rep + n_rep))

    def pipelined_iter(k_tile, commit, prefetch):
        read_tile_into(0, frag_A_ret, frag_B_ret, half_c, copy_chunks)
        _mma_groups(s, frag_A, frag_B, 0, p1_end)
        _schedule_phase(p1_end * mfma_per_g, n_dsrd_half, 0, 0)
        rocdl.sched_barrier(0)
        if const_expr(commit):
            rocdl.s_waitcnt(lgkmcnt=0)  # every wave is done reading tile k
            rocdl.s_barrier()
            commit_ab_to_lds(0)
            if const_expr(prefetch):
                load_ab_to_reg(k_tile + 2)
        _mma_groups(s, frag_A, frag_B, p1_end, p2_end)
        _schedule_phase(
            (p2_end - p1_end) * mfma_per_g,
            0,
            n_dswr if commit else 0,
            n_dswr if prefetch else 0,
        )
        rocdl.sched_barrier(0)
        if const_expr(commit):
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()  # tile k+1 visible before reading it
            read_tile_into(0, frag_A_ret, frag_B_ret, 0, half_c)
        _mma_groups(s, frag_A, frag_B, p2_end, mma_k_iters)
        _schedule_phase(
            (mma_k_iters - p2_end) * mfma_per_g,
            n_dsrd_half if commit else 0,
            0,
            0,
        )
        rocdl.sched_barrier(0)

    load_ab_to_reg(fx.Int32(0))
    rocdl.s_waitcnt(vmcnt=0)
    commit_ab_to_lds(0)
    load_ab_to_reg(fx.Int32(1))
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.s_barrier()
    read_tile_into(0, frag_A_ret, frag_B_ret, 0, half_c)
    rocdl.sched_barrier(0)
    for k_tile in range(0, k_tiles - 2, 1):
        pipelined_iter(k_tile, True, True)
    pipelined_iter(k_tiles - 2, True, False)
    pipelined_iter(k_tiles - 1, False, False)


def _schedule_buffered_single_stage(s: _KernelState):
    """Scheduler directives for one stages=1 buffered iteration.

    Forces each ds_write of the next tile into the tail of the MFMA run.
    """
    param = s.param
    m_rep = param.block_m // param.m_waves // param.mma_m
    n_rep = param.block_n // param.n_waves // param.mma_n
    n_mfma = m_rep * n_rep * (param.block_k // param.mma_k)
    n_dswr = param.ldg_a_iters + param.ldg_b_iters
    if n_mfma > n_dswr:
        rocdl.sched_mfma(n_mfma - n_dswr)
        for _ in range(n_dswr):
            rocdl.sched_dswr(1)
            rocdl.sched_mfma(1)


@flyc.jit
def _main_loop_buffered(s: _KernelState):
    """128-bit buffer_load (global->reg) + ds_write (reg->LDS), depth-1 prefetch.

    The next tile's vmem loads are in flight during the current tile's MFMAs;
    the ds_write commits them once vmcnt drains. stages=1 overwrites the single
    LDS buffer once the current tile is in registers; stages=2 double-buffers.
    """
    stages = s.param.stages
    k_tiles = s.k_tiles
    frag_A = s.frag_A
    frag_B = s.frag_B
    read_tile_into = s.read_tile_into
    load_ab_to_reg = s.load_ab_to_reg
    commit_ab_to_lds = s.commit_ab_to_lds
    mma_tile = s.mma_tile
    load_ab_to_reg(fx.Int32(0))
    rocdl.s_waitcnt(vmcnt=0)
    commit_ab_to_lds(0)
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.s_barrier()
    for k_tile in range(0, k_tiles - 1, 1):
        write_stage = (k_tile + 1) % stages
        # 128-bit buffer_load of tile k+1 (vmem, overlaps the MFMAs below)
        load_ab_to_reg(k_tile + 1)
        read_tile_into(k_tile % stages, s.frag_A_ret, s.frag_B_ret)
        if const_expr(stages == 1):
            # Single LDS buffer, but tile k's operands are already live
            # in registers. Once those ds_reads land, overwrite LDS with
            # tile k+1 while mma_tile consumes only registers.
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
            rocdl.s_waitcnt(vmcnt=0)
            commit_ab_to_lds(write_stage)
            mma_tile(frag_A, frag_B)
            _schedule_buffered_single_stage(s)
            rocdl.s_waitcnt(lgkmcnt=0)  # tile k+1 writes landed
            rocdl.s_barrier()  # writes visible before next read
        else:
            mma_tile(frag_A, frag_B)
            rocdl.s_waitcnt(vmcnt=0)
            # ds_write tile k+1 into the other buffer (double-buffered)
            commit_ab_to_lds(write_stage)
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
        rocdl.sched_barrier(0)
    read_tile_into((k_tiles - 1) % stages, s.frag_A_ret, s.frag_B_ret)
    mma_tile(frag_A, frag_B)


def _prime_async_stages(s: _KernelState):
    """Preload the first stages-1 K tiles with async global->LDS copies."""
    for stage in range(s.param.stages - 1):
        s.async_load_b_to_lds(stage, stage)
        s.async_load_a_to_lds(stage, stage)
        rocdl.asyncmark()
    rocdl.sched_barrier(0)


@flyc.jit
def _main_loop_async(s: _KernelState):
    """Async global->LDS DMA over ``stages`` buffers, reading the current tile inline.

    Needs k_tiles >= stages - 1 (checked on the host).
    """
    # Method calls on a name inside a dynamic loop mark it loop-carried, so
    # every helper the loops call is bound to a local first.
    stages = s.param.stages
    compute_stage = s.compute_stage
    async_load_a_to_lds = s.async_load_a_to_lds
    async_load_b_to_lds = s.async_load_b_to_lds
    _prime_async_stages(s)
    main_loop_end = s.k_tiles - (stages - 1)
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
    for i in range_constexpr(0, stages - 1):
        rocdl.wait_asyncmark(stages - 2 - i)
        rocdl.s_barrier()
        compute_stage(current_stage)
        current_stage = (current_stage + 1) % stages


@flyc.jit
def _main_loop_async_reg_db(s: _KernelState):
    """Async DMA with a register double buffer (stages >= 3).

    Tile k+1's LDS->reg read goes into the prefetch fragments *before* tile k's
    MFMAs, so the read stays in flight during the MFMAs and its lgkmcnt wait
    lands on the next iteration's work. stages >= 3 keeps the stage being read
    (k+1) apart from the one being async-written (k+stages-1).
    Needs k_tiles >= stages - 1 (checked on the host).
    """
    stages = s.param.stages
    frag_A = s.frag_A
    frag_B = s.frag_B
    frag_pf_A = s.frag_pf_A
    frag_pf_B = s.frag_pf_B
    read_tile_into = s.read_tile_into
    mma_tile = s.mma_tile
    async_load_a_to_lds = s.async_load_a_to_lds
    async_load_b_to_lds = s.async_load_b_to_lds
    _prime_async_stages(s)
    main_loop_end = s.k_tiles - (stages - 1)
    rocdl.wait_asyncmark(stages - 2)
    rocdl.s_barrier()
    read_tile_into(0, s.frag_A_ret, s.frag_B_ret)
    for k_tile in range(0, main_loop_end, 1):
        write_stage = (k_tile + stages - 1) % stages
        async_load_b_to_lds(k_tile + (stages - 1), write_stage)
        async_load_a_to_lds(k_tile + (stages - 1), write_stage)
        rocdl.asyncmark()
        rocdl.wait_asyncmark(stages - 2)
        rocdl.s_barrier()
        read_tile_into((k_tile + 1) % stages, s.frag_pf_A_ret, s.frag_pf_B_ret)
        mma_tile(frag_A, frag_B)
        # Rotate the just-read next tile into the MFMA fragments. The ds_read
        # above is only consumed here (after the MFMAs), so its lgkmcnt wait
        # is hidden behind the current tile's MFMAs.
        frag_A.store(frag_pf_A.load())
        frag_B.store(frag_pf_B.load())
        rocdl.sched_barrier(0)

    # Drain: compute the final stages-1 resident tiles, still prefetching one
    # tile ahead (no new async loads; progressively relax the wait count).
    for i in range_constexpr(0, stages - 1):
        if const_expr(i < stages - 2):
            rocdl.wait_asyncmark(stages - 2 - (i + 1))
            rocdl.s_barrier()
            read_tile_into(
                (main_loop_end + (i + 1)) % stages,
                s.frag_pf_A_ret,
                s.frag_pf_B_ret,
            )
        mma_tile(frag_A, frag_B)
        if const_expr(i < stages - 2):
            frag_A.store(frag_pf_A.load())
            frag_B.store(frag_pf_B.load())
        rocdl.sched_barrier(0)


@flyc.jit
def _reduce_k_waves(frag_C, k_waves, k_wave_idx, tid_mn, threads_per_k_slice, smem_red):
    """Tree-reduce the K waves' fp32 partial tiles into K wave 0.

    The exchange goes through k_waves // 2 LDS slots. Peer waves with the same
    tid_mn hold identically laid out fragments, so it is fragment-ordered in
    128-bit lane-interleaved chunks (ds_write_b128 / ds_read_b128). Waves that
    are already folded keep adding a peer slot; their sums are never read again.
    """
    n_vec4 = const_expr(fx.size(frag_C.shape).unpack() // 4)
    red = fx.make_view(
        smem_red,
        fx.make_layout(
            (4, threads_per_k_slice, n_vec4, k_waves // 2),
            (1, 4, 4 * threads_per_k_slice, 4 * threads_per_k_slice * n_vec4),
        ),
    )
    lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    frag_chunks = fx.make_layout((4, n_vec4), (1, 4))
    acc_chunks = fx.make_view(fx.get_iter(frag_C), frag_chunks)
    peer = fx.make_fragment_like(frag_C)
    peer_chunks = fx.make_view(fx.get_iter(peer), frag_chunks)

    for level in range_constexpr(k_waves.bit_length() - 1):
        active = const_expr(k_waves >> level)
        half = const_expr(active // 2)
        gpu.barrier()
        if (k_wave_idx >= half) & (k_wave_idx < active):
            fx.copy(lds_atom, acc_chunks, red[None, tid_mn, None, k_wave_idx - half])
        gpu.barrier()
        fx.copy(lds_atom, red[None, tid_mn, None, k_wave_idx % half], peer_chunks)
        frag_C.store(frag_C.load() + peer.load())


@flyc.jit
def _reduce_split_k(
    frag_C,
    owns_tile,
    tid,
    tid_mn,
    threads_per_k_slice,
    ks_idx,
    split_k,
    num_tiles,
    semaphore,
    workspace,
    smem_flag,
):
    """Sum the split-K partial tiles in the split that arrives last.

    Each split stores its fp32 partial tile in fragment order (128-bit,
    lane-interleaved, coalesced), bumps the tile's arrival counter, and the
    last split to arrive sums every partial. Partials are written and read with
    sc0 sc1 so they are coherent across XCDs without agent-scope fences: those
    lower to a full L2 writeback/invalidate that would evict every workgroup's
    A/B reuse. Returns (owns_tile, is_last) for the epilogue.
    """
    n_vec4 = const_expr(fx.size(frag_C.shape).unpack() // 4)
    tile_id = fx.block_idx.x
    # (4, lane, chunk, split, tile): each lane's fragment as 128-bit chunks,
    # lane-interleaved so a chunk index is one coalesced store per wave.
    ws_stride = 4 * threads_per_k_slice * n_vec4
    ws = fx.make_view(
        fx.get_iter(fx.rocdl.make_buffer_tensor(workspace, max_size=True)),
        fx.make_layout(
            (4, threads_per_k_slice, n_vec4, split_k, num_tiles),
            (1, 4, 4 * threads_per_k_slice, ws_stride, ws_stride * split_k),
        ),
    )
    ws_atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy128b(cache_modifier=CPOL_SC0_SC1), fx.Float32
    )
    frag_chunks = fx.make_layout((4, n_vec4), (1, 4))

    if owns_tile:
        fx.copy(
            ws_atom,
            fx.make_view(fx.get_iter(frag_C), frag_chunks),
            ws[None, tid_mn, None, ks_idx, tile_id],
        )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        # Agent scope: the arrival counter is shared across XCDs.
        sem_ptr = fx.add_offset(fx.get_iter(semaphore), tile_id)
        arrived = fx.llvm.atomic_add(
            sem_ptr,
            fx.Int32(1),
            syncscope=fx.rocdl.SyncScope.Agent,
            ordering=fx.AtomicOrdering.Monotonic,
        )
        fx.ptr_store(arrived, smem_flag)
        if arrived == split_k - 1:
            fx.llvm.generic_store(
                sem_ptr,
                fx.Int32(0),
                memory_order=fx.AtomicOrdering.Monotonic,
                syncscope=fx.rocdl.SyncScope.Agent,
            )
    gpu.barrier()
    is_last = fx.Int32(fx.ptr_load(smem_flag)) == split_k - 1
    owns_tile = owns_tile & is_last
    if is_last:
        peer = fx.make_fragment_like(frag_C)
        peer_chunks = fx.make_view(fx.get_iter(peer), frag_chunks)
        frag_C.fill(0.0)
        for split in range(0, split_k, 1):
            fx.copy(ws_atom, ws[None, tid_mn, None, split, tile_id], peer_chunks)
            frag_C.store(frag_C.load() + peer.load())
    return owns_tile, is_last


@flyc.jit
def _store_direct(frag_C, gC, thr_copy_C, c_r2g_atom, owns_tile, gated, is_fp32_output):
    """Store the MMA fragment straight to global memory, skipping the LDS
    staging and its barriers. Needs N to be a block_n multiple."""
    if const_expr(is_fp32_output):
        frag_C_out = frag_C
    else:
        frag_C_out = fx.make_fragment_like(frag_C, fx.BFloat16)
        frag_C_out.store(frag_C.load().to(fx.BFloat16))
    if const_expr(gated):
        c_src = thr_copy_C.retile(frag_C_out)
        c_dst = thr_copy_C.partition_S(gC)
        if owns_tile:
            fx.copy(c_r2g_atom, c_src, c_dst)
    else:
        fx.copy(
            c_r2g_atom,
            thr_copy_C.retile(frag_C_out),
            thr_copy_C.partition_S(gC),
        )


@flyc.jit
def _store_cshuffle(
    param: GemmA16W16Gfx942Param,
    frag_C,
    thr_copy_C_lds,
    sC_write,
    smem_c,
    out,
    out_buf,
    tid,
    m,
    n,
    block_m_offset,
    block_n_offset,
    owns_tile,
    is_last,
    gated,
    is_split_k,
    is_fp32_output,
    splitk_protocol,
    expected_signal,
):
    """Stage the C tile through LDS for coalesced, boundary-masked stores.

    Atomic split-K always uses it. The tile is staged in the output dtype, so
    fp32 output is not rounded to bf16. A tile over 64 KB goes through LDS in
    ``c_chunks`` row chunks.
    """
    block_m = param.block_m
    block_n = param.block_n
    block_threads = param.block_threads
    c_chunks = param.c_chunks
    cshuffle_r2g_vec_size = param.cshuffle_r2g_vec_size
    if const_expr(is_fp32_output):
        c_dtype = fx.Float32
        frag_C_out = frag_C
        c_s_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    else:
        c_dtype = fx.BFloat16
        frag_C_out = fx.make_fragment_like(frag_C, fx.BFloat16)
        frag_C_out.store(frag_C.load().to(fx.BFloat16))
        c_s_atom = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)
    # The (value, M repeat, N repeat) partitions put chunk c's rows
    # [c * chunk_rows, (c + 1) * chunk_rows) in M repeats
    # [c * chunk_reps, (c + 1) * chunk_reps).
    chunk_reps = block_m // (param.mma_m * param.m_waves) // c_chunks
    chunk_rows = block_m // c_chunks
    frag_src = thr_copy_C_lds.retile(frag_C_out)
    sC_dst = thr_copy_C_lds.partition_D(sC_write)
    cshuffle_r2g_x_threads = block_n // cshuffle_r2g_vec_size
    cshuffle_vectors = chunk_rows * block_n // cshuffle_r2g_vec_size
    cshuffle_iters = (cshuffle_vectors + block_threads - 1) // block_threads
    for chunk in range_constexpr(c_chunks):
        gpu.barrier()
        if const_expr(gated):
            if owns_tile:
                for j in range_constexpr(chunk_reps):
                    fx.copy(
                        c_s_atom,
                        frag_src[None, chunk * chunk_reps + j, None],
                        sC_dst[None, j, None],
                    )
        else:
            for j in range_constexpr(chunk_reps):
                fx.copy(
                    c_s_atom,
                    frag_src[None, chunk * chunk_reps + j, None],
                    sC_dst[None, j, None],
                )

        if const_expr(is_split_k and chunk == 0):
            splitk_protocol.wait_until_initialized(expected_signal)
        else:
            gpu.barrier()

        # Hand-indexed: row-major vectors go round-robin over all threads.
        # No TV layout expresses that when block_n // vec does not divide the
        # thread count (block_n 48, 96, 112, 160, 192); a tiled copy would
        # idle threads and add a pass.
        for i in range_constexpr(cshuffle_iters):
            vector_idx = block_threads * i + tid
            if vector_idx < cshuffle_vectors:
                local_row = vector_idx // cshuffle_r2g_x_threads
                local_col = vector_idx % cshuffle_r2g_x_threads * cshuffle_r2g_vec_size
                global_row = block_m_offset + chunk * chunk_rows + local_row
                global_col = block_n_offset + local_col
                global_offset = global_row * n + global_col
                in_rows = (global_row < m) and is_last
                if in_rows and (global_col + cshuffle_r2g_vec_size <= n):
                    c_vec = fx.ptr_load(
                        smem_c + local_row * block_n + local_col,
                        result_type=fx.Vector.make_type(cshuffle_r2g_vec_size, c_dtype),
                    )
                    write_cshuffle_vec_to_global(
                        out,
                        out_buf,
                        global_offset,
                        c_vec,
                        is_split_k,
                        is_fp32_output,
                    )
                # Ragged N: the vector straddling column n is stored element by
                # element, else it would spill into the next row.
                if (
                    in_rows
                    and (global_col < n)
                    and (global_col + cshuffle_r2g_vec_size > n)
                ):
                    for e in range_constexpr(cshuffle_r2g_vec_size):
                        if global_col + e < n:
                            c_elem = fx.ptr_load(
                                smem_c + local_row * block_n + local_col + e,
                                result_type=fx.Vector.make_type(1, c_dtype),
                            )
                            write_cshuffle_vec_to_global(
                                out,
                                out_buf,
                                global_offset + e,
                                c_elem,
                                is_split_k,
                                is_fp32_output,
                            )


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
    else:
        splitk_protocol = None

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
            rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(tid // threads_per_k_slice))
        )
    else:
        tid_mn = tid
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    # XCD-aware block remap, active when group_m > 0. NUM_PIDS_THRESHOLD=0
    # applies it to grids of any size (the grid must still divide across the
    # 8 XCDs): small narrow-N grids such as 32x6 blocks need it for L2 reuse.
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=0, GROUP_M=param.group_m
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
        c: fx.Array[
            global_output_dtype,
            1 if direct_store else block_m // c_chunks * block_n,
            16,
        ]
        red: fx.Array[fx.Float32, max(1, (k_waves // 2) * block_m * block_n), 16]
        flag: fx.Array[fx.Int32, 4, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    # LDS stage s of the A/B tiles starts at s * tile size from these bases.
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr
    smem_red = storage.red.peek().ptr
    smem_flag = storage.flag.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    # Direct stores rely on the descriptor's range check: rows past M fall
    # beyond the buffer end and are dropped, but columns past N would wrap
    # into the next row, so direct_store needs N to be a block_n multiple.
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=not direct_store)

    if const_expr(is_split_k):
        splitk_protocol.init(
            semaphore,
            signal,
            out,
            None,
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

    ctx = make_gfx942_load_context(
        elem_dtype, load_tid=tid, ks_begin=ks_begin, param=param
    )
    s2r_copy_atom = ctx.uni_copy_atom

    # The LDS writers address the whole (rows, block_k) tile; each K wave reads
    # its (rows, block_k_w) slice of the k_waves mode.
    a_lds_layout = make_k_sliced_lds_layout(block_m, block_k, k_waves)
    b_lds_layout = make_k_sliced_lds_layout(block_n, block_k, k_waves)
    k_slice = k_wave_idx if const_expr(k_waves > 1) else 0

    def k_wave_view(base, lds_layout):
        return fx.make_view(base, lds_layout)[None, (None, k_slice)]

    c_lds_layout = fx.make_layout((block_m // c_chunks, block_n), (block_n, 1))

    thr_mma = tiled_mma.thr_slice(tid_mn)
    thr_copy_A = fx.make_tiled_copy_A(ctx.buffer_copy_atom, tiled_mma).get_slice(tid_mn)
    thr_copy_B = fx.make_tiled_copy_B(ctx.buffer_copy_atom, tiled_mma).get_slice(tid_mn)

    gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]
    sC_write = fx.make_view(smem_c, c_lds_layout)

    sA0 = k_wave_view(smem_a, a_lds_layout)
    sB0 = k_wave_view(smem_b, b_lds_layout)
    frag_A = thr_mma.make_fragment_A(sA0)
    frag_B = thr_mma.make_fragment_B(sB0)
    frag_C = thr_mma.make_fragment_C(gC)
    frag_A_ret = thr_copy_A.retile(frag_A)
    frag_B_ret = thr_copy_B.retile(frag_B)
    thr_copy_C_direct = None
    c_r2g_atom = None
    if const_expr(direct_store):
        if const_expr(is_fp32_output):
            c_r2g_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        else:
            c_r2g_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), elem_dtype)
        thr_copy_C_direct = fx.make_tiled_copy_C(c_r2g_atom, tiled_mma).get_slice(
            tid_mn
        )
    # Register double buffer of the async pipeline with stages >= 3: the next
    # K-tile's LDS->reg read lands here while frag_A/frag_B feed the MFMAs.
    frag_pf_A = frag_pf_B = frag_pf_A_ret = frag_pf_B_ret = None
    if const_expr(stages >= 3):
        frag_pf_A = thr_mma.make_fragment_A(sA0)
        frag_pf_B = thr_mma.make_fragment_B(sB0)
        frag_pf_A_ret = thr_copy_A.retile(frag_pf_A)
        frag_pf_B_ret = thr_copy_B.retile(frag_pf_B)

    # Buffered-load tiled copies: 128-bit buffer_load global->reg, then a
    # layout-aware reg->LDS copy into the swizzled LDS tile.
    load_ab_to_reg = commit_ab_to_lds = None
    if const_expr(param.buffered_load):
        vec_per_thread = param.async_load_bytes // param.in_data_bytes
        buffer_atom = ctx.buffer_copy_atom
        uni_atom = ctx.uni_copy_atom

        def make_g2s_tv(rows):
            thrs_k = block_k // vec_per_thread
            thrs_r = block_threads // thrs_k
            tv = fx.make_layout(
                ((thrs_k, thrs_r), (1, vec_per_thread)),
                ((thrs_r * vec_per_thread, 1), (1, thrs_r)),
            )
            return tv, thrs_r

        # The descriptors span each operand's real extent, so edge tiles never
        # read past the allocation; rows past M/N only feed unstored outputs.
        # A: tile (block_m, block_k), global row-block bid_m, all K-tiles.
        a_tv, a_thrs_r = make_g2s_tv(block_m)
        a_bounded = fx.rocdl.make_buffer_tensor(a, max_size=False)
        gA_all = fx.flat_divide(a_bounded, (block_m, block_k))[None, None, bid_m, None]
        a_g2r = fx.make_tiled_copy(
            buffer_atom, a_tv, fx.make_tile(a_thrs_r, block_k)
        ).get_slice(tid)
        a_r2s = fx.make_tiled_copy(
            uni_atom, a_tv, fx.make_tile(a_thrs_r, block_k)
        ).get_slice(tid)
        gA_thr = a_g2r.partition_S(gA_all)
        a_cp_frag = fx.make_fragment_like(gA_thr[None, None, None, 0])
        a_cp_frag_ret = a_r2s.retile(a_cp_frag)

        # B: logical (N, K) view of the [K, N] weight (strides (b_leading_stride, 1)),
        # tile (block_n, block_k), global row-block bid_n, all K-tiles.
        b_nk_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(b), fx.make_layout((n, k), (b_leading_stride, 1))),
            max_size=False,
        )
        b_tv, b_thrs_r = make_g2s_tv(block_n)
        gB_all = fx.flat_divide(b_nk_buf, (block_n, block_k))[None, None, bid_n, None]
        b_g2r = fx.make_tiled_copy(
            buffer_atom, b_tv, fx.make_tile(b_thrs_r, block_k)
        ).get_slice(tid)
        b_r2s = fx.make_tiled_copy(
            uni_atom, b_tv, fx.make_tile(b_thrs_r, block_k)
        ).get_slice(tid)
        gB_thr = b_g2r.partition_S(gB_all)
        b_cp_frag = fx.make_fragment_like(gB_thr[None, None, None, 0])
        b_cp_frag_ret = b_r2s.retile(b_cp_frag)
        buffered_k_base = ks_begin // block_k

        def load_ab_to_reg(k_tile):
            global_k_tile = buffered_k_base + k_tile
            fx.copy(buffer_atom, gA_thr[None, None, None, global_k_tile], a_cp_frag)
            fx.copy(buffer_atom, gB_thr[None, None, None, global_k_tile], b_cp_frag)

        def commit_ab_to_lds(lds_stage):
            sA_w = a_r2s.partition_D(
                fx.make_view(smem_a + lds_stage * block_m * block_k, a_lds_layout)
            )
            sB_w = b_r2s.partition_D(
                fx.make_view(smem_b + lds_stage * block_n * block_k, b_lds_layout)
            )
            fx.copy(uni_atom, a_cp_frag_ret, sA_w)
            fx.copy(uni_atom, b_cp_frag_ret, sB_w)

    thr_copy_C_lds = fx.make_tiled_copy_C(
        fx.make_copy_atom(
            (
                fx.UniversalCopy32b()
                if const_expr(is_fp32_output)
                else fx.UniversalCopy16b()
            ),
            global_output_dtype,
        ),
        tiled_mma,
    ).get_slice(tid_mn)

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
            is_k_major=False,
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
            is_k_major=False,
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
            k_wave_view(smem_a + read_stage * block_m * block_k, a_lds_layout)
        )
        thr_sB = thr_copy_B.partition_S(
            k_wave_view(smem_b + read_stage * block_n * block_k, b_lds_layout)
        )
        for c in range_constexpr(c_begin, c_end):
            fx.copy(s2r_copy_atom, thr_sA[None, None, c], a_ret[None, None, c])
            fx.copy(s2r_copy_atom, thr_sB[None, None, c], b_ret[None, None, c])

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
        """Compute one LDS stage, issuing chunk c+1's ds_read before chunk c's MFMAs."""
        thr_sA = thr_copy_A.partition_S(
            k_wave_view(smem_a + read_stage * block_m * block_k, a_lds_layout)
        )
        thr_sB = thr_copy_B.partition_S(
            k_wave_view(smem_b + read_stage * block_n * block_k, b_lds_layout)
        )

        def rd(c):
            fx.copy(s2r_copy_atom, thr_sA[None, None, c], frag_A_ret[None, None, c])
            fx.copy(s2r_copy_atom, thr_sB[None, None, c], frag_B_ret[None, None, c])

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

    state = _KernelState(
        param=param,
        tiled_mma=tiled_mma,
        k_tiles=k_tiles,
        copy_chunks=copy_chunks,
        mma_k_iters=mma_k_iters,
        num_frgv=num_frgv,
        frag_A=frag_A,
        frag_B=frag_B,
        frag_A_ret=frag_A_ret,
        frag_B_ret=frag_B_ret,
        frag_pf_A=frag_pf_A,
        frag_pf_B=frag_pf_B,
        frag_pf_A_ret=frag_pf_A_ret,
        frag_pf_B_ret=frag_pf_B_ret,
        frag_C=frag_C,
        read_tile_into=read_tile_into,
        mma_tile=mma_tile,
        compute_stage=compute_stage,
        load_ab_to_reg=load_ab_to_reg,
        commit_ab_to_lds=commit_ab_to_lds,
        async_load_a_to_lds=async_load_a_to_lds,
        async_load_b_to_lds=async_load_b_to_lds,
    )

    frag_C.fill(0.0)
    expected_signal = None
    if const_expr(is_split_k):
        splitk_protocol.zero_c()
        expected_signal = splitk_protocol.take_ticket(split_k)
        splitk_protocol.publish(expected_signal)

    if const_expr(param.pingpong):
        _main_loop_pingpong(state)
    elif const_expr(param.buffered_load):
        _main_loop_buffered(state)
    elif const_expr(stages < 3):
        _main_loop_async(state)
    else:
        _main_loop_async_reg_db(state)

    if const_expr(k_waves > 1):
        _reduce_k_waves(
            frag_C, k_waves, k_wave_idx, tid_mn, threads_per_k_slice, smem_red
        )

    # Lanes that own the final tile: K wave 0, and with split_k_reduce only in
    # the split that arrives last for this tile.
    gated_epilogue = const_expr(k_waves > 1 or split_k_reduce)
    if const_expr(k_waves > 1):
        owns_tile = k_wave_idx == 0
    else:
        owns_tile = fx.Int32(1) == 1
    is_last = fx.Int32(1) == 1
    if const_expr(split_k_reduce):
        owns_tile, is_last = _reduce_split_k(
            frag_C,
            owns_tile,
            tid,
            tid_mn,
            threads_per_k_slice,
            ks_idx,
            split_k,
            num_pid_m * num_pid_n,
            semaphore,
            workspace,
            smem_flag,
        )

    if const_expr(direct_store):
        _store_direct(
            frag_C,
            gC,
            thr_copy_C_direct,
            c_r2g_atom,
            owns_tile,
            gated_epilogue,
            is_fp32_output,
        )
    else:
        _store_cshuffle(
            param,
            frag_C,
            thr_copy_C_lds,
            sC_write,
            smem_c,
            out,
            out_buf,
            tid,
            m,
            n,
            block_m_offset,
            block_n_offset,
            owns_tile,
            is_last,
            gated_epilogue,
            is_split_k,
            is_fp32_output,
            splitk_protocol,
            expected_signal,
        )


def named_gemm_a16w16_gfx942_kernel(name: str):
    """``gemm_a16w16_gfx942_kernel`` as a ``flyc.kernel`` named ``name`` for
    profilers. Each wraps a fresh copy of the function, since ``flyc.kernel``
    rewrites the code of the function it decorates.

    A plain function, not cached: FlyDSL's JIT cache key only follows plain
    functions and rejects mutated globals. It runs once per compile.
    """
    body = gemm_a16w16_gfx942_kernel
    fresh = types.FunctionType(
        body.__code__, body.__globals__, body.__name__, body.__defaults__
    )
    return flyc.kernel(fresh, name=name)


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
    a_leading_stride = fx.Int32(fx.get_scalar(a.stride[0]))
    b_leading_stride = fx.Int32(fx.get_scalar(b.stride[1]))
    mma_atom = fx.make_mma_atom(
        fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, fx.BFloat16)
    )
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
    named_gemm_a16w16_gfx942_kernel(make_gemm_a16w16_gfx942_kernel_name(param))(
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
    return flyc.from_dlpack(tensor.detach()).mark_layout_dynamic(
        leading_dim=leading_dim
    )


@functools.cache
def get_split_k_buffers(stream_handle: int, device, split_k, split_k_reduce):
    """Split-K tile semaphores and signals, one set per split_k and split_k_reduce.

    Also one per stream, since launches on two streams may overlap. Keyed by
    the raw stream handle (``cuda_stream``), so no stream object is kept alive.
    PyTorch recycles a fixed pool of handles per device, which bounds the sets
    of its streams; a stream reusing a destroyed stream's handle inherits its
    idle set. Sets (8-12 KB each) are never evicted: a captured CUDA graph
    keeps using their addresses.

    Zeroed once, never reset by the host. Atomic split-K (``SplitKProtocol``)
    never resets its 64-bit tickets: each launch advances them by split_k,
    which is how a split derives the launch index, so a set shared across
    split_k values leaves splits waiting for a tag that never comes. The
    split_k_reduce arrival counters are 32-bit and reset by the last split,
    which would corrupt the tickets, so they get their own set.
    """
    sem_dtype = torch.int32 if split_k_reduce else torch.int64
    with persistent_alloc(torch.device(device)):
        semaphore = torch.zeros((SPLIT_K_MAX_TILES,), dtype=sem_dtype, device=device)
        signal = torch.zeros((SPLIT_K_MAX_TILES,), dtype=torch.int32, device=device)
    return semaphore, signal


# Search space of the fallback config and of the tuner's pingpong screen:
# tiles with their (m, n, k) wave layouts. The cost model cannot tell
# wave layouts or block_k apart, so earlier entries win its ties.
CANDIDATE_TILES_M = (32, 64, 128, 256)
CANDIDATE_TILES_N = (32, 48, 64, 96, 112, 128, 160, 192, 256)
CANDIDATE_WAVES = ((2, 2, 1), (2, 1, 2), (1, 2, 2), (1, 4, 1), (4, 1, 1), (1, 1, 4))
CANDIDATE_MAX_SPLIT_K = 8
PINGPONG_PIPELINE = {
    "stages": 1,
    "buffered_load": True,
    "async_load_bytes": 16,
    "pingpong": True,
}


def pingpong_fill_cost_us(
    bm: int, bn: int, k_waves: int, split_k: int, k_tiles: int
) -> float:
    """Modelled runtime of one round of single-CU pingpong workgroups.

    Least-squares fit (14% rms) to ~1500 graph-timed pingpong configs on MI300X
    (N=256..2048, K=2048..10240): one bk128 tile costs 0.0205us per MFMA per
    SIMD plus 0.23us, and the split-K workspace reduction adds 2.86us plus
    54us per MB of fp32 partials. A tile is never faster than streaming its
    (bm + bn) x 128 bf16 operands at ~35 GB/s per CU; without that floor the
    fit prefers several rounds of 64x64 tiles over one of 128x128, which runs
    ~25% slower at M >= 768.
    """
    t_tile = max(0.0205 * (bm * bn / 128) + 0.23, (bm + bn) * 256 / 35e3)
    per_split = -(-k_tiles // split_k)
    if k_waves > 1 and split_k > 1 and bm * bn <= 64 * 64:
        # Small k_waves=2 tiles save ~7% per K tile but pay a fixed LDS
        # reduction of ~1.7 tiles, so they win on long splits.
        cost = 1.8 + (0.93 * per_split + 1.7) * t_tile
    else:
        if k_waves > 1 and split_k > 1:
            # Larger k_waves=2 tiles measured ~11% slower per K tile.
            t_tile *= 1.107
        cost = 1.8 + per_split * t_tile
    if split_k > 1:
        cost += 2.86 + 54.25e-6 * split_k * bm * bn * 4
    return cost


def model_cost_us(m: int, n: int, k: int, cfg: dict) -> float:
    """Busiest-CU estimate: rounds of workgroups x the fitted per-round cost."""
    tiles = -(-m // cfg["block_m"]) * -(-n // cfg["block_n"])
    rounds = -(-(tiles * cfg["split_k"]) // GFX942_NUM_CUS)
    return rounds * pingpong_fill_cost_us(
        cfg["block_m"], cfg["block_n"], cfg["k_waves"], cfg["split_k"], max(k // 128, 1)
    )


def shape_error(m: int, n: int, k: int, cfg: dict) -> str | None:
    """Why ``cfg`` cannot run an (m, n, k) problem beyond the param checks, or None."""
    cfg = {**CONFIG_DEFAULTS, **cfg}
    block_k = cfg["block_k"]
    split_k = cfg["split_k"]
    if k % block_k:
        return f"K={k} must be divisible by block_k={block_k} (K-tail unsupported)"
    if cfg["direct_store"] and n % cfg["block_n"]:
        return "direct_store requires N to be a block_n multiple"
    # SplitKProtocol.zero_c clears C in whole fp32x4 vectors.
    if split_k > 1 and not cfg["split_k_reduce"] and n % 4:
        return f"atomic split-K requires N to be a multiple of 4 (got N={n})"
    k_tiles = k // block_k
    last_split_tiles = k_tiles - (split_k - 1) * -(-k_tiles // split_k)
    if last_split_tiles < 1:
        return (
            f"split_k={split_k} leaves an empty K partition "
            f"(K={k}, block_k={block_k})"
        )
    if split_k > 1:
        tiles = -(-m // cfg["block_m"]) * -(-n // cfg["block_n"])
        if tiles > SPLIT_K_MAX_TILES:
            return (
                f"split-K needs {tiles} tile semaphores, "
                f"more than {SPLIT_K_MAX_TILES}"
            )
    if cfg["pingpong"]:
        if last_split_tiles < 2:
            return (
                f"pingpong needs at least 2 K tiles per split "
                f"(K={k}, split_k={split_k}, block_k={block_k})"
            )
    elif not cfg["buffered_load"]:
        stages = cfg["stages"]
        if last_split_tiles < stages - 1:
            return (
                f"async stages={stages} needs at least {stages - 1} K tiles per "
                f"split (K={k}, split_k={split_k}, block_k={block_k})"
            )
    return None


def config_is_valid(
    m: int, n: int, k: int, cfg: dict, out_dtype_id: int = GEMM_A16W16_DTYPE_BF16
) -> bool:
    if shape_error(m, n, k, cfg) is not None:
        return False
    try:
        make_gemm_a16w16_gfx942_param(out_dtype_id=out_dtype_id, **cfg)
    except (ValueError, AssertionError, NotImplementedError):
        return False
    return True


def pingpong_candidates(m: int, n: int, k: int) -> list[tuple[float, dict]]:
    """Pingpong configs of a shape with their modelled cost.

    Sorted cheapest first and not yet checked by ``config_is_valid``. Split-K
    only fills one round of workgroups, never more, and leaves every split at
    least two K tiles.
    """
    block_ks = (128, 64, 256) if k >= 2048 else (64, 128, 256)
    ranked = []
    order = 0
    for bm, bn, (mw, nw, kw), bk, sk in itertools.product(
        CANDIDATE_TILES_M,
        CANDIDATE_TILES_N,
        CANDIDATE_WAVES,
        block_ks,
        range(1, CANDIDATE_MAX_SPLIT_K + 1),
    ):
        order += 1
        if bm % (mw * 16) or bn % (nw * 16) or k % bk or bk % (kw * 64):
            continue
        if bm > max(2 * m, 32) or bn > max(2 * n, 32):
            continue
        tiles = -(-m // bm) * -(-n // bn)
        if sk > 1 and tiles * sk > GFX942_NUM_CUS:
            continue
        k_tiles = k // bk
        if k_tiles - (sk - 1) * -(-k_tiles // sk) < 2:
            continue
        cfg = {
            "block_m": bm,
            "block_n": bn,
            "block_k": bk,
            "split_k": sk,
            "m_waves": mw,
            "n_waves": nw,
            "k_waves": kw,
            "group_m": 4,
            "split_k_reduce": sk > 1,
            "direct_store": False,
            **PINGPONG_PIPELINE,
        }
        ranked.append((model_cost_us(m, n, k, cfg), order, cfg))
    ranked.sort(key=lambda item: item[:2])
    return [(cost, cfg) for cost, _, cfg in ranked]


def fallback_eligible(n: int, cfg: dict) -> bool:
    """Candidates the cost model misranks, kept out of the fallback only.

    Measured on 160 untuned shapes: per-wave tiles above 64x112 spill
    accumulators, tiles more than 3x wider than tall (or 4x taller than wide)
    lose to squarer ones, and non-power-of-two block_n that leaves a partial
    N tile loses to a power of two.
    """
    bm, bn = cfg["block_m"], cfg["block_n"]
    if (bm // cfg["m_waves"]) * (bn // cfg["n_waves"]) > 64 * 112:
        return False
    if bn > 3 * bm or bm > 4 * bn:
        return False
    return not (bn & (bn - 1) and n % bn)


@functools.cache
def _fallback_config(m: int, n: int, k: int, out_dtype_id: int) -> dict:
    for _, cfg in pingpong_candidates(m, n, k):
        if fallback_eligible(n, cfg) and config_is_valid(m, n, k, cfg, out_dtype_id):
            return cfg
    # Fewer than two block_k=64 tiles: the async pipeline, which needs one.
    return {
        "block_m": 32 if m <= 32 else 64,
        "block_n": 32 if n <= 32 else 64,
        "block_k": 64,
        "stages": 2,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 2,
        "k_waves": 1,
        "group_m": 0,
    }


def gfx942_fallback_config(
    m: int, n: int, k: int, out_dtype: torch.dtype = torch.bfloat16
) -> dict:
    """Config for a shape that is not in the tuned table: the cheapest valid,
    ``fallback_eligible`` pingpong config under ``model_cost_us``."""
    out_dtype_id = (
        GEMM_A16W16_DTYPE_FP32 if out_dtype == torch.float32 else GEMM_A16W16_DTYPE_BF16
    )
    return dict(_fallback_config(m, n, k, out_dtype_id))


_STRUCTURAL_KEYS = frozenset(
    ("block_m", "block_n", "block_k", "stages", "m_waves", "n_waves", "buffered_load")
)
_AUTO_ONLY_PIPELINE_KEYS = (
    "stages",
    "buffered_load",
    "async_load_bytes",
    "pingpong",
    "k_waves",
    "split_k_reduce",
    "direct_store",
)

# Config columns of the tuned table (aiter/configs/flydsl_gemm_a16w16_gfx942_tuned.csv).
TUNED_CONFIG_COLUMNS = tuple(CONFIG_DEFAULTS)
TUNED_BOOL_COLUMNS = frozenset(
    name for name, value in CONFIG_DEFAULTS.items() if isinstance(value, bool)
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
            cfg[name] = (
                value.lower() == "true" if name in TUNED_BOOL_COLUMNS else int(value)
            )
    return cfg


@functools.cache
def _load_tuned_table(path: str) -> dict:
    table = {}
    if not os.path.exists(path):
        return table
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["gfx"],
                int(row["cu_num"]),
                int(row["M"]),
                int(row["N"]),
                int(row["K"]),
                row["out_dtype"],
            )
            table[key] = parse_tuned_config(row)
    return table


@functools.cache
def tuned_table_path() -> str:
    """Resolved table path (env override + model_configs merge), cached per process."""
    from aiter.jit.core import AITER_CONFIGS

    return AITER_CONFIGS.AITER_CONFIG_FLYDSL_GEMM_A16W16_GFX942_FILE


def reload_tuned_table() -> None:
    """Re-read the table, e.g. after the tuner rewrote it or the env var changed."""
    tuned_table_path.cache_clear()
    _load_tuned_table.cache_clear()


@functools.cache
def _device_key(device_index: int) -> tuple[str, int]:
    props = torch.cuda.get_device_properties(device_index)
    return props.gcnArchName.split(":")[0], props.multi_processor_count


def gfx942_tuned_config(
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> dict | None:
    """The tuned-table config for this exact shape and device, or None."""
    if os.getenv("AITER_FLYDSL_GEMM_GFX942_SKIP_TABLE", "0") == "1":
        return None
    index = (
        torch.cuda.current_device()
        if device is None or device.index is None
        else device.index
    )
    gfx, cu_num = _device_key(index)
    cfg = _load_tuned_table(tuned_table_path()).get(
        (gfx, cu_num, m, n, k, tuned_out_dtype_name(out_dtype))
    )
    return dict(cfg) if cfg is not None else None


def gfx942_default_config(
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> dict:
    """What ``gemm_a16w16`` runs without overrides: the tuned row, else the
    cost-model fallback. ``AITER_FLYDSL_GEMM_GFX942_SKIP_TABLE=1`` skips the table.
    """
    cfg = gfx942_tuned_config(m, n, k, out_dtype, device)
    return cfg if cfg is not None else gfx942_fallback_config(m, n, k, out_dtype)


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

    ``layout`` follows the a16w16 convention (N = row-major, T = column-major);
    only ``nt``, the nn.Linear call (A [M,K] row-major, B = w.t() column-major),
    is supported on gfx942.

    ``user_kwargs`` override the shape's default config (``gfx942_default_config``).
    Setting a tile key (``_STRUCTURAL_KEYS``) also drops that config's pipeline
    flags, so the caller's tile runs the async pipeline unless ``user_kwargs``
    sets the flags too.
    """
    if bias is not None:
        raise NotImplementedError("gfx942 GEMM bias fusion is not implemented yet")
    if layout.lower() != "nt":
        raise ValueError(
            f"gfx942 GEMM supports the 'nt' layout only (got {layout!r}); "
            "transposed operands need CDNA4 ds_read_tr"
        )

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"a and b must be 2-D (got {a.ndim}-D and {b.ndim}-D)")
    device = a.device
    if device.type != "cuda" or b.device != device:
        raise ValueError(f"a and b must be on one GPU (got {device} and {b.device})")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"gfx942 kernel is bf16-only (got {a.dtype} and {b.dtype})")
    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError(
            f"a is [M, K] = {list(a.shape)}, b is [K, N] = {list(b.shape)}"
        )
    n = b.shape[1]
    if a.stride(1) != 1 or b.stride(0) != 1:
        raise ValueError(
            "the nt layout needs a row-major (a.stride(1) == 1) and b column-major "
            f"(b.stride(0) == 1); got strides {a.stride()} and {b.stride()}"
        )

    current = torch.cuda.current_stream()
    if stream is None:
        stream = (
            current if current.device == device else torch.cuda.current_stream(device)
        )
    elif stream.device != device:
        raise ValueError(f"stream is on {stream.device}, the operands on {device}")

    if out_dtype is None:
        out_dtype = a.dtype if out is None else out.dtype
    if out_dtype not in (a.dtype, torch.float32):
        raise ValueError(f"unsupported output dtype {out_dtype}")
    if out is not None:
        if out.dtype != out_dtype:
            raise TypeError(f"out is {out.dtype}, out_dtype is {out_dtype}")
        if out.device != device:
            raise ValueError(f"out is on {out.device}, the operands on {device}")
        if not out.is_contiguous() or out.numel() != m * n:
            raise ValueError(
                f"out must be contiguous with M * N = {m * n} elements "
                f"(got shape {list(out.shape)}, strides {out.stride()})"
            )

    # Param defaults, overlaid by the shape's tuned-table row (else the
    # cost-model fallback), then by any caller overrides.
    kwargs = dict(CONFIG_DEFAULTS)
    auto = gfx942_default_config(m, n, k, out_dtype, device)
    user_kwargs = dict(user_kwargs or {})
    if _STRUCTURAL_KEYS & user_kwargs.keys():
        # A caller-chosen tile/pipeline must not inherit flags that were only
        # valid for the default config's own tile/pipeline.
        for key in _AUTO_ONLY_PIPELINE_KEYS:
            auto.pop(key, None)
    kwargs.update(auto)
    kwargs.update(user_kwargs)
    kwargs["in_dtype_id"] = GEMM_A16W16_DTYPE_BF16
    split_k = kwargs["split_k"]
    if split_k == 1:
        kwargs["split_k_reduce"] = False
    split_k_reduce = kwargs["split_k_reduce"]

    # gfx942/CDNA3 has no packed-bf16 buffer atomic add, so bf16-output atomic
    # split-K accumulates through an f32 scratch buffer and casts to bf16 at the
    # end. f32-output split-K accumulates in place with f32 atomics.
    # split_k_reduce writes the final dtype directly.
    f32_accum_for_bf16 = (
        split_k > 1 and not split_k_reduce and out_dtype is torch.bfloat16
    )
    kwargs["out_dtype_id"] = (
        GEMM_A16W16_DTYPE_FP32
        if f32_accum_for_bf16 or out_dtype is torch.float32
        else GEMM_A16W16_DTYPE_BF16
    )

    error = shape_error(m, n, k, kwargs)
    if error is not None:
        raise ValueError(error)
    param = make_gemm_a16w16_gfx942_param(**kwargs)

    # Allocate on the launch stream, so the caching allocator orders reuse of
    # these buffers after the kernel, and copy back on it. Entering the stream
    # context costs ~8 us of host time, so it is skipped when already current.
    on_stream = (
        contextlib.nullcontext() if stream == current else torch.cuda.stream(stream)
    )
    with on_stream:
        if out is None:
            out = torch.empty((m, n), dtype=out_dtype, device=device)
        out = out.view(m, n)
        if f32_accum_for_bf16:
            kernel_out = torch.empty((m, n), dtype=torch.float32, device=device)
        else:
            kernel_out = out
        semaphore, signal = get_split_k_buffers(
            stream.cuda_stream, device, split_k, split_k_reduce
        )
        if split_k_reduce:
            # Partial tiles of every split; the last split of a tile reduces them.
            num_tiles = ((m + param.block_m - 1) // param.block_m) * (
                (n + param.block_n - 1) // param.block_n
            )
            workspace = torch.empty(
                (num_tiles * split_k * param.block_m * param.block_n,),
                dtype=torch.float32,
                device=device,
            )
        else:
            workspace = signal
        dispatch_args = (
            _dynamic_tensor_arg(kernel_out, 1),
            _dynamic_tensor_arg(a, 1),
            _dynamic_tensor_arg(b, 0),
            semaphore,
            signal,
            workspace,
            split_k,
            param,
            stream,
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
