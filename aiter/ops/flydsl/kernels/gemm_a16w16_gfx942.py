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

from dataclasses import dataclass
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

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
    get_wave_lds_offset,
)
from .gr_combine_and_mix.mfma_config import ab_k_perm
from .kernels_common import run_cached

WAVE_SIZE = GFX950_WAVE_SIZE  # 64 lanes on CDNA3
SMEM_CAPACITY_GFX942 = 65536  # 64 KB LDS / CU
GFX942_NUM_CUS = 304  # MI300X compute units (grid-fill threshold for stage picking)

GEMM_A16W16_DTYPE_BF16 = 2
GEMM_A16W16_DTYPE_FP32 = 1


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
    # derived
    async_load_bytes: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
    out_data_bytes: fx.Constexpr[int]
    cshuffle_r2g_vec_size: fx.Constexpr[int]
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
        if split_k != 1:
            raise ValueError("buffered_load currently supports split_k=1")
    if k_waves != 1:
        raise ValueError("gfx942 kernel supports k_waves=1 only (use split_k for K parallelism)")
    if split_k < 1:
        raise ValueError("split_k must be >= 1")
    if m_waves <= 0 or n_waves <= 0:
        raise ValueError("m_waves and n_waves must be positive")
    if m_waves * n_waves > 16:
        raise ValueError("the workgroup cannot contain more than 16 waves")

    in_dbytes = 2
    out_dbytes = 4 if out_dtype_id == GEMM_A16W16_DTYPE_FP32 else 2
    block_threads = m_waves * n_waves * k_waves * WAVE_SIZE

    # Register->global store vector width for the cshuffle epilogue. Split-K caps it
    # at 4 so the atomic-add granularity stays modest.
    max_cshuffle_vec = 16 // out_dbytes
    cshuffle_r2g_vec_size = min(max_cshuffle_vec, 4) if split_k > 1 else max_cshuffle_vec
    if block_n % cshuffle_r2g_vec_size != 0:
        raise ValueError("block_n must be divisible by the cshuffle store vector size")

    # LDS budget: two AB tiles per stage.
    smem_bytes = stages * (block_m + block_n) * block_k * in_dbytes
    if smem_bytes > SMEM_CAPACITY_GFX942:
        raise ValueError(
            f"staged LDS buffers exceed 64 KB: stages={stages}, block_m={block_m}, "
            f"block_n={block_n}, block_k={block_k}, smem_bytes={smem_bytes}"
        )

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
        async_load_bytes=async_load_bytes,
        in_data_bytes=in_dbytes,
        out_data_bytes=out_dbytes,
        cshuffle_r2g_vec_size=cshuffle_r2g_vec_size,
        ldg_x_threads=ldg_x_threads,
        block_threads=block_threads,
        ldg_a_iters=ldg_a_iters,
        ldg_b_iters=ldg_b_iters,
    )


def make_gemm_a16w16_gfx942_kernel_name(param: GemmA16W16Gfx942Param) -> str:
    out_suffix = "_fp32" if param.out_dtype_id == GEMM_A16W16_DTYPE_FP32 else ""
    sk = "_ksd" if param.is_split_k else "_ks1"
    dma = f"buf{param.async_load_bytes}" if param.buffered_load else str(param.async_load_bytes)
    return (
        f"hgemm_bf16{out_suffix}_gfx942_"
        f"t{param.block_m}x{param.block_n}x{param.block_k}x{param.stages}{sk}_"
        f"w{param.m_waves}x{param.n_waves}x{param.k_waves}_"
        f"gm{param.group_m}_dma{dma}"
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
    is_split_k = param.is_split_k
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    stages = param.stages
    block_threads = param.block_threads
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters
    cshuffle_r2g_vec_size = param.cshuffle_r2g_vec_size
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
        c: fx.Array[elem_dtype, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)

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

    a_lds_layout, b_lds_layout = make_gemm_ab_lds_layouts(
        block_m, block_n, block_k, param.a_is_transposed, param.b_is_transposed
    )
    c_lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))

    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(ctx.a_tiled_copy_atom, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(ctx.b_tiled_copy_atom, tiled_mma).get_slice(tid)

    gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]
    sC_write = fx.make_view(smem_c, c_lds_layout)

    sA0 = fx.make_view(smem_a, a_lds_layout)
    sB0 = fx.make_view(smem_b, b_lds_layout)
    frag_A = thr_mma.make_fragment_A(sA0)
    frag_B = thr_mma.make_fragment_B(sB0)
    frag_C = thr_mma.make_fragment_C(gC)
    frag_A_ret = thr_copy_A.retile(frag_A)
    frag_B_ret = thr_copy_B.retile(frag_B)
    # Register double-buffer (stages>=3): a second A/B fragment pair holds the
    # next K-tile's LDS->reg read while frag_A/frag_B feed the current MFMAs.
    if const_expr(stages >= 3):
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

        # A: tile (block_m, block_k), global row-block bid_m, all K-tiles.
        _a_tv, _a_thrs_r = _make_g2s_tv(block_m)
        _gA_all = fx.flat_divide(a_buf, (block_m, block_k))[None, None, bid_m, None]
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
            max_size=True,
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

        def load_ab_to_reg(k_tile):
            fx.copy(_buf_atom, _gA_thr[None, None, None, k_tile], a_cp_frag)
            fx.copy(_buf_atom, _gB_thr[None, None, None, k_tile], b_cp_frag)

        def commit_ab_to_lds(lds_stage):
            sA_w = _a_r2s.partition_D(
                fx.make_view(smem_a + lds_stage * block_m * block_k, a_lds_layout)
            )
            sB_w = _b_r2s.partition_D(
                fx.make_view(smem_b + lds_stage * block_n * block_k, b_lds_layout)
            )
            fx.copy(_uni_atom, a_cp_frag_ret, sA_w)
            fx.copy(_uni_atom, b_cp_frag_ret, sB_w)

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
    mma_k_iters = const_expr(block_k // mma_k)
    copy_chunks = const_expr(mma_k_iters // num_frgv)

    def read_tile_into(read_stage, a_ret, b_ret):
        """ds_read one K-tile from LDS stage ``read_stage`` into reg fragments."""
        thr_sA = thr_copy_A.partition_S(
            fx.make_view(smem_a + read_stage * block_m * block_k, a_lds_layout)
        )
        thr_sB = thr_copy_B.partition_S(
            fx.make_view(smem_b + read_stage * block_n * block_k, b_lds_layout)
        )
        for c in range_constexpr(copy_chunks):
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

    if const_expr(param.buffered_load):
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
            mma_tile(frag_A, frag_B)
            rocdl.s_waitcnt(vmcnt=0)
            if const_expr(stages == 1):
                # single LDS buffer: every wave must finish reading tile k
                # before tile k+1's ds_write overwrites it.
                rocdl.s_barrier()
            # ds_write tile k+1 into LDS (the other buffer when double-buffered)
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

    # Epilogue: stage the f32 accumulator (as bf16) through LDS, then do coalesced
    # vector stores to global. Split-K partitions atomically accumulate.
    frag_C_out = fx.make_fragment_like(frag_C, elem_dtype)
    frag_C_out.store(frag_C.load().to(elem_dtype))
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
    # Align each split-K partition to the async load granularity and block_k.
    split_alignment = param.async_load_bytes // param.in_data_bytes
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


def gfx942_auto_config(m: int, n: int, k: int) -> dict:
    """Best-known tile/wave/split-K config for a shape (from Phase-3 rocprofv3 tuning).

    Picks per regime: tiny-M decode/GEMV leans on split-K along K (narrow-N) or
    small tiles (wide-N); narrow-N tall-skinny ("hc-down") grows the tile with M
    and uses modest split-K at small M; wide-N short-K ("hc-up") uses big tiles;
    square compute-bound uses a 128x128 tile with 4x4 waves to hide the 32-bit
    async-DMA latency. The returned dict is a ``user_kwargs`` for ``gemm_a16w16``.
    """
    bk = 64

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

    if narrow_n:  # tall-skinny "hc-down", larger M
        # Narrow-N grids have few N-tiles, so without XCD-aware block swizzling
        # the workgroups sharing an A row-block scatter across XCDs and miss in
        # L2 (measured 44% hit vs hipBLASLt 81%). group_m groups M-tiles onto the
        # same XCD, lifting L2 hit to ~77% and cutting HBM traffic ~2.5x.
        if m <= 1024:
            # 32x32 fills the machine via split-K along the long K while the grid
            # underfills (<=304 tiles, M<=~680). Once the 32x32 grid already
            # overfills the CUs (M~1024: 384 tiles), split-K only oversubscribes;
            # a wider-N tile with deep block_k=128 instead amortizes the 32x32
            # tile's ~3.3x VALU/MFMA addressing overhead (b34: M1024 61->53us,
            # 1.90x->1.62x of hipBLASLt).
            tiles32 = ((m + 31) // 32) * ((n + 31) // 32)
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
        return dict(block_m=bm, block_n=128, block_k=bk, stages=2, split_k=1,
                    m_waves=waves, n_waves=waves)

    # square / general compute-bound
    return dict(block_m=128, block_n=128, block_k=bk, stages=2, split_k=1,
                m_waves=4, n_waves=4)


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

    # Base defaults, overlaid by the per-shape tuned auto-config (Phase-3
    # rocprofv3 tuning: tile/wave/split-K + XCD swizzle group_m), then by any
    # caller overrides. Without the auto-config the narrow-N "hc-down" grids run
    # the generic 128x128 no-swizzle default (~4x hipBLASLt at M2048); the
    # auto-config picks block_m=64/stages=4/group_m and closes most of that gap.
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
    kwargs.update(gfx942_auto_config(m, n, k))
    kwargs.update(user_kwargs or {})
    kwargs["a_is_transposed"] = a_is_transposed
    kwargs["b_is_transposed"] = b_is_transposed
    kwargs["in_dtype_id"] = GEMM_A16W16_DTYPE_BF16
    split_k = kwargs["split_k"]

    # gfx942/CDNA3 has no packed-bf16 buffer atomic add, so bf16-output split-K
    # accumulates through an f32 scratch buffer and casts to bf16 at the end.
    # f32-output split-K accumulates in place with f32 atomics.
    f32_accum_for_bf16 = split_k > 1 and out.dtype is torch.bfloat16
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
    if split_k > 1:
        # Keep every split-K partition an exact, block_k-aligned slice (no K-tail).
        if k % (split_k * kwargs["block_k"]) != 0:
            raise ValueError(
                f"K={k} must be divisible by split_k*block_k="
                f"{split_k * kwargs['block_k']} for split-K"
            )
        num_pid_m = (m + kwargs["block_m"] - 1) // kwargs["block_m"]
        num_pid_n = (n + kwargs["block_n"] - 1) // kwargs["block_n"]
        if num_pid_m * num_pid_n > SPLIT_K_SEMAPHORE_MAX_LEN:
            raise ValueError(
                f"split-K needs {num_pid_m * num_pid_n} tile semaphores, "
                f"more than {SPLIT_K_SEMAPHORE_MAX_LEN}"
            )

    param = make_gemm_a16w16_gfx942_param(**kwargs)

    semaphore, signal = get_split_k_buffers(stream, a.device)
    a_arg = _dynamic_tensor_arg(a, 0 if a_is_transposed else 1)
    b_arg = _dynamic_tensor_arg(b, 0 if b_is_transposed else 1)
    out_arg = _dynamic_tensor_arg(kernel_out, 1)
    dispatch_args = (out_arg, a_arg, b_arg, semaphore, signal, split_k, param, stream)
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
