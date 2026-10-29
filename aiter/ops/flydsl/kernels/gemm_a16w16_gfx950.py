# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
from dataclasses import dataclass
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch

from aiter.utility.graph_alloc import persistent_alloc

from .gemm_a16w16_gfx950_utils import (
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    SPLIT_K_COUNTER_STRIDE,
    SPLIT_K_SEMAPHORE_MAX_LEN,
    BlockSwizzle,
    SplitKReduction,
    get_wave_lds_offset,
    make_lds_layout,
    make_transposed_lds_layout,
    swizzled_col_idx,
    transposed_contiguous_idx,
    wait_vmcnt_and_barrier,
)
from .kernels_common import run_cached
from .tensor_shim import buf_base_i64

GEMM_A16W16_DTYPE_FP32 = 1
GEMM_A16W16_DTYPE_BF16 = 2
GEMM_A16W16_DTYPE_FP16 = 3
# Split-K tiles at least this large keep their own partial in registers and skip
# rows past M in the partial traffic; smaller ones gain nothing from it.
SPLIT_K_TRIM_MIN_TILE_ELEMS = 1024


def load_kernargs_up_front(tensors, scalars):
    """Load the tensors' base addresses and the Int32 scalar arguments together.

    The scalars pass through an empty asm that also uses the base addresses, so
    every later use of them waits for all of these kernarg loads at once: one
    scalar round trip before the first global load instead of two. Returns the
    scalars.
    """
    n = len(scalars)
    ptrs = [fx.as_ir_value(buf_base_i64(t)) for t in tensors]
    struct = ir.Type.parse("!llvm.struct<(" + ", ".join(["i32"] * n) + ")>")
    op = llvm.InlineAsmOp(
        struct,
        [fx.as_ir_value(v) for v in scalars] + ptrs,
        "",
        ",".join(["=s"] * n + [str(i) for i in range(n)] + ["s"] * len(ptrs)),
        has_side_effects=True,
    )
    i32 = ir.IntegerType.get_signless(32)
    return tuple(fx.Int32(llvm.extractvalue(i32, op.res, [i])) for i in range(n))


def c_lds_elems(block_m, block_n, k_waves, out_data_bytes, is_split_k):
    """Shared C elements (output dtype): the C tile, or the fp32 slice-K exchange
    tile, plus 16 bytes for the split-K arrival flag."""
    tile_bytes = block_m * block_n * (4 if k_waves > 1 else out_data_bytes)
    return (tile_bytes + (16 if is_split_k else 0)) // out_data_bytes


@fx.struct
class GemmA16W16Gfx950Param:
    in_dtype_id: fx.Constexpr[int]
    out_dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    split_k: fx.Constexpr[int]
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
    if out_dtype_id not in (in_dtype_id, GEMM_A16W16_DTYPE_FP32):
        raise ValueError(
            f"unsupported out_dtype_id={out_dtype_id} for in_dtype_id={in_dtype_id}"
        )
    if block_m <= 0 or block_n <= 0 or block_k <= 0 or stages <= 0 or split_k <= 0:
        raise ValueError(
            "block_m, block_n, block_k, stages, and split_k must be positive"
        )
    if (mma_m, mma_n, mma_k) != (16, 16, 32):
        raise ValueError("the gfx950 layout kernel currently requires mma=16x16x32")
    if stages < 2:
        raise ValueError("stages must be at least 2 for the staged LDS pipeline")
    if m_waves <= 0 or n_waves <= 0 or k_waves <= 0:
        raise ValueError("m_waves, n_waves, and k_waves must be positive")
    if m_waves * n_waves * k_waves > 16:
        raise ValueError("the workgroup cannot contain more than 16 waves")
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
    smem_bytes = max(
        smem_bytes,
        c_lds_elems(block_m, block_n, k_waves, out_dbytes, split_k > 1) * out_dbytes,
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
        split_k=split_k,
        m_waves=m_waves,
        n_waves=n_waves,
        k_waves=k_waves,
        group_m=group_m,
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
    # "fp32" and "f32" mark the fp32 split-K and slice-K reductions (cache key).
    name += f"_ks{param.split_k}" + ("fp32" if param.split_k > 1 else "")
    name += f"_w{param.m_waves}x{param.n_waves}x{param.k_waves}"
    name += "f32" if param.k_waves > 1 else ""
    name += f"_gm{param.group_m}"
    name += f"_bias{int(param.has_bias)}"
    a_layout = "t" if param.a_is_transposed else "n"
    b_layout = "t" if param.b_is_transposed else "n"
    name += f"_l{a_layout}{b_layout}"
    name += "_phti" if param.use_half_tile_interleaved else "_pft"
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
        fx.copy_atom_call(context.async_g2s_copy_atom, src, dst)
        rocdl.sched_barrier(0)
        if i < load_iters - 1:
            lds_ptr = lds_ptr + block_threads * async_load_vec_size


@flyc.jit
def write_cshuffle_vec_to_global(out, global_offset, c_vec, is_fp32_output):
    if const_expr(is_fp32_output):
        c_vec = c_vec.to(fx.Float32)
    fx.ptr_store(c_vec, fx.get_iter(out) + global_offset)


@flyc.kernel
def gemm_a16w16_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    semaphore: fx.Tensor,
    workspace: fx.Tensor,
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
    is_split_k = param.split_k > 1
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
    m, n, k, working_k = load_kernargs_up_front(
        (a, b, out)
        + ((semaphore, workspace) if const_expr(param.split_k > 1) else ())
        + ((bias,) if const_expr(param.has_bias) else ()),
        (m, n, k, working_k),
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
    if const_expr(param.group_m <= 0):
        # Grid (n tiles, m tiles, splits): the same dispatch order as the linear
        # tile id, without dividing it by num_pid_n.
        bid_n = fx.block_idx.x
        bid_m = fx.block_idx.y
        ks_idx = fx.block_idx.z
        tile_idx = bid_m * num_pid_n + bid_n
    else:
        bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
        ks_idx = fx.block_idx.y
        tile_idx = fx.block_idx.x
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
            c_lds_elems(block_m, block_n, k_waves, param.out_data_bytes, is_split_k),
            16,
        ]

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
        splitk_reduction = SplitKReduction(
            param.split_k, block_m * block_n, threads_per_k_slice
        )
        splitk_reduction.init(
            semaphore,
            workspace,
            fx.recast_iter(fx.Int32, smem_c + block_m * block_n),
            tid_in_k_slice,
            tile_idx,
            ks_idx,
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
    sC_write = fx.make_view(smem_c, c_lds_layout)

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

    def init_frag_C():
        if const_expr(param.has_bias):
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

    if const_expr(is_split_k):
        frag_C.fill(0.0)
    else:
        init_frag_C()

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
            context=ab_load_context,
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

    # Prime the staged LDS pipeline: preload the first `stages - 1` K tiles
    # before entering the main loop that overlaps async loads with compute.
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

    if const_expr(is_slice_k):
        gpu.barrier()
        # Add the k-wave partials into k-wave 0 in fp32 and in k-wave order, one
        # k-wave at a time through LDS, so C is rounded to the output dtype once.
        smem_x = fx.recast_iter(fx.Float32, smem_c)
        frag_C_groups = fx.size(frag_C.shape).unpack() // 4

        def store_k_slice_partial():
            for g in range_constexpr(frag_C_groups):
                x_vec = fx.Vector.from_elements(
                    [frag_C[g * 4 + e] for e in range_constexpr(4)], fx.Float32
                )
                fx.ptr_store(
                    x_vec, smem_x + (g * threads_per_k_slice + tid_in_k_slice) * 4
                )

        def add_k_slice_partial():
            for g in range_constexpr(frag_C_groups):
                x_vec = fx.ptr_load(
                    smem_x + (g * threads_per_k_slice + tid_in_k_slice) * 4,
                    result_type=fx.Vector.make_type(4, fx.Float32),
                )
                for e in range_constexpr(4):
                    frag_C[g * 4 + e] = frag_C[g * 4 + e] + x_vec[e]

        for k_slice in range_constexpr(1, k_waves):
            if k_wave_idx == k_slice:
                store_k_slice_partial()
            gpu.barrier()
            if k_wave_idx == 0:
                add_k_slice_partial()
            gpu.barrier()

    def store_c_tile_to_lds():
        frag_C_out = fx.make_fragment_like(frag_C, global_output_dtype)
        frag_C_out.store(frag_C.load().to(global_output_dtype))
        for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
            row = fx.get_scalar(thr_mma_cRow[i])
            col = fx.get_scalar(thr_mma_cCol[i])
            sC_write[row, col] = frag_C_out[i]

    def store_c_tile_to_global():
        if const_expr(is_slice_k):
            if k_wave_idx == 0:
                store_c_tile_to_lds()
        else:
            store_c_tile_to_lds()
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
                            cshuffle_r2g_vec_size, global_output_dtype
                        ),
                    )
                    write_cshuffle_vec_to_global(
                        out,
                        global_row * n + global_col,
                        c_vec,
                        param.out_dtype_id == GEMM_A16W16_DTYPE_FP32,
                    )

    if const_expr(is_split_k):
        # Calls through plain names: a dynamic `if` treats `obj.method()` as
        # state that must be yielded.
        store_partial = splitk_reduction.store_partial
        accumulate = splitk_reduction.accumulate

        # Tiles this large are worth trimming the reduction's memory traffic:
        # skip rows past M and keep this split's own partial in registers.
        trim = block_m * block_n >= SPLIT_K_TRIM_MIN_TILE_ELEMS

        def group_live(group):
            # Groups are 4 rows of one column; skip the ones entirely past M.
            return block_m_offset + fx.get_scalar(thr_mma_cRow[group * 4]) < m

        live = group_live if const_expr(trim) else None

        def reduce_tile():
            if const_expr(trim):
                # One copy per split index, so this split's partial comes from
                # registers and is added in its place without per-element selects.
                for own_split in range_constexpr(param.split_k):
                    if ks_idx == own_split:
                        accumulate(frag_C, 0, init_frag_C, own_split, live)
            else:
                accumulate(frag_C, 0, init_frag_C)

        if const_expr(is_slice_k):
            if k_wave_idx == 0:
                store_partial((frag_C,), live)
        else:
            store_partial((frag_C,), live)
        if splitk_reduction.arrive():
            if const_expr(is_slice_k):
                if k_wave_idx == 0:
                    reduce_tile()
            else:
                reduce_tile()
            store_c_tile_to_global()
    else:
        if const_expr(not is_slice_k):
            gpu.barrier()
        store_c_tile_to_global()


@flyc.kernel
def gemm_a16w16_hti_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    semaphore: fx.Tensor,
    workspace: fx.Tensor,
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
    is_split_k = param.split_k > 1
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
    m, n, k, working_k = load_kernargs_up_front(
        (a, b, out)
        + ((semaphore, workspace) if const_expr(param.split_k > 1) else ())
        + ((bias,) if const_expr(param.has_bias) else ()),
        (m, n, k, working_k),
    )
    tid = fx.thread_idx.x
    wid = tid // GFX950_WAVE_SIZE
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    )
    if const_expr(param.group_m <= 0):
        # Grid (n tiles, m tiles, splits): the same dispatch order as the linear
        # tile id, without dividing it by num_pid_n.
        bid_n = fx.block_idx.x
        bid_m = fx.block_idx.y
        ks_idx = fx.block_idx.z
        tile_idx = bid_m * num_pid_n + bid_n
    else:
        bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
        ks_idx = fx.block_idx.y
        tile_idx = fx.block_idx.x
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
            c_lds_elems(block_m, block_n, 1, param.out_data_bytes, is_split_k),
            16,
        ]

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
        splitk_reduction = SplitKReduction(
            param.split_k, block_m * block_n, block_threads
        )
        splitk_reduction.init(
            semaphore,
            workspace,
            fx.recast_iter(fx.Int32, smem_c + block_m * block_n),
            tid,
            tile_idx,
            ks_idx,
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
        async_load_to_lds(
            tile=AsyncLoadTile(
                lds_base=half_b_base(stage, n_part),
                src_base=fx.get_iter(b_buf),
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
                    write_cshuffle_vec_to_global(
                        out,
                        global_row * n + global_col,
                        c_vec,
                        param.out_dtype_id == GEMM_A16W16_DTYPE_FP32,
                    )

    c00 = make_c_fragment(0, 0)
    c01 = make_c_fragment(0, 1)
    c10 = make_c_fragment(1, 0)
    c11 = make_c_fragment(1, 1)

    def init_c_fragment(frag, n_part, with_bias):
        if const_expr(param.has_bias and with_bias):
            for i in range_constexpr(fx.size(frag.shape).unpack()):
                col_idx = fx.get_scalar(thr_mma_cCol[i])
                global_n_idx = block_n_offset + n_part * half_block_n + col_idx
                safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
                frag[i] = bias_buf[safe_global_n_idx].to(fx.Float32)
        else:
            frag.fill(0.0)

    # Split-K adds the bias once, when the partials are reduced.
    init_c_fragment(c00, 0, not is_split_k)
    init_c_fragment(c01, 1, not is_split_k)
    init_c_fragment(c10, 0, not is_split_k)
    init_c_fragment(c11, 1, not is_split_k)

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
    if const_expr(not is_split_k):
        store_half_tile_to_lds(0, 0, c00)
    consume(c10, a1, b0, False)
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    if const_expr(not is_split_k):
        store_half_tile_to_lds(0, 1, c01)
    consume(c11, a1, b1, False)
    rocdl.sched_barrier(0)
    if const_expr(is_split_k):
        if wid // n_waves == 0:
            rocdl.s_barrier()
        rocdl.s_barrier()
        store_partial = splitk_reduction.store_partial
        accumulate = splitk_reduction.accumulate
        group = fx.size(c00.shape).unpack() // 4
        store_partial((c00, c01, c10, c11))
        if splitk_reduction.arrive():
            accumulate(c00, 0, lambda: init_c_fragment(c00, 0, True))
            accumulate(c01, group, lambda: init_c_fragment(c01, 1, True))
            accumulate(c10, 2 * group, lambda: init_c_fragment(c10, 0, True))
            accumulate(c11, 3 * group, lambda: init_c_fragment(c11, 1, True))
            store_half_tile_to_lds(0, 0, c00)
            store_half_tile_to_lds(0, 1, c01)
            store_half_tile_to_lds(1, 0, c10)
            store_half_tile_to_lds(1, 1, c11)
            gpu.barrier()
            store_half_tile_to_global(0, 0)
            store_half_tile_to_global(0, 1)
            store_half_tile_to_global(1, 0)
            store_half_tile_to_global(1, 1)
    else:
        wait_vmcnt_and_barrier(0)
        store_half_tile_to_global(0, 0)
        store_half_tile_to_global(0, 1)
        wait_vmcnt_and_barrier(0)
        store_half_tile_to_lds(1, 0, c10)
        store_half_tile_to_lds(1, 1, c11)
        wait_vmcnt_and_barrier(0)
        store_half_tile_to_global(1, 0)
        store_half_tile_to_global(1, 1)


@flyc.jit
def gemm_a16w16_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    semaphore: fx.Tensor,
    workspace: fx.Tensor,
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
    # Whole K tiles per split; the last split takes the remainder.
    working_k = (k + split_k - 1) // split_k
    working_k = (working_k + param.block_k - 1) // param.block_k * param.block_k
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
        grid=(
            (num_pid_n, num_pid_m, split_k)
            if const_expr(param.group_m <= 0)
            else (num_pid_m * num_pid_n, split_k, 1)
        ),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


@functools.cache
def _gemm_a16w16_gfx950_param_for(items):
    # Building the param struct coerces every field through FlyDSL's typing
    # layer, which costs more host time than the launch itself; it depends
    # only on the config, so one struct per config serves every M.
    return make_gemm_a16w16_gfx950_param(**dict(items))


def make_gemm_a16w16_param_and_validate(m, n, k, kwargs):
    result = None
    try:
        result = _gemm_a16w16_gfx950_param_for(tuple(sorted(kwargs.items())))
    except Exception:  # noqa: BLE001 - invalid policies are filtered here
        return None
    split_k = kwargs.get("split_k", 1)
    try:
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
        workspace_bytes = (
            num_pid_m * num_pid_n * split_k * result.block_m * result.block_n * 4
        )
        if (
            num_pid_m * num_pid_n > SPLIT_K_SEMAPHORE_MAX_LEN
            or result.block_m * result.block_n % c_elements_per_iteration != 0
            or workspace_bytes >= 2**31
        ):
            return None
    return result


def assert_no_k_tail(k: int, kwargs: dict):
    split_k = kwargs["split_k"]
    block_k = kwargs["block_k"]
    stages = kwargs["stages"]
    use_half_tile_interleaved = kwargs["use_half_tile_interleaved"]
    working_k = (k + split_k - 1) // split_k
    working_k = (working_k + block_k - 1) // block_k * block_k
    last_working_k = k - (split_k - 1) * working_k
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


@functools.lru_cache(maxsize=128)
def get_split_k_buffers(stream, device):
    # One arrival counter per output tile, which the last split to arrive resets.
    with persistent_alloc(torch.device(device)):
        return torch.zeros(
            (SPLIT_K_SEMAPHORE_MAX_LEN * SPLIT_K_COUNTER_STRIDE,),
            dtype=torch.int32,
            device=device,
        )


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
    Set ``user_kwargs["split_k"]`` above 1 to reduce K partitions in fp32.
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
    assert_no_k_tail(k, kwargs)

    if bias is not None:
        assert bias.shape[0] == n
        assert bias.dtype == a.dtype

    param = make_gemm_a16w16_param_and_validate(m, n, k, kwargs)
    assert param is not None, "unsupported gemm_a16w16_gfx950 shape/config"
    semaphore = get_split_k_buffers(stream, device)
    a_arg = _dynamic_tensor_arg(a, 0 if a_is_transposed else 1)
    b_arg = _dynamic_tensor_arg(b, 0 if b_is_transposed else 1)
    out_arg = _dynamic_tensor_arg(out, 1)
    bias_arg = a_arg if bias is None else _dynamic_tensor_arg(bias, 0)
    if split_k > 1:
        # One fp32 partial tile per (output tile, split), reduced by the last split.
        tiles = -(-m // param.block_m) * -(-n // param.block_n)
        workspace = torch.empty(
            (split_k * tiles * param.block_m * param.block_n,),
            dtype=torch.float32,
            device=device,
        )
        workspace_arg = _dynamic_tensor_arg(workspace, 0)
    else:
        workspace_arg = out_arg
    dispatch_args = (
        out_arg,
        a_arg,
        b_arg,
        bias_arg,
        semaphore,
        workspace_arg,
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
