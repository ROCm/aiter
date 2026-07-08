# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import functools
import itertools
from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T
from flydsl.expr import (
    range_constexpr,
    const_expr,
    arith,
    vector,
    gpu,
    rocdl,
    buffer_ops,
)
from flydsl._mlir import ir
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, SMEM_CAPACITY_MAP
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir.dialects import llvm, memref

from .hgemm_wmma_gfx950_utils import (
    get_dtype_in_kernel,
    _run_compiled,
    WmmaHalf_m16n16k16,
    WmmaHalf_m16n16k32,
    WmmaFp8_m16n16k128,
    swizzle_xor16,
    swizzle_fp8_128,
    get_llvm_ptr,
    __barrier,
    __s_barrier,
    buffer_load_lds_inline,
    SplitKProtocol,
    BlockSwizzle,
)

SPLIT_K_SEMAPHORE_MAX_LEN = 256


@functools.lru_cache(maxsize=16384)
def compile_hgemm_wmma_kernel(
    DTYPE: str,
    TILE_M: int = 128,
    TILE_N: int = 128,
    TILE_K: int = 64,
    STAGES: int = 2,
    SPLIT_K: int = 1,
    BLOCK_M_WARPS: int = 2,
    BLOCK_N_WARPS: int = 2,
    BLOCK_K_WARPS: int = 1,
    HAS_BIAS: bool = False,
    HAS_K_TAIL: bool = False,
    GROUP_M: int = 0,
    USE_HALF_TILE_INTERLEAVED: bool = False,
):
    assert BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS <= 16
    assert STAGES >= 2
    IS_SPLIT_K = SPLIT_K > 1
    IS_SLICE_K = BLOCK_K_WARPS > 1
    IS_HT = USE_HALF_TILE_INTERLEAVED
    BLOCK_M = TILE_M
    BLOCK_N = TILE_N
    BLOCK_K = TILE_K
    HALF_BLOCK_M = BLOCK_M // 2
    HALF_BLOCK_N = BLOCK_N // 2
    IS_FP8 = "fp8" in DTYPE
    IS_FP8_PTPC = DTYPE == "fp8_ptpc"
    WARP_SIZE = 64
    IN_DTYPE_BYTES = 1 if IS_FP8 else 2
    OUT_DTYPE_BYTES = 2
    LDG_VEC_SIZE = 16 if IS_FP8 else 8
    GPU_ARCH = get_rocm_arch()

    if IS_FP8:
        assert BLOCK_K >= 128
        WMMA_IMPL = WmmaFp8_m16n16k128()
        DMA_BYTES = 16
        MFMA_PER_WARP_K = 1
    elif GPU_ARCH == "gfx942":
        assert BLOCK_K >= 16
        WMMA_IMPL = WmmaHalf_m16n16k16(DTYPE)
        DMA_BYTES = 4
        MFMA_PER_WARP_K = 2
    else:
        assert BLOCK_K >= 32
        WMMA_IMPL = WmmaHalf_m16n16k32(DTYPE)
        DMA_BYTES = 16
        MFMA_PER_WARP_K = 1

    if IS_HT:
        assert STAGES == 2
        assert BLOCK_M_WARPS == 2
        assert BLOCK_K_WARPS == 1
        assert HALF_BLOCK_M * 2 == BLOCK_M
        assert HALF_BLOCK_N * 2 == BLOCK_N

    WMMA_M = WMMA_IMPL.WMMA_M
    WMMA_N = WMMA_IMPL.WMMA_N
    WMMA_K = WMMA_IMPL.WMMA_K
    WMMA_A_FRAG_VALUES = WMMA_IMPL.WMMA_A_FRAG_VALUES
    WMMA_B_FRAG_VALUES = WMMA_IMPL.WMMA_B_FRAG_VALUES
    WMMA_C_FRAG_VALUES = WMMA_IMPL.WMMA_C_FRAG_VALUES
    WARP_ATOM_M = WMMA_M
    WARP_ATOM_N = WMMA_N
    WARP_ATOM_K = WMMA_K * MFMA_PER_WARP_K
    WARP_GROUP_K = BLOCK_K_WARPS * WARP_ATOM_K

    WARP_K_STEPS = BLOCK_K // WARP_GROUP_K
    assert WARP_K_STEPS * WARP_GROUP_K == BLOCK_K

    K_SLICE = BLOCK_K // BLOCK_K_WARPS
    assert K_SLICE * BLOCK_K_WARPS == BLOCK_K
    assert K_SLICE % WARP_ATOM_K == 0

    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS * WARP_SIZE
    BLOCK_MN_WARPS = BLOCK_M_WARPS * BLOCK_N_WARPS
    if IS_HT:
        WARP_M_STEPS = HALF_BLOCK_M // BLOCK_M_WARPS // WARP_ATOM_M
        WARP_N_STEPS = HALF_BLOCK_N // BLOCK_N_WARPS // WARP_ATOM_N
        assert (WARP_M_STEPS >= 1) and (WARP_N_STEPS >= 1)
        assert HALF_BLOCK_M % (BLOCK_M_WARPS * WARP_ATOM_M) == 0
        assert HALF_BLOCK_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0
    else:
        WARP_M_STEPS = BLOCK_M // BLOCK_M_WARPS // WARP_ATOM_M
        WARP_N_STEPS = BLOCK_N // BLOCK_N_WARPS // WARP_ATOM_N
        assert (WARP_M_STEPS >= 1) and (WARP_N_STEPS >= 1)
        assert BLOCK_M % (BLOCK_M_WARPS * WARP_ATOM_M) == 0
        assert BLOCK_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0
    WARP_M = WARP_M_STEPS * WARP_ATOM_M
    WARP_N = WARP_N_STEPS * WARP_ATOM_N
    if IS_HT:
        assert HALF_BLOCK_M == BLOCK_M_WARPS * WARP_M
        assert HALF_BLOCK_N == BLOCK_N_WARPS * WARP_N
    else:
        assert BLOCK_M == BLOCK_M_WARPS * WARP_M
        assert BLOCK_N == BLOCK_N_WARPS * WARP_N
    BLOCK_MK_SIZE = BLOCK_M * BLOCK_K
    BLOCK_NK_SIZE = BLOCK_N * BLOCK_K
    BLOCK_MN_SIZE = BLOCK_M * BLOCK_N

    if IS_HT:
        STG_SIZE_PER_M_STEP = BLOCK_M_WARPS * WARP_ATOM_M * HALF_BLOCK_N
        STG_WORK_SIZE_PER_M_STEP = STG_SIZE_PER_M_STEP // BLOCK_THREADS
        assert (STG_SIZE_PER_M_STEP % BLOCK_THREADS == 0) and (
            STG_WORK_SIZE_PER_M_STEP >= 1
        )
        STG_VEC_SIZE = 8 if STG_WORK_SIZE_PER_M_STEP >= 8 else STG_WORK_SIZE_PER_M_STEP
        assert STG_VEC_SIZE in [8, 4]
        assert (STG_WORK_SIZE_PER_M_STEP % STG_VEC_SIZE == 0) and (
            STG_WORK_SIZE_PER_M_STEP // STG_VEC_SIZE >= 1
        )
    else:
        STG_VEC_SIZE = 8

    BLOCK_VECS_STG = STG_VEC_SIZE * BLOCK_THREADS
    STG_C_X_THREADS = BLOCK_N // STG_VEC_SIZE
    STG_C_QUAD_X_THREADS = HALF_BLOCK_N // STG_VEC_SIZE
    assert STG_C_X_THREADS * STG_VEC_SIZE == BLOCK_N
    STG_C_ITERS = BLOCK_MN_SIZE // BLOCK_VECS_STG
    assert STG_C_ITERS * BLOCK_VECS_STG == BLOCK_MN_SIZE
    BLOCK_K_BYTES = BLOCK_K * IN_DTYPE_BYTES
    LDG_ASYNC_VEC_SIZE = DMA_BYTES // IN_DTYPE_BYTES
    assert LDG_ASYNC_VEC_SIZE * IN_DTYPE_BYTES == DMA_BYTES
    LDG_A_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    assert LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE == BLOCK_K
    LDG_B_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    assert LDG_B_X_THREADS_AS * LDG_ASYNC_VEC_SIZE == BLOCK_K
    assert (BLOCK_M * BLOCK_N) % (BLOCK_THREADS * STG_VEC_SIZE) == 0

    if IS_HT:
        LDG_A_ITERS_AS = HALF_BLOCK_M * BLOCK_K // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
        assert (
            LDG_A_ITERS_AS * LDG_ASYNC_VEC_SIZE * BLOCK_THREADS
            == HALF_BLOCK_M * BLOCK_K
        )
        LDG_B_ITERS_AS = HALF_BLOCK_N * BLOCK_K // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
        assert (
            LDG_B_ITERS_AS * LDG_ASYNC_VEC_SIZE * BLOCK_THREADS
            == HALF_BLOCK_N * BLOCK_K
        )
        assert (2 * LDG_B_ITERS_AS + 2 * LDG_A_ITERS_AS) < 63
        assert HALF_BLOCK_N % STG_VEC_SIZE == 0
        A_FRAGS_LEN = WARP_K_STEPS * WARP_M_STEPS
        B_FRAGS_LEN = WARP_K_STEPS * WARP_N_STEPS
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
    else:
        LDG_A_ITERS_AS = BLOCK_MK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
        assert LDG_A_ITERS_AS * LDG_ASYNC_VEC_SIZE * BLOCK_THREADS == BLOCK_MK_SIZE
        LDG_B_ITERS_AS = BLOCK_NK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
        assert LDG_B_ITERS_AS * LDG_ASYNC_VEC_SIZE * BLOCK_THREADS == BLOCK_NK_SIZE
        LDG_WAIT_COUNT = LDG_B_ITERS_AS + LDG_A_ITERS_AS
        assert ((STAGES - 2) * LDG_WAIT_COUNT) < 63

    # LDS parameters:
    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem")
    smem_a_offset = allocator._align(allocator.ptr, 16)
    AS_BYTES = STAGES * BLOCK_M * BLOCK_K * IN_DTYPE_BYTES
    allocator.ptr = smem_a_offset + AS_BYTES
    SMEM_USE = AS_BYTES
    smem_b_offset = allocator._align(allocator.ptr, 16)
    BS_BYTES = STAGES * BLOCK_N * BLOCK_K * IN_DTYPE_BYTES
    allocator.ptr = smem_b_offset + BS_BYTES
    SMEM_USE += BS_BYTES
    SMEM_USE_ = max(SMEM_USE, BLOCK_K_WARPS * BLOCK_M * BLOCK_N * OUT_DTYPE_BYTES)
    allocator.ptr += SMEM_USE_ - SMEM_USE
    assert SMEM_USE_ <= SMEM_CAPACITY_MAP[GPU_ARCH]

    KERNEL_NAME = f"HGEMM_{DTYPE}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}x{STAGES}_SPK{SPLIT_K}_W{BLOCK_M_WARPS}x{BLOCK_N_WARPS}x{BLOCK_K_WARPS}_TN"
    KERNEL_NAME += "_HT" if IS_HT else "_FT"
    KERNEL_NAME += f"_GM{max(GROUP_M, 0)}"
    KERNEL_NAME += "_BIAS" if HAS_BIAS else ""
    KERNEL_NAME += "_KTAIL" if HAS_K_TAIL else ""
    KERNEL_NAME = KERNEL_NAME.upper()

    splitk_protocol = SplitKProtocol(
        SPLIT_K,
        BLOCK_M,
        BLOCK_N,
        STG_VEC_SIZE,
        OUT_DTYPE_BYTES,
        BLOCK_THREADS,
        HAS_BIAS,
    )
    block_swizzle = BlockSwizzle(NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=GROUP_M)

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hgemm_kernel(
        c_ptr: fx.Pointer,
        a_ptr: fx.Pointer,
        b_ptr: fx.Pointer,
        scale_a_ptr: fx.Pointer,
        scale_b_ptr: fx.Pointer,
        bias_ptr: fx.Pointer,
        semaphore_ptr: fx.Pointer,
        signal_ptr: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        working_k: fx.Int32,
        num_pid_m: fx.Int32,
        num_pid_n: fx.Int32,
        c_stride: fx.Int32,
        a_stride: fx.Int32,
        b_stride: fx.Int32,
    ):
        input_dtype_ = (
            fx.Float8E4M3FN.ir_type if IS_FP8_PTPC else get_dtype_in_kernel(DTYPE)
        )
        output_dtype_ = T.bf16 if const_expr(IS_FP8) else input_dtype_
        smem_input_dtype_ = T.i8 if const_expr(IS_FP8) else input_dtype_

        c_rsrc = buffer_ops.create_buffer_resource(c_ptr, max_size=True)
        a_rsrc = buffer_ops.create_buffer_resource(a_ptr, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(b_ptr, max_size=True)
        if const_expr(IS_FP8_PTPC):
            scale_a_rsrc = buffer_ops.create_buffer_resource(scale_a_ptr, max_size=True)
            scale_b_rsrc = buffer_ops.create_buffer_resource(scale_b_ptr, max_size=True)
        else:
            scale_a_rsrc = scale_b_rsrc = None
        if const_expr(HAS_BIAS):
            bias_rsrc = buffer_ops.create_buffer_resource(bias_ptr, max_size=True)
        else:
            bias_rsrc = None

        base_ptr = allocator.get_base()
        smem_a_ptr = SmemPtr(
            base_ptr,
            smem_a_offset,
            smem_input_dtype_,
            shape=(STAGES * BLOCK_M * BLOCK_K,),
        )
        smem_b_ptr = SmemPtr(
            base_ptr,
            smem_b_offset,
            smem_input_dtype_,
            shape=(STAGES * BLOCK_N * BLOCK_K,),
        )
        smem_c_ptr = SmemPtr(
            base_ptr,
            smem_a_offset,
            output_dtype_,
            shape=(BLOCK_K_WARPS * BLOCK_M * BLOCK_N,),
        )

        tid = fx.thread_idx.x
        wid = tid // WARP_SIZE
        wid_mn = wid % BLOCK_MN_WARPS
        wid_k = wid // BLOCK_MN_WARPS
        w_tid = tid % WARP_SIZE

        block_m_idx, block_n_idx = block_swizzle.swizzle(
            num_pid_m, num_pid_n, fx.block_idx.x
        )
        ks_idx = fx.block_idx.y
        ks_begin = ks_idx * working_k
        ks_end = ks_begin + working_k
        ks_end = (ks_end < k).select(ks_end, k)

        block_m_offset = block_m_idx * BLOCK_M
        block_n_offset = block_n_idx * BLOCK_N
        k_blocks16 = BLOCK_K_BYTES // 16

        warp_m_idx = wid_mn // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid_mn % BLOCK_N_WARPS * WARP_N
        ldmatrix_a_m_idx = w_tid % WMMA_M
        ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K
        ldmatrix_b_n_idx = w_tid % WMMA_N
        ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
        warp_k_slice_base = wid_k * K_SLICE

        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
        c_frags = [acc_init] * C_FRAGS_LEN
        stmatrix_c_n_idx = w_tid % WMMA_N

        if const_expr(HAS_BIAS and not IS_SPLIT_K and not IS_FP8_PTPC):
            bias_frags = [acc_init] * WARP_N_STEPS
            for jj in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                lds_n_idx = warp_atom_n_idx + stmatrix_c_n_idx
                global_n_idx = block_n_offset + lds_n_idx
                safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
                bias_val = buffer_ops.buffer_load(
                    bias_rsrc, safe_global_n_idx, vec_width=1, dtype=output_dtype_
                ).extf(T.f32)
                if const_expr(IS_SLICE_K):
                    is_first_k_slice = wid_k == 0
                    bias_val = is_first_k_slice.select(bias_val, fx.Float32(0.0))
                bias_frags[jj] = vector.broadcast(
                    T.vec(WMMA_C_FRAG_VALUES, T.f32), arith.unwrap(bias_val)
                )
            for ii in range_constexpr(WARP_M_STEPS):
                for jj in range_constexpr(WARP_N_STEPS):
                    c_frags[ii * WARP_N_STEPS + jj] = bias_frags[jj]

        if const_expr(IS_SPLIT_K):
            signal_idx = fx.block_idx.x
            splitk_protocol.init(
                semaphore_ptr,
                signal_ptr,
                c_ptr,
                bias_rsrc,
                tid,
                ks_idx,
                m,
                n,
                block_m_offset,
                block_n_offset,
                output_dtype_,
                signal_idx,
                c_stride,
            )

        def get_dma_copy_warp_offset():
            return rocdl.readfirstlane(T.i64, fx.Int64(wid * WARP_SIZE * DMA_BYTES))

        warp_offset = get_dma_copy_warp_offset()

        def get_smem_warp_ptr(memptr):
            lds_base = memref.extract_aligned_pointer_as_index(memptr)
            lds_ptr_base = buffer_ops.create_llvm_ptr(
                arith.index_cast(T.i64, lds_base), address_space=3
            )
            return buffer_ops.get_element_ptr(lds_ptr_base, warp_offset)

        as_warp_ptr = get_smem_warp_ptr(smem_a_ptr.get())
        bs_warp_ptr = get_smem_warp_ptr(smem_b_ptr.get())

        def get_lds_ptr(base_ptr, static_bytes_offset, lds_ptr=None):
            if const_expr(lds_ptr is None):
                return buffer_ops.get_element_ptr(
                    base_ptr, byte_offset=static_bytes_offset
                )
            return buffer_ops.get_element_ptr(
                lds_ptr, static_byte_offset=BLOCK_THREADS * DMA_BYTES
            )

        def ldg_sts_a_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_tid = BLOCK_THREADS * ii + tid
            m_local_idx = global_tid // LDG_A_X_THREADS_AS
            k_local_idx = global_tid % LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            if const_expr(IS_FP8_PTPC):
                col_in_bytes = swizzle_fp8_128(m_local_idx, k_local_idx)
            else:
                col_in_bytes = k_local_idx * IN_DTYPE_BYTES
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
            global_m_idx = block_m_offset + m_local_idx
            safe_global_m_idx = (global_m_idx < m).select(global_m_idx, 0)
            global_k_idx = k_offset + col_in_bytes // IN_DTYPE_BYTES
            if const_expr(HAS_K_TAIL):
                safe_global_k_idx = (global_k_idx < ks_end).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            global_offset_in_bytes = (
                safe_global_m_idx * a_stride + safe_global_k_idx
            ) * IN_DTYPE_BYTES
            global_offset_in_bytes = arith.index_cast(T.i32, global_offset_in_bytes)
            dynamic_bytes_offset = write_stage * BLOCK_M * BLOCK_K * IN_DTYPE_BYTES
            lds_ptr = get_lds_ptr(as_warp_ptr, dynamic_bytes_offset, lds_ptr)
            buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset_in_bytes, DMA_BYTES)
            return lds_ptr

        def ldg_sts_b_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_tid = BLOCK_THREADS * ii + tid
            n_local_idx = global_tid // LDG_B_X_THREADS_AS
            k_local_idx = global_tid % LDG_B_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            if const_expr(IS_FP8_PTPC):
                col_in_bytes = swizzle_fp8_128(n_local_idx, k_local_idx)
            else:
                col_in_bytes = k_local_idx * IN_DTYPE_BYTES
                col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
            global_n_idx = block_n_offset + fx.Index(n_local_idx)
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            global_k_idx = k_offset + col_in_bytes // IN_DTYPE_BYTES
            if const_expr(HAS_K_TAIL):
                safe_global_k_idx = (global_k_idx < ks_end).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            global_offset_in_bytes = (
                safe_global_n_idx * b_stride + safe_global_k_idx
            ) * IN_DTYPE_BYTES
            global_offset_in_bytes = arith.index_cast(T.i32, global_offset_in_bytes)
            dynamic_bytes_offset = write_stage * BLOCK_N * BLOCK_K * IN_DTYPE_BYTES
            lds_ptr = get_lds_ptr(bs_warp_ptr, dynamic_bytes_offset, lds_ptr)
            buffer_load_lds_inline(b_rsrc, lds_ptr, global_offset_in_bytes, DMA_BYTES)
            return lds_ptr

        def ldg_sts_a_async(k_offset, lds_stage):
            lds_ptr = None
            for ii in range_constexpr(LDG_A_ITERS_AS):
                lds_ptr = ldg_sts_a_async_one(
                    ii, k_offset, lds_stage, lds_ptr if ii > 0 else None
                )

        def ldg_sts_b_async(k_offset, lds_stage):
            lds_ptr = None
            for ii in range_constexpr(LDG_B_ITERS_AS):
                lds_ptr = ldg_sts_b_async_one(
                    ii, k_offset, lds_stage, lds_ptr if ii > 0 else None
                )

        def mask_tail_k_frag(frag, k_base, frag_values):
            elems = [0] * frag_values
            for vi in range_constexpr(frag_values):
                global_k_idx = k_base + vi
                valid_k = global_k_idx < ks_end
                elem = vector.extract(frag, static_position=[vi], dynamic_position=[])
                elems[vi] = arith.select(
                    valid_k, elem, arith.constant(0, type=smem_input_dtype_)
                )
            return vector.from_elements(T.vec(frag_values, smem_input_dtype_), elems)

        def ldmatrix_compute_tile_streaming(
            lds_stage, k_offset, c_frags, mask_k_tail=False
        ):
            s = lds_stage
            c_frags_new = [cx for cx in c_frags]
            for kk in range_constexpr(WARP_K_STEPS):
                warp_atom_k_idx = warp_k_slice_base + kk * WARP_ATOM_K
                # ldmatrix b
                b_frags = [0] * WARP_N_STEPS
                for ii in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                    row = warp_atom_n_idx + ldmatrix_b_n_idx
                    if const_expr(IS_FP8_PTPC):

                        def load_i32x4_at_b(col_delta):
                            col_base = (
                                warp_atom_k_idx + (w_tid // WMMA_N) * 16 + col_delta
                            )
                            col_swz = swizzle_fp8_128(row, col_base)
                            flat_offset = (s * BLOCK_N + row) * BLOCK_K + col_swz
                            v16 = vector.load_op(
                                T.vec(LDG_VEC_SIZE, smem_input_dtype_),
                                smem_b_ptr.get(),
                                [fx.Index(flat_offset)],
                            )
                            if const_expr(mask_k_tail):
                                v16 = mask_tail_k_frag(
                                    v16, k_offset + col_base, LDG_VEC_SIZE
                                )
                            return fx.Vector(v16).bitcast(fx.Int32)

                        lo = load_i32x4_at_b(0)
                        hi = load_i32x4_at_b(64)
                        b_frags[ii] = fx.Vector(lo).shuffle(hi, list(range(8)))
                    else:
                        frag_k_base = warp_atom_k_idx + ldmatrix_b_k_vec_idx
                        col_in_bytes = frag_k_base * IN_DTYPE_BYTES
                        col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                        flat_offset = (
                            s * BLOCK_N + row
                        ) * BLOCK_K + col_in_bytes // IN_DTYPE_BYTES
                        vec = vector.load_op(
                            T.vec(
                                WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K, smem_input_dtype_
                            ),
                            smem_b_ptr.get(),
                            [fx.Index(flat_offset)],
                        )
                        if const_expr(mask_k_tail):
                            b_frags[ii] = mask_tail_k_frag(
                                vec,
                                k_offset + frag_k_base,
                                WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
                            )
                        else:
                            b_frags[ii] = vec
                a_frags = [0] * WARP_M_STEPS
                for ii in range_constexpr(WARP_M_STEPS):
                    # ldmatrix a
                    warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                    row = warp_atom_m_idx + ldmatrix_a_m_idx
                    if const_expr(IS_FP8_PTPC):

                        def load_i32x4_at_a(col_delta):
                            col_base = (
                                warp_atom_k_idx + (w_tid // WMMA_M) * 16 + col_delta
                            )
                            col_swz = swizzle_fp8_128(row, col_base)
                            flat_offset = (s * BLOCK_M + row) * BLOCK_K + col_swz
                            v16 = vector.load_op(
                                T.vec(LDG_VEC_SIZE, smem_input_dtype_),
                                smem_a_ptr.get(),
                                [fx.Index(flat_offset)],
                            )
                            if const_expr(mask_k_tail):
                                v16 = mask_tail_k_frag(
                                    v16, k_offset + col_base, LDG_VEC_SIZE
                                )
                            return fx.Vector(v16).bitcast(fx.Int32)

                        lo = load_i32x4_at_a(0)
                        hi = load_i32x4_at_a(64)
                        a_frags[ii] = fx.Vector(lo).shuffle(hi, list(range(8)))
                    else:
                        frag_k_base = warp_atom_k_idx + ldmatrix_a_k_vec_idx
                        col_in_bytes = frag_k_base * IN_DTYPE_BYTES
                        col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                        flat_offset = (
                            s * BLOCK_M + row
                        ) * BLOCK_K + col_in_bytes // IN_DTYPE_BYTES
                        vec = vector.load_op(
                            T.vec(
                                WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K, smem_input_dtype_
                            ),
                            smem_a_ptr.get(),
                            [fx.Index(flat_offset)],
                        )
                        if const_expr(mask_k_tail):
                            a_frags[ii] = mask_tail_k_frag(
                                vec,
                                k_offset + frag_k_base,
                                WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K,
                            )
                        else:
                            a_frags[ii] = vec
                rocdl.sched_barrier(0)
                for ii in range_constexpr(WARP_M_STEPS):
                    a_frag = a_frags[ii]
                    for jj in range_constexpr(WARP_N_STEPS):
                        b_frag = b_frags[jj]
                        if const_expr(MFMA_PER_WARP_K == 2):
                            # split a
                            a_i64x2 = vector.bitcast(T.i64x2, a_frag)
                            a0_i64 = vector.extract(
                                a_i64x2, static_position=[0], dynamic_position=[]
                            )
                            a1_i64 = vector.extract(
                                a_i64x2, static_position=[1], dynamic_position=[]
                            )
                            a_v0 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [a0_i64])
                            )
                            a_v1 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [a1_i64])
                            )
                            # split b
                            b_i64x2 = vector.bitcast(T.i64x2, b_frag)
                            b0_i64 = vector.extract(
                                b_i64x2, static_position=[0], dynamic_position=[]
                            )
                            b1_i64 = vector.extract(
                                b_i64x2, static_position=[1], dynamic_position=[]
                            )
                            b_v0 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [b0_i64])
                            )
                            b_v1 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [b1_i64])
                            )
                            # wmma
                            c_idx = ii * WARP_N_STEPS + jj
                            acc_in = c_frags_new[c_idx]
                            acc_mid = WMMA_IMPL(a_v0, b_v0, acc_in)
                            c_frags_new[c_idx] = WMMA_IMPL(a_v1, b_v1, acc_mid)
                        elif const_expr(MFMA_PER_WARP_K == 1):
                            c_idx = ii * WARP_N_STEPS + jj
                            c_frags_new[c_idx] = WMMA_IMPL(
                                a_frag, b_frag, c_frags_new[c_idx]
                            )
                        else:
                            raise NotImplementedError(
                                f"MFMA_PER_WARP_K={MFMA_PER_WARP_K} not supported"
                            )
            return c_frags_new

        if const_expr(IS_SPLIT_K):
            splitk_protocol.zero_c()

        for s in range_constexpr(STAGES - 1):
            ldg_sts_b_async(ks_begin + s * BLOCK_K, s)
            ldg_sts_a_async(ks_begin + s * BLOCK_K, s)
        rocdl.sched_barrier(0)

        def hot_loop_scheduler():
            # ================ Ordered ================
            for i in range_constexpr(LDG_B_ITERS_AS):
                rocdl.sched_vmem(1)  # ldg_sts_b_async next
            for i in range_constexpr(LDG_A_ITERS_AS):
                rocdl.sched_vmem(1)  # ldg_sts_a_async next
            for ki in range_constexpr(WARP_K_STEPS):
                for i in range_constexpr(WARP_N_STEPS):
                    rocdl.sched_dsrd(1)  # lds_matrix_b current
                for i in range_constexpr(WARP_M_STEPS):
                    rocdl.sched_dsrd(1)  # lds_matrix_a current
                for i in range_constexpr(WARP_M_STEPS):
                    rocdl.sched_mfma(WARP_N_STEPS)
            # ================ Reordered ================
            rocdl.sched_barrier(0)

        BLOCK_K_LOOPS = (working_k + BLOCK_K - 1) // BLOCK_K
        init_state = [ks_begin, fx.Int32(0)] + c_frags
        for _, state in range(0, BLOCK_K_LOOPS - (STAGES - 1), 1, init=init_state):
            k_offset = state[0]
            current_stage = state[1]
            c_frags = state[2:]
            next_stage = (current_stage + 1) % STAGES
            write_stage = (current_stage + STAGES - 1) % STAGES
            __barrier((STAGES - 2) * LDG_WAIT_COUNT)
            ldg_sts_b_async(k_offset + (STAGES - 1) * BLOCK_K, write_stage)
            ldg_sts_a_async(k_offset + (STAGES - 1) * BLOCK_K, write_stage)
            c_frags_new = ldmatrix_compute_tile_streaming(
                current_stage, k_offset, c_frags
            )
            k_offset_next = k_offset + BLOCK_K
            hot_loop_scheduler()
            results = yield [k_offset_next, next_stage] + c_frags_new
        current_stage = results[1]
        k_offset = results[0]
        c_frags = results[2:]
        for s in range_constexpr(0, STAGES - 1):
            __barrier((STAGES - 2 - s) * LDG_WAIT_COUNT)
            mask_k_tail = HAS_K_TAIL
            c_frags = ldmatrix_compute_tile_streaming(
                current_stage, k_offset, c_frags, mask_k_tail
            )
            k_offset = k_offset + BLOCK_K
            current_stage = (current_stage + 1) % STAGES

        # write to lds
        stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
        gpu.barrier()
        for ii in range_constexpr(WARP_M_STEPS):
            warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
            for jj in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                for kk in range_constexpr(WMMA_C_FRAG_VALUES):
                    lds_m_idx = fx.Index(warp_atom_m_idx + stmatrix_c_m_vec_idx + kk)
                    lds_n_idx = fx.Index(warp_atom_n_idx + stmatrix_c_n_idx)
                    val = vector.extract(
                        c_frags[ii * WARP_N_STEPS + jj],
                        static_position=[kk],
                        dynamic_position=[],
                    )
                    if const_expr(IS_FP8_PTPC):
                        row_global = block_m_offset + lds_m_idx
                        col_global = block_n_offset + lds_n_idx
                        scale_a_offset = (row_global < m).select(row_global, 0)
                        scale_a = buffer_ops.buffer_load(
                            scale_a_rsrc, scale_a_offset, vec_width=1, dtype=T.f32
                        )
                        scale_b_offset = (col_global < n).select(col_global, 0)
                        scale_b = buffer_ops.buffer_load(
                            scale_b_rsrc, scale_b_offset, vec_width=1, dtype=T.f32
                        )
                        val = val * scale_a * scale_b
                        if const_expr(HAS_BIAS and not IS_SPLIT_K):
                            bias = buffer_ops.buffer_load(
                                bias_rsrc,
                                scale_b_offset,
                                vec_width=1,
                                dtype=output_dtype_,
                            ).extf(T.f32)
                            val = val + bias
                    val = val.truncf(output_dtype_)
                    val = vector.from_elements(T.vec(1, output_dtype_), [val])
                    flat_offset = (wid_k * BLOCK_M + lds_m_idx) * BLOCK_N + lds_n_idx
                    vector.store(
                        val, smem_c_ptr.get(), [fx.Index(flat_offset)], alignment=16
                    )

        # write back to global
        if const_expr(IS_SPLIT_K):
            splitk_protocol.split_k_barrier()
        else:
            gpu.barrier()
        smem_c_ptr_ = smem_c_ptr.get()
        for i in range_constexpr(STG_C_ITERS):
            global_tid = BLOCK_THREADS * i + tid
            m_local_idx = global_tid // STG_C_X_THREADS
            n_local_idx = global_tid % STG_C_X_THREADS * STG_VEC_SIZE
            global_m_idx = block_m_offset + m_local_idx
            global_n_idx = block_n_offset + n_local_idx
            if (global_m_idx < m) and (global_n_idx < n):
                flat_offset = m_local_idx * BLOCK_N + n_local_idx
                c_vec = vector.load_op(
                    T.vec(STG_VEC_SIZE, output_dtype_),
                    smem_c_ptr_,
                    [fx.Index(flat_offset)],
                )
                for ksi in range_constexpr(1, BLOCK_K_WARPS):
                    peer_c_vec = vector.load_op(
                        T.vec(STG_VEC_SIZE, output_dtype_),
                        smem_c_ptr_,
                        [fx.Index(flat_offset + ksi * BLOCK_M * BLOCK_N)],
                    )
                    c_vec += peer_c_vec
                global_offset = global_m_idx * c_stride + global_n_idx
                if const_expr(IS_SPLIT_K):
                    # split to vec2s
                    vec2_ty = T.vec(2, output_dtype_)
                    for vec_idx in range_constexpr(STG_VEC_SIZE // 2):
                        e0 = vector.extract(
                            c_vec, static_position=[vec_idx * 2], dynamic_position=[]
                        )
                        e1 = vector.extract(
                            c_vec,
                            static_position=[vec_idx * 2 + 1],
                            dynamic_position=[],
                        )
                        pair = vector.from_elements(vec2_ty, [e0, e1])
                        pair_v = (
                            pair._value if const_expr(hasattr(pair, "_value")) else pair
                        )
                        pair_ptr_v = get_llvm_ptr(
                            c_ptr,
                            global_offset + vec_idx * 2,
                            OUT_DTYPE_BYTES,
                            ir.Type.parse("!llvm.ptr<1>"),
                        )
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            pair_ptr_v,
                            pair_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=4,
                        )
                else:
                    buffer_ops.buffer_store(c_vec, c_rsrc, global_offset)
        return

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hgemm_ht_kernel(
        c_ptr: fx.Pointer,
        a_ptr: fx.Pointer,
        b_ptr: fx.Pointer,
        scale_a_ptr: fx.Pointer,
        scale_b_ptr: fx.Pointer,
        bias_ptr: fx.Pointer,
        semaphore_ptr: fx.Pointer,
        signal_ptr: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        working_k: fx.Int32,
        num_pid_m: fx.Int32,
        num_pid_n: fx.Int32,
        c_stride: fx.Int32,
        a_stride: fx.Int32,
        b_stride: fx.Int32,
    ):
        input_dtype_ = (
            fx.Float8E4M3FN.ir_type if IS_FP8_PTPC else get_dtype_in_kernel(DTYPE)
        )
        output_dtype_ = T.bf16 if const_expr(IS_FP8) else input_dtype_
        smem_input_dtype_ = T.i8 if const_expr(IS_FP8) else input_dtype_
        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))

        c_rsrc = buffer_ops.create_buffer_resource(c_ptr, max_size=True)
        a_rsrc = buffer_ops.create_buffer_resource(a_ptr, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(b_ptr, max_size=True)
        if const_expr(IS_FP8_PTPC):
            scale_a_rsrc = buffer_ops.create_buffer_resource(scale_a_ptr, max_size=True)
            scale_b_rsrc = buffer_ops.create_buffer_resource(scale_b_ptr, max_size=True)
        else:
            scale_a_rsrc = scale_b_rsrc = None
        if const_expr(HAS_BIAS):
            bias_rsrc = buffer_ops.create_buffer_resource(bias_ptr, max_size=True)
        else:
            bias_rsrc = None

        base_ptr = allocator.get_base()
        smem_a_ptr = SmemPtr(
            base_ptr,
            smem_a_offset,
            smem_input_dtype_,
            shape=(STAGES * BLOCK_M * BLOCK_K,),
        )
        smem_b_ptr = SmemPtr(
            base_ptr,
            smem_b_offset,
            smem_input_dtype_,
            shape=(STAGES * BLOCK_N * BLOCK_K,),
        )
        smem_c_ptr = SmemPtr(
            base_ptr,
            smem_a_offset,
            output_dtype_,
            shape=(BLOCK_K_WARPS * BLOCK_M * BLOCK_N,),
        )

        tid = fx.thread_idx.x
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE

        block_m_idx, block_n_idx = block_swizzle.swizzle(
            num_pid_m, num_pid_n, fx.block_idx.x
        )
        ks_idx = fx.block_idx.y
        ks_begin = ks_idx * working_k
        ks_end = ks_begin + working_k
        ks_end = (ks_end < k).select(ks_end, k)
        block_m_offset = block_m_idx * BLOCK_M
        block_n_offset = block_n_idx * BLOCK_N
        k_blocks16 = BLOCK_K_BYTES // 16

        warp_m_idx = wid // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid % BLOCK_N_WARPS * WARP_N
        ldmatrix_a_m_idx = w_tid % WMMA_M
        ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K
        ldmatrix_b_n_idx = w_tid % WMMA_N
        ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K

        c_frags = [acc_init] * (4 * C_FRAGS_LEN)
        stmatrix_c_n_idx = w_tid % WMMA_N
        stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES

        if const_expr(HAS_BIAS and not IS_SPLIT_K and not IS_FP8_PTPC):
            bias_frags = [acc_init] * (2 * WARP_N_STEPS)
            for n_part in range_constexpr(2):
                for ni in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + ni * WARP_ATOM_N
                    lds_n_idx = (
                        n_part * HALF_BLOCK_N + warp_atom_n_idx + stmatrix_c_n_idx
                    )
                    global_n_idx = block_n_offset + lds_n_idx
                    safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
                    bias_val = buffer_ops.buffer_load(
                        bias_rsrc, safe_global_n_idx, vec_width=1, dtype=output_dtype_
                    ).extf(T.f32)
                    bias_frags[n_part * WARP_N_STEPS + ni] = vector.broadcast(
                        T.vec(WMMA_C_FRAG_VALUES, T.f32), bias_val
                    )
            for m_part in range_constexpr(2):
                for n_part in range_constexpr(2):
                    for mi in range_constexpr(WARP_M_STEPS):
                        for ni in range_constexpr(WARP_N_STEPS):
                            c_frags[
                                (m_part * 2 + n_part) * C_FRAGS_LEN
                                + mi * WARP_N_STEPS
                                + ni
                            ] = bias_frags[n_part * WARP_N_STEPS + ni]

        swizzle_cache_a = [0] * LDG_A_ITERS_AS
        for i in range_constexpr(LDG_A_ITERS_AS):
            global_tid = BLOCK_THREADS * i + tid
            m_local_idx = global_tid // LDG_A_X_THREADS_AS
            k_local_idx = global_tid % LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            if const_expr(IS_FP8_PTPC):
                col_in_bytes = swizzle_fp8_128(m_local_idx, k_local_idx)
            else:
                col_in_bytes = swizzle_xor16(
                    m_local_idx, k_local_idx * IN_DTYPE_BYTES, k_blocks16
                )
            swizzle_cache_a[i] = col_in_bytes // IN_DTYPE_BYTES

        swizzle_cache_b = [0] * LDG_B_ITERS_AS
        for i in range_constexpr(LDG_B_ITERS_AS):
            global_tid = BLOCK_THREADS * i + tid
            n_local_idx = global_tid // LDG_B_X_THREADS_AS
            k_local_idx = global_tid % LDG_B_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            if const_expr(IS_FP8_PTPC):
                col_in_bytes = swizzle_fp8_128(n_local_idx, k_local_idx)
            else:
                col_in_bytes = swizzle_xor16(
                    n_local_idx, k_local_idx * IN_DTYPE_BYTES, k_blocks16
                )
            swizzle_cache_b[i] = col_in_bytes // IN_DTYPE_BYTES

        if const_expr(IS_SPLIT_K):
            signal_idx = fx.block_idx.x
            splitk_protocol.init(
                semaphore_ptr,
                signal_ptr,
                c_ptr,
                bias_rsrc,
                tid,
                ks_idx,
                m,
                n,
                block_m_offset,
                block_n_offset,
                output_dtype_,
                signal_idx,
                c_stride,
            )

        def get_dma_copy_warp_offset():
            return rocdl.readfirstlane(T.i64, fx.Int64(wid * WARP_SIZE * DMA_BYTES))

        warp_offset = get_dma_copy_warp_offset()

        def get_smem_warp_ptr(memptr):
            lds_base = memref.extract_aligned_pointer_as_index(memptr)
            lds_ptr_base = buffer_ops.create_llvm_ptr(
                arith.index_cast(T.i64, lds_base), address_space=3
            )
            return buffer_ops.get_element_ptr(lds_ptr_base, warp_offset)

        as_warp_ptr = get_smem_warp_ptr(smem_a_ptr.get())
        bs_warp_ptr = get_smem_warp_ptr(smem_b_ptr.get())

        def get_lds_ptr(base_ptr, static_bytes_offset, lds_ptr=None):
            if const_expr(lds_ptr is None):
                return buffer_ops.get_element_ptr(
                    base_ptr, static_byte_offset=static_bytes_offset
                )
            return buffer_ops.get_element_ptr(
                lds_ptr, static_byte_offset=BLOCK_THREADS * DMA_BYTES
            )

        def ldg_sts_a_async_one(m_part, k_buf, k_offset, ii, lds_ptr=None):
            global_tid = BLOCK_THREADS * ii + tid
            m_local_idx = global_tid // LDG_A_X_THREADS_AS
            global_m_idx = block_m_offset + m_part * HALF_BLOCK_M + m_local_idx
            safe_global_m_idx = (global_m_idx < m).select(global_m_idx, 0)
            global_k_idx = k_offset + k_buf * BLOCK_K + swizzle_cache_a[ii]
            if const_expr(HAS_K_TAIL):
                safe_global_k_idx = (global_k_idx < ks_end).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            global_offset = (
                safe_global_m_idx * a_stride + safe_global_k_idx
            ) * IN_DTYPE_BYTES
            global_offset = arith.index_cast(T.i32, global_offset)
            static_bytes_offset = (
                (k_buf * 2 + m_part) * HALF_BLOCK_M * BLOCK_K * IN_DTYPE_BYTES
            )
            lds_ptr = get_lds_ptr(as_warp_ptr, static_bytes_offset, lds_ptr)
            buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset, DMA_BYTES)
            return lds_ptr

        def ldg_sts_b_async_one(n_part, k_buf, k_offset, ii, lds_ptr=None):
            global_tid = BLOCK_THREADS * ii + tid
            n_local_idx = global_tid // LDG_B_X_THREADS_AS
            global_n_idx = block_n_offset + n_part * HALF_BLOCK_N + n_local_idx
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            global_k_idx = k_offset + k_buf * BLOCK_K + swizzle_cache_b[ii]
            if const_expr(HAS_K_TAIL):
                safe_global_k_idx = (global_k_idx < ks_end).select(global_k_idx, 0)
            else:
                safe_global_k_idx = global_k_idx
            global_offset = (
                safe_global_n_idx * b_stride + safe_global_k_idx
            ) * IN_DTYPE_BYTES
            global_offset = arith.index_cast(T.i32, global_offset)
            static_bytes_offset = (
                (k_buf * 2 + n_part) * HALF_BLOCK_N * BLOCK_K * IN_DTYPE_BYTES
            )
            lds_ptr = get_lds_ptr(bs_warp_ptr, static_bytes_offset, lds_ptr)
            buffer_load_lds_inline(b_rsrc, lds_ptr, global_offset, DMA_BYTES)
            return lds_ptr

        def ldg_sts_a_async(m_part, k_buf, k_offset):
            lds_ptr = None
            for i in range_constexpr(LDG_A_ITERS_AS):
                lds_ptr = ldg_sts_a_async_one(
                    m_part, k_buf, k_offset, i, lds_ptr if i > 0 else None
                )

        def ldg_sts_b_async(n_part, k_buf, k_offset):
            lds_ptr = None
            for i in range_constexpr(LDG_B_ITERS_AS):
                lds_ptr = ldg_sts_b_async_one(
                    n_part, k_buf, k_offset, i, lds_ptr if i > 0 else None
                )

        def mask_tail_k_frag(frag, k_base, frag_values):
            elems = [0] * frag_values
            for vi in range_constexpr(frag_values):
                global_k_idx = k_base + vi
                valid_k = global_k_idx < ks_end
                elem = vector.extract(frag, static_position=[vi], dynamic_position=[])
                elems[vi] = arith.select(
                    valid_k, elem, arith.constant(0, type=smem_input_dtype_)
                )
            return vector.from_elements(T.vec(frag_values, smem_input_dtype_), elems)

        def ldmatrix_a(m_part, k_buf, k_tile_offset, mask_k_tail=False):
            a_frags = [0] * A_FRAGS_LEN
            for mi in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + mi * WARP_ATOM_M
                for ki in range_constexpr(WARP_K_STEPS):
                    if const_expr(IS_FP8_PTPC):
                        row = warp_atom_m_idx + ldmatrix_a_m_idx
                    else:
                        row = warp_atom_m_idx + ldmatrix_a_m_idx
                        warp_atom_k_idx = ki * WARP_ATOM_K
                        col_in_bytes = (
                            warp_atom_k_idx + ldmatrix_a_k_vec_idx
                        ) * IN_DTYPE_BYTES
                    if const_expr(IS_FP8_PTPC):

                        def load_i32x4_at(col_delta):
                            col_base = ki * WMMA_K + (w_tid // WMMA_M) * 16
                            col_swz = swizzle_fp8_128(row, col_base + col_delta)
                            flat_offset = (
                                (k_buf * 2 + m_part) * HALF_BLOCK_M + row
                            ) * BLOCK_K + col_swz
                            v16 = vector.load_op(
                                T.vec(LDG_VEC_SIZE, smem_input_dtype_),
                                smem_a_ptr.get(),
                                [fx.Index(flat_offset)],
                            )
                            if const_expr(mask_k_tail):
                                v16 = mask_tail_k_frag(
                                    v16,
                                    k_tile_offset + col_base + col_delta,
                                    LDG_VEC_SIZE,
                                )
                            return fx.Vector(v16).bitcast(fx.Int32)

                        lo = load_i32x4_at(0)
                        hi = load_i32x4_at(64)
                        a_frags[ki * WARP_M_STEPS + mi] = fx.Vector(lo).shuffle(
                            hi, list(range(8))
                        )
                    else:
                        col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                        flat_offset = (
                            (k_buf * 2 + m_part) * HALF_BLOCK_M + row
                        ) * BLOCK_K + col_in_bytes // IN_DTYPE_BYTES
                        vec = vector.load_op(
                            T.vec(
                                WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K, smem_input_dtype_
                            ),
                            smem_a_ptr.get(),
                            [fx.Index(flat_offset)],
                        )
                        if const_expr(mask_k_tail):
                            a_frags[ki * WARP_M_STEPS + mi] = mask_tail_k_frag(
                                vec,
                                k_tile_offset + warp_atom_k_idx + ldmatrix_a_k_vec_idx,
                                WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K,
                            )
                        else:
                            a_frags[ki * WARP_M_STEPS + mi] = vec
            return a_frags

        def ldmatrix_b(n_part, k_buf, k_tile_offset, mask_k_tail=False):
            b_frags = [0] * B_FRAGS_LEN
            for ni in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + ni * WARP_ATOM_N
                for ki in range_constexpr(WARP_K_STEPS):
                    if const_expr(IS_FP8_PTPC):
                        row = warp_atom_n_idx + ldmatrix_b_n_idx
                    else:
                        row = warp_atom_n_idx + ldmatrix_b_n_idx
                        warp_atom_k_idx = ki * WARP_ATOM_K
                        col_in_bytes = (
                            warp_atom_k_idx + ldmatrix_b_k_vec_idx
                        ) * IN_DTYPE_BYTES
                    if const_expr(IS_FP8_PTPC):

                        def load_i32x4_at(col_delta):
                            col_base = ki * WMMA_K + (w_tid // WMMA_N) * 16
                            col_swz = swizzle_fp8_128(row, col_base + col_delta)
                            flat_offset = (
                                (k_buf * 2 + n_part) * HALF_BLOCK_N + row
                            ) * BLOCK_K + col_swz
                            v16 = vector.load_op(
                                T.vec(LDG_VEC_SIZE, smem_input_dtype_),
                                smem_b_ptr.get(),
                                [fx.Index(flat_offset)],
                            )
                            if const_expr(mask_k_tail):
                                v16 = mask_tail_k_frag(
                                    v16,
                                    k_tile_offset + col_base + col_delta,
                                    LDG_VEC_SIZE,
                                )
                            return fx.Vector(v16).bitcast(fx.Int32)

                        lo = load_i32x4_at(0)
                        hi = load_i32x4_at(64)
                        b_frags[ki * WARP_N_STEPS + ni] = fx.Vector(lo).shuffle(
                            hi, list(range(8))
                        )
                    else:
                        col_in_bytes = swizzle_xor16(
                            row, col_in_bytes, fx.Int32(k_blocks16)
                        )
                        flat_offset = (
                            (k_buf * 2 + n_part) * HALF_BLOCK_N + row
                        ) * BLOCK_K + col_in_bytes // IN_DTYPE_BYTES
                        vec = vector.load_op(
                            T.vec(
                                WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K, smem_input_dtype_
                            ),
                            smem_b_ptr.get(),
                            [fx.Index(flat_offset)],
                        )
                        if const_expr(mask_k_tail):
                            b_frags[ki * WARP_N_STEPS + ni] = mask_tail_k_frag(
                                vec,
                                k_tile_offset + warp_atom_k_idx + ldmatrix_b_k_vec_idx,
                                WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
                            )
                        else:
                            b_frags[ki * WARP_N_STEPS + ni] = vec
            return b_frags

        def consume(m_part, n_part, a_frags, b_frags, c_frags_in, emit_sched_barrier):
            c_frags_out = [cx for cx in c_frags_in]
            if const_expr(emit_sched_barrier):
                rocdl.sched_barrier(0)
            for mi in range_constexpr(WARP_M_STEPS):
                for ni in range_constexpr(WARP_N_STEPS):
                    for ki in range_constexpr(WARP_K_STEPS):
                        c_idx = (
                            (m_part * 2 + n_part) * C_FRAGS_LEN + mi * WARP_N_STEPS + ni
                        )
                        c_frags_out[c_idx] = WMMA_IMPL(
                            a_frags[ki * WARP_M_STEPS + mi],
                            b_frags[ki * WARP_N_STEPS + ni],
                            c_frags_out[c_idx],
                        )
            if const_expr(emit_sched_barrier):
                rocdl.sched_barrier(0)
            return c_frags_out

        if const_expr(IS_SPLIT_K):
            splitk_protocol.zero_c()

        ldg_sts_b_async(0, 0, ks_begin)
        ldg_sts_a_async(0, 0, ks_begin)
        ldg_sts_b_async(1, 0, ks_begin)
        ldg_sts_a_async(1, 0, ks_begin)
        rocdl.sched_barrier(0)
        if wid // BLOCK_N_WARPS == 1:
            __s_barrier()
        rocdl.sched_barrier(0)
        __s_barrier()
        rocdl.sched_barrier(0)
        ldg_sts_b_async(0, 1, ks_begin)
        ldg_sts_a_async(0, 1, ks_begin)
        ldg_sts_b_async(1, 1, ks_begin)
        __barrier(1 * LDG_B_ITERS_AS + 1 * LDG_A_ITERS_AS)

        def compute_double_tile(k_offset, c_frags_in):
            next_k_offset = k_offset + 2 * BLOCK_K
            # 0
            b0_frags = ldmatrix_b(0, 0, k_offset)
            a0_frags = ldmatrix_a(0, 0, k_offset)
            ldg_sts_a_async(1, 1, k_offset)
            __s_barrier()
            c_frags_out = consume(0, 0, a0_frags, b0_frags, c_frags_in, True)
            __s_barrier()
            b1_frags = ldmatrix_b(1, 0, k_offset)
            ldg_sts_b_async(0, 0, next_k_offset)
            __s_barrier()
            c_frags_out = consume(0, 1, a0_frags, b1_frags, c_frags_out, True)
            __s_barrier()
            a1_frags = ldmatrix_a(1, 0, k_offset)
            ldg_sts_a_async(0, 0, next_k_offset)
            __s_barrier()
            c_frags_out = consume(1, 0, a1_frags, b0_frags, c_frags_out, True)
            __s_barrier()
            b0_frags = ldmatrix_b(0, 1, k_offset + BLOCK_K)
            ldg_sts_b_async(1, 0, next_k_offset)
            __barrier(2 * LDG_B_ITERS_AS + 1 * LDG_A_ITERS_AS)
            c_frags_out = consume(1, 1, a1_frags, b1_frags, c_frags_out, True)
            __s_barrier()
            # 1
            a0_frags = ldmatrix_a(0, 1, k_offset + BLOCK_K)
            ldg_sts_a_async(1, 0, next_k_offset)
            __s_barrier()
            c_frags_out = consume(0, 0, a0_frags, b0_frags, c_frags_out, True)
            __s_barrier()
            b1_frags = ldmatrix_b(1, 1, k_offset + BLOCK_K)
            ldg_sts_b_async(0, 1, next_k_offset)
            __s_barrier()
            c_frags_out = consume(0, 1, a0_frags, b1_frags, c_frags_out, True)
            __s_barrier()
            a1_frags = ldmatrix_a(1, 1, k_offset + BLOCK_K)
            ldg_sts_a_async(0, 1, next_k_offset)
            __s_barrier()
            c_frags_out = consume(1, 0, a1_frags, b0_frags, c_frags_out, True)
            __s_barrier()
            ldg_sts_b_async(1, 1, next_k_offset)
            __barrier(1 * LDG_B_ITERS_AS + 1 * LDG_A_ITERS_AS)
            c_frags_out = consume(1, 1, a1_frags, b1_frags, c_frags_out, True)
            __s_barrier()
            return c_frags_out

        BLOCK_K_LOOPS = (working_k + BLOCK_K - 1) // BLOCK_K
        loop_end = (BLOCK_K_LOOPS > 2).select(BLOCK_K_LOOPS - 2, 0)
        init_state = [ks_begin] + c_frags
        for _, state in range(0, loop_end, 2, init=init_state):
            k_offset = state[0]
            c_frags = state[1:]
            c_frags = compute_double_tile(k_offset, c_frags)
            results = yield [k_offset + 2 * BLOCK_K] + c_frags
        k_offset = results[0]
        c_frags = results[1:]

        def load_scale_b(n_part):
            scale_b_frags = [fx.Float32(0.0)] * WARP_N_STEPS
            for ni in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + ni * WARP_ATOM_N
                local_n_idx = fx.Index(
                    n_part * HALF_BLOCK_N + warp_atom_n_idx + stmatrix_c_n_idx
                )
                global_n_offset = block_n_offset + local_n_idx
                scale_b_offset = (global_n_offset < n).select(global_n_offset, 0)
                scale_b = buffer_ops.buffer_load(
                    scale_b_rsrc, scale_b_offset, vec_width=1, dtype=T.f32
                )
                scale_b_frags[ni] = scale_b
            return scale_b_frags

        def store_matrix_to_lds(m_, n_, c_frags, scale_b_frags=None):
            c_base = (m_ * 2 + n_) * C_FRAGS_LEN
            for mi in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + mi * WARP_ATOM_M
                lds_m_base = m_ * HALF_BLOCK_M + warp_atom_m_idx + stmatrix_c_m_vec_idx
                for ni in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + ni * WARP_ATOM_N
                    c_idx = c_base + mi * WARP_N_STEPS + ni
                    lds_n_idx = n_ * HALF_BLOCK_N + warp_atom_n_idx + stmatrix_c_n_idx
                    col_global = block_n_offset + lds_n_idx
                    safe_col_global = (col_global < n).select(col_global, 0)
                    if const_expr(HAS_BIAS and IS_FP8_PTPC and not IS_SPLIT_K):
                        bias = buffer_ops.buffer_load(
                            bias_rsrc,
                            safe_col_global,
                            vec_width=1,
                            dtype=output_dtype_,
                        ).extf(T.f32)
                    if const_expr(IS_FP8_PTPC):
                        scale_b = scale_b_frags[ni]
                    for kk in range_constexpr(WMMA_C_FRAG_VALUES):
                        lds_m_idx = lds_m_base + kk
                        val = vector.extract(
                            c_frags[c_idx],
                            static_position=[kk],
                            dynamic_position=[],
                        )
                        if const_expr(IS_FP8_PTPC):
                            row_global = block_m_offset + lds_m_idx
                            scale_a_offset = (row_global < m).select(row_global, 0)
                            scale_a = buffer_ops.buffer_load(
                                scale_a_rsrc,
                                scale_a_offset,
                                vec_width=1,
                                dtype=T.f32,
                            )
                            val = val * scale_a * scale_b
                            if const_expr(HAS_BIAS and not IS_SPLIT_K):
                                val = val + bias
                        val = val.truncf(output_dtype_)
                        val = vector.from_elements(T.vec(1, output_dtype_), [val])
                        flat_offset = lds_m_idx * BLOCK_N + lds_n_idx
                        vector.store(
                            val, smem_c_ptr.get(), [fx.Index(flat_offset)], alignment=16
                        )

        def atomic_add_vec_to_c(global_m_idx, global_n_idx, vec):
            global_offset = global_m_idx * c_stride + global_n_idx
            vec2_ty = T.vec(2, output_dtype_)
            for vec_idx in range_constexpr(STG_VEC_SIZE // 2):
                e0 = vector.extract(
                    vec, static_position=[vec_idx * 2], dynamic_position=[]
                )
                e1 = vector.extract(
                    vec,
                    static_position=[vec_idx * 2 + 1],
                    dynamic_position=[],
                )
                pair = vector.from_elements(vec2_ty, [e0, e1])
                pair_v = pair._value if const_expr(hasattr(pair, "_value")) else pair
                pair_ptr_v = get_llvm_ptr(
                    c_ptr,
                    global_offset + vec_idx * 2,
                    OUT_DTYPE_BYTES,
                    ir.Type.parse("!llvm.ptr<1>"),
                )
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    pair_ptr_v,
                    pair_v,
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )

        def store_vec_to_c(global_m_idx, global_n_idx, vec):
            if const_expr(IS_SPLIT_K):
                atomic_add_vec_to_c(global_m_idx, global_n_idx, vec)
            else:
                global_offset = global_m_idx * c_stride + global_n_idx
                buffer_ops.buffer_store(vec, c_rsrc, global_offset)

        def store_matrix_from_lds(m_, n_):
            smem_c_ptr_ = smem_c_ptr.get()
            for mi in range_constexpr(WARP_M_STEPS):
                for i in range_constexpr(STG_WORK_SIZE_PER_M_STEP // STG_VEC_SIZE):
                    global_tid = BLOCK_THREADS * i + tid
                    m_band_idx = global_tid // STG_C_QUAD_X_THREADS
                    n_local_idx = global_tid % STG_C_QUAD_X_THREADS * STG_VEC_SIZE
                    warp_m_band = m_band_idx // WARP_ATOM_M
                    atom_m_idx = m_band_idx % WARP_ATOM_M
                    m_tile_idx = (
                        m_ * HALF_BLOCK_M
                        + warp_m_band * WARP_M
                        + mi * WARP_ATOM_M
                        + atom_m_idx
                    )
                    n_tile_idx = n_ * HALF_BLOCK_N + n_local_idx
                    global_m_idx = block_m_offset + m_tile_idx
                    global_n_idx = block_n_offset + n_tile_idx

                    if (global_m_idx < m) and (global_n_idx < n):
                        flat_offset = m_tile_idx * BLOCK_N + n_tile_idx
                        c_vec = vector.load_op(
                            T.vec(STG_VEC_SIZE, output_dtype_),
                            smem_c_ptr_,
                            [fx.Index(flat_offset)],
                        )
                        store_vec_to_c(
                            global_m_idx,
                            global_n_idx,
                            c_vec,
                        )

        def compute_final_tile_and_epilogue(k_offset, c_frags_in):
            # 0
            b0_frags = ldmatrix_b(0, 0, k_offset, HAS_K_TAIL)
            a0_frags = ldmatrix_a(0, 0, k_offset, HAS_K_TAIL)
            ldg_sts_a_async(1, 1, k_offset)
            __s_barrier()
            c_frags_out = consume(0, 0, a0_frags, b0_frags, c_frags_in, True)
            __s_barrier()
            b1_frags = ldmatrix_b(1, 0, k_offset, HAS_K_TAIL)
            __s_barrier()
            c_frags_out = consume(0, 1, a0_frags, b1_frags, c_frags_out, True)
            __s_barrier()
            a1_frags = ldmatrix_a(1, 0, k_offset, HAS_K_TAIL)
            __s_barrier()
            c_frags_out = consume(1, 0, a1_frags, b0_frags, c_frags_out, True)
            __s_barrier()
            b0_frags = ldmatrix_b(0, 1, k_offset + BLOCK_K, HAS_K_TAIL)
            __s_barrier()
            c_frags_out = consume(1, 1, a1_frags, b1_frags, c_frags_out, True)
            # 1
            __barrier(0)
            a0_frags = ldmatrix_a(0, 1, k_offset + BLOCK_K, HAS_K_TAIL)
            __s_barrier()
            c_frags_out = consume(0, 0, a0_frags, b0_frags, c_frags_out, True)
            __s_barrier()
            b1_frags = ldmatrix_b(1, 1, k_offset + BLOCK_K, HAS_K_TAIL)
            __s_barrier()
            c_frags_out = consume(0, 1, a0_frags, b1_frags, c_frags_out, True)
            __s_barrier()
            a1_frags = ldmatrix_a(1, 1, k_offset + BLOCK_K, HAS_K_TAIL)
            __s_barrier()
            rocdl.sched_barrier(0)
            if const_expr(IS_FP8_PTPC):
                scale_b0 = load_scale_b(0)
                store_matrix_to_lds(0, 0, c_frags_out, scale_b0)
                scale_b1 = load_scale_b(1)
                store_matrix_to_lds(0, 1, c_frags_out, scale_b1)
            else:
                store_matrix_to_lds(0, 0, c_frags_out)
                store_matrix_to_lds(0, 1, c_frags_out)
            c_frags_out = consume(1, 0, a1_frags, b0_frags, c_frags_out, False)
            rocdl.sched_barrier(0)
            __s_barrier()
            rocdl.sched_barrier(0)
            if const_expr(IS_SPLIT_K):
                splitk_protocol.split_k_barrier()
            store_matrix_from_lds(0, 0)
            store_matrix_from_lds(0, 1)
            if const_expr(IS_FP8_PTPC):
                store_matrix_to_lds(1, 0, c_frags_out, scale_b0)
            else:
                store_matrix_to_lds(1, 0, c_frags_out)
            c_frags_out = consume(1, 1, a1_frags, b1_frags, c_frags_out, False)
            rocdl.sched_barrier(0)
            __s_barrier()
            store_matrix_from_lds(1, 0)
            if const_expr(IS_FP8_PTPC):
                store_matrix_to_lds(1, 1, c_frags_out, scale_b1)
            else:
                store_matrix_to_lds(1, 1, c_frags_out)
            __s_barrier()
            store_matrix_from_lds(1, 1)
            return c_frags_out

        c_frags = compute_final_tile_and_epilogue(k_offset, c_frags)
        return

    @flyc.jit
    def launch_hgemm_wmma_kernel(
        c_ptr: fx.Pointer,
        a_ptr: fx.Pointer,
        b_ptr: fx.Pointer,
        scale_a_ptr: fx.Pointer,
        scale_b_ptr: fx.Pointer,
        bias_ptr: fx.Pointer,
        semaphore_ptr: fx.Pointer,
        signal_ptr: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        working_k = (k + SPLIT_K - 1) // SPLIT_K
        num_pid_m = (m + BLOCK_M - 1) // BLOCK_M
        num_pid_n = (n + BLOCK_N - 1) // BLOCK_N
        c_stride = n
        a_stride = k
        b_stride = k
        hgemm_kernel_impl = hgemm_ht_kernel if IS_HT else hgemm_kernel
        hgemm_kernel_impl._func.__name__ = KERNEL_NAME
        hgemm_kernel_impl(
            c_ptr,
            a_ptr,
            b_ptr,
            scale_a_ptr,
            scale_b_ptr,
            bias_ptr,
            semaphore_ptr,
            signal_ptr,
            m,
            n,
            k,
            working_k,
            num_pid_m,
            num_pid_n,
            c_stride,
            a_stride,
            b_stride,
        ).launch(
            grid=(num_pid_m * num_pid_n, SPLIT_K, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_hgemm_wmma_kernel


def get_default_b16_kwargs(m, n, k):
    kwargs = {
        "TILE_M": 256,
        "TILE_N": 256,
        "TILE_K": 64,
        "STAGES": 2,
        "SPLIT_K": 1,
        "BLOCK_M_WARPS": 2,
        "BLOCK_N_WARPS": 4,
        "BLOCK_K_WARPS": 1,
        "USE_HALF_TILE_INTERLEAVED": True,
        "GROUP_M": 0,
    }
    if m == 2048 and n == 2048 and k == 2048:
        kwargs["TILE_M"] = 128
        kwargs["TILE_N"] = 128
        kwargs["TILE_K"] = 64
        kwargs["STAGES"] = 4
        kwargs["SPLIT_K"] = 1
        kwargs["BLOCK_M_WARPS"] = 4
        kwargs["BLOCK_N_WARPS"] = 4
        kwargs["BLOCK_K_WARPS"] = 1
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    elif m <= 32 and n == 384 and k == 7168:
        kwargs["TILE_M"] = 32
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 64
        kwargs["STAGES"] = 5
        kwargs["SPLIT_K"] = 16
        kwargs["BLOCK_M_WARPS"] = 2
        kwargs["BLOCK_N_WARPS"] = 2
        kwargs["BLOCK_K_WARPS"] = 1
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    elif m <= 32 and n == 7168 and k == 2048:
        kwargs["TILE_M"] = 16
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 128
        kwargs["STAGES"] = 4
        kwargs["SPLIT_K"] = 1
        kwargs["BLOCK_M_WARPS"] = 1
        kwargs["BLOCK_N_WARPS"] = 1
        kwargs["BLOCK_K_WARPS"] = 2
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    elif m <= 32 and n == 384 and k == 16384:
        kwargs["TILE_M"] = 32
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 256
        kwargs["STAGES"] = 3
        kwargs["SPLIT_K"] = 16
        kwargs["BLOCK_M_WARPS"] = 1
        kwargs["BLOCK_N_WARPS"] = 4
        kwargs["BLOCK_K_WARPS"] = 1
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    elif m <= 16 and n == 5120 and k == 2880:
        kwargs["TILE_M"] = 16
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 64
        kwargs["STAGES"] = 5
        kwargs["SPLIT_K"] = 3
        kwargs["BLOCK_M_WARPS"] = 1
        kwargs["BLOCK_N_WARPS"] = 2
        kwargs["BLOCK_K_WARPS"] = 1
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    elif m <= 32 and n == 2880 and k == 2048:
        kwargs["TILE_M"] = 16
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 128
        kwargs["STAGES"] = 5
        kwargs["SPLIT_K"] = 2
        kwargs["BLOCK_M_WARPS"] = 1
        kwargs["BLOCK_N_WARPS"] = 2
        kwargs["BLOCK_K_WARPS"] = 1
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    elif m <= 800 and n == 384 and k == 7168:
        kwargs["TILE_M"] = 32
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 128
        kwargs["STAGES"] = 6
        kwargs["SPLIT_K"] = 1
        kwargs["BLOCK_M_WARPS"] = 1
        kwargs["BLOCK_N_WARPS"] = 2
        kwargs["BLOCK_K_WARPS"] = 2
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    return kwargs


def get_default_b8_kwargs(m, n, k):
    kwargs = {
        "TILE_M": 256,
        "TILE_N": 256,
        "TILE_K": 128,
        "STAGES": 2,
        "SPLIT_K": 1,
        "BLOCK_M_WARPS": 2,
        "BLOCK_N_WARPS": 4,
        "BLOCK_K_WARPS": 1,
        "USE_HALF_TILE_INTERLEAVED": True,
        "GROUP_M": 0,
    }
    if m == 2048 and n == 2048 and k == 2048:
        kwargs["TILE_M"] = 128
        kwargs["TILE_N"] = 128
        kwargs["TILE_K"] = 128
        kwargs["STAGES"] = 4
        kwargs["SPLIT_K"] = 1
        kwargs["BLOCK_M_WARPS"] = 4
        kwargs["BLOCK_N_WARPS"] = 4
        kwargs["BLOCK_K_WARPS"] = 1
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    elif m <= 32 and n == 384 and k == 7168:
        kwargs["TILE_M"] = 32
        kwargs["TILE_N"] = 64
        kwargs["TILE_K"] = 128
        kwargs["STAGES"] = 5
        kwargs["SPLIT_K"] = 8
        kwargs["BLOCK_M_WARPS"] = 2
        kwargs["BLOCK_N_WARPS"] = 2
        kwargs["BLOCK_K_WARPS"] = 1
        kwargs["USE_HALF_TILE_INTERLEAVED"] = False
    return kwargs


def hgemm_validate(dtype_str, m, n, k, kwargs):
    TILE_M = kwargs["TILE_M"]
    TILE_N = kwargs["TILE_N"]
    TILE_K = kwargs["TILE_K"]
    STAGES = kwargs["STAGES"]
    SPLIT_K = kwargs["SPLIT_K"]
    BLOCK_M_WARPS = kwargs["BLOCK_M_WARPS"]
    BLOCK_N_WARPS = kwargs["BLOCK_N_WARPS"]
    BLOCK_K_WARPS = kwargs["BLOCK_K_WARPS"]
    USE_HALF_TILE_INTERLEAVED = kwargs.get("USE_HALF_TILE_INTERLEAVED", False)
    GPU_ARCH = get_rocm_arch()
    IS_FP8 = "fp8" in dtype_str
    DTYPE_BYTES = 1 if IS_FP8 else 2

    if USE_HALF_TILE_INTERLEAVED:
        if not (STAGES == 2 and BLOCK_K_WARPS == 1 and BLOCK_M_WARPS == 2):
            return False

    def get_stage_smem_use(stages_):
        SMEM_USE = stages_ * TILE_M * TILE_K * DTYPE_BYTES
        SMEM_USE += stages_ * TILE_N * TILE_K * DTYPE_BYTES
        SMEM_USE = max(SMEM_USE, BLOCK_K_WARPS * TILE_M * TILE_N * DTYPE_BYTES)
        return SMEM_USE

    smem_use_s0 = get_stage_smem_use(STAGES)
    # smem_use_s1 = get_stage_smem_use(STAGES + 3)
    smem_cap = SMEM_CAPACITY_MAP[GPU_ARCH]
    if not (smem_use_s0 <= smem_cap):
        return False
    if m >= 4096 and n >= 4096 and k >= 4096:
        if not (
            TILE_M == 256 and TILE_N == 256 and TILE_K == 128
            if IS_FP8
            else 64
            and STAGES == 2
            and SPLIT_K == 1
            and BLOCK_M_WARPS == 2
            and BLOCK_N_WARPS == 4
            and BLOCK_K_WARPS == 1
        ):
            return False
    if kwargs["SPLIT_K"] > 1:
        global SPLIT_K_SEMAPHORE_MAX_LEN
        bm = (m + TILE_M - 1) // TILE_M
        bn = (n + TILE_N - 1) // TILE_N
        if not (bm * bn <= SPLIT_K_SEMAPHORE_MAX_LEN):
            return False
    if not ((n % 16 == 0) and (k % 16 == 0)):
        return False
    return True


def hgemm_get_configs(m, n, k):
    selections = {
        "TILE_M": [16, 32, 48, 64, 80, 96, 128, 256],
        "TILE_N": [64, 80, 96, 128, 256],
        "TILE_K": [64, 128, 256],
        "STAGES": [i for i in range(2, 9)],
        "SPLIT_K": [i for i in range(1, 14)],
        "BLOCK_M_WARPS": [1, 2, 4],
        "BLOCK_N_WARPS": [1, 2, 4],
        "BLOCK_K_WARPS": [1, 2, 4],
        "GROUP_M": [0, 4],
        "USE_HALF_TILE_INTERLEAVED": [False, True],
    }
    keys = selections.keys()
    values = selections.values()
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    configs = [config for config in configs if hgemm_validate(m, n, k, config)]
    return configs


@functools.lru_cache(maxsize=128)
def get_semaphore(stream, device):
    semaphore = torch.zeros(
        (SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=device
    )
    signal = torch.zeros((SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=device)
    return semaphore, signal


def hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    user_kwargs: dict = {},
    scale_a: Optional[torch.Tensor] = None,
    scale_b: Optional[torch.Tensor] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    if stream is None:
        stream = torch.cuda.current_stream()
    global SPLIT_K_SEMAPHORE_MAX_LEN
    device = a.device
    semaphore, signal = get_semaphore(stream, device)
    k = a.shape[-1]
    a = a.view(-1, k)
    m = a.shape[0]
    n = b.shape[0]
    assert a.device == b.device
    assert a.dtype == b.dtype
    assert b.shape[1] == k
    assert b.numel() == n * k

    is_fp8_ptpc = scale_a is not None or scale_b is not None

    if c is None:
        if is_fp8_ptpc:
            c = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        else:
            c = torch.empty((m, n), dtype=a.dtype, device=a.device)
    c = c.view(-1, n)
    assert c.shape[0] == m
    assert a.device == c.device

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    kwargs = (
        get_default_b8_kwargs(m, n, k)
        if is_fp8_ptpc
        else get_default_b16_kwargs(m, n, k)
    )
    kwargs.update(user_kwargs)

    working_k = (k + kwargs["SPLIT_K"] - 1) // kwargs["SPLIT_K"]
    last_working_k = k - (kwargs["SPLIT_K"] - 1) * working_k
    kwargs["HAS_K_TAIL"] = (working_k % kwargs["TILE_K"] != 0) or (
        last_working_k % kwargs["TILE_K"] != 0
    )
    if kwargs["USE_HALF_TILE_INTERLEAVED"]:
        working_k_tiles = (working_k + kwargs["TILE_K"] - 1) // kwargs["TILE_K"]
        last_working_k_tiles = (last_working_k + kwargs["TILE_K"] - 1) // kwargs[
            "TILE_K"
        ]
        kwargs["HAS_K_TAIL"] = (
            kwargs["HAS_K_TAIL"]
            or (working_k_tiles % 2 != 0)
            or (last_working_k_tiles % 2 != 0)
        )

    kwargs["HAS_BIAS"] = False if bias is None else True

    bias_tensor = a if bias is None else bias
    if bias is not None:
        assert bias.shape[0] == n

    if is_fp8_ptpc:
        assert scale_a is not None and scale_b is not None
        assert scale_a.shape[0] == m
        assert scale_b.shape[0] == n
        assert c.dtype == torch.bfloat16
        assert hgemm_validate("fp8_ptpc", m, n, k, kwargs)
        exe = compile_hgemm_wmma_kernel("fp8_ptpc", **kwargs)
        _run_compiled(
            exe,
            c,
            a,
            b,
            scale_a,
            scale_b,
            bias_tensor,
            semaphore,
            signal,
            m,
            n,
            k,
            stream,
        )
    else:
        if a.dtype == torch.half:
            assert c.dtype == torch.half
            assert hgemm_validate("fp16", m, n, k, kwargs)
            exe = compile_hgemm_wmma_kernel("f16", **kwargs)
        elif a.dtype == torch.bfloat16:
            assert c.dtype == torch.bfloat16
            assert hgemm_validate("bf16", m, n, k, kwargs)
            exe = compile_hgemm_wmma_kernel("bf16", **kwargs)
        else:
            raise NotImplementedError()
        _run_compiled(
            exe, c, a, b, a, b, bias_tensor, semaphore, signal, m, n, k, stream
        )
    return c
