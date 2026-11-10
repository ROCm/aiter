# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from the MIT-licensed band-stream pipeline in gemm_a16w16_band_gfx950.py.

"""Static-ring BF16 projection GEMM; split partials stay FP32.

Each K step prefetches its successor fragment and interleaves the next tile's
DMA with MFMAs. Stage-ring unrolling removes dynamic LDS address selection.
Explicit vmcnt/barrier and inline DMA boundaries must remain paired.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl

from .gemm_a16w16_band_gfx950 import (
    _bstore,
    _gep,
    _lds_ptr3,
    _rsrc,
    _swizzle_xor16,
)
from .tensor_shim import AITER_FLYDSL_KERNARG_PRELOAD


@functools.lru_cache(maxsize=32)
def build_projection(
    m,
    n,
    k,
    TILE_M,
    TILE_N,
    BK,
    STAGES,
    BLOCK_M_WARPS,
    BLOCK_N_WARPS,
    SPLIT_K,
    K_STAGGER,
    UF,
    SCHED,
    REVERSE,
    STAGGER_AXIS,
    EXPERT,
):
    C_AUX = 0
    A_BITS = "" if n == 4480 else "sc0"
    B_BITS = "sc0"
    WARP_SIZE = 64
    VMCNT_MAX = 63
    DTYPE_BYTES = 2
    LDG_VEC_SIZE = 8
    DMA_BYTES = 16
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 32
    WMMA_FRAG_VALUES = 8
    WMMA_C_FRAG_VALUES = 4
    BLOCK_K = BK
    assert k % (BK * SPLIT_K) == 0
    PART_K = k // SPLIT_K
    BLOCK_K_LOOPS = PART_K // BLOCK_K
    assert k % BLOCK_K == 0 and BLOCK_K_LOOPS >= STAGES
    WARP_K_STEPS = BLOCK_K // WMMA_K
    assert BLOCK_K % (WMMA_K) == 0
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE
    WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WMMA_M
    WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WMMA_N
    assert WARP_M_STEPS >= 1 and WARP_N_STEPS >= 1
    WARP_M = WARP_M_STEPS * WMMA_M
    WARP_N = WARP_N_STEPS * WMMA_N
    BLOCK_M = BLOCK_M_WARPS * WARP_M
    BLOCK_N = BLOCK_N_WARPS * WARP_N
    N_REAL = TILE_N
    N_BLOCKS = n // TILE_N
    assert n % 8 == 0
    M_BLOCKS = (m + TILE_M - 1) // TILE_M
    assert m % TILE_M == 0 and n % TILE_N == 0
    assert BLOCK_M == TILE_M and BLOCK_N == TILE_N
    OUT_DTYPE = fx.Float32 if SPLIT_K > 1 else fx.BFloat16
    OUT_BYTES = 4 if SPLIT_K > 1 else 2
    C_VEC = 4 if SPLIT_K > 1 else 8
    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES
    AS_ELEMS = STAGES * BLOCK_M * BLOCK_K
    C_STRIDE = BLOCK_N + LDG_VEC_SIZE
    CMN_ELEMS = BLOCK_M * C_STRIDE
    LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
    LDG_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    ROWS_PER_LOAD = BLOCK_THREADS // LDG_X_THREADS_AS
    LDG_REG_B_COUNT_AS = (N_REAL + ROWS_PER_LOAD - 1) // ROWS_PER_LOAD
    B_STAGE_ROWS = LDG_REG_B_COUNT_AS * ROWS_PER_LOAD
    BS_ELEMS = STAGES * B_STAGE_ROWS * BLOCK_K
    LDS_ELEMS = max(AS_ELEMS + BS_ELEMS, CMN_ELEMS * OUT_BYTES // 2)
    assert LDS_ELEMS * 2 <= 163840
    B_LAST_PARTIAL = B_STAGE_ROWS > N_REAL
    A_CHUNKS = BLOCK_M * BLOCK_K // LDG_ASYNC_VEC_SIZE
    assert A_CHUNKS % BLOCK_THREADS == 0
    A_LOADS = min(
        A_CHUNKS // BLOCK_THREADS, (BLOCK_M + ROWS_PER_LOAD - 1) // ROWS_PER_LOAD
    )
    LDG_WAIT_COUNT = LDG_REG_B_COUNT_AS + A_LOADS
    assert (STAGES - 2) * LDG_WAIT_COUNT < VMCNT_MAX
    LdsArr = fx.Array[fx.BFloat16, LDS_ELEMS, 16]
    PIPE_ITEMS = [("a", i) for i in range(A_LOADS)] + [
        ("b", i) for i in range(LDG_REG_B_COUNT_AS)
    ]
    # Four K steps in the fused tile: spread its DMA through all four.
    # The other two configurations have only two K steps.
    _nd = len(PIPE_ITEMS)
    dma_steps = WARP_K_STEPS
    PIPE_BOUNDS = [
        min(_nd, (_nd * q + dma_steps - 1) // dma_steps)
        for q in range(WARP_K_STEPS + 1)
    ]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def gemm_projection(C: fx.Pointer, A: fx.Pointer, B: fx.Pointer):
        acc_init = fx.full(WMMA_C_FRAG_VALUES, 0.0, fx.Float32)
        bf16x8_ty = fx.Vector.make_type(LDG_VEC_SIZE, fx.BFloat16)
        a_rsrc = _rsrc(A, m * k * DTYPE_BYTES)
        b_rsrc = _rsrc(B, n * k * DTYPE_BYTES)
        c_rsrc = _rsrc(C, SPLIT_K * m * n * OUT_BYTES)
        lds_base = fx.SharedAllocator().allocate(LdsArr).peek().ptr
        c_lds_base = fx.recast_iter(OUT_DTYPE, lds_base)
        b_lds_ptr = lds_base + AS_ELEMS
        a_lds_i64 = fx.Int64(fx.ptrtoint(lds_base))
        b_lds_i64 = a_lds_i64 + fx.Int64(AS_ELEMS * DTYPE_BYTES)

        def as_load(stage, row, col):
            elem_off = (
                fx.Int64(stage) * (BLOCK_M * BLOCK_K)
                + fx.Int64(row) * BLOCK_K
                + fx.Int64(col)
            )
            return fx.ptr_load(lds_base + elem_off, result_type=bf16x8_ty)

        def bs_load(stage, row, col):
            elem_off = (
                fx.Int64(stage) * (B_STAGE_ROWS * BLOCK_K)
                + fx.Int64(row) * BLOCK_K
                + fx.Int64(col)
            )
            return fx.ptr_load(b_lds_ptr + elem_off, result_type=bf16x8_ty)

        tid = fx.thread_idx.x
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE
        pid = fx.block_idx.x
        split_id = pid % SPLIT_K
        tile_n = pid // SPLIT_K % N_BLOCKS
        tile_m = pid // (SPLIT_K * N_BLOCKS)
        m_base = tile_m * TILE_M
        n_base = tile_n * TILE_N
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)
        warp_m_idx = wid // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid % BLOCK_N_WARPS * WARP_N
        ldmatrix_m_idx = w_tid % WMMA_M
        ldmatrix_k_vec_idx = w_tid // WMMA_M * WMMA_FRAG_VALUES
        c_frags = [acc_init] * (WARP_M_STEPS * WARP_N_STEPS)

        def mfma(a_frag, b_frag, c_frag):
            return rocdl.mfma_f32_16x16x32_bf16(
                fx.Vector.make_type(WMMA_C_FRAG_VALUES, fx.Float32),
                [b_frag, a_frag, c_frag, 0, 0, 0],
            )

        def wait_barrier(vmcnt):
            llvm.InlineAsmOp(
                None,
                [],
                f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier",
                "",
                has_side_effects=True,
            )

        warp_offset = rocdl.readfirstlane(
            fx.Int64.ir_type, fx.Int64(wid * WARP_SIZE * DMA_BYTES)
        )

        def buffer_load_lds_inline(rsrc, lds_ptr, global_offset, bits):
            llvm.InlineAsmOp(
                None,
                [lds_ptr, global_offset, rsrc],
                "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, 0 offen "
                + bits
                + " lds",
                "s,v,s",
                has_side_effects=True,
            )

        def ldg_sts_a_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_tid = BLOCK_THREADS * ii + tid
            m_local_idx = global_tid // LDG_X_THREADS_AS
            k_local_idx = global_tid % LDG_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            col_in_bytes = k_local_idx * DTYPE_BYTES
            col_in_bytes = _swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
            row_idx = m_base + m_local_idx
            col_idx = split_id * PART_K + k_offset + col_in_bytes // DTYPE_BYTES
            global_offset = (row_idx * k + col_idx) * DTYPE_BYTES
            global_offset = fx.Int32(global_offset).ir_value()
            if const_expr(lds_ptr is None):
                stage_off = fx.Int64(write_stage) * (BLOCK_M * BLOCK_K * DTYPE_BYTES)
                lds_ptr = _gep(_lds_ptr3(a_lds_i64, stage_off), warp_offset)
            else:
                lds_ptr = _gep(lds_ptr, BLOCK_THREADS * DMA_BYTES)
            buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset, bits=A_BITS)
            return lds_ptr

        def ldg_sts_a_async(k_offset, lds_stage):
            lds_ptr = None
            for i in range_constexpr(A_LOADS):
                lds_ptr = ldg_sts_a_async_one(i, k_offset, lds_stage, lds_ptr)

        def ldg_sts_b_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_tid = BLOCK_THREADS * ii + tid
            n_local_idx = global_tid // LDG_X_THREADS_AS
            k_local_idx = global_tid % LDG_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            col_in_bytes = k_local_idx * DTYPE_BYTES
            col_in_bytes = _swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
            row_idx = n_base + n_local_idx
            col_idx = split_id * PART_K + k_offset + col_in_bytes // DTYPE_BYTES
            global_offset = (row_idx * k + col_idx) * DTYPE_BYTES
            global_offset = fx.Int32(global_offset).ir_value()
            if const_expr(B_LAST_PARTIAL and ii == LDG_REG_B_COUNT_AS - 1):
                global_offset = fx.arith.unwrap(
                    (fx.Int32(n_local_idx) < fx.Int32(N_REAL)).select(
                        fx.Int32(global_offset), fx.Int32(0x7FFFFFF0)
                    )
                )
            if const_expr(lds_ptr is None):
                stage_off = fx.Int64(write_stage) * (
                    B_STAGE_ROWS * BLOCK_K * DTYPE_BYTES
                )
                lds_ptr = _gep(_lds_ptr3(b_lds_i64, stage_off), warp_offset)
            else:
                lds_ptr = _gep(lds_ptr, BLOCK_THREADS * DMA_BYTES)
            buffer_load_lds_inline(b_rsrc, lds_ptr, global_offset, bits=B_BITS)
            return lds_ptr

        def ldg_sts_b_async(k_offset, lds_stage):
            lds_ptr = None
            for i in range_constexpr(LDG_REG_B_COUNT_AS):
                lds_ptr = ldg_sts_b_async_one(i, k_offset, lds_stage, lds_ptr)

        def load_b_frag(s, n_step, warp_atom_k_idx):
            row = warp_n_idx + n_step * WMMA_N + ldmatrix_m_idx
            col_in_bytes = (warp_atom_k_idx + ldmatrix_k_vec_idx) * DTYPE_BYTES
            col_in_bytes = _swizzle_xor16(row, col_in_bytes, k_blocks16)
            return bs_load(s, row, col_in_bytes // DTYPE_BYTES)

        def load_a_frag(s, m_step, warp_atom_k_idx):
            row = warp_m_idx + m_step * WMMA_M + ldmatrix_m_idx
            col_in_bytes = (warp_atom_k_idx + ldmatrix_k_vec_idx) * DTYPE_BYTES
            col_in_bytes = _swizzle_xor16(row, col_in_bytes, k_blocks16)
            return as_load(s, row, col_in_bytes // DTYPE_BYTES)

        def frag_loads(s, kk):
            wk = kk * WMMA_K
            a_f = [load_a_frag(s, ii, wk) for ii in range_constexpr(WARP_M_STEPS)]
            b_f = [load_b_frag(s, ii, wk) for ii in range_constexpr(WARP_N_STEPS)]
            return (a_f, b_f)

        def pipe_compute(lds_stage, c_frags, dma=None):
            s = fx.Index(lds_stage)
            c_new = list(c_frags)
            ptrs = {"a": None, "b": None}
            cur = frag_loads(s, 0)
            for kk in range_constexpr(WARP_K_STEPS):
                if const_expr(dma is not None):
                    for j in range_constexpr(PIPE_BOUNDS[kk], PIPE_BOUNDS[kk + 1]):
                        kind, i = PIPE_ITEMS[j]
                        if const_expr(kind == "a"):
                            ptrs["a"] = ldg_sts_a_async_one(
                                i, dma[0], dma[1], ptrs["a"]
                            )
                        else:
                            ptrs["b"] = ldg_sts_b_async_one(
                                i, dma[0], dma[1], ptrs["b"]
                            )
                nxt = None
                if const_expr(kk + 1 < WARP_K_STEPS):
                    nxt = frag_loads(s, kk + 1)
                if const_expr(SCHED):
                    rocdl.sched_barrier(0)
                a_f, b_f = cur
                for ii in range_constexpr(WARP_M_STEPS):
                    for jj in range_constexpr(WARP_N_STEPS):
                        c_idx = ii * WARP_N_STEPS + jj
                        if const_expr(REVERSE):
                            rev_i = WARP_M_STEPS - 1 - ii
                            rev_j = WARP_N_STEPS - 1 - jj
                            c_idx = rev_i * WARP_N_STEPS + rev_j
                            c_new[c_idx] = mfma(a_f[rev_i], b_f[rev_j], c_new[c_idx])
                        else:
                            c_new[c_idx] = mfma(a_f[ii], b_f[jj], c_new[c_idx])
                if const_expr(SCHED):
                    rocdl.sched_barrier(0)
                cur = nxt
            return c_new

        def kwrap(off):
            return (off >= fx.Int32(PART_K)).select(off - fx.Int32(PART_K), off)

        stagger_id = tile_n if const_expr(STAGGER_AXIS == 1) else pid
        ks_begin = fx.Int32(stagger_id * K_STAGGER % BLOCK_K_LOOPS * BLOCK_K)
        for s in range_constexpr(STAGES - 1):
            ks = kwrap(ks_begin + s * BLOCK_K)
            ldg_sts_a_async(ks, s)
            ldg_sts_b_async(ks, s)
        rocdl.sched_barrier(0)
        # Complete ring periods have compile-time LDS addresses. The remainder
        # and drain are also unrolled; no stage modulo remains in the runtime loop.
        MAIN = BLOCK_K_LOOPS - (STAGES - 1)
        for chunk, state in range(
            0, MAIN // (STAGES * UF), 1, init=[ks_begin] + c_frags
        ):
            k_offset = state[0]
            cc = state[1:]
            for ss in range_constexpr(STAGES * UF):
                wait_barrier((STAGES - 2) * LDG_WAIT_COUNT)
                k_load = kwrap(k_offset + (STAGES - 1) * BLOCK_K)
                cc = pipe_compute(ss % STAGES, cc, (k_load, (ss + STAGES - 1) % STAGES))
                k_offset = kwrap(k_offset + BLOCK_K)
            rr = yield ([k_offset] + cc)
        k_offset = rr[0]
        c_frags = rr[1:]
        for ss in range_constexpr(MAIN % (STAGES * UF)):
            wait_barrier((STAGES - 2) * LDG_WAIT_COUNT)
            k_load = kwrap(k_offset + (STAGES - 1) * BLOCK_K)
            c_frags = pipe_compute(
                ss % STAGES, c_frags, (k_load, (ss + STAGES - 1) % STAGES)
            )
            k_offset = kwrap(k_offset + BLOCK_K)
        for ss in range_constexpr(STAGES - 1):
            wait_barrier((STAGES - 2 - ss) * LDG_WAIT_COUNT)
            c_frags = pipe_compute((MAIN + ss) % STAGES, c_frags)
        c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
        c_n_idx = w_tid % WMMA_N
        # The last LDS readers must retire before stage 0 becomes C-shuffle.
        fx.gpu.barrier()
        for ii in range_constexpr(WARP_M_STEPS):
            for jj in range_constexpr(WARP_N_STEPS):
                cast = fx.Vector(c_frags[ii * WARP_N_STEPS + jj]).to(OUT_DTYPE)
                lds_m = fx.Int64(warp_m_idx + ii * WMMA_M + c_n_idx)
                lds_n = fx.Int64(warp_n_idx + jj * WMMA_N + c_m_vec_idx)
                fx.ptr_store(cast, c_lds_base + (lds_m * C_STRIDE + lds_n))
        fx.gpu.barrier()
        for i in range_constexpr(
            (BLOCK_M * BLOCK_N // C_VEC + BLOCK_THREADS - 1) // BLOCK_THREADS
        ):
            vi = BLOCK_THREADS * i + tid
            mr = vi // (BLOCK_N // C_VEC)
            nc = vi % (BLOCK_N // C_VEC) * C_VEC
            if mr < BLOCK_M:
                vec = fx.ptr_load(
                    c_lds_base + (fx.Int64(mr) * C_STRIDE + fx.Int64(nc)),
                    result_type=fx.Vector.make_type(C_VEC, OUT_DTYPE),
                )
                off = split_id * (m * n) + (m_base + mr) * n + n_base + nc
                _bstore(vec, c_rsrc, off, C_AUX)

    @flyc.jit
    def launch(C: fx.Pointer, A: fx.Pointer, B: fx.Pointer, stream: fx.Stream):
        gemm_projection(C, A, B).launch(
            grid=(N_BLOCKS * M_BLOCKS * SPLIT_K, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": 6,
        }
    }
    if EXPERT:
        launch.compile_hints["llvm_options"]["amdgpu-expert-scheduling-mode"] = True
    return launch
