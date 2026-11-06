# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# Adapted from AMD-AGI/AKLR BF16 decode kernels (MIT), revision 551613c81ea1f232cadb72e6f4647f96152e67d4.

"""BF16 decode band-stream policy for gfx950.

Each of 256 workgroups
owns 64 line-aligned columns plus a strip grouped by XCD. A/B use staged
LDS DMA, with successor fragment reads and DMA interleaved with MFMAs.
B is streamed; A stays cached. No cross-workgroup reduction or scratch.

The inline LDS DMA is a deliberate scheduling boundary: explicit per-wave
vmcnt waits order the ring, including waves that skip the final A transfer.
MFMAs consume raw pre-swizzled fragments. Keep those boundaries and the
barriers paired when changing the pipeline.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as rocdl_d
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector

from aiter.ops.flydsl.kernels.tensor_shim import AITER_FLYDSL_KERNARG_PRELOAD


def _as_i32(v):
    v = fx.arith.unwrap(v)
    t = v.type
    if isinstance(t, ir.IntegerType):
        if t.width == 32:
            return v
        if t.width > 32:
            return std_arith.TruncIOp(T.i32, v).result
        return std_arith.ExtSIOp(T.i32, v).result
    return std_arith.IndexCastOp(T.i32, v).result


def _rsrc(ptr, nbytes):
    return rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(ptr, num_records_bytes=nbytes))


def _bstore(data, rsrc, elem_off, aux):
    data = fx.arith.unwrap(data)
    off = std_arith.MulIOp(
        _as_i32(elem_off), fx.arith.constant(data.type.element_type.width // 8)
    ).result
    rocdl_d.RawPtrBufferStoreOp(
        data, rsrc, off, fx.arith.constant(0), aux=ir.IntegerAttr.get(T.i32, aux)
    )


def _swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)


def _lds_ptr3(base_i64, byte_off_i64):
    v = fx.arith.unwrap(base_i64 + byte_off_i64)
    return llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), v).result


def _gep(base_ptr, byte_offset):
    if isinstance(byte_offset, int):
        dyn, raw = [], [byte_offset]
    else:
        dyn, raw = [fx.arith.unwrap(byte_offset)], [-(2**31)]
    return llvm.GEPOp(base_ptr.type, base_ptr, dyn, raw, T.i8, None).result


@functools.lru_cache(maxsize=512)
def build_band_stream(
    m,
    n,
    k,
    TILE_M,
    TILE_N,
    STAGES,
    BLOCK_M_WARPS,
    BLOCK_N_WARPS,
    K_STAGGER,
    C_AUX,
    A_WSKIP,
):
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
    BLOCK_K = 128
    BLOCK_K_LOOPS = k // BLOCK_K
    assert k % BLOCK_K == 0 and BLOCK_K_LOOPS >= STAGES
    WARP_K_STEPS = BLOCK_K // WMMA_K
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE
    NUM_WAVES = BLOCK_M_WARPS * BLOCK_N_WARPS
    WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WMMA_M
    WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WMMA_N
    assert WARP_M_STEPS >= 1 and WARP_N_STEPS >= 1
    WARP_M = WARP_M_STEPS * WMMA_M
    WARP_N = WARP_N_STEPS * WMMA_N
    BLOCK_M = BLOCK_M_WARPS * WARP_M
    BLOCK_N = BLOCK_N_WARPS * WARP_N
    N_REAL = n // 256
    N_BLOCKS = 256
    STRIP = N_REAL - 64
    assert BLOCK_M == TILE_M and BLOCK_N == TILE_N and m <= BLOCK_M
    assert N_BLOCKS * N_REAL == n and N_BLOCKS % 8 == 0 and N_REAL <= BLOCK_N
    assert STRIP > 0 and STRIP % LDG_VEC_SIZE == 0
    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES
    AS_ELEMS = STAGES * BLOCK_M * BLOCK_K
    # C is staged in LDS at a row stride padded by one store vector, keeping vector reads aligned.
    C_STRIDE = BLOCK_N + LDG_VEC_SIZE
    CMN_ELEMS = BLOCK_M * C_STRIDE
    assert CMN_ELEMS <= BLOCK_M * BLOCK_K
    LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
    LDG_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    ROWS_PER_LOAD = BLOCK_THREADS // LDG_X_THREADS_AS
    # A B stage holds only the rows the DMAs write; MFMA rows past them feed unstored columns.
    LDG_REG_B_COUNT_AS = (N_REAL + ROWS_PER_LOAD - 1) // ROWS_PER_LOAD
    B_STAGE_ROWS = LDG_REG_B_COUNT_AS * ROWS_PER_LOAD
    BS_ELEMS = STAGES * B_STAGE_ROWS * BLOCK_K
    B_PAD_ELEMS = max(0, BLOCK_N - B_STAGE_ROWS) * BLOCK_K
    LDS_ELEMS = AS_ELEMS + BS_ELEMS + B_PAD_ELEMS
    B_LAST_PARTIAL = B_STAGE_ROWS > N_REAL
    A_CHUNKS = BLOCK_M * BLOCK_K // LDG_ASYNC_VEC_SIZE
    assert A_CHUNKS % BLOCK_THREADS == 0
    A_LOADS = min(A_CHUNKS // BLOCK_THREADS, (m + ROWS_PER_LOAD - 1) // ROWS_PER_LOAD)
    A_WCUT = NUM_WAVES
    if A_WSKIP:
        A_ROWS_PER_WAVE = WARP_SIZE // LDG_X_THREADS_AS
        last_rows = m - (A_LOADS - 1) * ROWS_PER_LOAD
        A_WCUT = (last_rows + A_ROWS_PER_WAVE - 1) // A_ROWS_PER_WAVE
    A_GUARDED = A_WCUT < NUM_WAVES
    A_UNGUARDED = A_LOADS - (1 if A_GUARDED else 0)
    # Waiting on the skipping waves' smaller load count is only stricter for the other waves.
    LDG_WAIT_COUNT = LDG_REG_B_COUNT_AS + A_UNGUARDED
    assert (STAGES - 2) * LDG_WAIT_COUNT < VMCNT_MAX
    LdsArr = fx.Array[fx.BFloat16, LDS_ELEMS, 16]
    PIPE_ITEMS = [("a", i) for i in range(A_UNGUARDED)] + [
        ("b", i) for i in range(LDG_REG_B_COUNT_AS)
    ]
    # Half a stage's DMAs (A first) go in K step 0 and the rest in K step 1, overlapping the MFMAs.
    _nd = len(PIPE_ITEMS)
    PIPE_BOUNDS = [0, (_nd + 1) // 2, _nd] + [_nd] * (WARP_K_STEPS - 2)

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def gemm_band_stream(C: fx.Pointer, A: fx.Pointer, B: fx.Pointer):
        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))
        bf16x8_ty = fx.Vector.make_type(LDG_VEC_SIZE, fx.BFloat16)
        a_rsrc = _rsrc(A, m * k * DTYPE_BYTES)
        b_rsrc = _rsrc(B, n * k * DTYPE_BYTES)
        c_rsrc = _rsrc(C, m * n * DTYPE_BYTES)
        lds_base = fx.SharedAllocator().allocate(LdsArr).peek().ptr
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
        n_main = fx.Index(pid * 64)
        strip_idx = (pid % 8) * (N_BLOCKS // 8) + pid // 8
        n_strip = fx.Index(strip_idx * STRIP + (N_BLOCKS * 64 - 64))
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)
        warp_m_idx = wid // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid % BLOCK_N_WARPS * WARP_N
        ldmatrix_m_idx = w_tid % WMMA_M
        ldmatrix_k_vec_idx = w_tid // WMMA_M * WMMA_FRAG_VALUES
        c_frags = [acc_init] * (WARP_M_STEPS * WARP_N_STEPS)

        def mfma(a_frag, b_frag, c_frag):
            return rocdl.mfma_f32_16x16x32_bf16(
                T.vec(WMMA_C_FRAG_VALUES, T.f32), [b_frag, a_frag, c_frag, 0, 0, 0]
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
            T.i64,
            arith.index_cast(
                T.i64,
                fx.Index(wid) * arith.constant(WARP_SIZE * DMA_BYTES, index=True),
            ),
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
            # Rows past m are out of the buffer bound, so they read zero with no memory traffic.
            row_idx = fx.Index(m_local_idx)
            col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
            global_offset = (row_idx * k + col_idx) * DTYPE_BYTES
            global_offset = arith.index_cast(T.i32, global_offset)
            if const_expr(lds_ptr is None):
                stage_off = arith.index_cast(
                    T.i64, fx.Index(write_stage) * (BLOCK_M * BLOCK_K * DTYPE_BYTES)
                )
                lds_ptr = _gep(_lds_ptr3(a_lds_i64, fx.Int64(stage_off)), warp_offset)
            else:
                lds_ptr = _gep(lds_ptr, BLOCK_THREADS * DMA_BYTES)
            buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset, bits="sc0")
            return lds_ptr

        wid_u = rocdl.readfirstlane(T.i32, _as_i32(wid))

        def ldg_sts_a_guarded(k_offset, lds_stage, lds_ptr):
            if fx.Int32(wid_u) < fx.Int32(A_WCUT):
                ldg_sts_a_async_one(A_LOADS - 1, k_offset, lds_stage, lds_ptr)

        def ldg_sts_a_async(k_offset, lds_stage):
            lds_ptr = None
            for i in range_constexpr(A_UNGUARDED):
                lds_ptr = ldg_sts_a_async_one(i, k_offset, lds_stage, lds_ptr)
            if const_expr(A_GUARDED):
                ldg_sts_a_guarded(k_offset, lds_stage, lds_ptr)

        def ldg_sts_b_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_tid = BLOCK_THREADS * ii + tid
            n_local_idx = global_tid // LDG_X_THREADS_AS
            k_local_idx = global_tid % LDG_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            col_in_bytes = k_local_idx * DTYPE_BYTES
            col_in_bytes = _swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
            lo_row = ii * ROWS_PER_LOAD
            hi_row = lo_row + ROWS_PER_LOAD
            if const_expr(hi_row <= 64):
                row_idx = n_main + fx.Index(n_local_idx)
            elif const_expr(lo_row >= 64):
                row_idx = n_strip + fx.Index(n_local_idx)
            else:
                row_idx = fx.Index(
                    (fx.Int32(n_local_idx) < fx.Int32(64)).select(
                        fx.Int32(n_main) + fx.Int32(n_local_idx),
                        fx.Int32(n_strip) + fx.Int32(n_local_idx),
                    )
                )
            col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
            global_offset = (row_idx * k + col_idx) * DTYPE_BYTES
            global_offset = arith.index_cast(T.i32, global_offset)
            if const_expr(B_LAST_PARTIAL and ii == LDG_REG_B_COUNT_AS - 1):
                # Rows past N_REAL belong to the next workgroup; out of bounds, they read zero.
                global_offset = fx.arith.unwrap(
                    (fx.Int32(n_local_idx) < fx.Int32(N_REAL)).select(
                        fx.Int32(global_offset), fx.Int32(0x7FFFFFF0)
                    )
                )
            if const_expr(lds_ptr is None):
                stage_off = arith.index_cast(
                    T.i64,
                    fx.Index(write_stage) * (B_STAGE_ROWS * BLOCK_K * DTYPE_BYTES),
                )
                lds_ptr = _gep(_lds_ptr3(b_lds_i64, fx.Int64(stage_off)), warp_offset)
            else:
                lds_ptr = _gep(lds_ptr, BLOCK_THREADS * DMA_BYTES)
            # B is read once; a non-temporal load keeps it from displacing the reused A lines.
            buffer_load_lds_inline(b_rsrc, lds_ptr, global_offset, bits="sc0 nt")
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
            return a_f, b_f

        def pipe_compute(lds_stage, c_frags, dma=None):
            # Fragments are read one K step ahead; the guarded A DMA goes last as its branch splits.
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
                rocdl.sched_barrier(0)
                a_f, b_f = cur
                for ii in range_constexpr(WARP_M_STEPS):
                    for jj in range_constexpr(WARP_N_STEPS):
                        c_idx = ii * WARP_N_STEPS + jj
                        c_new[c_idx] = mfma(a_f[ii], b_f[jj], c_new[c_idx])
                rocdl.sched_barrier(0)
                cur = nxt
            if const_expr(dma is not None and A_GUARDED):
                ldg_sts_a_guarded(dma[0], dma[1], ptrs["a"])
            return c_new

        def kwrap(off):
            return (off >= fx.Int32(k)).select(off - fx.Int32(k), off)

        # Staggered K starts spread concurrent A reads; each output keeps one fixed summation order.
        ks_begin = fx.Int32(pid * K_STAGGER % BLOCK_K_LOOPS * BLOCK_K)
        for s in range_constexpr(STAGES - 1):
            ks = kwrap(ks_begin + s * BLOCK_K)
            ldg_sts_a_async(ks, s)
            ldg_sts_b_async(ks, s)
        rocdl.sched_barrier(0)

        init_state = [ks_begin, arith.constant(0, index=True)] + c_frags
        for bki, state in range(0, BLOCK_K_LOOPS - (STAGES - 1), 1, init=init_state):
            k_offset = state[0]
            current_stage = fx.Index(state[1])
            c_frags = state[2:]
            next_stage = (current_stage + 1) % STAGES
            write_stage = (current_stage + STAGES - 1) % STAGES
            wait_barrier((STAGES - 2) * LDG_WAIT_COUNT)
            k_load = kwrap(k_offset + (STAGES - 1) * BLOCK_K)
            c_frags_new = pipe_compute(current_stage, c_frags, (k_load, write_stage))
            k_offset_next = kwrap(k_offset + fx.Int32(BLOCK_K))
            results = yield [k_offset_next, next_stage] + c_frags_new
        current_stage = fx.Index(results[1])
        c_frags = results[2:]
        for s in range_constexpr(0, STAGES - 1):
            wait_barrier((STAGES - 2 - s) * LDG_WAIT_COUNT)
            c_frags = pipe_compute(current_stage, c_frags)
            current_stage = (current_stage + 1) % STAGES

        c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
        c_n_idx = w_tid % WMMA_N
        # C reuses A stage 0, needing a barrier only if a tail step still reads that stage.
        tail_stages = [
            (BLOCK_K_LOOPS - (STAGES - 1) + j) % STAGES for j in range(STAGES - 1)
        ]
        if const_expr(0 in tail_stages):
            fx.gpu.barrier()
        for ii in range_constexpr(WARP_M_STEPS):
            for jj in range_constexpr(WARP_N_STEPS):
                cast = Vector(c_frags[ii * WARP_N_STEPS + jj]).to(fx.BFloat16)
                lds_m = fx.Int64(warp_m_idx + ii * WMMA_M + c_n_idx)
                lds_n = fx.Int64(warp_n_idx + jj * WMMA_N + c_m_vec_idx)
                fx.ptr_store(cast, lds_base + (lds_m * C_STRIDE + lds_n))
        fx.gpu.barrier()

        def store_c(m_local, n_local, base_col, aux):
            vec = fx.ptr_load(
                lds_base + (fx.Int64(m_local) * C_STRIDE + fx.Int64(n_local)),
                result_type=bf16x8_ty,
            )
            _bstore(vec, c_rsrc, m_local * n + fx.Int32(base_col) + n_local, aux)

        for col0, width, base_col, aux in (
            (0, 64, n_main, C_AUX),
            (64, STRIP, n_strip, 0),
        ):
            xt = width // LDG_VEC_SIZE
            nv = m * xt
            for i in range_constexpr((nv + BLOCK_THREADS - 1) // BLOCK_THREADS):
                global_tid = BLOCK_THREADS * i + tid
                m_local = global_tid // xt
                n_local = global_tid % xt * LDG_VEC_SIZE + col0
                if const_expr(nv % BLOCK_THREADS != 0 and i == nv // BLOCK_THREADS):
                    if fx.Int32(global_tid) < fx.Int32(nv):
                        store_c(m_local, n_local, base_col, aux)
                else:
                    store_c(m_local, n_local, base_col, aux)

    @flyc.jit
    def launch(C: fx.Pointer, A: fx.Pointer, B: fx.Pointer, stream: fx.Stream):
        gemm_band_stream(C, A, B).launch(
            grid=(N_BLOCKS, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": 6,
        }
    }
    return launch
