# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# jagged_dense_bmm_broadcast_add (jdbba) — FlyDSL layout-API kernel, GENERALIZED.
#
# Computes, for each group b over its packed row slice [s, e):
#     Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
#       (M_b x N)     (M_b x K)      (K x N)     (1 x N broadcast)
#
# Generalized from the N==K==128 prototype: N (output) and K (reduction) are now
# runtime-parametric. The @flyc.kernel is built per (N, K, tiling) by a memoized
# factory (mirrors splitk_hgemm.compile_hgemm_kernel) so each distinct shape gets
# its own compiled kernel with N/K baked in as closure constants. Public entry
# derives N, K from tensor shapes.
#
# HSTU (B,D,K,N) bench naming -> GEMM dims: reduction K = bench D, output N =
# bench K, grid M-envelope = bench N (max_seq_len). So a B1024_D512_K512_N16384
# cell runs with reduction K=512, output N=512, max_seq_len=16384.
#
# Dense is host pre-transposed to (B_groups * N, K) and consumed plain.
# bf16 in/out, fp32 accumulate. Carries the prototype's seq_start/seq_end
# scalarization (uniform C descriptor -> no epilogue store waterfall).

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx

# Default tiling (override via the factory args).
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
STAGES_A = 2
THREADS = 256


def make_bounded_buffer_tensor(tensor, num_records_bytes):
    """Like fx.rocdl.make_buffer_tensor but with a runtime byte bound, so the
    hardware OOB-drops stores past num_records_bytes."""
    from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
    from flydsl.expr.buffer_ops import _get_buffer_flags

    elem_ty = tensor.element_type
    ptr = fx.get_iter(tensor)
    layout = fx.get_layout(tensor)
    buf_ptr_ty = fx.PointerType.get(
        elem_ty=elem_ty.ir_type,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=ptr.alignment,
    )
    buf_ptr = fx.make_ptr(
        buf_ptr_ty,
        [
            ptr,
            fx.Int16(0).ir_value(),
            num_records_bytes.ir_value(),
            fx.Int32(_get_buffer_flags()).ir_value(),
        ],
    )
    return fx.make_view(buf_ptr, layout)


@functools.lru_cache(maxsize=None)
def _build_launcher(N, K, BLOCK_M, BLOCK_N, BLOCK_K, STAGES_A, THREADS):
    """Build (and memoize) a launch wrapper specialized to this (N, K, tiling).
    N (output) and K (reduction) are baked in as closure constants."""
    N_BLOCKS = N // BLOCK_N  # output column-tiles per group (compile-time)
    # Per-thread C accumulator length (fp32 lanes) = tile elems / threads.
    C_FRAG_LEN = BLOCK_M * BLOCK_N // THREADS
    smem_bytes = BLOCK_M * BLOCK_K * STAGES_A * 2  # bf16 A double-buffer staging

    @flyc.kernel
    def jdbba_kernel(
        C: fx.Tensor,            # out    (L, N)   bf16
        A: fx.Tensor,            # jagged (L, K)   bf16
        B: fx.Tensor,            # dense  (B_groups * N, K) bf16 (pre-transposed, tall)
        BIAS: fx.Tensor,         # bias   (B_groups * N,)   bf16
        SEQ_OFFSETS: fx.Tensor,  # (B_groups + 1,) int32
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s_A: fx.TiledCopy,
        tiled_copy_s2g_C: fx.TiledCopy,
    ):
        tid = fx.thread_idx.x
        pid_mn, _, off_b = fx.block_idx
        off_b = fx.Int32(off_b)

        block_m_idx = pid_mn // N_BLOCKS
        block_n_idx = pid_mn % N_BLOCKS

        # --- Device group resolution (read seq_offsets[b], seq_offsets[b+1]) ---
        seq_rsrc = fx.buffer_ops.create_buffer_resource(SEQ_OFFSETS, max_size=True)
        seq_start = fx.buffer_ops.buffer_load(
            seq_rsrc, fx.Int32(off_b), vec_width=1, dtype=fx.T.i32()
        )
        seq_end = fx.buffer_ops.buffer_load(
            seq_rsrc, fx.Int32(off_b) + fx.Int32(1), vec_width=1, dtype=fx.T.i32()
        )
        # Scalarize: keep M_b and all derived offsets / the C descriptor uniform
        # (SGPR), else the divergent C descriptor forces an epilogue store waterfall.
        seq_start = fx.rocdl.readfirstlane(fx.T.i32(), seq_start)
        seq_end = fx.rocdl.readfirstlane(fx.T.i32(), seq_end)
        M_b = seq_end - seq_start
        start_m = fx.Int32(block_m_idx) * fx.Int32(BLOCK_M)

        if start_m < M_b:
            # --- Per-group rebasing ---
            # i64 offset math: seq_start can reach ~L (millions of rows), so
            # seq_start*K / seq_start*N overflow i32 at large shapes (e.g.
            # B1024_D512: 7.86M*512 = 4.0e9 > 2^31). Cast to i64 BEFORE the
            # stride multiply so the product is computed in 64-bit.
            a_row_off = fx.Int64(seq_start) * fx.Int64(K)
            c_row_off = fx.Int64(seq_start) * fx.Int64(N)
            A_g = fx.make_view(fx.add_offset(fx.get_iter(A), fx.make_int_tuple(a_row_off)), fx.get_layout(A))
            C_g = fx.make_view(fx.add_offset(fx.get_iter(C), fx.make_int_tuple(c_row_off)), fx.get_layout(C))
            b_row_off = fx.Int64(off_b) * fx.Int64(N) * fx.Int64(K)
            B_g = fx.make_view(fx.add_offset(fx.get_iter(B), fx.make_int_tuple(b_row_off)), fx.get_layout(B))

            A_buf = fx.rocdl.make_buffer_tensor(A_g, max_size=True)
            B_buf = fx.rocdl.make_buffer_tensor(B_g, max_size=True)
            # Bound C to M_b rows so the hardware OOB-drops partial-tile stores.
            C_buf = make_bounded_buffer_tensor(C_g, fx.Int64(fx.Int32(M_b) * fx.Int32(N) * fx.Int32(2)))

            gA_k = fx.flat_divide(A_buf, (BLOCK_M, BLOCK_K))[None, None, block_m_idx, None]  # (BM, BK, k)
            gB_k = fx.flat_divide(B_buf, (BLOCK_N, BLOCK_K))[None, None, block_n_idx, None]  # (BN, BK, k)
            gC = fx.flat_divide(C_buf, (BLOCK_M, BLOCK_N))[None, None, block_m_idx, block_n_idx]  # (BM, BN)

            # Broadcast bias: group b's (N,) slice viewed as (BLOCK_M, N) M-stride 0.
            bias_elem_off = fx.Int32(off_b) * fx.Int32(N)
            BIAS_g = fx.make_view(fx.add_offset(fx.get_iter(BIAS), fx.make_int_tuple(bias_elem_off)), fx.get_layout(BIAS))
            BIAS_buf = fx.rocdl.make_buffer_tensor(BIAS_g, max_size=True)
            gBias2d = fx.make_view(fx.get_iter(BIAS_buf), fx.make_layout((BLOCK_M, N), (0, 1)))
            gBias = fx.flat_divide(gBias2d, (BLOCK_M, BLOCK_N))[None, None, 0, block_n_idx]  # (BM, BN)

            thr_mma = tiled_mma.thr_slice(tid)
            thr_copy_g2s_A = tiled_copy_g2s_A.get_slice(tid)

            uni_copy_128b = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
            buffer_copy_128b = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)

            thr_copy_s2r_A = fx.make_tiled_copy_A(buffer_copy_128b, tiled_mma).get_slice(tid)
            thr_copy_g2r_B = fx.make_tiled_copy_B(buffer_copy_128b, tiled_mma).get_slice(tid)

            composed_layout_A = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 3, 3)),
                fx.make_ordered_layout((BLOCK_M, BLOCK_K, STAGES_A), (1, 0, 2)),
            )
            sA = fx.make_view(fx.get_dyn_shared(fx.BFloat16), composed_layout_A)  # (BM, BK, STAGES_A)

            thr_gA_k = thr_copy_g2s_A.partition_S(gA_k)  # (VA, VM, VK, k)
            thr_sA = thr_copy_g2s_A.partition_D(sA)      # (VA, VM, VK, STAGES_A)
            thr_sA_s2r = thr_copy_s2r_A.partition_S(sA)  # (VA, VM, VK, STAGES_A)
            thr_gB_k = thr_copy_g2r_B.partition_S(gB_k)  # (VB, VN, VK, k)

            copy_frag_A = fx.make_fragment_like(thr_sA[None, None, None, 0])

            mma_frag_A = thr_mma.make_fragment_A(sA[None, None, 0])
            mma_frag_B = thr_mma.make_fragment_B(gB_k, stages=2)
            mma_frag_C = thr_mma.make_fragment_C(gC)

            mma_frag_A_retile = thr_copy_s2r_A.retile(mma_frag_A)
            mma_frag_B_retile = thr_copy_g2r_B.retile(mma_frag_B)

            gA_k_stride = fx.get_scalar(gA_k.stride[2])
            gB_k_stride = fx.get_scalar(gB_k.stride[2])

            def run_pipeline_stage(read_stage, next_k, read_next=True):
                write_stage = read_stage ^ 1
                if fx.const_expr(read_next):
                    next_k = fx.Int32(next_k)
                    fx.copy(
                        buffer_copy_128b,
                        thr_gA_k[None, None, None, 0],
                        copy_frag_A,
                        soffset=next_k * gA_k_stride,
                    )
                    fx.copy(
                        buffer_copy_128b,
                        thr_gB_k[None, None, None, 0],
                        mma_frag_B_retile[None, None, None, write_stage],
                        soffset=next_k * gB_k_stride,
                    )

                for block_k_iter in fx.range_constexpr(BLOCK_K // 32):
                    fx.copy(
                        uni_copy_128b,
                        thr_sA_s2r[None, None, block_k_iter, read_stage],
                        mma_frag_A_retile[None, None, block_k_iter],
                    )
                    fx.gemm(
                        tiled_mma,
                        mma_frag_C,
                        mma_frag_A[None, None, (None, block_k_iter)],
                        mma_frag_B[None, None, (None, block_k_iter), read_stage],
                        mma_frag_C,
                        traversal_order=fx.GemmTraversalOrder.KNM,
                    )

                fx.copy(uni_copy_128b, copy_frag_A, thr_sA[None, None, None, write_stage])
                fx.gpu.barrier()

            # Prologue: load K-tile 0 into the read buffer.
            fx.copy(buffer_copy_128b, thr_gA_k[None, None, None, 0], copy_frag_A)
            fx.copy(buffer_copy_128b, thr_gB_k[None, None, None, 0], mma_frag_B_retile[None, None, None, 0])
            mma_frag_C.fill(0)
            fx.copy(uni_copy_128b, copy_frag_A, thr_sA[None, None, None, 0])
            fx.gpu.barrier()

            # Main K-loop (double-buffered over K // BLOCK_K tiles). K>=128 here.
            for k_iter in range(0, K // BLOCK_K - 2, 2):
                run_pipeline_stage(read_stage=0, next_k=k_iter + 1)
                run_pipeline_stage(read_stage=1, next_k=k_iter + 2)
            run_pipeline_stage(read_stage=0, next_k=K // BLOCK_K - 1)
            run_pipeline_stage(read_stage=1, next_k=None, read_next=False)

            # --- Epilogue: fp32 accumulators -> bf16, broadcast bias, masked store ---
            # O4 LDS C-shuffle: the MFMA accumulator layout is M-major per lane
            # (a lane's contiguous fragment elements map to different output ROWS,
            # stride N in global). A direct vectorized N store is therefore
            # impossible from the fragment, so the baseline emitted 64
            # buffer_store_short (scalar 2-byte) per thread -- 38-62% of runtime
            # at these memory-bound shapes. We route C through LDS to transpose
            # the layout: write the fragment to a row-major shared C tile in its
            # natural MFMA layout, barrier, then re-read N-contiguous (8 bf16 /
            # thread) and store with buffer_store_dwordx4 (128b) to global.
            mma_frag_C_bf16 = fx.make_fragment_like(mma_frag_C, fx.BFloat16.ir_type)
            thr_copy_r2s_C = fx.make_tiled_copy_C(
                fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16), tiled_mma
            ).get_slice(tid)
            mma_frag_C_retile = thr_copy_r2s_C.retile(mma_frag_C_bf16)

            # Broadcast bias (read once via the natural MFMA layout) and fold into
            # the fragment before the LDS write.
            thr_gBias = thr_copy_r2s_C.partition_S(gBias)
            bias_frag = fx.make_fragment_like(thr_gBias)
            fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16), thr_gBias, bias_frag)
            bias_f32 = fx.arith.ExtFOp(fx.T.VectorType.get([C_FRAG_LEN], fx.T.f32()), bias_frag.load()).result
            mma_frag_C_bf16.store(
                fx.arith.trunc_f(
                    fx.T.VectorType.get([C_FRAG_LEN], fx.T.bf16()),
                    fx.arith.addf(mma_frag_C.load(), bias_f32),
                )
            )

            # Row-major shared C tile, reusing the dynamic shared memory that held
            # the A staging buffers (smem sized to max of the two).
            sC = fx.make_view(
                fx.get_dyn_shared(fx.BFloat16),
                fx.make_ordered_layout((BLOCK_M, BLOCK_N), (1, 0)),
            )  # (BM, BN) row-major
            thr_sC = thr_copy_r2s_C.partition_D(sC)  # natural MFMA layout into LDS

            # The A staging LDS reads (s2r) for the last K-tile must be done before
            # we overwrite that memory with C; the main-loop barrier guarantees it.
            fx.gpu.barrier()
            fx.copy(fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16), mma_frag_C_retile, thr_sC)
            fx.gpu.barrier()

            # Re-read N-contiguous from LDS and wide-store to global.
            thr_copy_s2g_C = tiled_copy_s2g_C.get_slice(tid)
            thr_sC_read = thr_copy_s2g_C.partition_S(sC)   # (V, VM, VN)
            thr_gC = thr_copy_s2g_C.partition_D(gC)        # (V, VM, VN) global
            cs_frag = fx.make_fragment_like(thr_sC_read)
            fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16), thr_sC_read, cs_frag)
            fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16), cs_frag, thr_gC)

    @flyc.jit
    def _launch(
        C: fx.Tensor,
        A: fx.Tensor,
        B: fx.Tensor,
        BIAS: fx.Tensor,
        SEQ_OFFSETS: fx.Tensor,
        n_groups: int,
        max_seq_len: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
            fx.make_layout((1, 4, 1), (0, 1, 0)),
            fx.make_tile(None, None, fx.make_layout((4, 4, 2), (1, 8, 4))),
        )
        val_per_thr = 8  # 16B / bf16
        thrs_col = BLOCK_K // val_per_thr
        thrs_row = THREADS // thrs_col
        tiled_copy_g2s_A = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16),
            fx.make_layout(((thrs_col, thrs_row), (1, val_per_thr)), ((thrs_row * val_per_thr, 1), (1, thrs_row))),
            fx.make_tile(thrs_row, BLOCK_K),
        )

        # O4 C-shuffle store copy: row-major (BLOCK_M, BLOCK_N), 8 bf16 / thread
        # along N (128b) so the global store coalesces to buffer_store_dwordx4.
        c_val = 8  # 16B / bf16
        c_thrs_n = BLOCK_N // c_val          # threads along N
        c_thrs_m = THREADS // c_thrs_n       # threads along M
        tiled_copy_s2g_C = fx.make_tiled_copy(
            fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16),
            fx.make_layout(((c_thrs_n, c_thrs_m), (1, c_val)), ((c_thrs_m * c_val, 1), (1, c_thrs_m))),
            fx.make_tile(c_thrs_m, BLOCK_N),
        )

        # smem holds either A double-buffer staging or the row-major C tile.
        epi_smem_bytes = max(smem_bytes, BLOCK_M * BLOCK_N * 2)

        bm = (max_seq_len + BLOCK_M - 1) // BLOCK_M
        jdbba_kernel(C, A, B, BIAS, SEQ_OFFSETS, tiled_mma, tiled_copy_g2s_A, tiled_copy_s2g_C).launch(
            grid=(bm * N_BLOCKS, 1, n_groups), block=(THREADS, 1, 1), smem=epi_smem_bytes, stream=stream
        )

    return _launch


def jagged_dense_bmm(
    C,
    A,
    B,
    BIAS,
    SEQ_OFFSETS,
    n_groups: int,
    max_seq_len: int,
    stream: fx.Stream = fx.Stream(None),
):
    """Public entry. Derives N (output) and K (reduction) from the tall dense B
    matrix shape (B_groups * N, K), then dispatches to the per-shape kernel."""
    N = B.shape[0] // n_groups
    K = B.shape[1]
    # Shape-dependent BLOCK_K: for small reduction K (<=256) a 2-iter K-loop with
    # BLOCK_K=128 has fewer barriers and wins (~4% on K=256 shapes); for larger K
    # the deeper BLOCK_K=64 pipeline keeps occupancy and is faster. (BLOCK_K=256
    # is unsafe: the 2-stage double-buffer epilogue mis-accumulates a single tile.)
    block_k = 128 if K <= 256 else BLOCK_K
    launch = _build_launcher(N, K, BLOCK_M, BLOCK_N, block_k, STAGES_A, THREADS)
    return launch(C, A, B, BIAS, SEQ_OFFSETS, n_groups, max_seq_len, stream=stream)
