# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# jagged_dense_bmm_broadcast_add (jdbba) — FlyDSL layout-API prototype.
#
# Computes, for each group b over its packed row slice [s, e):
#     Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
#       (M_b x N)     (M_b x K)      (K x N)     (1 x N broadcast)
#
# This is the clean layout-API prototype that validates the JAGGED logic
# (group axis, device seq_offsets read, per-group rebasing, runtime
# early-exit, broadcast bias, masked store) before porting onto the fast
# hand-tuned splitk_hgemm base. bf16 in/out, fp32 accumulate.
#
# Dense is host pre-transposed to (B_groups, N, K) and consumed plain
# (NO preshuffle, unlike examples/04-preshuffle_gemm.py).

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
STAGES_A = 2

# Problem shape (fixed across groups; only the per-group row count M_b varies).
N = 128
K = 128
N_BLOCKS = N // BLOCK_N  # column-tiles (compile-time; N static across groups)


def make_bounded_buffer_tensor(tensor, num_records_bytes):
    """Like fx.rocdl.make_buffer_tensor but with a runtime byte bound, so the
    hardware OOB-drops stores past num_records_bytes. Mirrors the installed
    make_buffer_tensor body (which hardcodes max_size)."""
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


@flyc.kernel
def jdbba_kernel(
    C: fx.Tensor,           # out    (L, N)   bf16
    A: fx.Tensor,           # jagged (L, K)   bf16
    B: fx.Tensor,           # dense  (B_groups * N, K) bf16  (pre-transposed, tall)
    BIAS: fx.Tensor,        # bias   (B_groups * N,)   bf16
    SEQ_OFFSETS: fx.Tensor,  # (B_groups + 1,) int32
    tiled_mma: fx.TiledMma,
    tiled_copy_g2s_A: fx.TiledCopy,
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
    # seq_start/seq_end are block-uniform (one group per block) but buffer_load
    # types them per-lane (VGPR). Scalarize so everything derived from them --
    # M_b, the A/C/B/bias base offsets, and the C buffer-descriptor bound -- is
    # uniform (SGPR). Otherwise the divergent C descriptor forces the epilogue
    # store into a per-lane readfirstlane/exec-mask waterfall.
    seq_start = fx.rocdl.readfirstlane(fx.T.i32(), seq_start)
    seq_end = fx.rocdl.readfirstlane(fx.T.i32(), seq_end)
    M_b = seq_end - seq_start
    start_m = fx.Int32(block_m_idx) * fx.Int32(BLOCK_M)

    # Runtime early-exit: tail tile fell off the end of a short group.
    # Positive guard -> scf.IfOp wrapping all compute + store (avoids bare-return
    # lowering issues with dynamic if).
    if start_m < M_b:
        # --- Per-group rebasing ---
        # A / C: shift base down by seq_start rows so local tiling restarts at
        # the group's own row 0 (handles unaligned seq_start).
        a_row_off = fx.Int32(seq_start) * fx.Int32(K)
        c_row_off = fx.Int32(seq_start) * fx.Int32(N)
        A_g = fx.make_view(fx.add_offset(fx.get_iter(A), fx.make_int_tuple(a_row_off)), fx.get_layout(A))
        C_g = fx.make_view(fx.add_offset(fx.get_iter(C), fx.make_int_tuple(c_row_off)), fx.get_layout(C))
        # B / bias: select group b's slice of the tall (B_groups*N, K) matrix.
        b_row_off = fx.Int32(off_b) * fx.Int32(N) * fx.Int32(K)
        B_g = fx.make_view(fx.add_offset(fx.get_iter(B), fx.make_int_tuple(b_row_off)), fx.get_layout(B))

        A_buf = fx.rocdl.make_buffer_tensor(A_g, max_size=True)
        B_buf = fx.rocdl.make_buffer_tensor(B_g, max_size=True)
        # Bound C's descriptor to exactly M_b rows so the hardware OOB-drops any
        # partial-tile store past this group's last row (prevents writing into
        # the next group's rows). bf16 = 2 bytes. Inlined buffer-desc build
        # because the installed make_buffer_tensor has no num_records_bytes arg.
        C_buf = make_bounded_buffer_tensor(C_g, fx.Int64(fx.Int32(M_b) * fx.Int32(N) * fx.Int32(2)))

        gA_k = fx.flat_divide(A_buf, (BLOCK_M, BLOCK_K))[None, None, block_m_idx, None]  # (BM, BK, k)
        gB_k = fx.flat_divide(B_buf, (BLOCK_N, BLOCK_K))[None, None, block_n_idx, None]  # (BN, BK, k)
        gC = fx.flat_divide(C_buf, (BLOCK_M, BLOCK_N))[None, None, block_m_idx, block_n_idx]  # (BM, BN)

        # Broadcast bias: group b's (N,) slice viewed as (BLOCK_M, N) with M-stride 0
        # (same vector for every row), then pick this block's N-tile.
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

        # Main K-loop (double-buffered over K // BLOCK_K tiles).
        for k_iter in range(0, K // BLOCK_K - 2, 2):
            run_pipeline_stage(read_stage=0, next_k=k_iter + 1)
            run_pipeline_stage(read_stage=1, next_k=k_iter + 2)
        run_pipeline_stage(read_stage=0, next_k=K // BLOCK_K - 1)
        run_pipeline_stage(read_stage=1, next_k=None, read_next=False)

        # --- Epilogue: fp32 accumulators -> bf16, broadcast bias, masked store ---
        mma_frag_C_bf16 = fx.make_fragment_like(mma_frag_C, fx.BFloat16.ir_type)
        thr_copy_r2g_C = fx.make_tiled_copy_C(
            fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16), tiled_mma
        ).get_slice(tid)
        mma_frag_C_retile = thr_copy_r2g_C.retile(mma_frag_C_bf16)
        thr_gC = thr_copy_r2g_C.partition_S(gC)

        # Load this thread's bias slice (same partition as C) into fp32 and add
        # to the accumulator before the bf16 truncation (broadcast over rows).
        thr_gBias = thr_copy_r2g_C.partition_S(gBias)
        bias_frag = fx.make_fragment_like(thr_gBias)
        fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16), thr_gBias, bias_frag)
        bias_f32 = fx.arith.ExtFOp(fx.T.VectorType.get([64], fx.T.f32()), bias_frag.load()).result
        mma_frag_C_bf16.store(
            fx.arith.trunc_f(
                fx.T.VectorType.get([64], fx.T.bf16()),
                fx.arith.addf(mma_frag_C.load(), bias_f32),
            )
        )
        fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16), mma_frag_C_retile, thr_gC)


@flyc.jit
def jagged_dense_bmm(
    C: fx.Tensor,
    A: fx.Tensor,
    B: fx.Tensor,
    BIAS: fx.Tensor,
    SEQ_OFFSETS: fx.Tensor,
    n_groups: int,
    max_seq_len: int,
    stream: fx.Stream = fx.Stream(None),
):
    # K-layout (4,4,2)/(1,8,4) is the natural MFMA fragment K-ordering (example
    # 04 applies it to A, which is never preshuffled). The B un-shuffle in ex.04
    # lives in its preshuffle_layout_B VIEW, which we omit (plain B).
    tiled_mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
        fx.make_layout((1, 4, 1), (0, 1, 0)),
        fx.make_tile(None, None, fx.make_layout((4, 4, 2), (1, 8, 4))),
    )
    val_per_thr = 8  # 16B / bf16
    thrs_col = BLOCK_K // val_per_thr
    thrs_row = 256 // thrs_col
    tiled_copy_g2s_A = fx.make_tiled_copy(
        fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16),
        fx.make_layout(((thrs_col, thrs_row), (1, val_per_thr)), ((thrs_row * val_per_thr, 1), (1, thrs_row))),
        fx.make_tile(thrs_row, BLOCK_K),
    )

    bm = (max_seq_len + BLOCK_M - 1) // BLOCK_M
    jdbba_kernel(C, A, B, BIAS, SEQ_OFFSETS, tiled_mma, tiled_copy_g2s_A).launch(
        grid=(bm * N_BLOCKS, 1, n_groups), block=(256, 1, 1), smem=32768, stream=stream
    )
