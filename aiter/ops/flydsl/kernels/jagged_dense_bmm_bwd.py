# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Backward pass of jagged_dense_bmm_broadcast_add (jdbba).
#
# Given the upstream gradient dOut (L, N) of the forward
#     Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
# this module produces, per group b over its packed row slice [s, e):
#     dJagged[s:e, :] = dOut[s:e, :] @ Dense[b].T        (M_b x K)
#     dDense[b]       = Jagged[s:e, :].T @ dOut[s:e, :]   (K x N)
#     dBias[b]        = sum_m dOut[s:e, :]                (N,)
#
# dJagged is a per-row-independent GEMM (contraction over the static N axis).
# dDense and dBias both contract over the dynamic sequence axis m, and are
# computed as a two-pass split-reduction over m (a partials kernel writing fp32
# scratch, then a reduce kernel) to avoid serializing the reduction.
#
# bf16 in/out, fp32 accumulate. Targets CDNA (gfx942 / gfx950) like the forward.

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.vector import full

# Sibling import (script dir is on sys.path[0]); shares the forward tile/shape
# constants and the runtime-bounded buffer helper so forward and backward stay
# in lockstep.
from jagged_dense_bmm import (  # noqa: F401
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    K,
    N,
    N_BLOCKS,
    STAGES_A,
    make_bounded_buffer_tensor,
)

# Split factor over the jagged (m) axis for the dDense / dBias reductions. Each
# (group, split) block owns one fp32 scratch slot; the reduce pass sums them.
SPLIT = 4

# The bias/reduce passes use one thread per output column n. AMDGPU caps a
# workgroup at 256 threads, so for N > 256 the N axis is tiled into NRED_COL_TILES
# blocks of NRED_BLK columns (a column-tile grid dim); column = col_tile*NRED_BLK +
# tid. For N <= 256 this is a single tile of N threads (unchanged behaviour).
NRED_BLK = N if N <= 256 else 256
NRED_COL_TILES = (N + NRED_BLK - 1) // NRED_BLK

# dJagged output is (M, K). Its second (column) axis is K, tiled by BLOCK_N; the
# contraction runs over N, tiled by BLOCK_K. With N == K == 128 these counts
# match the forward kernel exactly.
KOUT_BLOCKS = K // BLOCK_N  # column-tiles of the (M, K) output (compile-time)
NRED_TILES = N // BLOCK_K   # contraction tiles over N (compile-time)

# dDense partials tiling. The contraction dDense[k,n] = sum_m J[m,k]*dOut[m,n] is
# a transposed GEMM C[k,n] = sum_m A[k,m]*B[n,m] with A = J.T and B = dOut.T, i.e.
# the reduction axis m is the *contiguous* fragment axis of both operands. Each
# workgroup (256 threads = 4 waves) owns one (DDENSE_BK x DDENSE_BN) = 128x128
# output sub-tile of the group's (K, N) block and runs the MFMA 16x16x16 bf16 pipe
# (same tiled_mma as the forward kernel). The reduction is tiled DDENSE_BM rows of
# m at a time: each m-tile is staged transposed into LDS (J -> sJ as (k, m), dOut
# -> sD as (n, m), so m ends up contiguous and feeds the MFMA K-fragment), then a
# bf16 MFMA accumulates into one fp32 accumulator fragment carried across the whole
# (dynamic, split-strided) m-tile loop. Output-tiling keeps LDS fixed at 32 KB and
# the accumulator at one MFMA C-fragment per thread regardless of D = K = N.
DDENSE_BM = 64    # m-contraction tile staged per step (the MFMA K dimension)
DDENSE_BK = 128   # output K-tile per workgroup (MFMA M dimension)
DDENSE_BN = 128   # output N-tile per workgroup (MFMA N dimension)
DDENSE_THREADS = 256
NK_TILES = K // DDENSE_BK  # output K-tiles per group (compile-time)
NN_TILES = N // DDENSE_BN  # output N-tiles per group (compile-time)
_J_LDS_LOADS = (DDENSE_BM * DDENSE_BK) // DDENSE_THREADS  # 32
_D_LDS_LOADS = (DDENSE_BM * DDENSE_BN) // DDENSE_THREADS  # 32
_DDENSE_SMEM_BYTES = (DDENSE_BK * DDENSE_BM + DDENSE_BN * DDENSE_BM) * 2  # bf16 staging


def _load_scalar(copy_atom, elem_dtype, divided_tensor, index):
    """Load one element at column `index` from a row already divided by (1, 1)."""
    view = fx.slice(divided_tensor, (None, index))
    r = fx.make_rmem_tensor(1, elem_dtype)
    fx.copy_atom_call(copy_atom, view, r)
    return fx.memref_load_vec(r)[0]


def _store_scalar(copy_atom, store_dtype, divided_tensor, index, val):
    """Store scalar `val` to column `index` of a row already divided by (1, 1)."""
    r = fx.make_rmem_tensor(1, store_dtype)
    ts = full(1, store_dtype(val), store_dtype)
    fx.memref_store_vec(ts, r)
    view = fx.slice(divided_tensor, (None, index))
    fx.copy_atom_call(copy_atom, r, view)


@flyc.kernel
def grad_jagged_kernel(
    C: fx.Tensor,            # out    dJagged (L, K)          bf16
    A: fx.Tensor,            # grad   dOut    (L, N)          bf16
    B: fx.Tensor,            # dense  (n_groups * K, N)       bf16  (plain, K-major per group)
    SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
    tiled_mma: fx.TiledMma,
    tiled_copy_g2s_A: fx.TiledCopy,
):
    # dJagged[m,k] = sum_n dOut[m,n] * Dense[b][k,n]. In MFMA C[i,j]=sum_l A[i,l]
    # B[j,l] form: i=m (rows), j=k (output column), l=n (contraction). So A is the
    # dOut group view (M_b, N) and B is Dense[b] read in its plain (K, N) layout.
    tid = fx.thread_idx.x
    pid_mn, _, off_b = fx.block_idx
    off_b = fx.Int32(off_b)

    block_m_idx = pid_mn // KOUT_BLOCKS
    block_n_idx = pid_mn % KOUT_BLOCKS

    # Device group resolution; scalarize to keep group-derived values uniform.
    seq_rsrc = fx.buffer_ops.create_buffer_resource(SEQ_OFFSETS, max_size=True)
    seq_start = fx.buffer_ops.buffer_load(seq_rsrc, fx.Int32(off_b), vec_width=1, dtype=fx.T.i32())
    seq_end = fx.buffer_ops.buffer_load(seq_rsrc, fx.Int32(off_b) + fx.Int32(1), vec_width=1, dtype=fx.T.i32())
    seq_start = fx.rocdl.readfirstlane(fx.T.i32(), seq_start)
    seq_end = fx.rocdl.readfirstlane(fx.T.i32(), seq_end)
    M_b = seq_end - seq_start
    start_m = fx.Int32(block_m_idx) * fx.Int32(BLOCK_M)

    # Runtime early-exit: tail tile fell off the end of a short group.
    if start_m < M_b:
        # Rebase A (dOut, N cols/row) and C (dJagged, K cols/row) to this group's
        # local row 0; select B's (K, N) slice for group off_b.
        a_row_off = fx.Int32(seq_start) * fx.Int32(N)
        c_row_off = fx.Int32(seq_start) * fx.Int32(K)
        A_g = fx.make_view(fx.add_offset(fx.get_iter(A), fx.make_int_tuple(a_row_off)), fx.get_layout(A))
        C_g = fx.make_view(fx.add_offset(fx.get_iter(C), fx.make_int_tuple(c_row_off)), fx.get_layout(C))
        b_row_off = fx.Int32(off_b) * fx.Int32(K) * fx.Int32(N)
        B_g = fx.make_view(fx.add_offset(fx.get_iter(B), fx.make_int_tuple(b_row_off)), fx.get_layout(B))

        A_buf = fx.rocdl.make_buffer_tensor(A_g, max_size=True)
        B_buf = fx.rocdl.make_buffer_tensor(B_g, max_size=True)
        # Bound C to exactly M_b rows (K cols, bf16=2B) so partial tail-tile
        # stores are HW-dropped instead of corrupting the next group's rows.
        C_buf = make_bounded_buffer_tensor(C_g, fx.Int64(fx.Int32(M_b) * fx.Int32(K) * fx.Int32(2)))

        gA_k = fx.flat_divide(A_buf, (BLOCK_M, BLOCK_K))[None, None, block_m_idx, None]  # (BM, BK, n)
        gB_k = fx.flat_divide(B_buf, (BLOCK_N, BLOCK_K))[None, None, block_n_idx, None]  # (BN, BK, n)
        gC = fx.flat_divide(C_buf, (BLOCK_M, BLOCK_N))[None, None, block_m_idx, block_n_idx]  # (BM, BN)

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

        thr_gA_k = thr_copy_g2s_A.partition_S(gA_k)  # (VA, VM, VK, n)
        thr_sA = thr_copy_g2s_A.partition_D(sA)      # (VA, VM, VK, STAGES_A)
        thr_sA_s2r = thr_copy_s2r_A.partition_S(sA)  # (VA, VM, VK, STAGES_A)
        thr_gB_k = thr_copy_g2r_B.partition_S(gB_k)  # (VB, VN, VK, n)

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

        # Prologue: load contraction-tile 0 into the read buffer.
        fx.copy(buffer_copy_128b, thr_gA_k[None, None, None, 0], copy_frag_A)
        fx.copy(buffer_copy_128b, thr_gB_k[None, None, None, 0], mma_frag_B_retile[None, None, None, 0])
        mma_frag_C.fill(0)
        fx.copy(uni_copy_128b, copy_frag_A, thr_sA[None, None, None, 0])
        fx.gpu.barrier()

        # Main loop over the N contraction (double-buffered over N // BLOCK_K tiles).
        for k_iter in range(0, NRED_TILES - 2, 2):
            run_pipeline_stage(read_stage=0, next_k=k_iter + 1)
            run_pipeline_stage(read_stage=1, next_k=k_iter + 2)
        run_pipeline_stage(read_stage=0, next_k=NRED_TILES - 1)
        run_pipeline_stage(read_stage=1, next_k=None, read_next=False)

        # Epilogue: fp32 accumulators -> bf16, masked store (no bias in backward).
        mma_frag_C_bf16 = fx.make_fragment_like(mma_frag_C, fx.BFloat16.ir_type)
        thr_copy_r2g_C = fx.make_tiled_copy_C(
            fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16), tiled_mma
        ).get_slice(tid)
        mma_frag_C_retile = thr_copy_r2g_C.retile(mma_frag_C_bf16)
        thr_gC = thr_copy_r2g_C.partition_S(gC)

        mma_frag_C_bf16.store(
            fx.arith.trunc_f(fx.T.VectorType.get([64], fx.T.bf16()), mma_frag_C.load())
        )
        fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16), mma_frag_C_retile, thr_gC)


@flyc.jit
def grad_jagged(
    dJagged: fx.Tensor,      # out    (L, K)            bf16
    dOut: fx.Tensor,         # grad   (L, N)            bf16
    DENSE: fx.Tensor,        # dense  (n_groups * K, N) bf16  (plain, K-major per group)
    SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
    n_groups: int,
    max_seq_len: int,
    stream: fx.Stream = fx.Stream(None),
):
    """dJagged[s:e, :] = dOut[s:e, :] @ Dense[b].T, per group.

    Contraction is over the static N axis, so this is a clean per-row GEMM that
    reuses the forward kernel's MFMA + double-buffered pipeline (N == K == 128).
    """
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
    grad_jagged_kernel(dJagged, dOut, DENSE, SEQ_OFFSETS, tiled_mma, tiled_copy_g2s_A).launch(
        grid=(bm * KOUT_BLOCKS, 1, n_groups), block=(256, 1, 1), smem=32768, stream=stream
    )


@flyc.kernel
def grad_bias_partials_kernel(
    PARTIALS: fx.Tensor,     # out    (n_groups * SPLIT, N)  fp32
    DOUT: fx.Tensor,         # grad   (L, N)                 bf16
    SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
):
    # dBias[b][n] = sum_m dOut[m,n]. This pass sums one split's strided subset of
    # the group's rows; thread `tid` owns column n. Split s accumulates local rows
    # r = s, s + SPLIT, s + 2*SPLIT, ... < M_b, so the SPLIT blocks together cover
    # every row exactly once.
    tid = fx.thread_idx.x
    col_tile, off_s, off_b = fx.block_idx
    off_b = fx.Int32(off_b)
    off_s = fx.Int32(off_s)
    col = fx.Int32(col_tile) * fx.Int32(NRED_BLK) + tid  # output column n

    seq_rsrc = fx.buffer_ops.create_buffer_resource(SEQ_OFFSETS, max_size=True)
    seq_start = fx.buffer_ops.buffer_load(seq_rsrc, off_b, vec_width=1, dtype=fx.T.i32())
    seq_end = fx.buffer_ops.buffer_load(seq_rsrc, off_b + fx.Int32(1), vec_width=1, dtype=fx.T.i32())
    seq_start = fx.rocdl.readfirstlane(fx.T.i32(), seq_start)
    seq_end = fx.rocdl.readfirstlane(fx.T.i32(), seq_end)
    M_b = seq_end - seq_start

    # Rebase dOut to this group's local row 0; loop bound M_b keeps reads in-range.
    a_row_off = fx.Int32(seq_start) * fx.Int32(N)
    DOUT_g = fx.make_view(fx.add_offset(fx.get_iter(DOUT), fx.make_int_tuple(a_row_off)), fx.get_layout(DOUT))
    DOUT_buf = fx.rocdl.make_buffer_tensor(DOUT_g, max_size=True)
    copy_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)

    acc0 = fx.Float32(0.0)
    for r, state in range(off_s, M_b, fx.Int32(SPLIT), init=[acc0]):
        acc = state[0]
        row_view = fx.slice(DOUT_buf, (fx.Int32(r), None))
        row_div = fx.logical_divide(row_view, fx.make_layout(1, 1))
        val = _load_scalar(copy_bf16, fx.BFloat16, row_div, col)
        acc = acc + val.to(fx.Float32)
        results = yield [acc]
    acc_final = results

    PART_buf = fx.rocdl.make_buffer_tensor(PARTIALS, max_size=True)
    part_row = off_b * fx.Int32(SPLIT) + off_s
    part_div = fx.logical_divide(fx.slice(PART_buf, (part_row, None)), fx.make_layout(1, 1))
    copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    _store_scalar(copy_f32, fx.Float32, part_div, col, acc_final)


@flyc.kernel
def grad_bias_reduce_kernel(
    DBIAS: fx.Tensor,     # out    (n_groups, N)          bf16
    PARTIALS: fx.Tensor,  # in     (n_groups * SPLIT, N)  fp32
):
    # Sum this group's SPLIT fp32 partials and write bf16 dBias[b]. Thread tid
    # owns column n = col_tile*NRED_BLK + tid.
    tid = fx.thread_idx.x
    col_tile = fx.Int32(fx.block_idx.x)
    off_b = fx.Int32(fx.block_idx.y)
    col = col_tile * fx.Int32(NRED_BLK) + tid

    PART_buf = fx.rocdl.make_buffer_tensor(PARTIALS, max_size=True)
    DBIAS_buf = fx.rocdl.make_buffer_tensor(DBIAS, max_size=True)
    copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    copy_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)

    acc = fx.Float32(0.0)
    for s in fx.range_constexpr(SPLIT):
        part_row = off_b * fx.Int32(SPLIT) + fx.Int32(s)
        row_div = fx.logical_divide(fx.slice(PART_buf, (part_row, None)), fx.make_layout(1, 1))
        acc = acc + _load_scalar(copy_f32, fx.Float32, row_div, col)

    out_div = fx.logical_divide(fx.slice(DBIAS_buf, (off_b, None)), fx.make_layout(1, 1))
    _store_scalar(copy_bf16, fx.BFloat16, out_div, col, acc.to(fx.BFloat16))


@flyc.jit
def grad_bias(
    dBias: fx.Tensor,        # out    (n_groups, N)            bf16
    dOut: fx.Tensor,         # grad   (L, N)                   bf16
    SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
    partials: fx.Tensor,     # fp32 scratch (n_groups * SPLIT, N)
    n_groups: int,
    max_seq_len: int,
    stream: fx.Stream = fx.Stream(None),
):
    """dBias[b] = sum_m dOut[s:e, :], per group.

    Two-pass split-reduction over m: a partials pass writes SPLIT fp32 partial
    row-sums per group, then a reduce pass sums them into bf16 dBias.
    """
    grad_bias_partials_kernel(partials, dOut, SEQ_OFFSETS).launch(
        grid=(NRED_COL_TILES, SPLIT, n_groups), block=(NRED_BLK, 1, 1), stream=stream
    )
    grad_bias_reduce_kernel(dBias, partials).launch(
        grid=(NRED_COL_TILES, n_groups, 1), block=(NRED_BLK, 1, 1), stream=stream
    )


@flyc.kernel
def grad_dense_partials_kernel(
    PARTIALS: fx.Tensor,     # out    (n_groups * SPLIT * K, N)  fp32
    JAGGED: fx.Tensor,       # jagged (L, K)                     bf16
    DOUT: fx.Tensor,         # grad   (L, N)                     bf16
    SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
    tiled_mma: fx.TiledMma,
):
    # dDense[b][k,n] = sum_m Jagged[m,k] * dOut[m,n]. Written as the MFMA atom form
    # C[i,j] = sum_l A[i,l]*B[j,l] with i=k, j=n, l=m: A[k,m] = J[m,k] (= J.T) and
    # B[n,m] = dOut[m,n] (= dOut.T). Both operands therefore carry the reduction
    # axis m as their *contiguous* fragment (K) axis, so each m-tile is staged
    # transposed into LDS (sJ as (k, m), sD as (n, m)) and fed to bf16 MFMA. Split
    # s reduces a strided subset of the group's m-tiles; block_idx.x selects this
    # workgroup's (DDENSE_BK x DDENSE_BN) output sub-tile of the (K, N) block.
    tid = fx.thread_idx.x
    pid_kn, off_s, off_b = fx.block_idx
    off_b = fx.Int32(off_b)
    off_s = fx.Int32(off_s)
    pid_kn = fx.Int32(pid_kn)
    k_off = (pid_kn // fx.Int32(NN_TILES)) * fx.Int32(DDENSE_BK)
    n_off = (pid_kn % fx.Int32(NN_TILES)) * fx.Int32(DDENSE_BN)

    seq_rsrc = fx.buffer_ops.create_buffer_resource(SEQ_OFFSETS, max_size=True)
    seq_start = fx.buffer_ops.buffer_load(seq_rsrc, off_b, vec_width=1, dtype=fx.T.i32())
    seq_end = fx.buffer_ops.buffer_load(seq_rsrc, off_b + fx.Int32(1), vec_width=1, dtype=fx.T.i32())
    seq_start = fx.rocdl.readfirstlane(fx.T.i32(), seq_start)
    seq_end = fx.rocdl.readfirstlane(fx.T.i32(), seq_end)
    M_b = seq_end - seq_start

    # Group-rebased buffers bounded to M_b rows: any local row >= M_b zero-fills
    # (CDNA OOB-load == 0), so tail rows contribute 0 to the contraction.
    j_rsrc = fx.buffer_ops.create_buffer_resource(
        JAGGED,
        max_size=False,
        num_records_bytes=fx.Int64(fx.Int32(M_b) * fx.Int32(K) * fx.Int32(2)),
        base_byte_offset=fx.Int32(seq_start) * fx.Int32(K) * fx.Int32(2),
    )
    d_rsrc = fx.buffer_ops.create_buffer_resource(
        DOUT,
        max_size=False,
        num_records_bytes=fx.Int64(fx.Int32(M_b) * fx.Int32(N) * fx.Int32(2)),
        base_byte_offset=fx.Int32(seq_start) * fx.Int32(N) * fx.Int32(2),
    )

    # LDS staging tiles, both (out_dim, m) with m (the contraction) contiguous so
    # the s2r feed lands m on the MFMA K-fragment. Same swizzle as the forward sA.
    composed_J = fx.make_composed_layout(
        fx.static(fx.SwizzleType.get(3, 3, 3)),
        fx.make_ordered_layout((DDENSE_BK, DDENSE_BM), (1, 0)),
    )
    composed_D = fx.make_composed_layout(
        fx.static(fx.SwizzleType.get(3, 3, 3)),
        fx.make_ordered_layout((DDENSE_BN, DDENSE_BM), (1, 0)),
    )
    smem = fx.get_dyn_shared(fx.BFloat16)
    sJ = fx.make_view(smem, composed_J)                                   # (k, m)
    smem_d = fx.add_offset(smem, fx.make_int_tuple(DDENSE_BK * DDENSE_BM))
    sD = fx.make_view(smem_d, composed_D)                                 # (n, m)

    thr_mma = tiled_mma.thr_slice(tid)
    uni_copy_128b = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
    buffer_copy_128b = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    thr_copy_s2r_A = fx.make_tiled_copy_A(buffer_copy_128b, tiled_mma).get_slice(tid)
    thr_copy_s2r_B = fx.make_tiled_copy_B(buffer_copy_128b, tiled_mma).get_slice(tid)

    thr_sJ_s2r = thr_copy_s2r_A.partition_S(sJ)  # (VA, VM, VK)
    thr_sD_s2r = thr_copy_s2r_B.partition_S(sD)  # (VB, VN, VK)

    # Output (K, N) sub-tile of this workgroup, viewed in the fp32 partials scratch.
    part_off = ((off_b * fx.Int32(SPLIT) + off_s) * fx.Int32(K) + k_off) * fx.Int32(N) + n_off
    PART_g = fx.make_view(fx.add_offset(fx.get_iter(PARTIALS), fx.make_int_tuple(part_off)), fx.get_layout(PARTIALS))
    PART_buf = fx.rocdl.make_buffer_tensor(PART_g, max_size=True)
    gPart = fx.make_view(fx.get_iter(PART_buf), fx.make_layout((DDENSE_BK, DDENSE_BN), (N, 1)))

    mma_frag_A = thr_mma.make_fragment_A(sJ)
    mma_frag_B = thr_mma.make_fragment_B(sD)
    mma_frag_C = thr_mma.make_fragment_C(gPart)
    mma_frag_A_retile = thr_copy_s2r_A.retile(mma_frag_A)
    mma_frag_B_retile = thr_copy_s2r_B.retile(mma_frag_B)

    num_tiles = (M_b + fx.Int32(DDENSE_BM - 1)) // fx.Int32(DDENSE_BM)

    # fp32 MFMA accumulator carried in place across the dynamic, split-strided
    # m-tile loop (AGPR accumulate; no SSA carry needed).
    mma_frag_C.fill(0)
    for m_tile in range(off_s, num_tiles, fx.Int32(SPLIT)):
        mt = fx.Int32(m_tile)
        # Transpose-stage this m-tile: read coalesced along the contiguous global
        # axis (k for J, n for dOut), store into LDS with m contiguous.
        for i in fx.range_constexpr(_J_LDS_LOADS):
            lin = tid + fx.Int32(i * DDENSE_THREADS)
            m_local = lin // fx.Int32(DDENSE_BK)
            k_local = lin % fx.Int32(DDENSE_BK)
            joff = (mt * fx.Int32(DDENSE_BM) + m_local) * fx.Int32(K) + (k_off + k_local)
            jval = fx.buffer_ops.buffer_load(j_rsrc, joff, vec_width=1, dtype=fx.T.bf16())
            fx.memref_store(jval, sJ, (k_local, m_local))
        for i in fx.range_constexpr(_D_LDS_LOADS):
            lin = tid + fx.Int32(i * DDENSE_THREADS)
            m_local = lin // fx.Int32(DDENSE_BN)
            n_local = lin % fx.Int32(DDENSE_BN)
            doff = (mt * fx.Int32(DDENSE_BM) + m_local) * fx.Int32(N) + (n_off + n_local)
            dval = fx.buffer_ops.buffer_load(d_rsrc, doff, vec_width=1, dtype=fx.T.bf16())
            fx.memref_store(dval, sD, (n_local, m_local))
        fx.gpu.barrier()

        for block_k_iter in fx.range_constexpr(DDENSE_BM // 32):
            fx.copy(uni_copy_128b, thr_sJ_s2r[None, None, block_k_iter], mma_frag_A_retile[None, None, block_k_iter])
            fx.copy(uni_copy_128b, thr_sD_s2r[None, None, block_k_iter], mma_frag_B_retile[None, None, block_k_iter])
            fx.gemm(
                tiled_mma,
                mma_frag_C,
                mma_frag_A[None, None, (None, block_k_iter)],
                mma_frag_B[None, None, (None, block_k_iter)],
                mma_frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )
        fx.gpu.barrier()

    # Epilogue: write the fp32 accumulator straight to the fp32 partials scratch
    # (no bf16 cast; the reduce pass does the final fp32 -> bf16).
    thr_copy_r2g_C = fx.make_tiled_copy_C(
        fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32), tiled_mma
    ).get_slice(tid)
    mma_frag_C_retile = thr_copy_r2g_C.retile(mma_frag_C)
    thr_gPart = thr_copy_r2g_C.partition_S(gPart)
    fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32), mma_frag_C_retile, thr_gPart)


@flyc.kernel
def grad_dense_reduce_kernel(
    DDENSE: fx.Tensor,    # out    (n_groups * K, N)          bf16
    PARTIALS: fx.Tensor,  # in     (n_groups * SPLIT * K, N)  fp32
):
    # Sum this group's SPLIT fp32 (K, N) partials and write bf16 dDense[b]. Block
    # (col-tile, k-row, group); thread tid owns column n = col_tile*NRED_BLK + tid.
    tid = fx.thread_idx.x
    col_tile = fx.Int32(fx.block_idx.x)
    off_k = fx.Int32(fx.block_idx.y)
    off_b = fx.Int32(fx.block_idx.z)
    col = col_tile * fx.Int32(NRED_BLK) + tid

    PART_buf = fx.rocdl.make_buffer_tensor(PARTIALS, max_size=True)
    DD_buf = fx.rocdl.make_buffer_tensor(DDENSE, max_size=True)
    copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    copy_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)

    acc = fx.Float32(0.0)
    for s in fx.range_constexpr(SPLIT):
        part_row = (off_b * fx.Int32(SPLIT) + fx.Int32(s)) * fx.Int32(K) + off_k
        row_div = fx.logical_divide(fx.slice(PART_buf, (part_row, None)), fx.make_layout(1, 1))
        acc = acc + _load_scalar(copy_f32, fx.Float32, row_div, col)

    out_row = off_b * fx.Int32(K) + off_k
    out_div = fx.logical_divide(fx.slice(DD_buf, (out_row, None)), fx.make_layout(1, 1))
    _store_scalar(copy_bf16, fx.BFloat16, out_div, col, acc.to(fx.BFloat16))


@flyc.jit
def grad_dense(
    dDense: fx.Tensor,       # out    (n_groups * K, N)            bf16
    JAGGED: fx.Tensor,       # jagged (L, K)                       bf16
    dOut: fx.Tensor,         # grad   (L, N)                       bf16
    SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
    partials: fx.Tensor,     # fp32 scratch (n_groups * SPLIT * K, N)
    n_groups: int,
    max_seq_len: int,
    stream: fx.Stream = fx.Stream(None),
):
    """dDense[b] = Jagged[s:e, :].T @ dOut[s:e, :], per group.

    Contraction is over the dynamic sequence axis m. Two-pass split-reduction:
    a partials pass accumulates fp32 (K, N) partials per (group, split) slot via
    bf16 MFMA on the transposed GEMM, then a reduce pass sums them into bf16 dDense.
    """
    # Same MFMA atom + tiling as the forward GEMM: 16x16x16 bf16, 4 N-atoms per
    # 256-thread workgroup, natural (4,4,2) K-fragment ordering. Here the operands
    # are the transposed J / dOut tiles and the contraction is the jagged axis m.
    tiled_mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
        fx.make_layout((1, 4, 1), (0, 1, 0)),
        fx.make_tile(None, None, fx.make_layout((4, 4, 2), (1, 8, 4))),
    )
    grad_dense_partials_kernel(partials, JAGGED, dOut, SEQ_OFFSETS, tiled_mma).launch(
        grid=(NK_TILES * NN_TILES, SPLIT, n_groups), block=(DDENSE_THREADS, 1, 1),
        smem=_DDENSE_SMEM_BYTES, stream=stream
    )
    grad_dense_reduce_kernel(dDense, partials).launch(
        grid=(NRED_COL_TILES, K, n_groups), block=(NRED_BLK, 1, 1), stream=stream
    )
