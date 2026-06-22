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

# dJagged output is (M, K). Its second (column) axis is K, tiled by BLOCK_N; the
# contraction runs over N, tiled by BLOCK_K. With N == K == 128 these counts
# match the forward kernel exactly.
KOUT_BLOCKS = K // BLOCK_N  # column-tiles of the (M, K) output (compile-time)
NRED_TILES = N // BLOCK_K   # contraction tiles over N (compile-time)

# dDense reduction tiling. The (K, N) output tile is owned by a 16x16 thread grid
# (256 threads); each thread accumulates a (KPT, NPT) register sub-block over the
# jagged axis. Rows are staged through LDS DDENSE_BM at a time.
DDENSE_BM = 64
DDENSE_TK = 16
DDENSE_TN = 16
KPT = K // DDENSE_TK  # 8
NPT = N // DDENSE_TN  # 8
DDENSE_THREADS = DDENSE_TK * DDENSE_TN  # 256
_J_LDS_LOADS = (DDENSE_BM * K) // DDENSE_THREADS
_D_LDS_LOADS = (DDENSE_BM * N) // DDENSE_THREADS
_DDENSE_SMEM_BYTES = (DDENSE_BM * K + DDENSE_BM * N) * 2  # bf16 staging for J and dOut


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
    _, off_s, off_b = fx.block_idx
    off_b = fx.Int32(off_b)
    off_s = fx.Int32(off_s)

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
        row_view = fx.slice(DOUT_buf, (r, None))
        row_div = fx.logical_divide(row_view, fx.make_layout(1, 1))
        val = _load_scalar(copy_bf16, fx.BFloat16, row_div, tid)
        acc = acc + val.to(fx.Float32)
        results = yield [acc]
    acc_final = results

    PART_buf = fx.rocdl.make_buffer_tensor(PARTIALS, max_size=True)
    part_row = off_b * fx.Int32(SPLIT) + off_s
    part_div = fx.logical_divide(fx.slice(PART_buf, (part_row, None)), fx.make_layout(1, 1))
    copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    _store_scalar(copy_f32, fx.Float32, part_div, tid, acc_final)


@flyc.kernel
def grad_bias_reduce_kernel(
    DBIAS: fx.Tensor,     # out    (n_groups, N)          bf16
    PARTIALS: fx.Tensor,  # in     (n_groups * SPLIT, N)  fp32
):
    # Sum this group's SPLIT fp32 partials and write bf16 dBias[b]. Thread tid
    # owns column n.
    tid = fx.thread_idx.x
    off_b = fx.Int32(fx.block_idx.x)

    PART_buf = fx.rocdl.make_buffer_tensor(PARTIALS, max_size=True)
    DBIAS_buf = fx.rocdl.make_buffer_tensor(DBIAS, max_size=True)
    copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    copy_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)

    acc = fx.Float32(0.0)
    for s in fx.range_constexpr(SPLIT):
        part_row = off_b * fx.Int32(SPLIT) + fx.Int32(s)
        row_div = fx.logical_divide(fx.slice(PART_buf, (part_row, None)), fx.make_layout(1, 1))
        acc = acc + _load_scalar(copy_f32, fx.Float32, row_div, tid)

    out_div = fx.logical_divide(fx.slice(DBIAS_buf, (off_b, None)), fx.make_layout(1, 1))
    _store_scalar(copy_bf16, fx.BFloat16, out_div, tid, acc.to(fx.BFloat16))


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
        grid=(1, SPLIT, n_groups), block=(N, 1, 1), stream=stream
    )
    grad_bias_reduce_kernel(dBias, partials).launch(
        grid=(n_groups, 1, 1), block=(N, 1, 1), stream=stream
    )


@flyc.kernel
def grad_dense_partials_kernel(
    PARTIALS: fx.Tensor,     # out    (n_groups * SPLIT * K, N)  fp32
    JAGGED: fx.Tensor,       # jagged (L, K)                     bf16
    DOUT: fx.Tensor,         # grad   (L, N)                     bf16
    SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
):
    # dDense[b][k,n] = sum_m Jagged[m,k] * dOut[m,n]. Split s reduces a strided
    # subset of the group's BM-row tiles; each thread owns a (KPT, NPT) register
    # block of the (K, N) output.
    tid = fx.thread_idx.x
    _, off_s, off_b = fx.block_idx
    off_b = fx.Int32(off_b)
    off_s = fx.Int32(off_s)
    tk = tid // fx.Int32(DDENSE_TN)
    tn = tid % fx.Int32(DDENSE_TN)

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

    smem = fx.get_dyn_shared(fx.BFloat16)
    sJ = fx.make_view(smem, fx.make_layout((DDENSE_BM, K), (K, 1)))
    smem_d = fx.add_offset(smem, fx.make_int_tuple(DDENSE_BM * K))
    sD = fx.make_view(smem_d, fx.make_layout((DDENSE_BM, N), (N, 1)))

    acc = fx.make_rmem_tensor(fx.make_layout((KPT, NPT), (NPT, 1)), fx.Float32)
    for ki in fx.range_constexpr(KPT):
        for ni in fx.range_constexpr(NPT):
            fx.memref_store(fx.Float32(0.0), acc, (ki, ni))

    num_tiles = (M_b + fx.Int32(DDENSE_BM - 1)) // fx.Int32(DDENSE_BM)

    for m_tile in range(off_s, num_tiles, fx.Int32(SPLIT)):
        for i in fx.range_constexpr(_J_LDS_LOADS):
            lin = tid + fx.Int32(i * DDENSE_THREADS)
            lrow = lin // fx.Int32(K)
            lcol = lin % fx.Int32(K)
            joff = (m_tile * DDENSE_BM + lrow) * fx.Int32(K) + lcol
            jval = fx.buffer_ops.buffer_load(j_rsrc, joff, vec_width=1, dtype=fx.T.bf16())
            fx.memref_store(jval, sJ, (lrow, lcol))
        for i in fx.range_constexpr(_D_LDS_LOADS):
            lin = tid + fx.Int32(i * DDENSE_THREADS)
            lrow = lin // fx.Int32(N)
            lcol = lin % fx.Int32(N)
            doff = (m_tile * DDENSE_BM + lrow) * fx.Int32(N) + lcol
            dval = fx.buffer_ops.buffer_load(d_rsrc, doff, vec_width=1, dtype=fx.T.bf16())
            fx.memref_store(dval, sD, (lrow, lcol))
        fx.gpu.barrier()

        a = [[fx.memref_load(acc, (ki, ni)) for ni in range(NPT)] for ki in range(KPT)]
        for m in fx.range_constexpr(DDENSE_BM):
            jv = [fx.memref_load(sJ, (m, tk * fx.Int32(KPT) + fx.Int32(ki))).to(fx.Float32) for ki in range(KPT)]
            dv = [fx.memref_load(sD, (m, tn * fx.Int32(NPT) + fx.Int32(ni))).to(fx.Float32) for ni in range(NPT)]
            for ki in fx.range_constexpr(KPT):
                for ni in fx.range_constexpr(NPT):
                    a[ki][ni] = a[ki][ni] + jv[ki] * dv[ni]
        for ki in fx.range_constexpr(KPT):
            for ni in fx.range_constexpr(NPT):
                fx.memref_store(a[ki][ni], acc, (ki, ni))
        fx.gpu.barrier()

    part_rsrc = fx.buffer_ops.create_buffer_resource(PARTIALS, max_size=True)
    base_part_row = (off_b * fx.Int32(SPLIT) + off_s) * fx.Int32(K)
    for ki in fx.range_constexpr(KPT):
        for ni in fx.range_constexpr(NPT):
            row = base_part_row + tk * fx.Int32(KPT) + fx.Int32(ki)
            off = row * fx.Int32(N) + tn * fx.Int32(NPT) + fx.Int32(ni)
            fx.buffer_ops.buffer_store(fx.memref_load(acc, (ki, ni)), part_rsrc, off)


@flyc.kernel
def grad_dense_reduce_kernel(
    DDENSE: fx.Tensor,    # out    (n_groups * K, N)          bf16
    PARTIALS: fx.Tensor,  # in     (n_groups * SPLIT * K, N)  fp32
):
    # Sum this group's SPLIT fp32 (K, N) partials and write bf16 dDense[b]. Block
    # (k-row, group); thread tid owns column n.
    tid = fx.thread_idx.x
    off_k = fx.Int32(fx.block_idx.x)
    off_b = fx.Int32(fx.block_idx.y)

    PART_buf = fx.rocdl.make_buffer_tensor(PARTIALS, max_size=True)
    DD_buf = fx.rocdl.make_buffer_tensor(DDENSE, max_size=True)
    copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    copy_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)

    acc = fx.Float32(0.0)
    for s in fx.range_constexpr(SPLIT):
        part_row = (off_b * fx.Int32(SPLIT) + fx.Int32(s)) * fx.Int32(K) + off_k
        row_div = fx.logical_divide(fx.slice(PART_buf, (part_row, None)), fx.make_layout(1, 1))
        acc = acc + _load_scalar(copy_f32, fx.Float32, row_div, tid)

    out_row = off_b * fx.Int32(K) + off_k
    out_div = fx.logical_divide(fx.slice(DD_buf, (out_row, None)), fx.make_layout(1, 1))
    _store_scalar(copy_bf16, fx.BFloat16, out_div, tid, acc.to(fx.BFloat16))


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
    a partials pass accumulates fp32 (K, N) partials per (group, split) slot,
    then a reduce pass sums them into bf16 dDense.
    """
    grad_dense_partials_kernel(partials, JAGGED, dOut, SEQ_OFFSETS).launch(
        grid=(1, SPLIT, n_groups), block=(DDENSE_THREADS, 1, 1), smem=_DDENSE_SMEM_BYTES, stream=stream
    )
    grad_dense_reduce_kernel(dDense, partials).launch(
        grid=(K, n_groups, 1), block=(N, 1, 1), stream=stream
    )
