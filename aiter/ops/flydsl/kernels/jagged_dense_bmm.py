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

        #@Sami: creating the view offsets for the different sequences in the batch
        #@Claude_review: Mostly correct. Each group b owns a contiguous row slice
        # [seq_start, seq_end) of the packed (L, K)/(L, N) jagged tensors, so this
        # shifts A's and C's base pointer by seq_start*K / seq_start*N ELEMENTS
        # (add_offset is in elements, K cols per A-row, N cols per C-row). The result
        # A_g/C_g is a logical view that starts at this group's local row 0. Note B_g
        # (next line) is the DENSE weight, not a jagged sequence: it is offset by group
        # index off_b into the tall (B_groups*N, K) matrix, not by seq_start.
        a_row_off = fx.Int32(seq_start) * fx.Int32(K)
        c_row_off = fx.Int32(seq_start) * fx.Int32(N)
        A_g = fx.make_view(fx.add_offset(fx.get_iter(A), fx.make_int_tuple(a_row_off)), fx.get_layout(A))
        C_g = fx.make_view(fx.add_offset(fx.get_iter(C), fx.make_int_tuple(c_row_off)), fx.get_layout(C))
        # B / bias: select group b's slice of the tall (B_groups*N, K) matrix.
        b_row_off = fx.Int32(off_b) * fx.Int32(N) * fx.Int32(K)
        B_g = fx.make_view(fx.add_offset(fx.get_iter(B), fx.make_int_tuple(b_row_off)), fx.get_layout(B))

        #@Sami: creating the buffer tensors for the input data at the DRAM level
        #@Claude_review: Partially correct. These wrap the global-memory views in AMD
        # hardware *buffer descriptors* (V# resources), which is what enables the
        # buffer_load/buffer_store path with hardware out-of-bounds checking. "DRAM
        # level" is the right idea (these address global/HBM memory). Two corrections:
        # (1) it is not only "input data" -- C_buf is the OUTPUT, and it is built with
        # an explicit byte bound (M_b*N*2) so partial-tile stores past the group's last
        # row are HW-dropped instead of corrupting the next group. (2) A/B use
        # max_size=True (no tight bound); only C is bounded.
        A_buf = fx.rocdl.make_buffer_tensor(A_g, max_size=True)
        B_buf = fx.rocdl.make_buffer_tensor(B_g, max_size=True)
        # Bound C's descriptor to exactly M_b rows so the hardware OOB-drops any
        # partial-tile store past this group's last row (prevents writing into
        # the next group's rows). bf16 = 2 bytes. Inlined buffer-desc build
        # because the installed make_buffer_tensor has no num_records_bytes arg.
        C_buf = make_bounded_buffer_tensor(C_g, fx.Int64(fx.Int32(M_b) * fx.Int32(N) * fx.Int32(2)))

        #@Sami: Start specifying the partition at the thread block level
        #@Claude_review: Correct. flat_divide tiles the whole tensor into BLOCK_M x BLOCK_K
        # (etc.) tiles, then the [None, None, idx, None] indexing selects THIS workgroup's
        # tile along the M/N tile axis while keeping the intra-tile dims (None) and the
        # K-tile iteration axis (the trailing None for gA_k/gB_k). So gA_k = (BM, BK, k)
        # = this block's full K-strip; gC = (BM, BN) = this block's single output tile.
        # This is block/workgroup-level selection; per-thread partitioning happens later
        # via partition_S/partition_D. Good mental model.
        gA_k = fx.flat_divide(A_buf, (BLOCK_M, BLOCK_K))[None, None, block_m_idx, None]  # (BM, BK, k)
        gB_k = fx.flat_divide(B_buf, (BLOCK_N, BLOCK_K))[None, None, block_n_idx, None]  # (BN, BK, k)
        gC = fx.flat_divide(C_buf, (BLOCK_M, BLOCK_N))[None, None, block_m_idx, block_n_idx]  # (BM, BN)

        # Broadcast bias: group b's (N,) slice viewed as (BLOCK_M, N) with M-stride 0
        # (same vector for every row), then pick this block's N-tile.
        #@Sami: Broadcast the bias for the different elements in the batch
        #@Claude_review: Correct, with a nuance about WHAT is broadcast. bias_elem_off
        # selects group b's own (N,) bias vector (per-group, indexed by off_b). The
        # "broadcast" is the make_layout((BLOCK_M, N), (0, 1)): the M (row) stride is 0,
        # so every one of the BLOCK_M rows reads the SAME bias[n] vector. So it is
        # broadcast over ROWS within a tile (the bias is added to every output row),
        # not "over the batch" -- each batch group still has a distinct bias slice.
        bias_elem_off = fx.Int32(off_b) * fx.Int32(N)
        BIAS_g = fx.make_view(fx.add_offset(fx.get_iter(BIAS), fx.make_int_tuple(bias_elem_off)), fx.get_layout(BIAS))
        BIAS_buf = fx.rocdl.make_buffer_tensor(BIAS_g, max_size=True)
        gBias2d = fx.make_view(fx.get_iter(BIAS_buf), fx.make_layout((BLOCK_M, N), (0, 1)))
        gBias = fx.flat_divide(gBias2d, (BLOCK_M, BLOCK_N))[None, None, 0, block_n_idx]  # (BM, BN)

        #@Sami: Specifying the MFMA layout at the thread level
        #@Claude_review: Half right. Line 1 (thr_mma) is correct: thr_slice(tid) projects
        # the tiled MMA down to THIS lane's view, so make_fragment_A/B/C below know which
        # MFMA register slots this thread owns. Line 2 is NOT about MFMA: it is the
        # per-thread slice of the global->shared (g2s) COPY for A, i.e. which elements of
        # the A tile this thread loads. So: thr_mma = thread's MMA view; thr_copy_g2s_A =
        # thread's copy view. Both are "thread-level", but only the first is the MFMA layout.
        thr_mma = tiled_mma.thr_slice(tid)
        thr_copy_g2s_A = tiled_copy_g2s_A.get_slice(tid)

        #@Sami: Specify how the copy should be done
        #@Claude_review: Correct. A copy atom is the *instruction-level* description of a
        # single copy: the hardware op + element dtype + access width. UniversalCopy128b is
        # a generic 128-bit vectorized load/store (used here LDS<->register); BufferCopy128b
        # is the AMD buffer-descriptor 128-bit load (global<->register, with HW bounds). The
        # atom is the "how" (one transfer); the TiledCopy/partition_* built later is the
        # "who/which-elements" (mapping threads to data).
        uni_copy_128b = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        buffer_copy_128b = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)

        #@Sami: Create the specification at the thread level based on the copy definition above
        #@Claude_review: Correct, with an important detail: these TiledCopies are derived
        # from the *MMA operand layout* (make_tiled_copy_A/B(..., tiled_mma)), NOT from the
        # g2s copy above. They guarantee the copy lands data in exactly the register layout
        # the MFMA expects. s2r_A = shared->register feed for MMA operand A; g2r_B =
        # global->register feed for MMA operand B (B skips LDS here). .get_slice(tid) then
        # gives this thread's portion. So "thread-level spec matched to the MMA" is right.
        thr_copy_s2r_A = fx.make_tiled_copy_A(buffer_copy_128b, tiled_mma).get_slice(tid)
        thr_copy_g2r_B = fx.make_tiled_copy_B(buffer_copy_128b, tiled_mma).get_slice(tid)

        #@Sami: ???
        #@Claude_review: This allocates A's LDS (shared-memory) staging buffer and gives it
        # a swizzled layout. make_ordered_layout((BLOCK_M, BLOCK_K, STAGES_A), (1,0,2)) is a
        # (BM, BK, 2) tile -- the trailing 2 is DOUBLE BUFFERING (STAGES_A=2 ping-pong slots).
        # SwizzleType.get(3,3,3) is an XOR swizzle composed onto that layout to avoid LDS
        # bank conflicts on the ds_read into MMA. make_composed_layout(swizzle, base) means a
        # coord is mapped by base then permuted by swizzle. sA = make_view over get_dyn_shared
        # = the actual LDS tensor (BM, BK, STAGES_A) that A is staged through before MFMA.
        composed_layout_A = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(3, 3, 3)),
            fx.make_ordered_layout((BLOCK_M, BLOCK_K, STAGES_A), (1, 0, 2)),
        )
        sA = fx.make_view(fx.get_dyn_shared(fx.BFloat16), composed_layout_A)  # (BM, BK, STAGES_A)

        #@Sami: Specify that we partition the data within the tile. Why are there two variants partition_S and partition_D?
        #@Claude_review: Correct on intent. Answer to the question: partition_S = the SOURCE
        # tensor of a copy, partition_D = the DESTINATION tensor, under the SAME TiledCopy
        # thread-value mapping. They are separate because source and destination usually have
        # different memory layouts, so the per-thread slice differs even though each thread
        # moves the same logical elements. Here the g2s copy is global gA_k (src) -> shared sA
        # (dst): thr_gA_k = partition_S(gA_k), thr_sA = partition_D(sA). thr_sA_s2r =
        # partition_S(sA) re-partitions the SAME sA as the SOURCE of the later shared->register
        # copy (different TiledCopy, thr_copy_s2r_A). thr_gB_k = partition_S(gB_k) is B's global
        # source (its register destination is the retiled mma_frag_B, see below).
        thr_gA_k = thr_copy_g2s_A.partition_S(gA_k)  # (VA, VM, VK, k)
        thr_sA = thr_copy_g2s_A.partition_D(sA)      # (VA, VM, VK, STAGES_A)
        thr_sA_s2r = thr_copy_s2r_A.partition_S(sA)  # (VA, VM, VK, STAGES_A)
        thr_gB_k = thr_copy_g2r_B.partition_S(gB_k)  # (VB, VN, VK, k)

        #@Sami: Speficy how the layout at register level should be based on the tile partition defined above
        #@Claude_review: Correct. make_fragment_like allocates a REGISTER tensor whose layout
        # matches one stage (index 0 of the STAGES_A axis) of the partitioned shared tensor
        # thr_sA. copy_frag_A is the per-thread register buffer that holds A on the way from
        # global -> (register) -> LDS: the prologue/pipeline does copy(global -> copy_frag_A)
        # then copy(copy_frag_A -> thr_sA). So yes: a register-level layout derived from the
        # thread's tile partition.
        copy_frag_A = fx.make_fragment_like(thr_sA[None, None, None, 0])

        #@Sami: Specify the register level layout but for what???
        #@Claude_review: These are the MMA OPERAND register fragments -- the actual VGPR tiles
        # each lane feeds into / gets out of the MFMA instruction. mma_frag_A = A operand
        # fragment (sourced from LDS tile sA[...,0]); mma_frag_B = B operand fragment, double
        # buffered (stages=2) since B streams straight from global; mma_frag_C = the fp32
        # ACCUMULATOR fragment (output, shaped from gC, filled with 0 before the K-loop). So:
        # register layouts "for the MFMA inputs (A, B) and output accumulator (C)".
        mma_frag_A = thr_mma.make_fragment_A(sA[None, None, 0])
        mma_frag_B = thr_mma.make_fragment_B(gB_k, stages=2)
        mma_frag_C = thr_mma.make_fragment_C(gC)

        #@Sami: Update the layout for the thread copy based on the MMA layout for the operations (reshuffling???)
        #@Claude_review: Right idea, but "reshuffling" is misleading -- retile moves NO data and
        # emits no instructions. It is a pure re-VIEW: it re-expresses the MMA fragment
        # (mma_frag_A/B, laid out the way MFMA wants) under the COPY's thread-value tiling, so
        # the copy atom can write directly into those same registers. Think "relabel the same
        # VGPRs so both the copy and the MFMA agree on them", not "rearrange the values".
        mma_frag_A_retile = thr_copy_s2r_A.retile(mma_frag_A)
        mma_frag_B_retile = thr_copy_g2r_B.retile(mma_frag_B)

        #@Sami: Get the stride at the block level
        #@Claude_review: Imprecise. stride[2] is specifically the stride of the K-TILE
        # iteration axis (mode 2 of gA_k=(BM, BK, k) / gB_k=(BN, BK, k)) -- i.e. how far apart
        # consecutive BLOCK_K tiles are in memory, as a scalar. It is used in run_pipeline_stage
        # as soffset = next_k * stride to advance the buffer_load to the next K tile, instead of
        # rebuilding a view each iteration. So it is the "K-tile step stride", not a generic
        # block-level stride.
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
        #@Sami2: Prepare the output tiles where we move the result of the MFMA tile in LDS into the output tensor in registers
        #@Claude_review: Two corrections. (1) "in LDS" is wrong: the MFMA result lives in
        # REGISTERS -- mma_frag_C is the fp32 accumulator fragment (VGPRs) from
        # make_fragment_C, and this kernel does a direct register->global epilogue (no
        # CShuffle-through-LDS). (2) This block does not yet MOVE anything; it only SETS UP
        # the store: mma_frag_C_bf16 is a new bf16 register fragment, thr_copy_r2g_C is the
        # register->global (r2g) tiled copy for C, mma_frag_C_retile re-views the bf16 frag
        # under that copy's layout, and thr_gC = partition_S(gC) is this thread's slice of
        # the global C tile (the eventual store destination). The actual convert+store happens
        # in the two @Sami2 blocks below.
        mma_frag_C_bf16 = fx.make_fragment_like(mma_frag_C, fx.BFloat16.ir_type)
        thr_copy_r2g_C = fx.make_tiled_copy_C(
            fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16), tiled_mma
        ).get_slice(tid)
        mma_frag_C_retile = thr_copy_r2g_C.retile(mma_frag_C_bf16)
        thr_gC = thr_copy_r2g_C.partition_S(gC)

        # Load this thread's bias slice (same partition as C) into fp32 and add
        # to the accumulator before the bf16 truncation (broadcast over rows).
        #@Sami2: Effectively loads and adds biases to the output tile registers
        #@Claude_review: Correct in spirit; here is the precise split. This @Sami2 block only
        # LOADS the bias: partition_S(gBias) slices the (broadcast) bias tile with the SAME
        # thread mapping as C (so each lane gets exactly the bias for its C elements), the copy
        # pulls it into bias_frag (bf16), and ExtFOp widens it to fp32 (bias_f32). The ADD
        # itself is the next statement: mma_frag_C_bf16.store(trunc(mma_frag_C.load() +
        # bias_f32)) -- i.e. bias is added to the fp32 ACCUMULATOR, then truncated to bf16 into
        # mma_frag_C_bf16. So "loads and adds bias" is right; just note add+truncate happen in
        # the store expression, and the accumulate is done in fp32 before the bf16 cast.
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
        #@Sami2: Write back the result to the output buffer in DRAM
        #@Claude_review: Correct. This copies the bf16 register fragment (mma_frag_C_retile)
        # to the global C tile (thr_gC) via a BufferCopy16b store -- register->global ("DRAM").
        # Worth noting: because C_buf was built with the M_b-row byte bound, this is implicitly
        # a MASKED store -- the hardware buffer descriptor OOB-drops any rows past the group's
        # last row, so a partial tail tile cannot corrupt the next group's rows. (thr_gC is
        # named partition_S only by API convention; it is the store destination here.)
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
