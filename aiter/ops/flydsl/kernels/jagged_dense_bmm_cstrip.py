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

# --- HipKittens Algorithm 1 chiplet (XCD) block-ID remap knobs ---
# MI355X / CDNA4 has 8 XCDs; the hardware routes raw block id `xy` to XCD
# `xy % NXCD`. We invert that round-robin so a group's M-tiles (which all share
# that group's Dense[b] weight matrix) co-locate on the same XCD's private L2,
# cutting the redundant HBM re-reads of Dense[b]. Measured: L2 hit 49->76% and
# -5% device time on the D=512 shapes (the bigger the reduction K / weight, the
# bigger the win). The sweet-spot chunk size C is weight-size dependent: small K
# (small weight, fits one XCD's L2 in a small chunk) wants small C; larger K
# wants a larger chunk -- see XCD_C below. W=8 is a flat optimum across shapes.
NXCD = 8
XCD_W = 8  # knob W: window height (M-tiles) for the 2D windowed traversal
XCD_C_SMALL_K = 32  # knob C for reduction K <= 256 (small weight)
XCD_C_LARGE_K = 60  # knob C for reduction K  > 256 (large weight)


def _xcd_remap(xy, num_rows, num_cols, C, W, nXCD=NXCD):
    """HipKittens Algorithm 1 remap. `xy` is the raw hardware linear block id
    (fx.Int32, uniform across the wave). `num_rows`, `num_cols`, `C`, `W`, `nXCD`
    are compile-time Python ints. Returns (row, col) output-tile coords, both
    uniform fx.Int32. row in [0, num_rows), col in [0, num_cols).

    Logical 2D tile grid for THIS kernel: row = global M-tile index
    (= off_b * bm + block_m_idx), col = block_n_idx. Phase 1 (C) clusters
    consecutive rows -- i.e. consecutive M-tiles of one group -- onto one XCD;
    Phase 2 (W) walks each XCD's chunk in W-row vertical windows so the chunk
    reuses both the group's Dense rows and its bias/columns from the fast L2.

    Both phases are exact permutations of [0, num_rows*num_cols) for ANY total /
    C / W (phase 1 identity-maps the non-(nXCD*C)-divisible tail; phase 2's
    min() window tail handles num_rows % W != 0), so the remap never drops or
    duplicates an output tile -- verified by cos=1.0 across all headline shapes.
    """
    total = num_rows * num_cols  # compile-time
    period = nXCD * C  # compile-time
    prefix = total - (total % period)  # largest [0,prefix) divisible by period
    xy = fx.Int32(xy)
    cnXCD = fx.Int32(nXCD)
    cC = fx.Int32(C)

    # Phase 1: XCD grouping. The naive HipKittens map is a bijection only on a
    # range that is a multiple of (nXCD*C); apply it to the divisible prefix and
    # identity-map the (< period) tail blocks so the whole map stays a bijection
    # on [0, total) for ANY C (e.g. C=25 with total not a multiple of 200).
    xcd = xy % cnXCD
    local = xy // cnXCD
    chunk_idx = local // cC
    pos = local % cC
    xy_g_remap = chunk_idx * (cnXCD * cC) + xcd * cC + pos
    if fx.const_expr(prefix == total):
        xy_g = xy_g_remap
    else:
        xy_g = (xy < fx.Int32(prefix)).select(xy_g_remap, xy)

    # Phase 2: windowed 2D traversal (W rows tall) within the chunk.
    cW = fx.Int32(W)
    c_num_rows = fx.Int32(num_rows)
    c_num_cols = fx.Int32(num_cols)
    tids_per_grp = cW * c_num_cols
    group_id = xy_g // tids_per_grp
    first_row = group_id * cW
    remaining = c_num_rows - first_row
    # win_h = min(num_rows - first_row, W) -- the last window may be short.
    win_h = (remaining < cW).select(remaining, cW)
    l = xy_g % tids_per_grp
    row = first_row + (l % win_h)
    col = l // win_h
    return row, col


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


# --- Epilogue C-shuffle N-strip count (occupancy experiment) ---
# The O4 LDS C-shuffle stages a row-major (BLOCK_M, BLOCK_N) bf16 tile through
# LDS = BLOCK_M*BLOCK_N*2 bytes (32 KB at default tiling), which dominates the
# epilogue LDS footprint. EPI_STRIPS=S processes the BLOCK_N tile in S strips of
# width BLOCK_N//S, so the staged sC tile shrinks to BLOCK_M*(BLOCK_N//S)*2,
# halving (S=2) or quartering (S=4) the epilogue LDS. Trade-off: S barriers and
# S store iterations instead of 1. S=1 reproduces the production full-tile path.
EPI_STRIPS = 2


@functools.lru_cache(maxsize=None)
def _build_launcher(N, K, BLOCK_M, BLOCK_N, BLOCK_K, STAGES_A, THREADS, BM_TILES, N_GROUPS, XCD_C, XCD_W, EPI_STRIPS):
    """Build (and memoize) a launch wrapper specialized to this (N, K, tiling).
    N (output) and K (reduction) are baked in as closure constants. BM_TILES
    (= ceil(max_seq_len/BLOCK_M)) and N_GROUPS are also baked in so the chiplet
    (XCD) block-ID remap dimensions are compile-time; this makes each distinct
    (shape, XCD_C, XCD_W) memoize its own compiled kernel. EPI_STRIPS is the
    epilogue C-shuffle N-strip count (see module constant)."""
    assert BLOCK_N % EPI_STRIPS == 0, "EPI_STRIPS must divide BLOCK_N"
    BLOCK_N_S = BLOCK_N // EPI_STRIPS  # epilogue strip width along N
    N_BLOCKS = N // BLOCK_N  # output column-tiles per group (compile-time)
    # Logical 2D tile grid for the chiplet remap: rows = all M-tiles of all
    # groups stacked (group-major), cols = N column-tiles of one group.
    XCD_NUM_ROWS = BM_TILES * N_GROUPS
    XCD_NUM_COLS = N_BLOCKS
    # Per-thread C accumulator length (fp32 lanes) = tile elems / threads.
    C_FRAG_LEN = BLOCK_M * BLOCK_N // THREADS
    smem_bytes = BLOCK_M * BLOCK_K * STAGES_A * 2  # bf16 A double-buffer staging

    @flyc.kernel
    def jdbba_kernel(
        C: fx.Tensor,  # out    (L, N)   bf16
        A: fx.Tensor,  # jagged (L, K)   bf16
        B: fx.Tensor,  # dense  (B_groups * N, K) bf16 (pre-transposed, tall)
        BIAS: fx.Tensor,  # bias   (B_groups * N,)   bf16
        SEQ_OFFSETS: fx.Tensor,  # (B_groups + 1,) int32
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s_A: fx.TiledCopy,
        tiled_copy_s2g_C: fx.TiledCopy,
    ):
        tid = fx.thread_idx.x
        pid_mn, _, raw_b = fx.block_idx

        # --- Chiplet (XCD) block-ID remap (HipKittens Algorithm 1) ---
        # The raw hardware linear block id for grid (gx,1,gz) is
        #   xy = raw_b * gx + pid_mn,  gx = BM_TILES * N_BLOCKS,
        # which is exactly the id the hardware uses to pick XCD = xy % 8. Remap
        # it to a (row, col) so a group's M-tiles co-locate on the same XCD.
        raw_xy = fx.Int32(raw_b) * fx.Int32(BM_TILES * N_BLOCKS) + fx.Int32(pid_mn)
        row, col = _xcd_remap(raw_xy, XCD_NUM_ROWS, XCD_NUM_COLS, XCD_C, XCD_W)
        # Keep the remapped indices uniform across the wave so all derived
        # offsets / buffer descriptors stay in SGPRs (no store waterfall).
        row = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), row))
        col = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), col))

        off_b = row // fx.Int32(BM_TILES)
        block_m_idx = row % fx.Int32(BM_TILES)
        block_n_idx = col

        # --- Device group resolution (read seq_offsets[b], seq_offsets[b+1]) ---
        seq_rsrc = fx.buffer_ops.create_buffer_resource(SEQ_OFFSETS, max_size=True)
        seq_start = fx.buffer_ops.buffer_load(seq_rsrc, fx.Int32(off_b), vec_width=1, dtype=fx.T.i32())
        seq_end = fx.buffer_ops.buffer_load(seq_rsrc, fx.Int32(off_b) + fx.Int32(1), vec_width=1, dtype=fx.T.i32())
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
            BIAS_g = fx.make_view(
                fx.add_offset(fx.get_iter(BIAS), fx.make_int_tuple(bias_elem_off)), fx.get_layout(BIAS)
            )
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
            thr_sA = thr_copy_g2s_A.partition_D(sA)  # (VA, VM, VK, STAGES_A)
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
            # the WHOLE fragment before the per-strip LDS writes.
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

            # --- N-STRIP C-shuffle (occupancy experiment) ---
            # Process the BLOCK_N output tile in EPI_STRIPS strips of BLOCK_N_S
            # columns, each routed through a SMALLER row-major (BLOCK_M, BLOCK_N_S)
            # sC tile. The retiled bf16 fragment's TRAILING axis is the N-repeat
            # (verified: splitting BLOCK_N halves that axis), so strip s is exactly
            # mma_frag_C_retile[:, :, s]. The s2g copy + gC/gBias divides use the
            # strip width, so re-reads stay N-contiguous (8 bf16/thread, 128b).
            sC = fx.make_view(
                fx.get_dyn_shared(fx.BFloat16),
                fx.make_ordered_layout((BLOCK_M, BLOCK_N_S), (1, 0)),
            )  # (BM, BN_S) row-major -- BLOCK_N_S = BLOCK_N // EPI_STRIPS
            thr_sC = thr_copy_r2s_C.partition_D(sC)  # ((1,4), 8, 1) natural MFMA
            thr_copy_s2g_C = tiled_copy_s2g_C.get_slice(tid)

            # Strip-width global C / bias divides: strip s of this block's N-tile is
            # the (block_n_idx*EPI_STRIPS + s)-th BLOCK_N_S-wide column tile.
            gC_strips = fx.flat_divide(C_buf, (BLOCK_M, BLOCK_N_S))[None, None, block_m_idx, None]  # (BM, BN_S, ns)

            # The A staging LDS reads (s2r) for the last K-tile must be done before
            # we overwrite that memory with C; the main-loop barrier guarantees it.
            fx.gpu.barrier()
            for s in fx.range_constexpr(EPI_STRIPS):
                if fx.const_expr(s > 0):
                    # Reuse the same sC for the next strip: ensure all threads
                    # finished re-reading the previous strip before overwriting.
                    fx.gpu.barrier()
                # Write this strip's fragment columns (retile last axis = N-repeat).
                fx.copy(
                    fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16),
                    mma_frag_C_retile[None, None, s],
                    thr_sC[None, None, 0],
                )
                fx.gpu.barrier()
                # Re-read N-contiguous from LDS and wide-store this strip to global.
                gC_strip = gC_strips[None, None, fx.Int32(block_n_idx) * fx.Int32(EPI_STRIPS) + s]
                thr_sC_read = thr_copy_s2g_C.partition_S(sC)  # (V, VM, VN_S)
                thr_gC = thr_copy_s2g_C.partition_D(gC_strip)  # (V, VM, VN_S) global
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

        # O4 C-shuffle store copy: row-major (BLOCK_M, BLOCK_N_S) STRIP, 8 bf16 /
        # thread along N (128b) so the global store coalesces to
        # buffer_store_dwordx4. Built at the strip width so the s2g re-read tiles
        # the smaller sC; the kernel loops EPI_STRIPS times reusing this copy.
        c_val = 8  # 16B / bf16
        c_thrs_n = BLOCK_N_S // c_val  # threads along N (strip width)
        c_thrs_m = THREADS // c_thrs_n  # threads along M
        tiled_copy_s2g_C = fx.make_tiled_copy(
            fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16),
            fx.make_layout(((c_thrs_n, c_thrs_m), (1, c_val)), ((c_thrs_m * c_val, 1), (1, c_thrs_m))),
            fx.make_tile(c_thrs_m, BLOCK_N_S),
        )

        # smem holds either A double-buffer staging or the strip-sized row-major C
        # tile. The C-shuffle now stages only one BLOCK_N_S-wide strip at a time.
        epi_smem_bytes = max(smem_bytes, BLOCK_M * BLOCK_N_S * 2)

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
    xcd_c: int | None = None,
    xcd_w: int = XCD_W,
    epi_strips: int = EPI_STRIPS,
):
    """Public entry. Derives N (output) and K (reduction) from the tall dense B
    matrix shape (B_groups * N, K), then dispatches to the per-shape kernel.

    xcd_c / xcd_w are the chiplet (XCD) remap knobs (C and W); they are baked
    into the compiled kernel, so each distinct (shape, xcd_c, xcd_w) memoizes a
    separate kernel. xcd_c defaults to a weight-size-dependent value (see the
    XCD_C_* module constants). epi_strips is the epilogue C-shuffle N-strip count
    (occupancy experiment; halves/quarters the epilogue LDS at S=2/4)."""
    N = B.shape[0] // n_groups
    K = B.shape[1]
    # Shape-dependent BLOCK_K: for small reduction K (<=256) a 2-iter K-loop with
    # BLOCK_K=128 has fewer barriers and wins (~4% on K=256 shapes); for larger K
    # the deeper BLOCK_K=64 pipeline keeps occupancy and is faster. (BLOCK_K=256
    # is unsafe: the 2-stage double-buffer epilogue mis-accumulates a single tile.)
    block_k = 128 if K <= 256 else BLOCK_K
    # Weight-size-dependent XCD chunk: small reduction K -> small Dense[b] that
    # fits one XCD's L2 in a small chunk (large C starves the shared LLC); larger
    # K wants the bigger chunk. Swept per headline shape.
    if xcd_c is None:
        xcd_c = XCD_C_SMALL_K if K <= 256 else XCD_C_LARGE_K
    bm = (max_seq_len + BLOCK_M - 1) // BLOCK_M
    launch = _build_launcher(
        N, K, BLOCK_M, BLOCK_N, block_k, STAGES_A, THREADS, bm, n_groups, xcd_c, xcd_w, epi_strips
    )
    return launch(C, A, B, BIAS, SEQ_OFFSETS, n_groups, max_seq_len, stream=stream)
