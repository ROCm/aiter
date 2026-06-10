# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# jagged_dense_bmm_broadcast_add (jdbba) — FlyDSL layout-API kernel,
# M-REGISTER-TILING variant (clone of jagged_dense_bmm_gen).
#
# This variant has each thread block compute MM (=2) consecutive M-subtiles of
# BLOCK_M rows each (MM*BLOCK_M logical output rows per block). The dense weight
# B K-tile is loaded into registers ONCE per K-iter and reused across all MM
# A-subtiles' MFMAs (classic register-blocking). This cuts the number of M-tile
# blocks per group by MMx, so each group's Dense[b] is re-fetched ~MMx fewer
# times -> less HBM traffic on the bottleneck (B re-reads).
#
# Cost: the A LDS staging grows by MMx (MM concurrent swizzled sA buffers held
# across the K-loop). The C-shuffle epilogue tile is REUSED sequentially across
# the MM subtiles, so it does NOT grow. Net LDS for MM=2: ~64KB (<160KB).
# The grid M dimension shrinks by MMx: bm = ceil(Mi/(MM*BLOCK_M)). The XCD remap
# is kept unchanged and fed the reduced bm. The MM physical M-tiles of a super-
# tile are at (super_m_idx*MM + j) for j in [0,MM); a subtile fully past M_b is
# masked off (its start_m >= M_b), partial tiles handled by the bounded C buffer.
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
MM = 2  # M register-tiling factor: M-subtiles per block sharing one B fragment.

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


@functools.lru_cache(maxsize=None)
def _build_launcher(
    N, K, BLOCK_M, BLOCK_N, BLOCK_K, STAGES_A, THREADS, BM_TILES, N_GROUPS, XCD_C, XCD_W, MM
):
    """Build (and memoize) a launch wrapper specialized to this (N, K, tiling).
    N (output) and K (reduction) are baked in as closure constants. BM_TILES
    (= ceil(max_seq_len/BLOCK_M)) and N_GROUPS are also baked in so the chiplet
    (XCD) block-ID remap dimensions are compile-time; this makes each distinct
    (shape, XCD_C, XCD_W) memoize its own compiled kernel."""
    N_BLOCKS = N // BLOCK_N  # output column-tiles per group (compile-time)
    # Logical 2D tile grid for the chiplet remap: rows = all M-tiles of all
    # groups stacked (group-major), cols = N column-tiles of one group.
    XCD_NUM_ROWS = BM_TILES * N_GROUPS
    XCD_NUM_COLS = N_BLOCKS
    # Per-thread C accumulator length (fp32 lanes) = tile elems / threads.
    C_FRAG_LEN = BLOCK_M * BLOCK_N // THREADS
    # MM concurrent A double-buffers (one per M-subtile) held across the K-loop.
    smem_bytes = BLOCK_M * BLOCK_K * STAGES_A * 2 * MM  # bf16 A staging x MM

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
        super_m_idx = row % fx.Int32(BM_TILES)  # super-tile index (MM*BLOCK_M rows)
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
        # Super-tile's first physical M-tile; guard the whole block on subtile 0.
        super_start_m = super_m_idx * fx.Int32(MM * BLOCK_M)

        if super_start_m < M_b:
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
            # Bound C to M_b rows so the hardware OOB-drops partial-tile stores
            # (this also masks any fully-OOB subtile -- all its rows are dropped).
            C_buf = make_bounded_buffer_tensor(C_g, fx.Int64(fx.Int32(M_b) * fx.Int32(N) * fx.Int32(2)))

            # MM physical M-tile indices of this super-tile. Each must stay uniform
            # across the wave (it feeds buffer descriptors / soffsets).
            block_m_idx = [
                fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), super_m_idx * fx.Int32(MM) + fx.Int32(j)))
                for j in range(MM)
            ]

            # --- Per-subtile global views (A jagged rows, C out rows). B and bias
            # are shared across subtiles (same group, same N column-tile). ---
            gA_k = [
                fx.flat_divide(A_buf, (BLOCK_M, BLOCK_K))[None, None, block_m_idx[j], None]  # (BM, BK, k)
                for j in range(MM)
            ]
            gB_k = fx.flat_divide(B_buf, (BLOCK_N, BLOCK_K))[None, None, block_n_idx, None]  # (BN, BK, k)
            gC = [
                fx.flat_divide(C_buf, (BLOCK_M, BLOCK_N))[None, None, block_m_idx[j], block_n_idx]  # (BM, BN)
                for j in range(MM)
            ]

            # Broadcast bias: group b's (N,) slice viewed as (BLOCK_M, N) M-stride 0
            # (shared across all MM subtiles).
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

            # MM concurrent A staging buffers as an extra outermost LDS dim. The
            # swizzle (8-elem mask) only touches the inner (BM,BK) tile; each
            # subtile region is BLOCK_M*BLOCK_K*STAGES_A elems apart (a multiple of
            # the swizzle period), so subtile swizzles stay independent.
            composed_layout_A = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 3, 3)),
                fx.make_ordered_layout((BLOCK_M, BLOCK_K, STAGES_A, MM), (1, 0, 2, 3)),
            )
            sA_all = fx.make_view(fx.get_dyn_shared(fx.BFloat16), composed_layout_A)  # (BM,BK,ST,MM)
            sA = [sA_all[None, None, None, j] for j in range(MM)]  # MM x (BM, BK, STAGES_A)

            thr_gA_k = [thr_copy_g2s_A.partition_S(gA_k[j]) for j in range(MM)]  # (VA,VM,VK,k)
            thr_sA = [thr_copy_g2s_A.partition_D(sA[j]) for j in range(MM)]  # (VA,VM,VK,ST)
            thr_sA_s2r = [thr_copy_s2r_A.partition_S(sA[j]) for j in range(MM)]  # (VA,VM,VK,ST)
            thr_gB_k = thr_copy_g2r_B.partition_S(gB_k)  # (VB, VN, VK, k) -- shared B

            copy_frag_A = [fx.make_fragment_like(thr_sA[j][None, None, None, 0]) for j in range(MM)]

            mma_frag_A = [thr_mma.make_fragment_A(sA[j][None, None, 0]) for j in range(MM)]
            mma_frag_B = thr_mma.make_fragment_B(gB_k, stages=2)  # one B fragment, reused
            mma_frag_C = [thr_mma.make_fragment_C(gC[j]) for j in range(MM)]

            mma_frag_A_retile = [thr_copy_s2r_A.retile(mma_frag_A[j]) for j in range(MM)]
            mma_frag_B_retile = thr_copy_g2r_B.retile(mma_frag_B)

            gA_k_stride = [fx.get_scalar(gA_k[j].stride[2]) for j in range(MM)]
            gB_k_stride = fx.get_scalar(gB_k.stride[2])

            def run_pipeline_stage(read_stage, next_k, read_next=True):
                write_stage = read_stage ^ 1
                if fx.const_expr(read_next):
                    next_k = fx.Int32(next_k)
                    # One B K-tile prefetch (shared); MM A K-tile prefetches.
                    fx.copy(
                        buffer_copy_128b,
                        thr_gB_k[None, None, None, 0],
                        mma_frag_B_retile[None, None, None, write_stage],
                        soffset=next_k * gB_k_stride,
                    )
                    for j in fx.range_constexpr(MM):
                        fx.copy(
                            buffer_copy_128b,
                            thr_gA_k[j][None, None, None, 0],
                            copy_frag_A[j],
                            soffset=next_k * gA_k_stride[j],
                        )

                for block_k_iter in fx.range_constexpr(BLOCK_K // 32):
                    # Read the shared B fragment once per k-subtile; issue MM MFMAs
                    # (one per A-subtile) against it -> B load amortized over MM.
                    for j in fx.range_constexpr(MM):
                        fx.copy(
                            uni_copy_128b,
                            thr_sA_s2r[j][None, None, block_k_iter, read_stage],
                            mma_frag_A_retile[j][None, None, block_k_iter],
                        )
                        fx.gemm(
                            tiled_mma,
                            mma_frag_C[j],
                            mma_frag_A[j][None, None, (None, block_k_iter)],
                            mma_frag_B[None, None, (None, block_k_iter), read_stage],
                            mma_frag_C[j],
                            traversal_order=fx.GemmTraversalOrder.KNM,
                        )

                for j in fx.range_constexpr(MM):
                    fx.copy(uni_copy_128b, copy_frag_A[j], thr_sA[j][None, None, None, write_stage])
                fx.gpu.barrier()

            # Prologue: load K-tile 0 into the read buffer (B shared, MM A subtiles).
            fx.copy(buffer_copy_128b, thr_gB_k[None, None, None, 0], mma_frag_B_retile[None, None, None, 0])
            for j in fx.range_constexpr(MM):
                fx.copy(buffer_copy_128b, thr_gA_k[j][None, None, None, 0], copy_frag_A[j])
                mma_frag_C[j].fill(0)
                fx.copy(uni_copy_128b, copy_frag_A[j], thr_sA[j][None, None, None, 0])
            fx.gpu.barrier()

            # Main K-loop (double-buffered over K // BLOCK_K tiles). K>=128 here.
            for k_iter in range(0, K // BLOCK_K - 2, 2):
                run_pipeline_stage(read_stage=0, next_k=k_iter + 1)
                run_pipeline_stage(read_stage=1, next_k=k_iter + 2)
            run_pipeline_stage(read_stage=0, next_k=K // BLOCK_K - 1)
            run_pipeline_stage(read_stage=1, next_k=None, read_next=False)

            # --- Epilogue: fp32 accumulators -> bf16, broadcast bias, masked store ---
            # O4 LDS C-shuffle (see jagged_dense_bmm_gen): write the M-major MFMA
            # fragment to a row-major LDS tile, barrier, re-read N-contiguous, store
            # 128b. Here we run the epilogue PER SUBTILE, REUSING one sC tile
            # (BLOCK_M, BLOCK_N) sequentially so the C-shuffle LDS does NOT grow.
            thr_copy_r2s_C = fx.make_tiled_copy_C(
                fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16), tiled_mma
            ).get_slice(tid)
            thr_copy_s2g_C = tiled_copy_s2g_C.get_slice(tid)

            # Broadcast bias read once (shared across subtiles) via natural MFMA layout.
            thr_gBias = thr_copy_r2s_C.partition_S(gBias)
            bias_frag = fx.make_fragment_like(thr_gBias)
            fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16), thr_gBias, bias_frag)
            bias_f32 = fx.arith.ExtFOp(fx.T.VectorType.get([C_FRAG_LEN], fx.T.f32()), bias_frag.load()).result

            # Row-major shared C tile, reusing the dynamic shared memory.
            sC = fx.make_view(
                fx.get_dyn_shared(fx.BFloat16),
                fx.make_ordered_layout((BLOCK_M, BLOCK_N), (1, 0)),
            )  # (BM, BN) row-major
            thr_sC = thr_copy_r2s_C.partition_D(sC)  # natural MFMA layout into LDS
            thr_sC_read = thr_copy_s2g_C.partition_S(sC)  # (V, VM, VN)

            for j in fx.range_constexpr(MM):
                mma_frag_C_bf16 = fx.make_fragment_like(mma_frag_C[j], fx.BFloat16.ir_type)
                mma_frag_C_retile = thr_copy_r2s_C.retile(mma_frag_C_bf16)
                mma_frag_C_bf16.store(
                    fx.arith.trunc_f(
                        fx.T.VectorType.get([C_FRAG_LEN], fx.T.bf16()),
                        fx.arith.addf(mma_frag_C[j].load(), bias_f32),
                    )
                )
                # Barrier before reusing sC (also guards the last K-tile s2r reads
                # before we overwrite that LDS with C on the first iteration).
                fx.gpu.barrier()
                fx.copy(fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16), mma_frag_C_retile, thr_sC)
                fx.gpu.barrier()

                thr_gC = thr_copy_s2g_C.partition_D(gC[j])  # (V, VM, VN) global
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
        c_thrs_n = BLOCK_N // c_val  # threads along N
        c_thrs_m = THREADS // c_thrs_n  # threads along M
        tiled_copy_s2g_C = fx.make_tiled_copy(
            fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16),
            fx.make_layout(((c_thrs_n, c_thrs_m), (1, c_val)), ((c_thrs_m * c_val, 1), (1, c_thrs_m))),
            fx.make_tile(c_thrs_m, BLOCK_N),
        )

        # smem holds either the MM-wide A double-buffer staging or the (reused)
        # row-major C tile.
        epi_smem_bytes = max(smem_bytes, BLOCK_M * BLOCK_N * 2)

        # M register-tiling: each block covers MM*BLOCK_M rows, so the M grid
        # dimension shrinks by MMx. BM_TILES (= bm) is the super-tile count.
        bm = (max_seq_len + MM * BLOCK_M - 1) // (MM * BLOCK_M)
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
    mm: int = MM,
):
    """Public entry. Derives N (output) and K (reduction) from the tall dense B
    matrix shape (B_groups * N, K), then dispatches to the per-shape kernel.

    xcd_c / xcd_w are the chiplet (XCD) remap knobs (C and W); they are baked
    into the compiled kernel, so each distinct (shape, xcd_c, xcd_w) memoizes a
    separate kernel. xcd_c defaults to a weight-size-dependent value (see the
    XCD_C_* module constants).

    mm is the M register-tiling factor (M-subtiles per block sharing one B
    fragment); also baked into the compiled kernel."""
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
    # bm = number of super-tiles (MM*BLOCK_M rows each) per group; feeds the XCD
    # remap row dimension (BM_TILES) so the remap operates on super-tiles.
    bm = (max_seq_len + mm * BLOCK_M - 1) // (mm * BLOCK_M)
    launch = _build_launcher(
        N, K, BLOCK_M, BLOCK_N, block_k, STAGES_A, THREADS, bm, n_groups, xcd_c, xcd_w, mm
    )
    return launch(C, A, B, BIAS, SEQ_OFFSETS, n_groups, max_seq_len, stream=stream)
