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
from flydsl.runtime.device import get_rocm_arch

# Default tiling (override via the factory args).
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
STAGES_A = 2
THREADS = 256


def _use_mfma_k32(arch: str | None = None) -> bool:
    """CDNA4 (gfx950) has v_mfma_f32_16x16x32_bf16 (twice the K per instruction
    vs the gfx942 16x16x16 atom). It halves MFMA issue and is ~4-7% faster here,
    bit-exact. gfx942/MI300 lacks the 32-K atom, so fall back to 16x16x16."""
    if arch is None:
        arch = get_rocm_arch()
    return bool(arch) and arch.lower().startswith("gfx95")


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
XCD_C_LARGE_K = 120  # knob C for reduction K  > 256 (large weight; autotuned)


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
def _build_launcher(N, K, BLOCK_M, BLOCK_N, BLOCK_K, STAGES_A, THREADS, BM_TILES, N_GROUPS, XCD_C, XCD_W, USE_MFMA_K32):
    """Build (and memoize) a launch wrapper specialized to this (N, K, tiling).
    N (output) and K (reduction) are baked in as closure constants. BM_TILES
    (= ceil(max_seq_len/BLOCK_M)) and N_GROUPS are also baked in so the chiplet
    (XCD) block-ID remap dimensions are compile-time; this makes each distinct
    (shape, XCD_C, XCD_W, USE_MFMA_K32) memoize its own compiled kernel."""
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

            # Bound A and C to M_b rows so the hardware OOB-drops the partial
            # bottom-tile accesses. A is allocated exactly L rows, so the LAST
            # group's partial tile (M_b not a multiple of BLOCK_M) would otherwise
            # read up to BLOCK_M-1 rows past the allocation end via the unbounded
            # max_size descriptor -- a data-dependent OOB that GPU-faults when
            # those rows land on an unmapped page. Bounding drops them (read as 0,
            # harmless: the matching output rows are masked out of the C store).
            A_buf = make_bounded_buffer_tensor(A_g, fx.Int64(fx.Int32(M_b) * fx.Int32(K) * fx.Int32(2)))
            B_buf = fx.rocdl.make_buffer_tensor(B_g, max_size=True)
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
            mma_frag_B = thr_mma.make_fragment_B(gB_k, stages=3)
            mma_frag_C = thr_mma.make_fragment_C(gC)

            mma_frag_A_retile = thr_copy_s2r_A.retile(mma_frag_A)
            mma_frag_B_retile = thr_copy_g2r_B.retile(mma_frag_B)

            gA_k_stride = fx.get_scalar(gA_k.stride[2])
            gB_k_stride = fx.get_scalar(gB_k.stride[2])

            # B is register-double/triple-buffered with a DEEPER prefetch than A.
            # A (LDS) stays 1-ahead/2-deep on a_read_stage^1. B uses 3 register
            # slots and prefetches 2 tiles ahead: while consuming logical tile i
            # from b_read_stage, it loads tile i+2 into b_write_stage = (i+2)%3.
            def run_pipeline_stage(a_read_stage, b_read_stage, a_next_k, b_next_k,
                                   read_next_a=True, read_next_b=True):
                a_write_stage = a_read_stage ^ 1
                b_write_stage = (b_read_stage + 2) % 3
                if fx.const_expr(read_next_a):
                    a_next_k = fx.Int32(a_next_k)
                    fx.copy(
                        buffer_copy_128b,
                        thr_gA_k[None, None, None, 0],
                        copy_frag_A,
                        soffset=a_next_k * gA_k_stride,
                    )
                if fx.const_expr(read_next_b):
                    b_next_k = fx.Int32(b_next_k)
                    fx.copy(
                        buffer_copy_128b,
                        thr_gB_k[None, None, None, 0],
                        mma_frag_B_retile[None, None, None, b_write_stage],
                        soffset=b_next_k * gB_k_stride,
                    )

                for block_k_iter in fx.range_constexpr(BLOCK_K // 32):
                    fx.copy(
                        uni_copy_128b,
                        thr_sA_s2r[None, None, block_k_iter, a_read_stage],
                        mma_frag_A_retile[None, None, block_k_iter],
                    )
                    # K=32 atom: one atom spans tile_K_perm=32, so the fragment K
                    # dim is FLAT (index by block_k_iter). The K=16 atom packs 2
                    # atoms per 32-wide perm group -> hierarchical (None, iter).
                    if fx.const_expr(USE_MFMA_K32):
                        frag_A_k = mma_frag_A[None, None, block_k_iter]
                        frag_B_k = mma_frag_B[None, None, block_k_iter, b_read_stage]
                    else:
                        frag_A_k = mma_frag_A[None, None, (None, block_k_iter)]
                        frag_B_k = mma_frag_B[None, None, (None, block_k_iter), b_read_stage]
                    fx.gemm(
                        tiled_mma,
                        mma_frag_C,
                        frag_A_k,
                        frag_B_k,
                        mma_frag_C,
                        traversal_order=fx.GemmTraversalOrder.KNM,
                    )

                fx.copy(uni_copy_128b, copy_frag_A, thr_sA[None, None, None, a_write_stage])
                fx.gpu.barrier()

            # Prologue: load K-tile 0 (A LDS slot 0, B reg slot 0) and prefetch
            # K-tile 1 into B reg slot 1 (B is 2-ahead, so slot 1 must be primed).
            fx.copy(buffer_copy_128b, thr_gA_k[None, None, None, 0], copy_frag_A)
            fx.copy(buffer_copy_128b, thr_gB_k[None, None, None, 0], mma_frag_B_retile[None, None, None, 0])
            mma_frag_C.fill(0)
            fx.copy(uni_copy_128b, copy_frag_A, thr_sA[None, None, None, 0])
            if fx.const_expr(K // BLOCK_K >= 2):
                fx.copy(
                    buffer_copy_128b,
                    thr_gB_k[None, None, None, 0],
                    mma_frag_B_retile[None, None, None, 1],
                    soffset=fx.Int32(1) * gB_k_stride,
                )
            fx.gpu.barrier()

            # Main K-loop. A toggles 0/1 (mod 2); B rotates 0/1/2 (mod 3).
            # At logical tile i: read B slot i%3, prefetch tile i+2 -> (i+2)%3.
            n_tiles = K // BLOCK_K
            for i in range(0, n_tiles - 1):
                a_rs = i % 2
                b_rs = i % 3
                # A prefetch: tile i+1 (1-ahead). B prefetch: tile i+2 (2-ahead),
                # only while i+2 is a valid tile.
                b_has_next = (i + 2) < n_tiles
                run_pipeline_stage(
                    a_read_stage=a_rs,
                    b_read_stage=b_rs,
                    a_next_k=i + 1,
                    b_next_k=i + 2,
                    read_next_a=True,
                    read_next_b=b_has_next,
                )
            # Final tile (i = n_tiles-1): no prefetch.
            last = n_tiles - 1
            run_pipeline_stage(
                a_read_stage=last % 2,
                b_read_stage=last % 3,
                a_next_k=None,
                b_next_k=None,
                read_next_a=False,
                read_next_b=False,
            )

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
            thr_sC_read = thr_copy_s2g_C.partition_S(sC)  # (V, VM, VN)
            thr_gC = thr_copy_s2g_C.partition_D(gC)  # (V, VM, VN) global
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
        # MFMA atom: CDNA4 (gfx950) uses the 16x16x32 bf16 atom (one 32-wide-K
        # atom covers tile_K_perm=32 -> K-permute (8,4),(1,8), flat fragment K).
        # gfx942 falls back to 16x16x16 (2 atoms per 32-wide perm -> (4,4,2)).
        if fx.const_expr(USE_MFMA_K32):
            tiled_mma = fx.make_tiled_mma(
                fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16)),
                fx.make_layout((1, 4, 1), (0, 1, 0)),
                fx.make_tile(None, None, fx.make_layout((8, 4), (1, 8))),
            )
        else:
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
    xcd_c: int | None = None,
    xcd_w: int | None = None,
    use_mfma_k32: bool | None = None,
    uniform_seqlen: bool = True,
):
    """Public entry. Derives N (output) and K (reduction) from the tall dense B
    matrix shape (B_groups * N, K), then dispatches to the per-shape kernel.

    xcd_c / xcd_w are the chiplet (XCD) remap knobs (C and W); they are baked
    into the compiled kernel, so each distinct (shape, xcd_c, xcd_w) memoizes a
    separate kernel. When left as None they default per (K, uniform_seqlen):

      uniform_seqlen=True: remap ON. K>256 uses C=XCD_C_LARGE_K, K<=256 uses
        C=XCD_C_SMALL_K; W=XCD_W. Clustering a group's M-tiles on one XCD's L2
        gives +~5-8% (the bigger the reduction K / weight, the bigger the win),
        because every M-tile is occupied so the chiplets stay balanced.
      uniform_seqlen=False (SKEW): remap OFF (identity C=1, W=1) for ALL K. Under
        skew the per-group M_i are wildly uneven (~20-30% empty groups), so
        clustering a group's M-tiles onto a single XCD OVERLOADS that chiplet
        while others idle -- a load-imbalance penalty that swamps the L2-reuse
        win. Measured on D=512 skew: remap-off is 1.4-1.6x FASTER than remap-on
        (the earlier "+2% on D512 varlen" note was wrong). Round-robin (identity)
        spreads the surviving tiles evenly across all 8 XCDs, which is what skew
        needs. (An explicit xcd_c still overrides this.)

    An explicit xcd_c / xcd_w always overrides the gate. use_mfma_k32 selects the
    CDNA4 16x16x32 bf16 atom; defaults to auto-detect (on gfx950, off gfx942)."""
    N = B.shape[0] // n_groups
    K = B.shape[1]
    # BLOCK_K=64 for ALL reduction K on gfx942. The smaller A double-buffer
    # (BLOCK_M*BLOCK_K*STAGES_A*2 = 32KB at K<=256, vs 64KB for BLOCK_K=128)
    # leaves headroom under gfx942's 64KB LDS ceiling, doubling occupancy: +11-19%
    # on the D256 shapes (uniform B120 1.11x, B1024 1.14x; skew B120 1.19x), cos=1.0.
    # (The old "BLOCK_K=128 wins ~4% on K<=256" was a gfx950 result; gfx950's larger
    # LDS hides the occupancy cost that dominates on gfx942's smaller LDS. BLOCK_K=256
    # remains unsafe: the 2-stage double-buffer epilogue mis-accumulates a single tile.)
    block_k = BLOCK_K
    # XCD remap defaults (see docstring). Remap ON only for uniform seqlens, where
    # every M-tile is occupied so chiplets stay balanced and the Dense[b] L2-reuse
    # win lands. Under SKEW the remap clusters a group's M-tiles onto one XCD and
    # overloads it while others idle (load imbalance >> L2 reuse), so fall back to
    # identity (C=1, W=1) round-robin for ALL K. Measured: remap-off is 1.4-1.6x
    # faster than remap-on on D=512 skew.
    if xcd_c is None:
        if not uniform_seqlen:
            xcd_c = 1
        elif K > 256:
            xcd_c = XCD_C_LARGE_K
        else:
            xcd_c = XCD_C_SMALL_K
    if xcd_w is None:
        xcd_w = 1 if xcd_c == 1 else XCD_W
    if use_mfma_k32 is None:
        use_mfma_k32 = _use_mfma_k32()
    bm = (max_seq_len + BLOCK_M - 1) // BLOCK_M
    launch = _build_launcher(
        N, K, BLOCK_M, BLOCK_N, block_k, STAGES_A, THREADS, bm, n_groups, xcd_c, xcd_w, use_mfma_k32
    )
    return launch(C, A, B, BIAS, SEQ_OFFSETS, n_groups, max_seq_len, stream=stream)
