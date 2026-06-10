# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# jagged_dense_bmm_broadcast_add (jdbba) — PERSISTENT problem-visitor variant,
# ON-DEVICE tile mapping (Design B-1, NO host work-list build).
#
# Computes, for each group b over its packed row slice [s, e):
#     Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
#       (M_b x N)     (M_b x K)      (K x N)     (1 x N broadcast)
#
# === What is different from jagged_dense_bmm_persist.py (Design A) ===
# Design A (jagged_dense_bmm_persist.py) built a flat int32 work-list
# WORK_ITEMS[num_tiles, 3] on the HOST via seq_offsets.cpu() + a Python loop.
# That host round-trip is a hard sync that erases the kernel-only win end-to-end.
#
# This variant (Design B-1) is the true CUTLASS grouped-GEMM problem visitor:
# NOTHING is materialized on the host. Instead a tiny per-group cumulative
# tile-count prefix array CUM[n_groups+1] is built ON DEVICE with a pure-GPU
# torch expression (no .cpu()):
#     mb    = seq_offsets[1:] - seq_offsets[:-1]              # M_b per group
#     tiles = ((mb + BLOCK_M-1)//BLOCK_M).clamp(min=0) * N_BLOCKS
#     CUM   = [0, *tiles.cumsum(0)]                           # int32, on GPU
# so CUM[b+1]-CUM[b] = occupied tiles of group b (empty groups contribute 0) and
# CUM[n_groups] = total_tiles. This stays entirely on the GPU stream — ZERO host
# sync — and is trivially cheap.
#
# The persistent kernel takes CUM (device tensor) instead of WORK_ITEMS. It
# launches a FIXED PERSIST_BLOCKS grid; each block reads total_tiles =
# CUM[n_groups] into a register (buffer_load + readfirstlane) and loops
#     for t in (blk_id, blk_id+grid_dim, ...) while t < total_tiles:
# mapping each linear tile id t -> (off_b, block_m_idx, block_n_idx) by an
# IN-KERNEL SEARCH over CUM: find group b with CUM[b] <= t < CUM[b+1] (a
# compile-time-unrolled linear scan over the known n_groups, accumulating with
# select so it stays uniform/branchless), then local = t - CUM[b],
# block_m_idx = local // N_BLOCKS, block_n_idx = local % N_BLOCKS. Empty groups
# have CUM[b]==CUM[b+1] (zero-width interval) so the `CUM[b] <= t < CUM[b+1]`
# test is never true for them — they are skipped automatically. Linear t is
# group-major over CUM, so the GROUP-MAJOR L2-locality ordering is preserved for
# free (no WORK_ITEMS list needed). The per-tile compute body is IDENTICAL to
# Design A; only the source of (off_b, block_m_idx, block_n_idx) changed.
#
# Dense is host pre-transposed to (B_groups * N, K) and consumed plain.
# bf16 in/out, fp32 accumulate. Carries the prototype's seq_start/seq_end
# scalarization (uniform C descriptor -> no epilogue store waterfall).

from __future__ import annotations

import functools

import torch

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


@functools.lru_cache(maxsize=None)
def _build_launcher(N, K, BLOCK_M, BLOCK_N, BLOCK_K, STAGES_A, THREADS, N_GROUPS, XCD_C):
    """Build (and memoize) a launch wrapper specialized to this (N, K, tiling).
    N (output) and K (reduction) are baked in as closure constants.

    XCD-AWARE persistent visiting order (the varlen-skew lever). The baseline
    persistent loop has block ``blk_id`` claim linear tile ids
    ``blk_id, blk_id+grid, blk_id+2*grid, ...``. The grid is a multiple of NXCD
    (8), and the hardware routes block ``blk_id`` to XCD ``blk_id % NXCD``, so a
    block's XCD == ``wi % NXCD`` for EVERY id it claims. But linear ids are
    group-major (a group's tiles are a contiguous run sharing ``Dense[b]``), so
    consecutive ids go to consecutive blocks == consecutive XCDs -> every
    ``Dense[b]`` is pulled redundantly into all 8 XCD L2 slices (the same waste the
    static kernel's XCD remap fixes for uniform). Fix: apply HipKittens phase-1
    chunking to the VISITING index ``v`` so a run of ``XCD_C`` consecutive
    group-major tiles co-locates on one XCD's L2. XCD_C<=1 disables it (identity,
    == baseline persist_dev)."""
    N_BLOCKS = N // BLOCK_N  # output column-tiles per group (compile-time)
    # Per-thread C accumulator length (fp32 lanes) = tile elems / threads.
    C_FRAG_LEN = BLOCK_M * BLOCK_N // THREADS
    smem_bytes = BLOCK_M * BLOCK_K * STAGES_A * 2  # bf16 A double-buffer staging

    from flydsl.utils.smem_allocator import SmemPtr

    @flyc.kernel
    def jdbba_kernel(
        C: fx.Tensor,  # out    (L, N)   bf16
        A: fx.Tensor,  # jagged (L, K)   bf16
        B: fx.Tensor,  # dense  (B_groups * N, K) bf16 (pre-transposed, tall)
        BIAS: fx.Tensor,  # bias   (B_groups * N,)   bf16
        SEQ_OFFSETS: fx.Tensor,  # (B_groups + 1,) int32
        CUM: fx.Tensor,  # (B_groups + 1,) int32: cumulative occupied-tile counts
    ):
        tid = fx.thread_idx.x
        blk_id = fx.block_idx.x
        n_blocks_grid = fx.grid_dim.x

        # Buffer resources reused every persistent iteration.
        cum_rsrc = fx.buffer_ops.create_buffer_resource(CUM, max_size=True)
        seq_rsrc = fx.buffer_ops.create_buffer_resource(SEQ_OFFSETS, max_size=True)

        # total_tiles = CUM[n_groups] (occupied tiles across all groups). Read it
        # on-device into a uniform register so the persistent loop bound needs no
        # host scalar. ZERO host sync.
        total_tiles = fx.buffer_ops.buffer_load(
            cum_rsrc, fx.Int32(N_GROUPS), vec_width=1, dtype=fx.T.i32()
        )
        total_tiles = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), total_tiles))

        # XCD chunking: runtime prefix = largest [0,prefix) divisible by NXCD*C.
        # The remap (below) is a bijection on [0,prefix); the (<period) tail is
        # identity-mapped so the whole permutation still covers [0,total_tiles).
        # period / prefix are RUNTIME here (total_tiles is device-read), unlike the
        # static kernel where they are compile-time.
        if fx.const_expr(XCD_C > 1):
            c_period = fx.Int32(NXCD * XCD_C)
            xcd_prefix = fx.Int32(total_tiles - (total_tiles % c_period))
            xcd_prefix = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), xcd_prefix))

        # --- PERSISTENT problem-visitor loop ---
        # Each block strides the VISITING index v over [0, total_tiles); v % NXCD
        # == blk_id % NXCD == this block's physical XCD (grid is a multiple of
        # NXCD). The actual tile id wi = remap(v) applies HipKittens phase-1
        # chunking so a run of XCD_C consecutive group-major tile ids (sharing
        # Dense[b]) co-locates on one XCD's L2 instead of scattering across all 8.
        for v in range(fx.Int32(blk_id), fx.Int32(total_tiles), fx.Int32(n_blocks_grid)):
            # Recreate shared-memory views each iteration -> reset the view cache
            # so the second+ iteration's fx.get_dyn_shared views don't reuse the
            # previous iteration's SSA values (MLIR dominance).
            SmemPtr._view_cache = None

            # v -> wi (tile id). xcd = v % NXCD == physical XCD; the remap places
            # a chunk of XCD_C consecutive tile ids onto that one XCD.
            if fx.const_expr(XCD_C > 1):
                v_i = fx.Int32(v)
                cnXCD = fx.Int32(NXCD)
                cC = fx.Int32(XCD_C)
                xcd = v_i % cnXCD
                local = v_i // cnXCD
                chunk_idx = local // cC
                pos = local % cC
                wi_remap = chunk_idx * (cnXCD * cC) + xcd * cC + pos
                wi = (v_i < xcd_prefix).select(wi_remap, v_i)
                wi = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), wi))
            else:
                wi = fx.Int32(v)

            # Construct the tiled MMA / copies INSIDE the loop body. If built
            # before the loop and merely read inside, the DSL auto-promotes them
            # to scf.for iter_args (block args), and then the gemm-op lowering's
            # getDefiningOp<MakeTiledMma> fails (mma_atom_call keeps the whole
            # tiled_mma as operand 0). Building them in-region keeps make_tiled_mma
            # as the gemm's directly-visible defining op. They are loop-invariant,
            # so LICM-style passes can still hoist the cheap setup later.
            tiled_mma = fx.make_tiled_mma(
                fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
                fx.make_layout((1, 4, 1), (0, 1, 0)),
                fx.make_tile(None, None, fx.make_layout((4, 4, 2), (1, 8, 4))),
            )
            _val_per_thr = 8  # 16B / bf16
            _thrs_col = BLOCK_K // _val_per_thr
            _thrs_row = THREADS // _thrs_col
            tiled_copy_g2s_A = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16),
                fx.make_layout(
                    ((_thrs_col, _thrs_row), (1, _val_per_thr)),
                    ((_thrs_row * _val_per_thr, 1), (1, _thrs_row)),
                ),
                fx.make_tile(_thrs_row, BLOCK_K),
            )
            _c_val = 8  # 16B / bf16
            _c_thrs_n = BLOCK_N // _c_val
            _c_thrs_m = THREADS // _c_thrs_n
            tiled_copy_s2g_C = fx.make_tiled_copy(
                fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16),
                fx.make_layout(
                    ((_c_thrs_n, _c_thrs_m), (1, _c_val)),
                    ((_c_thrs_m * _c_val, 1), (1, _c_thrs_m)),
                ),
                fx.make_tile(_c_thrs_m, BLOCK_N),
            )

            # --- In-kernel CUM search: linear tile id `wi` -> (off_b, local) ---
            # Find the group off_b = largest b in [0, N_GROUPS) with CUM[b] <= wi
            # (equivalently CUM[off_b] <= wi < CUM[off_b+1]). A compile-time-
            # unrolled BINARY SEARCH over the known index range: log2(N_GROUPS)
            # buffer_loads per tile instead of O(N_GROUPS) — critical because this
            # runs every persistent iteration (a linear scan over 1024 groups was
            # the kernel-only bottleneck). Each step probes CUM[mid+1]: if
            # wi >= CUM[mid+1] the answer is > mid, so lo := mid+1, else hi := mid.
            # All values are uniform (CUM loaded then readfirstlane'd) so off_b
            # stays in an SGPR. Empty groups have CUM[b]==CUM[b+1] (zero-width); wi
            # can never satisfy CUM[b] <= wi < CUM[b+1] for them so the search
            # always lands on an occupied group — they are skipped automatically.
            wi_i32 = fx.Int32(wi)
            _bits = max(1, (N_GROUPS - 1).bit_length())  # ceil(log2(N_GROUPS))
            lo = fx.Int32(0)
            hi = fx.Int32(N_GROUPS - 1)  # search over group indices [0, N_GROUPS)
            for _step in fx.range_constexpr(_bits):
                mid = fx.Int32((lo + hi) >> fx.Int32(1))
                cum_mid1 = fx.buffer_ops.buffer_load(
                    cum_rsrc, fx.Int32(mid) + fx.Int32(1), vec_width=1, dtype=fx.T.i32()
                )
                cum_mid1 = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), cum_mid1))
                go_right = wi_i32 >= cum_mid1  # answer is in (mid, hi]
                lo = go_right.select(mid + fx.Int32(1), lo)
                hi = go_right.select(hi, mid)
            off_b = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), lo))  # lo == hi == owning group
            cum_at_off = fx.buffer_ops.buffer_load(
                cum_rsrc, fx.Int32(off_b), vec_width=1, dtype=fx.T.i32()
            )
            cum_at_off = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), cum_at_off))
            # local = wi - CUM[off_b]; m-tile is outer, n-tile inner (group-major).
            local = wi_i32 - cum_at_off
            block_m_idx = fx.Int32(local) // fx.Int32(N_BLOCKS)
            block_n_idx = fx.Int32(local) % fx.Int32(N_BLOCKS)
            block_m_idx = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), block_m_idx))
            block_n_idx = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), block_n_idx))

            # --- Device group resolution (read seq_offsets[b], seq_offsets[b+1]) ---
            seq_start = fx.buffer_ops.buffer_load(seq_rsrc, fx.Int32(off_b), vec_width=1, dtype=fx.T.i32())
            seq_end = fx.buffer_ops.buffer_load(seq_rsrc, fx.Int32(off_b) + fx.Int32(1), vec_width=1, dtype=fx.T.i32())
            # Scalarize: keep M_b and all derived offsets / the C descriptor uniform
            # (SGPR), else the divergent C descriptor forces an epilogue store waterfall.
            seq_start = fx.rocdl.readfirstlane(fx.T.i32(), seq_start)
            seq_end = fx.rocdl.readfirstlane(fx.T.i32(), seq_end)
            M_b = seq_end - seq_start
            start_m = fx.Int32(block_m_idx) * fx.Int32(BLOCK_M)

            # Work list only contains valid tiles (start_m < M_b); guard kept for
            # safety and to reuse the _gen body verbatim.
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
        CUM: fx.Tensor,
        persist_blocks: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        # smem holds either A double-buffer staging or the row-major C tile.
        epi_smem_bytes = max(smem_bytes, BLOCK_M * BLOCK_N * 2)

        # Persistent grid: 1D, FIXED persist_blocks. No host scalar needed; each
        # block reads total_tiles = CUM[n_groups] on-device and self-limits its
        # loop. The tiled MMA / copies are built inside the kernel (so the gemm op
        # inside the scf.for lowers correctly).
        jdbba_kernel(
            C, A, B, BIAS, SEQ_OFFSETS, CUM,
        ).launch(
            grid=(persist_blocks, 1, 1), block=(THREADS, 1, 1), smem=epi_smem_bytes, stream=stream
        )

    return _launch


# Default persistent grid size. MI355X / CDNA4 has ~256 CUs; with this kernel's
# LDS / register footprint roughly 2 blocks co-reside per CU, so ~512 persistent
# blocks saturate the machine. Unlike Design A we launch a FIXED grid (no host
# scalar to cap it); blocks with t >= total_tiles simply never enter the loop.
PERSIST_BLOCKS = 512


def build_cum(SEQ_OFFSETS, n_groups, N, block_m=BLOCK_M, block_n=BLOCK_N):
    """On-DEVICE cumulative occupied-tile-count prefix array (Design B-1).

    CUM[b+1]-CUM[b] = ceil(M_b/BLOCK_M) * N_BLOCKS = occupied tiles of group b
    (empty groups contribute 0). CUM[n_groups] = total_tiles. Built with a pure
    torch expression on SEQ_OFFSETS' device — NO .cpu()/.item(), so it stays on
    the GPU stream with ZERO host sync. Returns an int32 tensor (n_groups+1,) on
    the same device.

    Group-major ordering is implicit: linear tile id t in [0, total_tiles) maps
    to group b = (CUM[b] <= t < CUM[b+1]) then local = t - CUM[b], so all of a
    group's tiles are a contiguous run of t and within it m-tiles are outer,
    n-tiles inner (local // N_BLOCKS, local % N_BLOCKS)."""
    n_blocks = N // block_n
    so = SEQ_OFFSETS.to(torch.int64)
    mb = so[1:] - so[:-1]  # M_b per group, on GPU
    tiles = ((mb + (block_m - 1)) // block_m).clamp(min=0) * n_blocks
    cum = torch.zeros(n_groups + 1, dtype=torch.int32, device=SEQ_OFFSETS.device)
    cum[1:] = tiles.cumsum(0).to(torch.int32)
    return cum


def jagged_dense_bmm(
    C,
    A,
    B,
    BIAS,
    SEQ_OFFSETS,
    n_groups: int,
    max_seq_len: int,
    stream: fx.Stream = fx.Stream(None),
    persist_blocks: int = PERSIST_BLOCKS,
    cum=None,
    xcd_c: int = None,
):
    """Public entry (PERSISTENT problem-visitor, Design B-1, ON-DEVICE mapping).
    Derives N (output) and K (reduction) from the tall dense B (B_groups * N, K).

    No host work-list build, NO seq_offsets.cpu(): the per-group cumulative
    tile-count array CUM is built on the GPU with build_cum() (a torch
    expression) and passed straight to the kernel, which reads total_tiles =
    CUM[n_groups] on-device and maps each persistent tile id via an in-kernel CUM
    search. Steady-state path has ZERO host syncs.

    Callers may pass a pre-built `cum` (e.g. a bench harness timing the
    kernel-only path) to skip the build, but the build is GPU-only and cheap so
    the default path already avoids any host round-trip."""
    N = B.shape[0] // n_groups
    K = B.shape[1]
    # Shape-dependent BLOCK_K (same rationale as _gen): K<=256 -> BLOCK_K=128.
    block_k = 128 if K <= 256 else BLOCK_K

    if cum is None:
        # --- ON-DEVICE prefix build (pure torch, no host sync) ---
        cum = build_cum(SEQ_OFFSETS, n_groups, N)

    if xcd_c is None:
        xcd_c = XCD_C_LARGE_K if K > 256 else XCD_C_SMALL_K

    tCUM = flyc.from_dlpack(cum)
    launch = _build_launcher(N, K, BLOCK_M, BLOCK_N, block_k, STAGES_A, THREADS, n_groups, xcd_c)
    return launch(C, A, B, BIAS, SEQ_OFFSETS, tCUM, persist_blocks, stream=stream)
