# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
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
#
# SHAPE GENERALIZATION: the square dense dim D (= K = N) and every
# D-derived tiling constant are baked in as closure constants by a memoized
# factory `build_backward(D, split, gj_stages_a, coarsen_m)` (mirrors the forward).
# Each distinct (D, knobs) memoizes its own compiled kernels, so ONE process can serve MULTIPLE D -- no module-global
# snapshot, no single-D-per-process constraint. The legacy module-level
# `configure_dim` / `grad_jagged` / `grad_dense_bias` remain as a thin single-D
# facade over the factory (for the profiler / example timing path); the production
# dispatch wrapper calls `build_backward` directly and is multi-D.

from __future__ import annotations

import collections
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.derived import make_layout_tv
from flydsl.expr.typing import full

from aiter.jit.utils.chip_info import (
    get_gfx_runtime,
    get_lds_capacity_bytes,
    get_num_xcds,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_scalar_load

# Forward tile/shape constants. Imported dual-path so this module works both as
# an aiter package submodule (production: ``aiter.ops.flydsl.kernels``) and as a
# bare sibling. The gen forward no longer exports module-level BLOCK_K / THREADS
# (those live on the dispatch defaults). grad_jagged keeps the historical 64 / 256
# tiles. Bounded views use fx.rocdl.make_buffer_tensor(..., num_records_bytes=).
try:
    from . import jagged_dense_bmm_gen as _fwd  # package-relative (aiter import path)
except ImportError:  # bare sibling import (kernels dir on sys.path[0])
    import jagged_dense_bmm_gen as _fwd

BLOCK_K = 64
BLOCK_M = _fwd.BLOCK_M
BLOCK_N = _fwd.BLOCK_N
THREADS = 256

# --- D-INDEPENDENT constants (module-level; the kernels snapshot these as fixed
# values regardless of D). ---

# dDense partials tiling. The contraction dDense[k,n] = sum_m J[m,k]*dOut[m,n] is
# a transposed GEMM C[k,n] = sum_m A[k,m]*B[n,m] with A = J.T and B = dOut.T, i.e.
# the reduction axis m is the *contiguous* fragment axis of both operands. Each
# workgroup (256 threads = 4 waves) owns one (DDENSE_BK x DDENSE_BN) = 128x128
# output sub-tile of the group's (K, N) block and runs the MFMA 16x16x16 bf16 pipe
# (same tiled_mma as the forward kernel). The reduction is tiled DDENSE_BM rows of
# m at a time: each m-tile is staged transposed into LDS (J -> sJ as (k, m), dOut
# -> sD as (n, m), so m ends up contiguous and feeds the MFMA K-fragment), then a
# bf16 MFMA accumulates into one fp32 accumulator fragment carried across the whole
# (dynamic, split-strided) m-tile loop. Output-tiling keeps one stage at 32 KB and
# the accumulator at one MFMA C-fragment per thread regardless of D = K = N.
# gfx950 D=256, D=384 and D=512 instead stage 128 m per step (64 KB), keep both tiles out_dim-
# contiguous and feeds a 32x32x16 MFMA through ds_read_tr16; see
# ``_dense_gfx950_128_tile_for``.
DDENSE_BM = 64  # m-contraction tile staged per step (the MFMA K dimension)
DDENSE_BK = 128  # output K-tile per workgroup (MFMA M dimension)
DDENSE_BN = 128  # output N-tile per workgroup (MFMA N dimension)
DDENSE_THREADS = 256
_DDENSE_SMEM_BYTES = (DDENSE_BK * DDENSE_BM + DDENSE_BN * DDENSE_BM) * 2  # bf16 staging

# dOut global->LDS staging map. Each thread loads _D_VEC contiguous n (one 128-bit
# buffer load) from _D_ROW_PASSES rows of the (DDENSE_BM, DDENSE_BN) m-tile, then
# scatters them into the (n, m) sD tile. A wave covers _D_LANE_CHUNKS n-chunks x
# _D_LANE_ROWS rows: fewer, wider chunks per row keep the transposed b16 LDS stores
# spread across banks (sD rows are 128 B apart), while each row still reads 64 B
# contiguous. Thread tid owns n-chunk
#   c = (tid // 64) * _D_LANE_CHUNKS + tid % _D_LANE_CHUNKS
# and rows m = pass * _D_LANE_ROWS + (tid % 64) // _D_LANE_CHUNKS.
_D_VEC = 8  # 16 B / bf16
_D_CHUNKS = DDENSE_BN // _D_VEC  # 16 n-chunks per row
_D_LANE_CHUNKS = 4
_D_LANE_ROWS = 64 // _D_LANE_CHUNKS  # 16
_D_ROW_PASSES = DDENSE_BM // _D_LANE_ROWS  # 4
assert _D_CHUNKS == (DDENSE_THREADS // 64) * _D_LANE_CHUNKS, "dOut staging map"
assert _D_ROW_PASSES * _D_LANE_ROWS == DDENSE_BM, "dOut staging map"

# dBias[b][n] = sum_m dOut[m,n] reduces over the same dynamic m axis as dDense, and
# the dDense partials kernel already streams every dOut element through LDS, so the
# bias column-sums are folded into that pass instead of re-reading dOut in a separate
# kernel. Under the staging map above, thread tid always owns the _D_VEC columns of
# n-chunk c, so each column is co-owned by the _D_LANE_ROWS threads that share c.
# Each thread carries _D_VEC fp32 running sums and the owners are combined once
# through LDS at the end. Only the k-tile-0 workgroups (k_off == 0) emit a column's
# bias partial, so every N-tile is written exactly once regardless of the K-tiling.

# Default schedule knobs (baked into a build unless the caller overrides). See
# build_backward for how they specialize per shape.
#  * COARSEN_M: grad_jagged M-coarsening. At large deployment shapes (B=1024,
#    Mi=7680) the per-row dJagged GEMM launches a huge grid of *tiny* workgroups
#    (each WG does only NRED_TILES=4 MFMA K-steps) and is dispatch/latency-bound.
#    Coarsening makes each WG process COARSEN_M consecutive BLOCK_M output tiles
#    (grid M-dim shrinks by COARSEN_M), so WGs live longer and fewer are
#    dispatched. COARSEN_M=1 is the un-coarsened grid (one BLOCK_M tile per WG).
#  * GJ_STAGES_A: grad_jagged A-staging depth. The double-buffered sA is
#    BLOCK_M*BLOCK_K*GJ_STAGES_A bf16 = 32 KB at 2 stages, which caps grad_jagged at
#    <=2 WG/CU (LDS-bound). =1 single-buffers it -> 16 KB -> up to 4 WG/CU, at the
#    cost of one extra barrier/iter. The production path resolves it per-D from the
#    dispatch JSON (1 @ D<=256, 2 @ D>256).
COARSEN_M = 2
GJ_STAGES_A = 1

_DEFAULT_STREAM = fx.Stream(None)


def _dense_gfx950_128_tile_for(D, arch):
    """Whether the gfx950 D=256/384/512 dense/bias kernel uses the CDNA4 tile."""
    return arch == "gfx950" and int(D) in (256, 384, 512)


def _split_for(D):
    """Default SPLIT over the jagged (m) axis for the dDense / dBias reductions.

    Each (group, split) block owns one fp32 scratch slot; the reduce pass sums them.
    SPLIT trades partials-pass m-parallelism / long-group load balance (higher)
    against the fp32 partials round-trip + wasted blocks on short/empty groups
    (lower). The partials base grid is NK_TILES*NN_TILES*n_groups = (D/128)^2 *
    n_groups, growing with D^2, so the best SPLIT shrinks as D grows: D=256 is
    fastest at 2, D=512 at 1 (both uniform + skew).
    """
    return 2 if D <= 256 else 1


def _load_scalar(copy_atom, elem_dtype, divided_tensor, index):
    """Load one element at column `index` from a row already divided by (1, 1)."""
    view = fx.slice(divided_tensor, (None, index))
    r = fx.make_rmem_tensor(1, elem_dtype)
    fx.copy(copy_atom, view, r)
    return fx.memref_load_vec(r)[0]


def _store_scalar(copy_atom, store_dtype, divided_tensor, index, val):
    """Store scalar `val` to column `index` of a row already divided by (1, 1)."""
    r = fx.make_rmem_tensor(1, store_dtype)
    ts = full(1, store_dtype(val), store_dtype)
    fx.memref_store_vec(ts, r)
    view = fx.slice(divided_tensor, (None, index))
    fx.copy(copy_atom, r, view)


# A built backward: the two host launchers + the resolved SPLIT for this shape.
Backward = collections.namedtuple(
    "Backward", ["grad_jagged", "grad_dense_bias", "split", "D"]
)


# maxsize=None (unbounded) is intentional: the key space is the handful of
# distinct (D, split, gj_stages_a, coarsen_m, num_xcds, arch) tuples a process
# actually launches (typically 1-2 D x a couple of knob sets), and each entry
# is a set of compiled kernels we WANT to keep resident for the process
# lifetime. No eviction needed.
def build_backward(D, split=None, gj_stages_a=None, coarsen_m=None, num_xcds=None):
    """Build (and memoize) the backward launchers specialized to this (D, knobs).

    D (= K = N) and every D-derived tiling constant are baked in as closure
    constants, so FlyDSL memoizes one compiled kernel set per (D, split,
    gj_stages_a, coarsen_m, num_xcds, arch) and MULTIPLE can coexist in one process
    (mirrors the forward gen kernel's factory). Returns a ``Backward`` namedtuple
    of the two host launchers plus the resolved SPLIT.

    ``split`` / ``gj_stages_a`` / ``coarsen_m`` default (None) to the D-derived
    SPLIT and the module knob defaults; the dispatch layer overrides them per shape.
    ``num_xcds`` defaults to the current device's XCD count and is part of the
    memo key, so a 4-XCD part does not reuse an 8-XCD schedule. Because they are
    closure constants (not module globals), the FlyDSL artifact cache keys on
    them correctly -- no cache-collision hazard.
    """
    if num_xcds is None:
        num_xcds = get_num_xcds()
    return _build_backward(
        D, split, gj_stages_a, coarsen_m, int(num_xcds), get_gfx_runtime()
    )


@functools.cache
def _build_backward(D, split, gj_stages_a, coarsen_m, num_xcds, arch):
    D = int(D)
    num_xcds = int(num_xcds)
    arch = str(arch)
    if num_xcds < 1:
        raise ValueError(f"num_xcds must be >= 1, got {num_xcds}")
    if D % DDENSE_BK or D % DDENSE_BN or D % BLOCK_N or D % BLOCK_K:
        raise ValueError(
            f"D={D} must be divisible by the tile sizes "
            f"(DDENSE_BK={DDENSE_BK}, DDENSE_BN={DDENSE_BN}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K})"
        )
    # D-derived compile-time constants (closure locals baked into the kernels).
    K = N = D
    SPLIT = _split_for(D) if split is None else int(split)
    if SPLIT < 1:
        raise ValueError(f"split must be >= 1, got {SPLIT}")
    NRED_BLK = min(N, 256)
    # NRED_COL_TILES > 1 when N > NRED_BLK; the last tile may be partial (N not a
    # multiple of NRED_BLK, e.g. a forced split=2 at D=384). The reduce kernels
    # guard col < N, so a partial last tile is correct -- no divisibility gate is
    # imposed here (rejecting it would forbid an otherwise-valid forced split).
    NRED_COL_TILES = (N + NRED_BLK - 1) // NRED_BLK
    # CDNA4 follows Triton's 128x256x32 dJagged tile.  The existing wave layout
    # supplies its 128 output columns through four 32x32 MFMA waves; the extra
    # output repeat is represented by the MMA K-mode tile layout below.
    gj_gfx950_tile = arch == "gfx950" and D in (256, 512)
    GJ_BLOCK_N = 256 if gj_gfx950_tile else BLOCK_N
    GJ_BLOCK_K = 32 if gj_gfx950_tile else BLOCK_K
    KOUT_BLOCKS = K // GJ_BLOCK_N  # column-tiles of the (M, K) dJagged output
    NRED_TILES = N // GJ_BLOCK_K  # contraction tiles over N
    NK_TILES = K // DDENSE_BK  # dDense output K-tiles per group
    NN_TILES = N // DDENSE_BN  # dDense output N-tiles per group
    COARSEN_M = COARSEN_M_default if coarsen_m is None else int(coarsen_m)
    GJ_STAGES_A = GJ_STAGES_A_default if gj_stages_a is None else int(gj_stages_a)
    GJ_STAGES = 2 if gj_gfx950_tile else GJ_STAGES_A
    dense_gfx950_128_tile = _dense_gfx950_128_tile_for(D, arch)
    gj_mfma32 = arch == "gfx950"
    dense_bm = 128 if dense_gfx950_128_tile else DDENSE_BM
    dense_row_passes = dense_bm // _D_LANE_ROWS
    dense_smem_bytes = (DDENSE_BK * dense_bm + DDENSE_BN * dense_bm) * 2
    if dense_row_passes * _D_LANE_ROWS != dense_bm:
        raise ValueError(f"dense m-tile {dense_bm} does not match the staging map")
    if dense_smem_bytes > get_lds_capacity_bytes(arch):
        raise ValueError(
            f"{arch} LDS is {get_lds_capacity_bytes(arch)} bytes; dense/bias "
            f"staging for D={D} needs {dense_smem_bytes}"
        )

    @flyc.kernel
    def grad_jagged_kernel(
        C: fx.Tensor,  # out    dJagged (L, K)          bf16
        A: fx.Tensor,  # grad   dOut    (L, N)          bf16
        B: fx.Tensor,  # dense  (n_groups * K, N)       bf16  (plain, K-major per group)
        SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s_A: fx.TiledCopy,
        n_groups: fx.Int32,
        tiles_per_group: fx.Int32,
    ):
        # dJagged[m,k] = sum_n dOut[m,n] * Dense[b][k,n]. In MFMA C[i,j]=sum_l A[i,l]
        # B[j,l] form: i=m (rows), j=k (output column), l=n (contraction). So A is the
        # dOut group view (M_b, N) and B is Dense[b] read in its plain (K, N) layout.
        tid = fx.thread_idx.x
        # Same XCD stripe as grad_dense_partials: column tiles of one row-group reread
        # the same dOut rows, and every row-group of one group rereads Dense[b].
        # linear = (g % num_xcds) + num_xcds * (tile + (g // num_xcds) * T).
        linear = fx.block_idx.x
        xcd = linear % fx.Int32(num_xcds)
        slot = linear // fx.Int32(num_xcds)
        pid_mn = slot % tiles_per_group
        off_b = xcd + (slot // tiles_per_group) * fx.Int32(num_xcds)
        in_range = off_b < n_groups
        off_b = in_range.select(off_b, fx.Int32(0))

        # One workgroup owns COARSEN_M consecutive BLOCK_M output row-tiles (all sharing
        # the same K-column tile / Dense slice). group_mn indexes the coarsened m-group;
        # the per-tile work below runs once per sub-tile. Group resolution is recomputed
        # per sub-tile (cheap scalar/SGPR work) to keep every value the scf.if branch
        # touches defined inside the branch (the rewriter downcasts hoisted copy slices).
        group_mn = pid_mn // KOUT_BLOCKS
        block_n_idx = pid_mn % KOUT_BLOCKS

        for m_sub in fx.range_constexpr(COARSEN_M):
            block_m_idx = group_mn * fx.Int32(COARSEN_M) + fx.Int32(m_sub)

            # off_b comes from the block id, so both offsets are wave-uniform and
            # land in SGPRs. A tail id past n_groups clamps M_b to 0, and the
            # start_m < M_b check below then stores nothing.
            seq_buf = fx.rocdl.make_buffer_tensor(SEQ_OFFSETS, max_size=True)
            seq_start = fx.Int32(buf_scalar_load(seq_buf, off_b))
            seq_end = fx.Int32(buf_scalar_load(seq_buf, off_b + fx.Int32(1)))
            M_b = seq_end - seq_start
            M_b = in_range.select(M_b, fx.Int32(0))
            start_m = block_m_idx * fx.Int32(BLOCK_M)

            # Runtime early-exit: tail tile fell off the end of a short group.
            if start_m < M_b:
                # Rebase A (dOut, N cols/row) and C (dJagged, K cols/row) to this group's
                # local row 0; select B's (K, N) slice for group off_b.
                # int64 element offsets: at the largest deployment shape (B=1024, Mi=7680)
                # seq_start reaches ~L ? 7.86M, so seq_start*K (or *N) ? 4G overflows
                # int32 once K/N >= 512, silently wrapping the (d)Jagged/dOut base pointer
                # (the masked store then writes to a wrong/read-only page). seq_start*K
                # at K=256 squeaks under 2^31, which is why 256 shapes never tripped it.
                # Mirrors the int64 base_byte_offset in grad_dense_partials_kernel.
                a_row_off = fx.Int64(seq_start) * fx.Int64(N)
                c_row_off = fx.Int64(seq_start) * fx.Int64(K)
                A_g = fx.make_view(
                    fx.add_offset(fx.get_iter(A), fx.make_int_tuple(a_row_off)),
                    fx.get_layout(A),
                )
                C_g = fx.make_view(
                    fx.add_offset(fx.get_iter(C), fx.make_int_tuple(c_row_off)),
                    fx.get_layout(C),
                )
                # int64 element offset (same as a_row_off/c_row_off): off_b*K*N scales
                # with the group count and dense tile size, not with L; the product can
                # exceed int32, so widen before multiplying.
                b_row_off = fx.Int64(off_b) * fx.Int64(K) * fx.Int64(N)
                B_g = fx.make_view(
                    fx.add_offset(fx.get_iter(B), fx.make_int_tuple(b_row_off)),
                    fx.get_layout(B),
                )

                # Bound A (dOut) reads to exactly M_b rows: a partial last row-tile spans
                # rows [M_b, ceil(M_b/BLOCK_M)*BLOCK_M), which for the LAST group runs past
                # the dOut allocation. An unbounded (max_size) descriptor lets that over-read
                # touch unmapped memory -> a data/size-dependent GPU fault. Bounding makes the
                # HW return 0 for the tail rows instead (they are masked out on the C store
                # anyway, so the result is unchanged). num_records is int32-safe (per-group
                # M_b*N*2 <= Mi*N*2). Mirrors grad_dense_partials_kernel's bounded reads.
                A_buf = fx.rocdl.make_buffer_tensor(
                    A_g,
                    num_records_bytes=fx.Int64(M_b * fx.Int32(N * 2)),
                )
                B_buf = fx.rocdl.make_buffer_tensor(B_g, max_size=True)
                # Bound C to exactly M_b rows (K cols, bf16=2B) so partial tail-tile
                # stores are HW-dropped instead of corrupting the next group's rows.
                C_buf = fx.rocdl.make_buffer_tensor(
                    C_g,
                    num_records_bytes=fx.Int64(M_b * fx.Int32(K * 2)),
                )

                gA_k = fx.flat_divide(A_buf, (BLOCK_M, GJ_BLOCK_K))[
                    None, None, block_m_idx, None
                ]  # (BM, BK, n)
                gB_k = fx.flat_divide(B_buf, (GJ_BLOCK_N, GJ_BLOCK_K))[
                    None, None, block_n_idx, None
                ]  # (BN, BK, n)
                gC = fx.flat_divide(C_buf, (BLOCK_M, GJ_BLOCK_N))[
                    None, None, block_m_idx, block_n_idx
                ]  # (BM, BN)

                thr_mma = tiled_mma.thr_slice(tid)
                thr_copy_g2s_A = tiled_copy_g2s_A.get_slice(tid)

                uni_copy_128b = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
                buffer_copy_128b = fx.make_copy_atom(
                    fx.rocdl.BufferCopy128b(), fx.BFloat16
                )

                thr_copy_s2r_A = fx.make_tiled_copy_A(
                    buffer_copy_128b, tiled_mma
                ).get_slice(tid)
                if fx.const_expr(gj_gfx950_tile):
                    copy_s2r_B = uni_copy_128b
                    thr_copy_s2r_B = fx.make_tiled_copy_B(
                        copy_s2r_B, tiled_mma
                    ).get_slice(tid)
                else:
                    copy_s2r_B = buffer_copy_128b
                    thr_copy_s2r_B = fx.make_tiled_copy_B(
                        buffer_copy_128b, tiled_mma
                    ).get_slice(tid)
                thr_copy_g2r_B = fx.make_tiled_copy_B(
                    buffer_copy_128b, tiled_mma
                ).get_slice(tid)

                composed_layout_A = fx.make_composed_layout(
                    fx.static(fx.SwizzleType.get(3, 3, 3)),
                    fx.make_ordered_layout((BLOCK_M, GJ_BLOCK_K, GJ_STAGES), (1, 0, 2)),
                )
                sA = fx.make_view(
                    fx.get_dyn_shared(fx.BFloat16), composed_layout_A
                )  # (BM, BK, GJ_STAGES)
                composed_layout_B = fx.make_composed_layout(
                    fx.static(fx.SwizzleType.get(2, 4, 4)),
                    fx.make_ordered_layout(
                        (GJ_BLOCK_N, GJ_BLOCK_K, GJ_STAGES), (1, 0, 2)
                    ),
                )
                smem_b = fx.add_offset(
                    fx.get_dyn_shared(fx.BFloat16),
                    fx.make_int_tuple(BLOCK_M * GJ_BLOCK_K * GJ_STAGES),
                )
                sB = fx.make_view(smem_b, composed_layout_B)  # (BN, BK, stages)

                thr_gA_k = thr_copy_g2s_A.partition_S(gA_k)  # (VA, VM, VK, n)
                thr_sA = thr_copy_g2s_A.partition_D(sA)  # (VA, VM, VK, stages)
                thr_sA_s2r = thr_copy_s2r_A.partition_S(sA)  # (VA, VM, VK, stages)
                thr_sB_s2r = thr_copy_s2r_B.partition_S(sB)  # (VB, VN, VK, stages)
                thr_gB_k = thr_copy_g2r_B.partition_S(gB_k)  # (VB, VN, VK, n)

                copy_frag_A = fx.make_fragment_like(thr_sA[None, None, None, 0])
                b_frags = [
                    fx.make_rmem_tensor(_D_VEC, fx.BFloat16)
                    for _ in fx.range_constexpr(4)
                ]
                gB_full = fx.make_view(
                    fx.add_offset(
                        fx.get_iter(B_buf),
                        fx.make_int_tuple(block_n_idx * fx.Int32(GJ_BLOCK_N * N)),
                    ),
                    fx.make_layout((GJ_BLOCK_N, N), (N, 1)),
                )

                def load_b_tile(k_tile):
                    for p in fx.range_constexpr(4):
                        chunk = tid + fx.Int32(p * THREADS)
                        row = chunk // fx.Int32(4)
                        vec = chunk % fx.Int32(4)
                        row_div = fx.logical_divide(
                            fx.slice(gB_full, (row, None)),
                            fx.make_layout(_D_VEC, 1),
                        )
                        fx.copy(
                            buffer_copy_128b,
                            fx.slice(
                                row_div,
                                (None, k_tile * fx.Int32(4) + vec),
                            ),
                            b_frags[p],
                        )

                def store_b_tile(stage):
                    for p in fx.range_constexpr(4):
                        chunk = tid + fx.Int32(p * THREADS)
                        row = chunk // fx.Int32(4)
                        vec = chunk % fx.Int32(4)
                        row_div = fx.logical_divide(
                            fx.slice(sB, (row, None, stage)),
                            fx.make_layout(_D_VEC, 1),
                        )
                        fx.copy(
                            uni_copy_128b,
                            b_frags[p],
                            fx.slice(row_div, (None, vec)),
                        )

                mma_frag_A = thr_mma.make_fragment_A(sA[None, None, 0])
                if fx.const_expr(gj_gfx950_tile):
                    mma_frag_B = thr_mma.make_fragment_B(sB[None, None, 0])
                    mma_frag_B_retile = thr_copy_s2r_B.retile(mma_frag_B)
                else:
                    mma_frag_B = thr_mma.make_fragment_B(gB_k, stages=2)
                    mma_frag_B_retile = thr_copy_g2r_B.retile(mma_frag_B)
                mma_frag_C = thr_mma.make_fragment_C(gC)

                mma_frag_A_retile = thr_copy_s2r_A.retile(mma_frag_A)

                gA_k_stride = fx.get_scalar(gA_k.stride[2])
                gB_k_stride = fx.get_scalar(gB_k.stride[2])

                def run_pipeline_stage(
                    read_stage,
                    next_k,
                    read_next=True,
                    *,
                    buffer_copy_128b=buffer_copy_128b,
                    thr_gA_k=thr_gA_k,
                    copy_frag_A=copy_frag_A,
                    gA_k_stride=gA_k_stride,
                    thr_gB_k=thr_gB_k,
                    mma_frag_B_retile=mma_frag_B_retile,
                    gB_k_stride=gB_k_stride,
                    uni_copy_128b=uni_copy_128b,
                    thr_sA_s2r=thr_sA_s2r,
                    mma_frag_A_retile=mma_frag_A_retile,
                    tiled_mma=tiled_mma,
                    mma_frag_C=mma_frag_C,
                    mma_frag_A=mma_frag_A,
                    mma_frag_B=mma_frag_B,
                    thr_sA=thr_sA,
                    thr_sB_s2r=thr_sB_s2r,
                ):
                    write_stage = read_stage ^ 1
                    # A's LDS stage folds modulo GJ_STAGES: 2 stages ping-pong across
                    # two buffers; 1 stage collapses to a single buffer.
                    a_read = read_stage % GJ_STAGES
                    a_write = write_stage % GJ_STAGES
                    if fx.const_expr(read_next):
                        next_k = fx.Int32(next_k)
                        fx.copy(
                            buffer_copy_128b,
                            thr_gA_k[None, None, None, 0],
                            copy_frag_A,
                            soffset=next_k * gA_k_stride,
                        )
                        if fx.const_expr(gj_gfx950_tile):
                            load_b_tile(next_k)
                        else:
                            fx.copy(
                                buffer_copy_128b,
                                thr_gB_k[None, None, None, 0],
                                mma_frag_B_retile[None, None, None, write_stage],
                                soffset=next_k * gB_k_stride,
                            )

                    for block_k_iter in fx.range_constexpr(
                        GJ_BLOCK_K // (16 if gj_mfma32 else 32)
                    ):
                        fx.copy(
                            uni_copy_128b,
                            thr_sA_s2r[None, None, block_k_iter, a_read],
                            mma_frag_A_retile[None, None, block_k_iter],
                        )
                        if fx.const_expr(gj_mfma32):
                            if fx.const_expr(gj_gfx950_tile):
                                fx.copy(
                                    copy_s2r_B,
                                    thr_sB_s2r[None, None, block_k_iter, a_read],
                                    mma_frag_B_retile[None, None, block_k_iter],
                                )
                                fx.gemm(
                                    tiled_mma,
                                    mma_frag_C,
                                    mma_frag_A[None, None, block_k_iter],
                                    mma_frag_B[None, None, block_k_iter],
                                    mma_frag_C,
                                    traversal_order=fx.GemmTraversalOrder.KNM,
                                )
                            else:
                                fx.gemm(
                                    tiled_mma,
                                    mma_frag_C,
                                    mma_frag_A[None, None, block_k_iter],
                                    mma_frag_B[None, None, block_k_iter, read_stage],
                                    mma_frag_C,
                                    traversal_order=fx.GemmTraversalOrder.KNM,
                                )
                        else:
                            fx.gemm(
                                tiled_mma,
                                mma_frag_C,
                                mma_frag_A[None, None, (None, block_k_iter)],
                                mma_frag_B[
                                    None, None, (None, block_k_iter), read_stage
                                ],
                                mma_frag_C,
                                traversal_order=fx.GemmTraversalOrder.KNM,
                            )

                    # Single-buffer (GJ_STAGES==1): the next-k write reuses the buffer we
                    # just read, so all threads must finish their s2r reads before any write.
                    if fx.const_expr(GJ_STAGES == 1):
                        fx.gpu.barrier()
                    fx.copy(
                        uni_copy_128b, copy_frag_A, thr_sA[None, None, None, a_write]
                    )
                    if fx.const_expr(gj_gfx950_tile):
                        store_b_tile(a_write)
                    fx.gpu.barrier()

                # Prologue: load contraction-tile 0 into the read buffer.
                fx.copy(buffer_copy_128b, thr_gA_k[None, None, None, 0], copy_frag_A)
                if fx.const_expr(gj_gfx950_tile):
                    load_b_tile(fx.Int32(0))
                else:
                    fx.copy(
                        buffer_copy_128b,
                        thr_gB_k[None, None, None, 0],
                        mma_frag_B_retile[None, None, None, 0],
                    )
                mma_frag_C.fill(0)
                fx.copy(uni_copy_128b, copy_frag_A, thr_sA[None, None, None, 0])
                if fx.const_expr(gj_gfx950_tile):
                    store_b_tile(fx.Int32(0))
                fx.gpu.barrier()

                # Main loop over the N contraction (double-buffered over N // BLOCK_K tiles).
                for k_iter in range(0, NRED_TILES - 2, 2):
                    run_pipeline_stage(read_stage=0, next_k=k_iter + 1)
                    run_pipeline_stage(read_stage=1, next_k=k_iter + 2)
                run_pipeline_stage(read_stage=0, next_k=NRED_TILES - 1)
                run_pipeline_stage(read_stage=1, next_k=None, read_next=False)

                # Epilogue: fp32 accumulators -> bf16; bounded C_buf drops OOB tail
                # stores in hardware (no bias in backward).
                mma_frag_C_bf16 = fx.make_fragment_like(mma_frag_C, fx.BFloat16.ir_type)
                thr_copy_r2g_C = fx.make_tiled_copy_C(
                    fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16), tiled_mma
                ).get_slice(tid)
                mma_frag_C_retile = thr_copy_r2g_C.retile(mma_frag_C_bf16)
                thr_gC = thr_copy_r2g_C.partition_S(gC)

                mma_frag_C_bf16.store(mma_frag_C.load().to(fx.BFloat16))
                fx.copy(
                    fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16),
                    mma_frag_C_retile,
                    thr_gC,
                )

    @flyc.jit
    def grad_jagged(
        dJagged: fx.Tensor,  # out    (L, K)            bf16
        dOut: fx.Tensor,  # grad   (L, N)            bf16
        DENSE: fx.Tensor,  # dense  (n_groups * K, N) bf16  (plain, K-major per group)
        SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
        n_groups: int,
        max_seq_len: int,
        stream: fx.Stream = _DEFAULT_STREAM,
    ):
        """dJagged[s:e, :] = dOut[s:e, :] @ Dense[b].T, per group.

        Contraction is over the static N axis, so this is a clean per-row GEMM that
        reuses the forward kernel's MFMA + double-buffered pipeline (N == K == 128).
        """
        if fx.const_expr(gj_mfma32):
            tiled_mma = fx.make_tiled_mma(
                fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16)),
                fx.make_layout((1, 4, 1), (0, 1, 0)),
                fx.make_tile(None, None, fx.make_layout((4, 4), (1, 4))),
            )
        else:
            tiled_mma = fx.make_tiled_mma(
                fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
                fx.make_layout((1, 4, 1), (0, 1, 0)),
                fx.make_tile(None, None, fx.make_layout((4, 4, 2), (1, 8, 4))),
            )
        val_per_thr = 8  # 16B / bf16
        thrs_col = GJ_BLOCK_K // val_per_thr
        thrs_row = THREADS // thrs_col
        tiled_copy_g2s_A = fx.make_tiled_copy_tv(
            fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16),
            fx.make_layout((thrs_row, thrs_col), (thrs_col, 1)),
            fx.make_layout((1, val_per_thr), (1, 1)),
        )
        bm = (max_seq_len + BLOCK_M - 1) // BLOCK_M
        bm_coarse = (bm + COARSEN_M - 1) // COARSEN_M  # M row-tiles per WG = COARSEN_M
        tiles_per_group = bm_coarse * KOUT_BLOCKS
        xcd_rounds = (n_groups + num_xcds - 1) // num_xcds
        gj_smem = BLOCK_M * GJ_BLOCK_K * GJ_STAGES * 2
        if fx.const_expr(gj_gfx950_tile):
            gj_smem = gj_smem + GJ_BLOCK_N * GJ_BLOCK_K * GJ_STAGES * 2
        grad_jagged_kernel(
            dJagged,
            dOut,
            DENSE,
            SEQ_OFFSETS,
            tiled_mma,
            tiled_copy_g2s_A,
            n_groups,
            tiles_per_group,
        ).launch(
            grid=(num_xcds * tiles_per_group * xcd_rounds, 1, 1),
            block=(THREADS, 1, 1),
            smem=gj_smem,
            stream=stream,
        )

    @flyc.kernel
    def grad_bias_reduce_kernel(
        DBIAS: fx.Tensor,  # out    (n_groups, N)          bf16
        PARTIALS: fx.Tensor,  # in     (n_groups * SPLIT, N)  fp32
    ):
        # Sum this group's SPLIT fp32 partials and write bf16 dBias[b]. Thread tid
        # owns column n = col_tile*NRED_BLK + tid.
        tid = fx.thread_idx.x
        col_tile = fx.block_idx.x
        off_b = fx.block_idx.y
        col = col_tile * fx.Int32(NRED_BLK) + tid

        PART_buf = fx.rocdl.make_buffer_tensor(PARTIALS, max_size=True)
        DBIAS_buf = fx.rocdl.make_buffer_tensor(DBIAS, max_size=True)
        copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        copy_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)

        # Guard col < N: when N is not a multiple of NRED_BLK the last column-tile
        # carries lanes with col >= N. DBIAS_buf is a WHOLE-TENSOR descriptor, so an
        # unguarded store from those lanes is NOT an OOB drop -- it spills into the
        # next row (dBias[b+1]). The D-derived split policy never trips this (SPLIT>=2
        # only at D<=256, where NRED_BLK == N), but a forced split kwarg at N%NRED_BLK
        # != 0 (e.g. split=2 at D=384) would; the guard makes any N correct.
        if col < fx.Int32(N):
            acc = fx.Float32(0.0)
            for s in fx.range_constexpr(SPLIT):
                part_row = off_b * fx.Int32(SPLIT) + fx.Int32(s)
                row_div = fx.logical_divide(
                    fx.slice(PART_buf, (part_row, None)), fx.make_layout(1, 1)
                )
                acc = acc + _load_scalar(copy_f32, fx.Float32, row_div, col)

            out_div = fx.logical_divide(
                fx.slice(DBIAS_buf, (off_b, None)), fx.make_layout(1, 1)
            )
            _store_scalar(copy_bf16, fx.BFloat16, out_div, col, acc.to(fx.BFloat16))

    @flyc.kernel
    def grad_dense_partials_kernel(
        PARTIALS: fx.Tensor,  # out  SPLIT>=2: (n_groups*SPLIT*K, N) fp32 scratch; SPLIT==1: bf16 dDense (n_groups*K, N)
        BIAS_PARTIALS: fx.Tensor,  # out  SPLIT>=2: (n_groups*SPLIT, N) fp32 (fused dBias); SPLIT==1: bf16 dBias (n_groups, N)
        JAGGED: fx.Tensor,  # jagged (L, K)                     bf16
        DOUT: fx.Tensor,  # grad   (L, N)                     bf16
        SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s_J: fx.TiledCopy,
        tiled_copy_s2s_J: fx.TiledCopy,
        n_groups: fx.Int32,
    ):
        # dDense[b][k,n] = sum_m Jagged[m,k] * dOut[m,n]. Written as the MFMA atom form
        # C[i,j] = sum_l A[i,l]*B[j,l] with i=k, j=n, l=m: A[k,m] = J[m,k] (= J.T) and
        # B[n,m] = dOut[m,n] (= dOut.T). Both operands therefore carry the reduction
        # axis m as their *contiguous* fragment (K) axis, so each m-tile is staged
        # transposed into LDS (sJ as (k, m), sD as (n, m)) and fed to bf16 MFMA. Split
        # s reduces a strided subset of the group's m-tiles. block_idx.x is the
        # XCD-striped launch id; it decodes to this workgroup's (DDENSE_BK x
        # DDENSE_BN) output sub-tile of the (K, N) block.
        tid = fx.thread_idx.x
        # The dispatcher deals workgroup i to XCD (i % num_xcds), and each XCD
        # has its own L2. This group's (K, N) output tiles reread the same
        # Jagged and dOut rows, so they have to share an XCD; groups themselves
        # stay striped (group g on XCD g % num_xcds) so a run of empty or full
        # groups cannot pile onto one die. Launch order is the inverse:
        # linear = (g % num_xcds) + num_xcds * (tile + (g // num_xcds) * T).
        # A group count that does not fill the last round of XCDs takes the
        # in_range == false path and stores nothing.
        tiles_kn = fx.Int32(NK_TILES * NN_TILES)
        tiles_per_group = tiles_kn * fx.Int32(SPLIT)
        linear = fx.block_idx.x
        xcd = linear % fx.Int32(num_xcds)
        slot = linear // fx.Int32(num_xcds)
        tile = slot % tiles_per_group
        off_b = xcd + (slot // tiles_per_group) * fx.Int32(num_xcds)
        in_range = off_b < n_groups
        off_b = in_range.select(off_b, fx.Int32(0))
        pid_kn = tile % tiles_kn
        off_s = tile // tiles_kn
        k_off = (pid_kn // fx.Int32(NN_TILES)) * fx.Int32(DDENSE_BK)
        n_off = (pid_kn % fx.Int32(NN_TILES)) * fx.Int32(DDENSE_BN)

        # off_b comes from the block id, so both offsets are wave-uniform and stay in SGPRs.
        seq_buf = fx.rocdl.make_buffer_tensor(SEQ_OFFSETS, max_size=True)
        seq_start = fx.Int32(buf_scalar_load(seq_buf, off_b))
        seq_end = fx.Int32(buf_scalar_load(seq_buf, off_b + fx.Int32(1)))
        M_b = seq_end - seq_start
        # Out-of-range tail ids clamp onto group 0 and then run an empty
        # contraction, so the epilogue below is what keeps them from storing.
        M_b = in_range.select(M_b, fx.Int32(0))

        # Group-rebased buffers bounded to M_b rows: any local row >= M_b zero-fills
        # (CDNA OOB-load == 0), so tail rows contribute 0 to the contraction.
        # base_byte_offset MUST be computed in int64: at the largest deployment shape seq_start
        # reaches ~L ? 7.86M rows, so seq_start*K*2 ? 4 GB overflows int32 (silently
        # wrapping the descriptor base). num_records_bytes stays in-range (per-group,
        # M_b <= Mi) so an int32 product is fine there.
        J_g = fx.make_view(
            fx.add_offset(
                fx.get_iter(JAGGED),
                fx.make_int_tuple(fx.Int64(seq_start) * fx.Int64(K)),
            ),
            fx.get_layout(JAGGED),
        )
        J_buf = fx.rocdl.make_buffer_tensor(
            J_g,
            num_records_bytes=fx.Int64(M_b * fx.Int32(K * 2)),
        )
        D_g = fx.make_view(
            fx.add_offset(
                fx.get_iter(DOUT),
                fx.make_int_tuple(fx.Int64(seq_start) * fx.Int64(N)),
            ),
            fx.get_layout(DOUT),
        )
        D_buf = fx.rocdl.make_buffer_tensor(
            D_g,
            num_records_bytes=fx.Int64(M_b * fx.Int32(N * 2)),
        )
        d_chunk = (tid // fx.Int32(64)) * fx.Int32(_D_LANE_CHUNKS) + tid % fx.Int32(
            _D_LANE_CHUNKS
        )
        d_row = (tid % fx.Int32(64)) // fx.Int32(_D_LANE_CHUNKS)

        if fx.const_expr(dense_gfx950_128_tile):
            # Both tiles keep their global orientation, (out_dim, m) with out_dim
            # contiguous, so 128-bit row loads store straight to LDS and
            # ds_read_tr16 transposes m onto the MFMA K-fragment. Swizzle(2,4,3)
            # keeps the 16-element groups ds_read_tr16 needs and XORs low m bits
            # into out_dim bits [4:6] (same as gemm_a16w16_gfx950's 128-row tile).
            composed_J = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(2, 4, 3)),
                fx.make_ordered_layout((DDENSE_BK, dense_bm), (0, 1)),
            )
            composed_D = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(2, 4, 3)),
                fx.make_ordered_layout((DDENSE_BN, dense_bm), (0, 1)),
            )
        else:
            # LDS staging tiles, both (out_dim, m) with m (the contraction) contiguous so
            # the s2r feed lands m on the MFMA K-fragment. Same swizzle as the forward sA.
            composed_J = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 3, 3)),
                fx.make_ordered_layout((DDENSE_BK, dense_bm), (1, 0)),
            )
            composed_D = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 3, 3)),
                fx.make_ordered_layout((DDENSE_BN, dense_bm), (1, 0)),
            )
        smem = fx.get_dyn_shared(fx.BFloat16)
        sJ = fx.make_view(smem, composed_J)  # (k, m)
        smem_d = fx.add_offset(smem, fx.make_int_tuple(DDENSE_BK * dense_bm))
        sD = fx.make_view(smem_d, composed_D)  # (n, m)

        thr_mma = tiled_mma.thr_slice(tid)
        uni_copy_128b = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        buffer_copy_128b = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        if fx.const_expr(dense_gfx950_128_tile):
            s2r_tiled_atom = fx.make_copy_atom(
                fx.rocdl.cdna4.LDSReadTrans16_64b(), fx.BFloat16
            )
            s2r_copy_atom = s2r_tiled_atom
        else:
            s2r_tiled_atom = buffer_copy_128b
            s2r_copy_atom = uni_copy_128b
        thr_copy_s2r_A = fx.make_tiled_copy_A(s2r_tiled_atom, tiled_mma).get_slice(tid)
        thr_copy_s2r_B = fx.make_tiled_copy_B(s2r_tiled_atom, tiled_mma).get_slice(tid)
        buffer_copy_16b = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        thr_copy_g2s_J = tiled_copy_g2s_J.get_slice(tid)
        thr_copy_s2s_J = tiled_copy_s2s_J.get_slice(tid)

        thr_sJ_s2r = thr_copy_s2r_A.partition_S(sJ)  # (VA, VM, VK)
        thr_sD_s2r = thr_copy_s2r_B.partition_S(sD)  # (VB, VN, VK)

        # Output (K, N) sub-tile of this workgroup, viewed in the fp32 partials scratch.
        # int64 element offset (same rationale as b_row_off): the partials-tile index
        # scales with the group count and dense tile size; widen before multiplying
        # to avoid int32 wrap.
        part_off = ((off_b * fx.Int64(SPLIT) + off_s) * fx.Int64(K) + k_off) * fx.Int64(
            N
        ) + n_off
        PART_g = fx.make_view(
            fx.add_offset(fx.get_iter(PARTIALS), fx.make_int_tuple(part_off)),
            fx.get_layout(PARTIALS),
        )
        PART_buf = fx.rocdl.make_buffer_tensor(PART_g, max_size=True)
        gPart = fx.make_view(
            fx.get_iter(PART_buf), fx.make_layout((DDENSE_BK, DDENSE_BN), (N, 1))
        )

        mma_frag_A = thr_mma.make_fragment_A(sJ)
        mma_frag_B = thr_mma.make_fragment_B(sD)
        mma_frag_C = thr_mma.make_fragment_C(gPart)
        mma_frag_A_retile = thr_copy_s2r_A.retile(mma_frag_A)
        mma_frag_B_retile = thr_copy_s2r_B.retile(mma_frag_B)

        num_tiles = (M_b + fx.Int32(dense_bm - 1)) // fx.Int32(dense_bm)

        # fp32 MFMA accumulator carried in place across the dynamic, split-strided
        # m-tile loop (AGPR accumulate; no SSA carry needed). The fused dBias column
        # partials are carried as an fp32 loop iter_arg: each thread sums the dOut
        # elements it already loads for the transpose-staging (_D_VEC running fp32 per
        # thread), so the bias reduction piggybacks on the dDense dOut traffic at no
        # extra global reads.
        mma_frag_C.fill(0)
        bias_acc0 = fx.Vector.filled(_D_VEC, 0.0, fx.Float32)
        copy_d_128b = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        for m_tile, _carry in range(
            off_s, num_tiles, fx.Int32(SPLIT), init=[bias_acc0]
        ):
            bias_acc = fx.Vector(_carry[0])
            mt = fx.Int32(m_tile)
            j_tile_off = fx.Int64(mt) * fx.Int64(dense_bm) * fx.Int64(K) + fx.Int64(
                k_off
            )
            if fx.const_expr(dense_gfx950_128_tile):
                # J rows are k-contiguous in global and in sJ, so J reuses the
                # dOut row map: one 128-bit load and one 128-bit LDS store per pass.
                gJ_rows = fx.make_view(
                    fx.add_offset(fx.get_iter(J_buf), fx.make_int_tuple(j_tile_off)),
                    fx.make_layout((dense_bm, DDENSE_BK), (K, 1)),
                )
                for p in fx.range_constexpr(dense_row_passes):
                    m_local = d_row + fx.Int32(p * _D_LANE_ROWS)
                    row_div = fx.logical_divide(
                        fx.slice(gJ_rows, (m_local, None)),
                        fx.make_layout(_D_VEC, 1),
                    )
                    frag_j = fx.make_rmem_tensor(_D_VEC, fx.BFloat16)
                    fx.copy(copy_d_128b, fx.slice(row_div, (None, d_chunk)), frag_j)
                    s_div = fx.logical_divide(
                        fx.slice(sJ, (None, m_local)), fx.make_layout(_D_VEC, 1)
                    )
                    fx.copy(uni_copy_128b, frag_j, fx.slice(s_div, (None, d_chunk)))
            else:
                # Transpose-stage J for this m-tile via tiled g2s. Global J is viewed in
                # logical (k, m) order for the workgroup's K sub-tile; partition_D lands in
                # swizzled sJ (k, m). The global m stride is K so the load stays scalar b16,
                # but each thread owns 32 contiguous m at one k, so the LDS store is 128-bit
                # (4 writes, conflict-free in the same Swizzle(3,3,3) as the forward sA)
                # instead of 32 scalar b16 writes that scatter across banks.
                gJ_tile = fx.make_view(
                    fx.add_offset(fx.get_iter(J_buf), fx.make_int_tuple(j_tile_off)),
                    fx.make_layout((DDENSE_BK, dense_bm), (1, K)),
                )
                thr_gJ = thr_copy_g2s_J.partition_S(gJ_tile)
                thr_sJ_g2s = thr_copy_s2s_J.partition_D(sJ)
                copy_frag_J = fx.make_fragment_like(thr_gJ)
                fx.copy(buffer_copy_16b, thr_gJ, copy_frag_J)
                fx.copy(uni_copy_128b, thr_copy_s2s_J.retile(copy_frag_J), thr_sJ_g2s)
            # Tail rows (m >= M_b) zero-fill via the bounded descriptor, so they add
            # 0 to the bias sums. On the default path sD is (n, m) with m contiguous,
            # so the 128-bit row loads are scattered into sD one element at a time.
            d_tile_off = fx.Int64(mt) * fx.Int64(dense_bm) * fx.Int64(N) + fx.Int64(
                n_off
            )
            gD_tile = fx.make_view(
                fx.add_offset(fx.get_iter(D_buf), fx.make_int_tuple(d_tile_off)),
                fx.make_layout((dense_bm, DDENSE_BN), (N, 1)),
            )
            for p in fx.range_constexpr(dense_row_passes):
                m_local = d_row + fx.Int32(p * _D_LANE_ROWS)
                row_div = fx.logical_divide(
                    fx.slice(gD_tile, (m_local, None)), fx.make_layout(_D_VEC, 1)
                )
                frag_d = fx.make_rmem_tensor(_D_VEC, fx.BFloat16)
                fx.copy(copy_d_128b, fx.slice(row_div, (None, d_chunk)), frag_d)
                dvec = fx.Vector(fx.memref_load_vec(frag_d))
                if fx.const_expr(dense_gfx950_128_tile):
                    s_div = fx.logical_divide(
                        fx.slice(sD, (None, m_local)), fx.make_layout(_D_VEC, 1)
                    )
                    fx.copy(uni_copy_128b, frag_d, fx.slice(s_div, (None, d_chunk)))
                else:
                    for j in fx.range_constexpr(_D_VEC):
                        fx.memref_store(
                            dvec[j],
                            sD,
                            (d_chunk * fx.Int32(_D_VEC) + fx.Int32(j), m_local),
                        )
                bias_acc = bias_acc + dvec.to(fx.Float32)
            fx.gpu.barrier()

            for block_k_iter in fx.range_constexpr(
                dense_bm // (16 if dense_gfx950_128_tile else 32)
            ):
                fx.copy(
                    s2r_copy_atom,
                    thr_sJ_s2r[None, None, block_k_iter],
                    mma_frag_A_retile[None, None, block_k_iter],
                )
                fx.copy(
                    s2r_copy_atom,
                    thr_sD_s2r[None, None, block_k_iter],
                    mma_frag_B_retile[None, None, block_k_iter],
                )
                if fx.const_expr(dense_gfx950_128_tile):
                    fx.gemm(
                        tiled_mma,
                        mma_frag_C,
                        mma_frag_A[None, None, block_k_iter],
                        mma_frag_B[None, None, block_k_iter],
                        mma_frag_C,
                        traversal_order=fx.GemmTraversalOrder.KNM,
                    )
                else:
                    fx.gemm(
                        tiled_mma,
                        mma_frag_C,
                        mma_frag_A[None, None, (None, block_k_iter)],
                        mma_frag_B[None, None, (None, block_k_iter)],
                        mma_frag_C,
                        traversal_order=fx.GemmTraversalOrder.KNM,
                    )
            fx.gpu.barrier()
            _carry_out = yield [bias_acc]
        bias_acc_final = _carry_out

        if in_range:
            # Epilogue. SPLIT==1 (e.g. D=512): each (K,N) tile is fully reduced inside this one
            # workgroup, so truncate the fp32 accumulator to bf16 and store it STRAIGHT to dDense
            # (PARTIALS *is* the bf16 dDense view in that case -- part_off collapses to the
            # dDense element offset since off_s==0). This skips the fp32 partials round-trip and
            # the separate grad_dense_reduce launch. SPLIT>=2 (e.g. D=256): write the fp32
            # accumulator to the partials scratch; the reduce pass sums the SPLIT partials and
            # casts to bf16.
            if fx.const_expr(SPLIT == 1):
                mma_frag_C_bf16 = fx.make_fragment_like(mma_frag_C, fx.BFloat16.ir_type)
                thr_copy_r2g_C = fx.make_tiled_copy_C(
                    fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16), tiled_mma
                ).get_slice(tid)
                mma_frag_C_retile = thr_copy_r2g_C.retile(mma_frag_C_bf16)
                thr_gPart = thr_copy_r2g_C.partition_S(gPart)
                mma_frag_C_bf16.store(mma_frag_C.load().to(fx.BFloat16))
                fx.copy(
                    fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16),
                    mma_frag_C_retile,
                    thr_gPart,
                )
            else:
                thr_copy_r2g_C = fx.make_tiled_copy_C(
                    fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32), tiled_mma
                ).get_slice(tid)
                mma_frag_C_retile = thr_copy_r2g_C.retile(mma_frag_C)
                thr_gPart = thr_copy_r2g_C.partition_S(gPart)
                fx.copy(
                    fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32),
                    mma_frag_C_retile,
                    thr_gPart,
                )

            # Fused dBias epilogue: combine the per-thread column partials through LDS (the
            # sJ/sD staging region is dead now, so reuse it as fp32 scratch -> no extra smem,
            # occupancy unchanged) and write one fp32 bias partial per output column for this
            # (group, split). Only the k-tile-0 workgroups emit it so each N-tile is written
            # once. M_b == 0 groups never enter the m-loop, so bias_acc stays 0 -> 0 bias.
            if k_off == fx.Int32(0):
                smem_f32 = fx.get_dyn_shared(fx.Float32)
                bias_lds = fx.make_view(
                    smem_f32,
                    fx.make_layout((_D_VEC, DDENSE_THREADS), (1, _D_VEC)),
                )
                bias_final = fx.Vector(bias_acc_final)
                fx.gpu.barrier()
                for j in fx.range_constexpr(_D_VEC):
                    fx.memref_store(bias_final[j], bias_lds, (fx.Int32(j), tid))
                fx.gpu.barrier()
                if tid < fx.Int32(DDENSE_BN):
                    # Column tid lives in n-chunk c at slot j; its owners are the
                    # _D_LANE_ROWS threads of wave c // _D_LANE_CHUNKS with that lane chunk.
                    col_c = tid // fx.Int32(_D_VEC)
                    col_j = tid % fx.Int32(_D_VEC)
                    owner0 = (col_c // fx.Int32(_D_LANE_CHUNKS)) * fx.Int32(
                        64
                    ) + col_c % fx.Int32(_D_LANE_CHUNKS)
                    col_sum = fx.Float32(0.0)
                    for r in fx.range_constexpr(_D_LANE_ROWS):
                        col_sum = col_sum + fx.Float32(
                            fx.memref_load(
                                bias_lds,
                                (col_j, owner0 + fx.Int32(r * _D_LANE_CHUNKS)),
                            )
                        )
                    BP_buf = fx.rocdl.make_buffer_tensor(BIAS_PARTIALS, max_size=True)
                    part_row = off_b * fx.Int32(SPLIT) + off_s
                    bp_div = fx.logical_divide(
                        fx.slice(BP_buf, (part_row, None)), fx.make_layout(1, 1)
                    )
                    if fx.const_expr(SPLIT == 1):
                        # SPLIT==1: BIAS_PARTIALS is the bf16 dBias view (part_row == off_b),
                        # so write the final column sum straight to dBias[b] -- no separate
                        # grad_bias_reduce launch.
                        copy_bf16 = fx.make_copy_atom(
                            fx.rocdl.BufferCopy16b(), fx.BFloat16
                        )
                        _store_scalar(
                            copy_bf16,
                            fx.BFloat16,
                            bp_div,
                            n_off + tid,
                            col_sum.to(fx.BFloat16),
                        )
                    else:
                        copy_f32 = fx.make_copy_atom(
                            fx.rocdl.BufferCopy32b(), fx.Float32
                        )
                        _store_scalar(
                            copy_f32, fx.Float32, bp_div, n_off + tid, col_sum
                        )

    @flyc.kernel
    def grad_dense_reduce_kernel(
        DDENSE: fx.Tensor,  # out    (n_groups * K, N)          bf16
        PARTIALS: fx.Tensor,  # in     (n_groups * SPLIT * K, N)  fp32
    ):
        # Sum this group's SPLIT fp32 (K, N) partials and write bf16 dDense[b]. Block
        # (col-tile, k-row, group); thread tid owns column n = col_tile*NRED_BLK + tid.
        tid = fx.thread_idx.x
        col_tile = fx.block_idx.x
        off_k = fx.block_idx.y
        off_b = fx.block_idx.z
        col = col_tile * fx.Int32(NRED_BLK) + tid

        PART_buf = fx.rocdl.make_buffer_tensor(PARTIALS, max_size=True)
        DD_buf = fx.rocdl.make_buffer_tensor(DDENSE, max_size=True)
        copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        copy_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)

        # Guard col < N (see grad_bias_reduce_kernel): DD_buf is a whole-tensor
        # descriptor, so the last column-tile's col >= N lanes must not store, or
        # they spill into the next (K, N) row instead of being OOB-dropped.
        if col < fx.Int32(N):
            acc = fx.Float32(0.0)
            for s in fx.range_constexpr(SPLIT):
                part_row = (off_b * fx.Int32(SPLIT) + fx.Int32(s)) * fx.Int32(K) + off_k
                row_div = fx.logical_divide(
                    fx.slice(PART_buf, (part_row, None)), fx.make_layout(1, 1)
                )
                acc = acc + _load_scalar(copy_f32, fx.Float32, row_div, col)

            out_row = off_b * fx.Int32(K) + off_k
            out_div = fx.logical_divide(
                fx.slice(DD_buf, (out_row, None)), fx.make_layout(1, 1)
            )
            _store_scalar(copy_bf16, fx.BFloat16, out_div, col, acc.to(fx.BFloat16))

    @flyc.jit
    def grad_dense_bias(
        dDense: fx.Tensor,  # out    (n_groups * K, N)            bf16
        dBias: fx.Tensor,  # out    (n_groups, N)               bf16
        JAGGED: fx.Tensor,  # jagged (L, K)                       bf16
        dOut: fx.Tensor,  # grad   (L, N)                       bf16
        SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
        partials: fx.Tensor,  # fp32 scratch (n_groups * SPLIT * K, N)
        bias_partials: fx.Tensor,  # fp32 scratch (n_groups * SPLIT, N)
        n_groups: int,
        max_seq_len: int,
        stream: fx.Stream = _DEFAULT_STREAM,
    ):
        """dDense[b] = Jagged[s:e, :].T @ dOut[s:e, :] and dBias[b] = sum_m dOut[s:e, :].

        Both reduce over the dynamic sequence axis m, so the dBias column-sums are fused
        into the dDense partials pass (which already streams dOut through LDS) instead of
        re-reading dOut in a separate kernel.

        Two launch schedules, chosen by the compile-time SPLIT:
          * SPLIT == 1 (e.g. D=512): each (K, N) tile is fully reduced inside one workgroup,
            so the partials pass truncates its fp32 accumulator to bf16 and writes dDense +
            dBias DIRECTLY -- no fp32 scratch round-trip and no reduce passes (1 launch).
          * SPLIT >= 2 (e.g. D=256): three launches -- one bf16-MFMA partials pass writes the
            fp32 (K, N) dDense partials and fp32 (N,) dBias partials per (group, split), then
            two light reduce passes sum the SPLIT partials into bf16 dDense and dBias.
        """
        # gfx950 D=512 uses its native 32x32x16 bf16 MFMA across a 2x2-wave
        # 128x128 tile. The remaining paths retain the gfx942-compatible 16x16
        # atom and layout.
        if fx.const_expr(dense_gfx950_128_tile):
            tiled_mma = fx.make_tiled_mma(
                fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16)),
                fx.make_layout((2, 2, 1), (2, 1, 0)),
                fx.make_tile(None, None, fx.make_layout((4, 4), (1, 4))),
            )
        else:
            tiled_mma = fx.make_tiled_mma(
                fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
                fx.make_layout((1, 4, 1), (0, 1, 0)),
                fx.make_tile(None, None, fx.make_layout((4, 4, 2), (1, 8, 4))),
            )
        # One TV layout shared by two copy atoms: the global J load is strided in m
        # (stride K) so it stays scalar b16, but each thread owns 32 contiguous m at one
        # k, so the swizzled sJ store is 128-bit. partition_S/partition_D retile the
        # value mode to the atom width, so the store needs its own 128b tiled copy.
        j_g2s_tiler_mn, j_g2s_tv = make_layout_tv(
            fx.make_layout((128, 2), (2, 1)),
            fx.make_layout((1, dense_bm // 2), (1, 1)),
        )
        tiled_copy_g2s_J = fx.make_tiled_copy(
            fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16),
            j_g2s_tv,
            j_g2s_tiler_mn,
        )
        tiled_copy_s2s_J = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16),
            j_g2s_tv,
            j_g2s_tiler_mn,
        )
        # 1D grid: the dispatcher round-robins the flattened id, and grid.y tops
        # out at 65535, which (D/128)^2 * SPLIT * ceil(n_groups/num_xcds) can pass.
        xcd_rounds = (n_groups + num_xcds - 1) // num_xcds
        partials_grid = (num_xcds * NK_TILES * NN_TILES * SPLIT * xcd_rounds, 1, 1)
        if SPLIT == 1:
            # SPLIT==1 fast path: partials kernel writes bf16 dDense + dBias directly
            # (dDense/dBias passed as the PARTIALS/BIAS_PARTIALS args); scratch unused.
            grad_dense_partials_kernel(
                dDense,
                dBias,
                JAGGED,
                dOut,
                SEQ_OFFSETS,
                tiled_mma,
                tiled_copy_g2s_J,
                tiled_copy_s2s_J,
                n_groups,
            ).launch(
                grid=partials_grid,
                block=(DDENSE_THREADS, 1, 1),
                smem=dense_smem_bytes,
                stream=stream,
            )
        else:
            grad_dense_partials_kernel(
                partials,
                bias_partials,
                JAGGED,
                dOut,
                SEQ_OFFSETS,
                tiled_mma,
                tiled_copy_g2s_J,
                tiled_copy_s2s_J,
                n_groups,
            ).launch(
                grid=partials_grid,
                block=(DDENSE_THREADS, 1, 1),
                smem=dense_smem_bytes,
                stream=stream,
            )
            grad_dense_reduce_kernel(dDense, partials).launch(
                grid=(NRED_COL_TILES, K, n_groups),
                block=(NRED_BLK, 1, 1),
                stream=stream,
            )
            grad_bias_reduce_kernel(dBias, bias_partials).launch(
                grid=(NRED_COL_TILES, n_groups, 1),
                block=(NRED_BLK, 1, 1),
                stream=stream,
            )

    return Backward(
        grad_jagged=grad_jagged, grad_dense_bias=grad_dense_bias, split=SPLIT, D=D
    )


# --- Module defaults referenced by build_backward for the None knob cases. ---
# (Named separately so build_backward's local COARSEN_M / GJ_STAGES_A can shadow
# the intended knob without recursing on the module global.)
COARSEN_M_default = COARSEN_M
GJ_STAGES_A_default = GJ_STAGES_A


# =========================================================================== #
# Legacy single-D facade over build_backward.                                 #
#                                                                             #
# The production path is jagged_dense_bmm_bwd_dispatch.py, which calls         #
# build_backward() directly and serves multiple D per process. These          #
# module-level names keep the profiler / example timing drivers (which call   #
# configure_dim + grad_jagged / grad_dense_bias and read SPLIT) working        #
# unchanged. Single-D: configure_dim rebinds the active build.                 #
# =========================================================================== #
# The gen forward (jagged_dense_bmm_gen) is parametrized per-call and exposes no
# module-level K/N/N_BLOCKS. Seed the single-D
# facade with a default D matching that forward's old K=N default; configure_dim(D)
# rebinds it. The production dispatch path bypasses this facade entirely.
K = getattr(_fwd, "K", BLOCK_N)
N = getattr(_fwd, "N", BLOCK_N)
SPLIT = _split_for(K)
_active: Backward | None = None


def configure_dim(D):
    """Legacy single-D facade: (re)build the active backward for D and rebind the
    module-level launchers + K/N/SPLIT. Uses the current module GJ_STAGES_A /
    COARSEN_M (so the profiler's ``_bwd.GJ_STAGES_A = X; configure_dim(D)`` still
    works). The production dispatch wrapper bypasses this and calls build_backward()
    directly, which is why the single-D constraint this used to impose is gone."""
    global K, N, SPLIT, _active
    D = int(D)
    _active = build_backward(
        D, split=None, gj_stages_a=GJ_STAGES_A, coarsen_m=COARSEN_M
    )
    K = N = D
    SPLIT = _active.split
    # Keep a non-gen forward module's K/N in sync for any legacy co-launch.
    # The gen forward is parametrized per-call and has no such globals, so only
    # sync when they already exist.
    if hasattr(_fwd, "K"):
        _fwd.K = _fwd.N = D
        _fwd.N_BLOCKS = D // BLOCK_N
    return D


def _ensure_active():
    if _active is None:
        configure_dim(K)
    return _active


def grad_jagged(*args, **kwargs):
    """Legacy facade -> the active build's grad_jagged (call configure_dim first)."""
    return _ensure_active().grad_jagged(*args, **kwargs)


def grad_dense_bias(*args, **kwargs):
    """Legacy facade -> the active build's grad_dense_bias (call configure_dim first)."""
    return _ensure_active().grad_dense_bias(*args, **kwargs)
