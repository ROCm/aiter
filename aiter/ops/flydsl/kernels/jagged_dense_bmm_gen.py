# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL kernel for jagged_dense_bmm_broadcast_add (grouped BF16 GEMM + bias).

For each group ``b`` over ``[seq_offsets[b], seq_offsets[b+1])``::

    Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b]

Dense weight is host-pre-transposed to ``(B * N, K)``. Supports a uniform
grid launch with optional XCD remap, and a compact ``TILE_MAP`` launch for
skewed sequence lengths.
"""

from __future__ import annotations

from collections import OrderedDict

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from ._buffer_utils import make_bounded_buffer_tensor

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
STAGES_A = 2
THREADS = 256
LDS_BYTES = 65536
K_TILE_GRAN = 32  # MFMA K-tile loop uses BLOCK_K // 32 stages

NXCD = 8
XCD_W = 8
XCD_C_SMALL_K = 32
XCD_C_LARGE_K = 120


def _xcd_remap(xy, num_rows, num_cols, C, W, nXCD=NXCD):
    xy = fx.Int32(xy)
    cnXCD = fx.Int32(nXCD)
    cC = fx.Int32(C)
    cW = fx.Int32(W)
    c_num_cols = fx.Int32(num_cols)
    c_num_rows = fx.Int32(num_rows)

    xcd = xy % cnXCD
    local = xy // cnXCD
    chunk_idx = local // cC
    pos = local % cC
    xy_g_remap = chunk_idx * (cnXCD * cC) + xcd * cC + pos

    if isinstance(num_rows, int):
        total = num_rows * num_cols  # compile-time
        period = nXCD * C  # compile-time
        prefix = total - (total % period)  # largest [0,prefix) divisible by period
        if fx.const_expr(prefix == total):
            xy_g = xy_g_remap
        else:
            xy_g = (xy < fx.Int32(prefix)).select(xy_g_remap, xy)
    else:
        total = c_num_rows * c_num_cols  # runtime: total launched blocks
        period = fx.Int32(nXCD * C)  # compile-time product
        prefix = total - (total % period)  # runtime largest multiple of period <= total
        xy_g = (xy < prefix).select(xy_g_remap, xy)

    tids_per_grp = cW * c_num_cols
    group_id = xy_g // tids_per_grp
    first_row = group_id * cW
    remaining = c_num_rows - first_row
    win_h = (remaining < cW).select(remaining, cW)  # last window may be short
    tid_in_grp = xy_g % tids_per_grp
    row = first_row + (tid_in_grp % win_h)
    col = tid_in_grp // win_h
    return row, col


_COMPILED_CACHE_MAX = 64  # skew compact keys include tot; bound growth
_COMPILED_CACHE: OrderedDict = OrderedDict()


def _build_launcher(
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    STAGES_A,
    THREADS,
    BM_TILES,
    N_GROUPS,
    XCD_C,
    XCD_W,
    USE_MFMA_K32,
    WAVES_PER_EU=0,
    B_STAGES=3,
    COMPACT=False,
):
    N_BLOCKS = N // BLOCK_N  # output column-tiles per group (compile-time)
    XCD_NUM_ROWS = BM_TILES * N_GROUPS
    XCD_NUM_COLS = N_BLOCKS
    C_FRAG_LEN = BLOCK_M * BLOCK_N // THREADS
    smem_bytes = BLOCK_M * BLOCK_K * STAGES_A * 2  # bf16 A double-buffer staging

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def jdbba_kernel(
        C: fx.Tensor,  # out    (L, N)   bf16
        A: fx.Tensor,  # jagged (L, K)   bf16
        B: fx.Tensor,  # dense  (B_groups * N, K) bf16 (pre-transposed, tall)
        BIAS: fx.Tensor,  # bias   (B_groups * N,)   bf16
        SEQ_OFFSETS: fx.Tensor,  # (B_groups + 1,) int32
        TILE_MAP: fx.Tensor,  # compact mode: (total_occ_tiles, 2) int32 [off_b, m_idx]; else dummy
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s_A: fx.TiledCopy,
        tiled_copy_s2g_C: fx.TiledCopy,
    ):
        tid = fx.thread_idx.x
        pid_mn, _, raw_b = fx.block_idx

        if fx.const_expr(COMPACT):
            tile_rsrc = fx.buffer_ops.create_buffer_resource(TILE_MAP, max_size=True)
            if fx.const_expr(XCD_C > 1):
                total_occ = fx.Int32(fx.grid_dim.x)
                raw_xy = fx.Int32(raw_b) * total_occ + fx.Int32(pid_mn)
                tile_row, n_col = _xcd_remap(raw_xy, total_occ, N_BLOCKS, XCD_C, XCD_W)
                tile_row = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), tile_row))
                block_n_idx = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), n_col))
            else:
                tile_row = fx.Int32(pid_mn)
                block_n_idx = fx.Int32(raw_b)
            base = tile_row * fx.Int32(2)
            off_b = fx.buffer_ops.buffer_load(
                tile_rsrc, base, vec_width=1, dtype=fx.T.i32()
            )
            block_m_idx = fx.buffer_ops.buffer_load(
                tile_rsrc, base + fx.Int32(1), vec_width=1, dtype=fx.T.i32()
            )
            off_b = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), off_b))
            block_m_idx = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), block_m_idx))
        else:
            raw_xy = fx.Int32(raw_b) * fx.Int32(BM_TILES * N_BLOCKS) + fx.Int32(pid_mn)
            row, col = _xcd_remap(raw_xy, XCD_NUM_ROWS, XCD_NUM_COLS, XCD_C, XCD_W)
            row = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), row))
            col = fx.Int32(fx.rocdl.readfirstlane(fx.T.i32(), col))

            off_b = row // fx.Int32(BM_TILES)
            block_m_idx = row % fx.Int32(BM_TILES)
            block_n_idx = col

        seq_rsrc = fx.buffer_ops.create_buffer_resource(SEQ_OFFSETS, max_size=True)
        seq_start = fx.buffer_ops.buffer_load(
            seq_rsrc, fx.Int32(off_b), vec_width=1, dtype=fx.T.i32()
        )
        seq_end = fx.buffer_ops.buffer_load(
            seq_rsrc, fx.Int32(off_b) + fx.Int32(1), vec_width=1, dtype=fx.T.i32()
        )
        seq_start = fx.rocdl.readfirstlane(fx.T.i32(), seq_start)
        seq_end = fx.rocdl.readfirstlane(fx.T.i32(), seq_end)
        M_b = seq_end - seq_start
        start_m = fx.Int32(block_m_idx) * fx.Int32(BLOCK_M)

        if start_m < M_b:
            a_row_off = fx.Int64(seq_start) * fx.Int64(K)
            c_row_off = fx.Int64(seq_start) * fx.Int64(N)
            A_g = fx.make_view(
                fx.add_offset(fx.get_iter(A), fx.make_int_tuple(a_row_off)),
                fx.get_layout(A),
            )
            C_g = fx.make_view(
                fx.add_offset(fx.get_iter(C), fx.make_int_tuple(c_row_off)),
                fx.get_layout(C),
            )
            b_row_off = fx.Int64(off_b) * fx.Int64(N) * fx.Int64(K)
            B_g = fx.make_view(
                fx.add_offset(fx.get_iter(B), fx.make_int_tuple(b_row_off)),
                fx.get_layout(B),
            )

            A_buf = make_bounded_buffer_tensor(
                A_g, fx.Int64(fx.Int32(M_b) * fx.Int32(K) * fx.Int32(2))
            )
            B_buf = fx.rocdl.make_buffer_tensor(B_g, max_size=True)
            C_buf = make_bounded_buffer_tensor(
                C_g, fx.Int64(fx.Int32(M_b) * fx.Int32(N) * fx.Int32(2))
            )

            gA_k = fx.flat_divide(A_buf, (BLOCK_M, BLOCK_K))[
                None, None, block_m_idx, None
            ]
            gB_k = fx.flat_divide(B_buf, (BLOCK_N, BLOCK_K))[
                None, None, block_n_idx, None
            ]
            gC = fx.flat_divide(C_buf, (BLOCK_M, BLOCK_N))[
                None, None, block_m_idx, block_n_idx
            ]

            bias_elem_off = fx.Int32(off_b) * fx.Int32(N)
            BIAS_g = fx.make_view(
                fx.add_offset(fx.get_iter(BIAS), fx.make_int_tuple(bias_elem_off)),
                fx.get_layout(BIAS),
            )
            BIAS_buf = fx.rocdl.make_buffer_tensor(BIAS_g, max_size=True)
            gBias2d = fx.make_view(
                fx.get_iter(BIAS_buf), fx.make_layout((BLOCK_M, N), (0, 1))
            )
            gBias = fx.flat_divide(gBias2d, (BLOCK_M, BLOCK_N))[
                None, None, 0, block_n_idx
            ]

            thr_mma = tiled_mma.thr_slice(tid)
            thr_copy_g2s_A = tiled_copy_g2s_A.get_slice(tid)

            uni_copy_128b = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
            buffer_copy_128b = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)

            thr_copy_s2r_A = fx.make_tiled_copy_A(
                buffer_copy_128b, tiled_mma
            ).get_slice(tid)
            thr_copy_g2r_B = fx.make_tiled_copy_B(
                buffer_copy_128b, tiled_mma
            ).get_slice(tid)

            composed_layout_A = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 3, 3)),
                fx.make_ordered_layout((BLOCK_M, BLOCK_K, STAGES_A), (1, 0, 2)),
            )
            sA = fx.make_view(fx.get_dyn_shared(fx.BFloat16), composed_layout_A)

            thr_gA_k = thr_copy_g2s_A.partition_S(gA_k)
            thr_sA = thr_copy_g2s_A.partition_D(sA)
            thr_sA_s2r = thr_copy_s2r_A.partition_S(sA)
            thr_gB_k = thr_copy_g2r_B.partition_S(gB_k)

            copy_frag_A = fx.make_fragment_like(thr_sA[None, None, None, 0])

            mma_frag_A = thr_mma.make_fragment_A(sA[None, None, 0])
            mma_frag_B = thr_mma.make_fragment_B(gB_k, stages=B_STAGES)
            mma_frag_C = thr_mma.make_fragment_C(gC)

            mma_frag_A_retile = thr_copy_s2r_A.retile(mma_frag_A)
            mma_frag_B_retile = thr_copy_g2r_B.retile(mma_frag_B)

            gA_k_stride = fx.get_scalar(gA_k.stride[2])
            gB_k_stride = fx.get_scalar(gB_k.stride[2])

            def run_pipeline_stage(
                a_read_stage, b_read_slot, a_next_k=None, b_next_k=None
            ):
                a_write_stage = a_read_stage ^ 1
                if fx.const_expr(a_next_k is not None):
                    fx.copy(
                        buffer_copy_128b,
                        thr_gA_k[None, None, None, 0],
                        copy_frag_A,
                        soffset=fx.Int32(a_next_k) * gA_k_stride,
                    )
                if fx.const_expr(b_next_k is not None):
                    b_fetch_slot = (
                        b_read_slot + (2 if fx.const_expr(B_STAGES > 2) else 1)
                    ) % B_STAGES
                    fx.copy(
                        buffer_copy_128b,
                        thr_gB_k[None, None, None, 0],
                        mma_frag_B_retile[None, None, None, b_fetch_slot],
                        soffset=fx.Int32(b_next_k) * gB_k_stride,
                    )

                for block_k_iter in fx.range_constexpr(BLOCK_K // 32):
                    fx.copy(
                        uni_copy_128b,
                        thr_sA_s2r[None, None, block_k_iter, a_read_stage],
                        mma_frag_A_retile[None, None, block_k_iter],
                    )
                    if fx.const_expr(USE_MFMA_K32):
                        frag_A_k = mma_frag_A[None, None, block_k_iter]
                        frag_B_k = mma_frag_B[None, None, block_k_iter, b_read_slot]
                    else:
                        frag_A_k = mma_frag_A[None, None, (None, block_k_iter)]
                        frag_B_k = mma_frag_B[
                            None, None, (None, block_k_iter), b_read_slot
                        ]
                    fx.gemm(
                        tiled_mma,
                        mma_frag_C,
                        frag_A_k,
                        frag_B_k,
                        mma_frag_C,
                        traversal_order=fx.GemmTraversalOrder.KNM,
                    )

                fx.copy(
                    uni_copy_128b, copy_frag_A, thr_sA[None, None, None, a_write_stage]
                )
                fx.gpu.barrier()

            n_tiles = K // BLOCK_K
            fx.copy(buffer_copy_128b, thr_gA_k[None, None, None, 0], copy_frag_A)
            fx.copy(
                buffer_copy_128b,
                thr_gB_k[None, None, None, 0],
                mma_frag_B_retile[None, None, None, 0],
            )
            if fx.const_expr(B_STAGES > 2 and n_tiles > 1):
                fx.copy(
                    buffer_copy_128b,
                    thr_gB_k[None, None, None, 0],
                    mma_frag_B_retile[None, None, None, 1],
                    soffset=fx.Int32(1) * gB_k_stride,
                )
            mma_frag_C.fill(0)
            fx.copy(uni_copy_128b, copy_frag_A, thr_sA[None, None, None, 0])
            fx.gpu.barrier()

            for k_tile in fx.range_constexpr(n_tiles):
                a_slot = k_tile % 2
                b_slot = k_tile % B_STAGES
                a_next = k_tile + 1 if k_tile + 1 < n_tiles else None
                b_next = (
                    (k_tile + B_STAGES - 1)
                    if (k_tile + B_STAGES - 1) < n_tiles
                    else None
                )
                run_pipeline_stage(a_slot, b_slot, a_next_k=a_next, b_next_k=b_next)

            mma_frag_C_bf16 = fx.make_fragment_like(mma_frag_C, fx.BFloat16.ir_type)
            thr_copy_r2s_C = fx.make_tiled_copy_C(
                fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16), tiled_mma
            ).get_slice(tid)
            mma_frag_C_retile = thr_copy_r2s_C.retile(mma_frag_C_bf16)

            thr_gBias = thr_copy_r2s_C.partition_S(gBias)
            bias_frag = fx.make_fragment_like(thr_gBias)
            fx.copy(
                fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16),
                thr_gBias,
                bias_frag,
            )
            bias_f32 = fx.arith.ExtFOp(
                fx.T.VectorType.get([C_FRAG_LEN], fx.T.f32()), bias_frag.load()
            ).result
            mma_frag_C_bf16.store(
                fx.arith.trunc_f(
                    fx.T.VectorType.get([C_FRAG_LEN], fx.T.bf16()),
                    fx.arith.addf(mma_frag_C.load(), bias_f32),
                )
            )

            sC = fx.make_view(
                fx.get_dyn_shared(fx.BFloat16),
                fx.make_ordered_layout((BLOCK_M, BLOCK_N), (1, 0)),
            )
            thr_sC = thr_copy_r2s_C.partition_D(sC)

            fx.gpu.barrier()
            fx.copy(
                fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16),
                mma_frag_C_retile,
                thr_sC,
            )
            fx.gpu.barrier()

            thr_copy_s2g_C = tiled_copy_s2g_C.get_slice(tid)
            thr_sC_read = thr_copy_s2g_C.partition_S(sC)
            thr_gC = thr_copy_s2g_C.partition_D(gC)
            cs_frag = fx.make_fragment_like(thr_sC_read)
            fx.copy(
                fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16),
                thr_sC_read,
                cs_frag,
            )
            fx.copy(
                fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16),
                cs_frag,
                thr_gC,
            )

    @flyc.jit
    def _launch(
        C: fx.Tensor,
        A: fx.Tensor,
        B: fx.Tensor,
        BIAS: fx.Tensor,
        SEQ_OFFSETS: fx.Tensor,
        TILE_MAP: fx.Tensor,
        total_occ_tiles: int,
        n_groups: int,
        max_seq_len: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        if fx.const_expr(USE_MFMA_K32):
            tiled_mma = fx.make_tiled_mma(
                fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16)),
                fx.make_layout((1, 4, 1), (0, 1, 0)),
                fx.make_tile(None, None, fx.make_layout((8, 4), (1, 8))),
            )
        else:
            n_warps_total = THREADS // 64
            n_warps = min(n_warps_total, BLOCK_N // 16)
            m_warps = n_warps_total // n_warps
            tiled_mma = fx.make_tiled_mma(
                fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
                fx.make_layout((m_warps, n_warps, 1), (1, m_warps, 0)),
                fx.make_tile(None, None, fx.make_layout((4, 4, 2), (1, 8, 4))),
            )
        val_per_thr = 8  # 16B / bf16
        thrs_col = BLOCK_K // val_per_thr
        thrs_row = THREADS // thrs_col
        tiled_copy_g2s_A = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16),
            fx.make_layout(
                ((thrs_col, thrs_row), (1, val_per_thr)),
                ((thrs_row * val_per_thr, 1), (1, thrs_row)),
            ),
            fx.make_tile(thrs_row, BLOCK_K),
        )

        c_val = 8
        c_thrs_n = BLOCK_N // c_val  # threads along N
        c_thrs_m = THREADS // c_thrs_n  # threads along M
        tiled_copy_s2g_C = fx.make_tiled_copy(
            fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16),
            fx.make_layout(
                ((c_thrs_n, c_thrs_m), (1, c_val)),
                ((c_thrs_m * c_val, 1), (1, c_thrs_m)),
            ),
            fx.make_tile(c_thrs_m, BLOCK_N),
        )

        epi_smem_bytes = max(smem_bytes, BLOCK_M * BLOCK_N * 2)

        bm = (max_seq_len + BLOCK_M - 1) // BLOCK_M
        if fx.const_expr(COMPACT):
            grid = (total_occ_tiles, 1, N_BLOCKS)
        else:
            grid = (bm * N_BLOCKS, 1, n_groups)
        if WAVES_PER_EU:
            with CompilationContext.compile_hints({"waves_per_eu": WAVES_PER_EU}):
                jdbba_kernel(
                    C,
                    A,
                    B,
                    BIAS,
                    SEQ_OFFSETS,
                    TILE_MAP,
                    tiled_mma,
                    tiled_copy_g2s_A,
                    tiled_copy_s2g_C,
                ).launch(
                    grid=grid, block=(THREADS, 1, 1), smem=epi_smem_bytes, stream=stream
                )
        else:
            jdbba_kernel(
                C,
                A,
                B,
                BIAS,
                SEQ_OFFSETS,
                TILE_MAP,
                tiled_mma,
                tiled_copy_g2s_A,
                tiled_copy_s2g_C,
            ).launch(
                grid=grid, block=(THREADS, 1, 1), smem=epi_smem_bytes, stream=stream
            )

    return _launch


def _drop_leaked_ir_contexts() -> None:
    try:
        from flydsl._mlir import ir

        while ir.Context.current is not None:
            ir.Context.current.__exit__(None, None, None)
    except Exception:
        pass


def _cache_get(key):
    cf = _COMPILED_CACHE.get(key)
    if cf is not None:
        _COMPILED_CACHE.move_to_end(key)
    return cf


def _cache_put(key, cf):
    _COMPILED_CACHE[key] = cf
    _COMPILED_CACHE.move_to_end(key)
    while len(_COMPILED_CACHE) > _COMPILED_CACHE_MAX:
        _COMPILED_CACHE.popitem(last=False)


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
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    waves_per_eu: int = 0,
    threads: int | None = None,
    tile_map=None,
    total_occ_tiles: int | None = None,
):
    N = B.shape[0] // n_groups
    K = B.shape[1]
    if use_mfma_k32 is None:
        use_mfma_k32 = False
    if block_k is None:
        if use_mfma_k32:
            block_k = 128 if K <= 256 else BLOCK_K
        else:
            block_k = BLOCK_K
    else:
        block_k = int(block_k)
        if block_k <= 0 or block_k % K_TILE_GRAN != 0:
            raise ValueError(
                f"block_k must be a positive multiple of {K_TILE_GRAN}, got {block_k}"
            )
        if not use_mfma_k32 and block_k < 64:
            raise ValueError(f"block_k must be >= 64 for MFMA 16x16x16, got {block_k}")
        a_smem = BLOCK_M * block_k * STAGES_A * 2
        epi_smem = max(a_smem, BLOCK_M * BLOCK_N * 2)
        if epi_smem > LDS_BYTES:
            raise ValueError(
                f"block_k={block_k} needs {epi_smem} bytes LDS (max {LDS_BYTES}); "
                f"A-stage staging is {a_smem} bytes"
            )
        if K % block_k != 0:
            raise ValueError(f"block_k={block_k} must divide K={K}")
    if xcd_c is None:
        if not uniform_seqlen:
            xcd_c = 1
        elif K > 256:
            xcd_c = XCD_C_LARGE_K
        else:
            xcd_c = XCD_C_SMALL_K
    if xcd_w is None:
        xcd_w = 1 if xcd_c == 1 else XCD_W
    bmn = BLOCK_M if block_m is None else block_m
    bnn = BLOCK_N if block_n is None else block_n
    bm = (max_seq_len + bmn - 1) // bmn
    b_stages = 2 if use_mfma_k32 else 3
    nthreads = THREADS if threads is None else threads
    compact = tile_map is not None

    tmap = SEQ_OFFSETS if tile_map is None else tile_map
    tot = 0 if total_occ_tiles is None else int(total_occ_tiles)

    key = (
        N,
        K,
        bmn,
        bnn,
        block_k,
        STAGES_A,
        nthreads,
        bm,
        n_groups,
        tot,
        xcd_c,
        xcd_w,
        use_mfma_k32,
        waves_per_eu,
        b_stages,
        compact,
    )
    launch_args = (C, A, B, BIAS, SEQ_OFFSETS, tmap, tot, n_groups, max_seq_len, stream)

    cached = _cache_get(key)
    if cached is not None:
        return cached(*launch_args)

    launch = _build_launcher(
        N,
        K,
        bmn,
        bnn,
        block_k,
        STAGES_A,
        nthreads,
        bm,
        n_groups,
        xcd_c,
        xcd_w,
        use_mfma_k32,
        waves_per_eu,
        b_stages,
        compact,
    )
    try:
        _cache_put(key, flyc.compile(launch, *launch_args))
        return None
    except Exception:
        # Mirrors moe_kernels._run_compiled: cleanup leaked ir.Context, re-raise.
        _drop_leaked_ir_contexts()
        raise
