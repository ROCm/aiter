# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Spike copy of the 16-row QSA indexer scorer for SILOTIGER-1481 phase 1.

The body is ``build_qsa_indexer_prefill_scores_module`` plus one compile-time
candidate array. The production module is not modified.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import (
    BFloat16,
    Float32,
    Int32,
    Int64,
    const_expr,
    gpu,
    range_constexpr,
)

from aiter.ops.flydsl.kernels.kernels_common import div_nonneg, kernel_signature
from aiter.ops.flydsl.kernels.qsa.indexer_select import (
    _D,
    _H,
    _LDS_ROW_PAD,
    _ROWS_BLOCK_N,
    _SCORE_HEADS,
    _compressed_span,
    _dense_indexer_strides,
    _k_page_row,
    _layout_elems,
    _neg_inf,
    _q_lds_layout,
    _qk_mfma,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_base_i64, buf_copy_atom


def build_qsa_indexer_prefill_scores_lds_probe(
    page_size: int,
    use_k32: bool,
    n_heads: int = _H,
    wide_cache: bool = False,
    k_strides: tuple[int, int, int] | None = None,
    block_n: int = _ROWS_BLOCK_N,
    fold_heads: bool = False,
    candidate_bytes: int = 0,
):
    """Spike copy of the 16-row MFMA scorer with a compile-time candidate array.

    ``candidate_bytes`` is 0, 16384, 32768, or 65536. Zero keeps the production
    LDS struct. A positive size adds an int8 array of that length. Thread 0
    loads both ends and writes one int into ``sink`` at this workgroup's
    slot, so the bytes stay live. The production module is not used for this
    launch.

    One wave per 16 columns, each over the whole head dim, so the four
    heads accumulate in registers and the wave writes its scores directly.
    A row tile may span requests. Q is staged once per row tile; each
    request in the tile gathers its own K tiles and writes only its own
    rows, so a request-major batch gathers each K tile once per request it
    touches. A workgroup walks ``tiles_per_wg`` consecutive
    column tiles, a launch argument, so one Q staging serves several K tiles.

    ``fold_heads`` (4 heads only) puts 4 rows x 4 heads on the MFMA M axis
    instead of 16 rows per head: a tile is 4 rows, one MFMA chain, and each
    lane holds one row's four heads, so the head sum stays in the lane. An
    MTP batch with 4 rows per request then has one request per tile.
    """
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if n_heads not in _SCORE_HEADS:
        raise ValueError(f"score heads must be {_SCORE_HEADS}, got {n_heads}")
    if block_n not in (32, 64):
        raise ValueError(f"16-row scorer block_n must be 32 or 64, got {block_n}")
    if fold_heads and n_heads != 4:
        raise ValueError(f"fold_heads needs 4 heads, got {n_heads}")
    if candidate_bytes not in (0, 16 * 1024, 32 * 1024, 64 * 1024):
        raise ValueError(
            f"candidate_bytes must be 0, 16384, 32768, or 65536, got {candidate_bytes}"
        )
    dense_k_strides = _dense_indexer_strides(page_size)
    if k_strides is None:
        k_strides = dense_k_strides
    else:
        k_strides = tuple(int(s) for s in k_strides)
    strided_wide = wide_cache and k_strides != dense_k_strides

    # Rows per tile; the MFMA M axis is always 16 (rows, or rows x heads).
    block_m = 16 // n_heads if fold_heads else 16
    mfma_chains = 1 if fold_heads else n_heads
    num_waves = block_n // 16
    block_threads = 64 * num_waves
    qk_k = 32 if use_k32 else 16
    qk_vec = qk_k // 4
    k_steps = _D // qk_k
    vec = 8
    vec_chunks = _D // vec
    q_vectors = block_m * n_heads * vec_chunks
    k_vectors = block_n * vec_chunks
    if q_vectors % block_threads or k_vectors % block_threads:
        raise ValueError("prefill Q/K vector counts must divide block threads")
    q_vectors_per_thread = q_vectors // block_threads
    k_vectors_per_thread = k_vectors // block_threads
    q_shape, q_stride = _q_lds_layout(block_m, n_heads, fold_heads)
    k_shape = (block_n, _D)
    k_stride = (_D + _LDS_ROW_PAD, 1)

    def _storage_type():
        if candidate_bytes == 0:

            @fx.struct
            class SharedStorage:
                q: fx.Array[BFloat16, _layout_elems(q_shape, q_stride), 16]
                k: fx.Array[BFloat16, _layout_elems(k_shape, k_stride), 16]
                visible: fx.Array[Int32, block_m, 16]
                req: fx.Array[Int32, block_m, 16]
                live: fx.Array[Int32, block_n, 16]

            return SharedStorage

        @fx.struct
        class SharedStorage:
            q: fx.Array[BFloat16, _layout_elems(q_shape, q_stride), 16]
            k: fx.Array[BFloat16, _layout_elems(k_shape, k_stride), 16]
            visible: fx.Array[Int32, block_m, 16]
            req: fx.Array[Int32, block_m, 16]
            live: fx.Array[Int32, block_n, 16]
            cand: fx.Array[fx.Int8, candidate_bytes, 16]

        return SharedStorage

    SharedStorage = _storage_type()

    @flyc.kernel(
        name="qsa_indexer_rows_scores_probe_"
        + kernel_signature(
            ps=page_size,
            bm=block_m,
            bn=block_n,
            h=n_heads,
            d=_D,
            blk=block_threads,
            qkk=qk_k,
            fh=int(fold_heads),
            wide=int(wide_cache),
            cb=candidate_bytes,
            **({"kvs": "x".join(str(s) for s in k_strides)} if strided_wide else {}),
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def qsa_indexer_prefill_scores_kernel(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        scores: fx.Tensor,
        row_lens: fx.Tensor,
        n_columns: Int32,
        rows: Int32,
        n_req: Int32,
        score_scale: Float32,
        n_cache_blocks: Int32,
        tiles_per_wg: Int32,
        sink: fx.Tensor,
    ):
        # Row tiles vary fastest, so one column group's K is reused across
        # its row tiles. Consecutive workgroup ids are dealt across the 8
        # XCDs, so that reuse is every 8th row tile on each XCD.
        # xcd_remap_bx_by group 8, which would keep a contiguous run of row
        # tiles on one XCD, was slower on gfx950 at M=512, cold, packed and
        # wide. Row-fast then remap, microseconds: L=8192 28.53/29.18 and
        # 28.96/29.44; L=32768 43.75/46.06 and 43.90/46.48. The launch stays
        # (row tiles, column groups). Each workgroup scores tiles_per_wg
        # consecutive column tiles with one Q staging.
        row_tile = fx.block_idx.x
        tile0 = fx.block_idx.y * tiles_per_wg
        tid = fx.thread_idx.x
        zero = Int32(0)
        one = Int32(1)
        page = Int32(page_size)
        wave = div_nonneg(tid, Int32(64))
        lane = tid - wave * Int32(64)
        vec_layout = fx.make_layout(vec, 1)
        g_copy = buf_copy_atom(16, BFloat16)
        # A buffer resource is wave-uniform, so a per-lane wide row would
        # waterfall. The wide gather is a raw 128-bit load instead.
        k_copy = (
            fx.make_copy_atom(fx.UniversalCopy128b(), BFloat16)
            if const_expr(wide_cache)
            else g_copy
        )
        q_buf = fx.rocdl.make_buffer_tensor(q)
        # Same 4 GiB split as the one-row scorer: one whole-cache descriptor,
        # or a separate compile that reads each gathered row raw.
        if const_expr(wide_cache):
            k_base = buf_base_i64(k_cache)
            row_ptr_ty = fx.PointerType.get(
                BFloat16.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=16,
            )
            page_elems64 = Int64(k_strides[0])
            token_elems64 = Int64(k_strides[1])
            head_elems64 = Int64(k_strides[2])

        else:
            k_buf = fx.rocdl.make_buffer_tensor(k_cache)

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        # The size test is compile-time and the lane test is not, so they stay split.
        if const_expr(candidate_bytes != 0):  # noqa: SIM102
            # Both ends, so the array keeps its full size. One store per
            # workgroup keeps that load live without a grid-wide race.
            if tid == zero:
                packed = fx.Int32(storage.cand[0]) + fx.Int32(
                    storage.cand[candidate_bytes - 1]
                )
                slot = fx.block_idx.x + fx.block_idx.y * gpu.grid_dim.x
                sink[slot] = packed
        q_lds = storage.q.view(fx.make_layout(q_shape, q_stride))
        k_lds = storage.k.view(fx.make_layout(k_shape, k_stride))
        visible_lds = storage.visible.view(fx.make_layout(block_m, 1))
        req_lds = storage.req.view(fx.make_layout(block_m, 1))
        live_lds = storage.live.view(fx.make_layout(block_n, 1))
        qk_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, qk_k, BFloat16))
        # One atom per wave. Its M and N projections are the lane's row and
        # column. Folded M is (row, head):(4, 1), so m = 4*row + head and
        # one chain holds four heads of one row.
        thr_qk = fx.make_tiled_mma(
            qk_mma, fx.make_layout((1, 1, 1), (0, 0, 0))
        ).get_slice(lane)
        c_m = thr_qk.partition_C(fx.make_view(0, fx.make_layout((16, 16), (1, 0))))
        c_n = thr_qk.partition_C(fx.make_view(0, fx.make_layout((16, 16), (0, 1))))
        a_m = thr_qk.partition_A(fx.make_view(0, fx.make_layout((16, qk_k), (1, 0))))
        a_k = thr_qk.partition_A(fx.make_view(0, fx.make_layout((16, qk_k), (0, 1))))
        b_n = thr_qk.partition_B(fx.make_view(0, fx.make_layout((16, qk_k), (1, 0))))
        atom_n = Int32(fx.get_scalar(c_n[0]))
        k_col = Int32(fx.get_scalar(b_n[0]))
        if const_expr(fold_heads):
            fold = fx.make_layout((block_m, n_heads), (n_heads, 1))
            a_crd = fx.idx2crd(fx.get_scalar(a_m[0]), fold)
            a_row = Int32(a_crd[0].unpack())
            a_head = Int32(a_crd[1].unpack())
        else:
            a_row = Int32(fx.get_scalar(a_m[0]))
        qk_a = fx.make_rmem_tensor(fx.make_layout(qk_vec, 1), BFloat16)
        qk_b = fx.make_rmem_tensor(fx.make_layout(qk_vec, 1), BFloat16)
        qk_c = fx.make_rmem_tensor(fx.make_layout(4, 1), Float32)

        if tid < Int32(block_m):
            row = row_tile * Int32(block_m) + tid
            row_live = row < rows
            safe_row = row_live.select(row, zero)
            req = token_to_req[safe_row]
            req_live = row_live & (req >= zero) & (req < n_req)
            safe_req = req_live.select(req, zero)
            qpos = query_positions[safe_row]
            slen = context_lens[safe_req]
            visible = fx.min(_compressed_span(qpos, slen), n_columns)
            visible = req_live.select(visible, zero)
            visible_lds[tid] = visible
            req_lds[tid] = req_live.select(req, Int32(-1))
            if (tile0 == zero) & row_live:
                row_lens[row] = visible
        gpu.barrier()
        # Every lane reads the same block_m entries, so these and every branch on
        # them below are workgroup-uniform.
        tile_visible = zero
        req_lo = n_req
        req_hi = Int32(-1)
        for r in range_constexpr(block_m):
            v = visible_lds[Int32(r)]
            rq = req_lds[Int32(r)]
            tile_visible = fx.max(v, tile_visible)
            req_lo = ((rq >= zero) & (rq < req_lo)).select(rq, req_lo)
            req_hi = fx.max(rq, req_hi)

        # As in the one-row scorer: skip tiles no row of this block can see.
        if tile0 * Int32(block_n) < tile_visible:
            # Q is staged once per row tile and reused by every request and tile
            # this workgroup scores: its mask is any live request, not this one.
            # The fill stays a thread-linear copy of the whole (row, head, d)
            # tile, eight bf16 at a time. partition_A is the MFMA A fragment each
            # lane later reads from this LDS, so it does not name this store:
            # storing through it would skip elements that fragment does not own
            # and would retile the global load.
            for qpart_q in range_constexpr(q_vectors_per_thread):
                qlinear_q = tid + Int32(qpart_q * block_threads)
                qrow_local_q = div_nonneg(qlinear_q, Int32(n_heads * vec_chunks))
                qrem_q = qlinear_q - qrow_local_q * Int32(n_heads * vec_chunks)
                qhead_q = div_nonneg(qrem_q, Int32(vec_chunks))
                qd_chunk_q = qrem_q - qhead_q * Int32(vec_chunks)
                qrow_q = row_tile * Int32(block_m) + qrow_local_q
                qrow_live_q = qrow_q < rows
                qsafe_row_q = qrow_live_q.select(qrow_q, zero)
                qreq_live_q = req_lds[qrow_local_q] >= zero
                qs_row_q = fx.logical_divide(
                    fx.slice(q_buf, (qsafe_row_q, qhead_q, None)), vec_layout
                )
                qs_src_q = fx.slice(qs_row_q, (None, qd_chunk_q))
                qs_frag_q = fx.make_fragment_like(qs_src_q)
                fx.copy(g_copy, qs_src_q, qs_frag_q)
                qs_vec_q = fx.Vector(fx.memref_load_vec(qs_frag_q))
                qd0_q = qd_chunk_q * Int32(vec)
                for iq_q in range_constexpr(vec):
                    qv_q = qreq_live_q.select(qs_vec_q[iq_q].to(Float32), Float32(0.0))
                    q_lds[qrow_local_q, qhead_q, qd0_q + Int32(iq_q)] = qv_q.to(
                        BFloat16
                    )
            gpu.barrier()
            # Names assigned in this loop are its own: a name it shares with
            # the code above would become an scf.for iter_arg.
            for cur_req in range(req_lo, req_hi + one, one):
                cur = Int32(cur_req)
                req_visible = zero
                for r_r in range_constexpr(block_m):
                    v_r = (req_lds[Int32(r_r)] == cur).select(
                        visible_lds[Int32(r_r)], zero
                    )
                    req_visible = fx.max(v_r, req_visible)
                if tile0 * Int32(block_n) < req_visible:
                    tile_end = div_nonneg(
                        req_visible + Int32(block_n - 1), Int32(block_n)
                    )
                    tile_end = fx.min(tile_end, tile0 + tiles_per_wg)
                    # The tile loop's names carry _t, for the same reason.
                    for cur_tile in range(tile0, tile_end, one):
                        tile_t = Int32(cur_tile)
                        for part_t in range_constexpr(k_vectors_per_thread):
                            linear_t = tid + Int32(part_t * block_threads)
                            col_t = div_nonneg(linear_t, Int32(vec_chunks))
                            d_chunk_t = linear_t - col_t * Int32(vec_chunks)
                            score_col_t = tile_t * Int32(block_n) + col_t
                            # req_visible is within n_columns and the context.
                            col_live_t = score_col_t < req_visible
                            safe_col_t = col_live_t.select(score_col_t, zero)
                            logical_page_t = div_nonneg(safe_col_t, page)
                            off_t = safe_col_t - logical_page_t * page
                            phys_t = page_table[cur, logical_page_t]
                            phys_live_t = (phys_t >= zero) & (phys_t < n_cache_blocks)
                            safe_phys_t = phys_live_t.select(phys_t, zero)
                            col_live_t = col_live_t & phys_live_t
                            if const_expr(wide_cache):
                                k_row_t = _k_page_row(
                                    k_base,
                                    safe_phys_t,
                                    off_t,
                                    page_elems64,
                                    token_elems64,
                                    head_elems64,
                                    zero,
                                    row_ptr_ty,
                                    vec_layout,
                                )
                            else:
                                k_row_t = fx.logical_divide(
                                    fx.slice(k_buf, (safe_phys_t, off_t, zero, None)),
                                    vec_layout,
                                )
                            k_src_t = fx.slice(k_row_t, (None, d_chunk_t))
                            k_frag_t = fx.make_fragment_like(k_src_t)
                            fx.copy(k_copy, k_src_t, k_frag_t)
                            k_vec_t = fx.Vector(fx.memref_load_vec(k_frag_t))
                            d0_t = d_chunk_t * Int32(vec)
                            for i_t in range_constexpr(vec):
                                kv_t = col_live_t.select(
                                    k_vec_t[i_t].to(Float32), Float32(0.0)
                                )
                                k_lds[col_t, d0_t + Int32(i_t)] = kv_t.to(BFloat16)
                            # Every d_chunk of a column stores the same flag.
                            live_lds[col_t] = col_live_t.select(one, zero)
                        gpu.barrier()

                        # This wave's 16 columns against the tile: 16 rows per
                        # head, or 4 rows x 4 heads in one chain when folded.
                        # K's column is the B atom's N. The score column is
                        # the C atom's N. Both are this lane's column.
                        n_row_t = wave * Int32(16) + k_col
                        out_n_t = wave * Int32(16) + atom_n
                        accs_t = []
                        for head_t in range_constexpr(mfma_chains):
                            acc4_t = fx.Vector.filled(4, 0.0, Float32)
                            a_head_t = a_head if const_expr(fold_heads) else head_t
                            for ks_t in range_constexpr(k_steps):
                                dk_t = Int32(ks_t * qk_k)
                                a_vec_t = fx.Vector.from_elements(
                                    [
                                        q_lds[
                                            a_row,
                                            a_head_t,
                                            dk_t + Int32(fx.get_scalar(a_k[i_t])),
                                        ]
                                        for i_t in range_constexpr(qk_vec)
                                    ],
                                    BFloat16,
                                )
                                b_vec_t = fx.Vector.from_elements(
                                    [
                                        k_lds[
                                            n_row_t,
                                            dk_t + Int32(fx.get_scalar(a_k[i_t])),
                                        ]
                                        for i_t in range_constexpr(qk_vec)
                                    ],
                                    BFloat16,
                                )
                                acc4_t = fx.Vector(
                                    _qk_mfma(
                                        qk_mma,
                                        qk_a,
                                        qk_b,
                                        qk_c,
                                        a_vec_t,
                                        b_vec_t,
                                        acc4_t,
                                    )
                                )
                            accs_t.append(acc4_t)

                        out_col_t = tile_t * Int32(block_n) + out_n_t
                        # A visible column on an invalid page scored a zeroed K;
                        # it must not tie into the top-512.
                        page_live_t = live_lds[n_row_t] != zero
                        if const_expr(fold_heads):
                            # The four C values are the four heads of one row.
                            row_local_t = Int32(
                                fx.idx2crd(fx.get_scalar(c_m[0]), fold)[0].unpack()
                            )
                            row_t = row_tile * Int32(block_m) + row_local_t
                            score_t = Float32(0.0)
                            for i_t in range_constexpr(4):
                                score_t = score_t + fx.max(accs_t[0][i_t], Float32(0.0))
                            # Rows of other requests are written by their own
                            # iteration.
                            mine_t = req_lds[row_local_t] == cur
                            if mine_t & (row_t < rows) & (out_col_t < n_columns):
                                live_t = (
                                    out_col_t < visible_lds[row_local_t]
                                ) & page_live_t
                                # ``scores`` must keep a dynamic row stride:
                                # a static layout, use_32bit_stride=True, or
                                # an Int32 linear index would wrap.
                                scores[row_t, out_col_t] = live_t.select(
                                    score_t * score_scale, _neg_inf()
                                )
                        else:
                            for i_t in range_constexpr(4):
                                row_local_t = Int32(fx.get_scalar(c_m[i_t]))
                                row_t = row_tile * Int32(block_m) + row_local_t
                                score_t = Float32(0.0)
                                for head_t in range_constexpr(n_heads):
                                    score_t = score_t + fx.max(
                                        accs_t[head_t][i_t], Float32(0.0)
                                    )
                                # Rows of other requests are written by their
                                # own iteration.
                                mine_t = req_lds[row_local_t] == cur
                                if mine_t & (row_t < rows) & (out_col_t < n_columns):
                                    live_t = (
                                        out_col_t < visible_lds[row_local_t]
                                    ) & page_live_t
                                    # ``scores`` must keep a dynamic row stride:
                                    # a static layout, use_32bit_stride=True, or
                                    # an Int32 linear index would wrap.
                                    scores[row_t, out_col_t] = live_t.select(
                                        score_t * score_scale, _neg_inf()
                                    )
                        # The next tile or request overwrites k_lds and live_lds.
                        gpu.barrier()

    @flyc.jit
    def launch_qsa_indexer_prefill_scores(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        scores: fx.Tensor,
        row_lens: fx.Tensor,
        n_columns: Int32,
        rows: Int32,
        n_req: Int32,
        score_scale: Float32,
        n_cache_blocks: Int32,
        row_tiles: Int32,
        tile_groups: Int32,
        tiles_per_wg: Int32,
        sink: fx.Tensor,
        stream: fx.Stream,
    ):
        qsa_indexer_prefill_scores_kernel(
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            scores,
            row_lens,
            n_columns,
            rows,
            n_req,
            score_scale,
            n_cache_blocks,
            tiles_per_wg,
            sink,
        ).launch(
            grid=(row_tiles, tile_groups, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_qsa_indexer_prefill_scores


build_qsa_indexer_prefill_scores_lds_probe = functools.lru_cache(maxsize=16)(
    build_qsa_indexer_prefill_scores_lds_probe
)
