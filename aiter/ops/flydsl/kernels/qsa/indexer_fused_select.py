# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused 16-row indexer score and split-local top-512.

One workgroup owns one 16-row tile and one column span. It stages Q once,
scores the span with the same bf16 MFMA and 4-head ReLU sum as the prefill
scorer, and keeps each row's top 512. Comparison uses the fp32 score. The
workspace record is the Phase 1 layout: a 16-bit order-preserving key and a
16-bit column offset within the span. The fp32 scores stay in the 32 KiB LDS
array so the 16-bit key is not the selection predicate. Offsets live in the
global candidate buffer until the epilogue packs them with the key.

The first-level radix histogram is 256 bins, the top 8 bits of that 16-bit
key. Inserts and evictions update it, so a bin count is the number of survivors
in the row's top 512. Invalid, incomplete, and missing-page columns are not
inserted. gfx950 only: the tile plus the 32 KiB scores is 71328 bytes.
"""

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
    _K,
    _LDS_ROW_PAD,
    _ROWS_BLOCK_N,
    _compressed_span,
    _dense_indexer_strides,
    _k_page_row,
    _layout_elems,
    _neg_inf,
    _q_lds_layout,
    _qk_mfma,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_base_i64, buf_copy_atom

# Trimmed 128K width. A second span starts here at 256K. Each span stays
# under 65536 columns, so the offset fits in 16 bits.
FUSED_SPLIT_COLUMNS = 32928
HIST_BINS = 256
_INVALID_PACKED = -1


def fused_split_spans(
    max_seq_len: int,
    n_columns: int,
    page_size: int,
    compress_ratio: int = 4,
) -> list[tuple[int, int]]:
    """Column spans for one step. Each is ``[begin, end)`` within the trimmed table."""
    if max_seq_len < 1 or n_columns < 1 or page_size < 1 or compress_ratio < 1:
        raise ValueError(
            f"split plan needs positive sizes, got max_seq_len={max_seq_len} "
            f"n_columns={n_columns} page_size={page_size} compress_ratio={compress_ratio}"
        )
    page_tokens = page_size * compress_ratio
    pages = (max_seq_len + page_tokens - 1) // page_tokens
    trimmed = min(pages * page_size, n_columns)
    begin = 0
    spans = []
    while begin < trimmed:
        end = min(begin + FUSED_SPLIT_COLUMNS, trimmed)
        spans.append((begin, end))
        begin = end
    return spans


def _score_key16(score):
    """High 16 bits of an order-preserving fp32 key.

    Positive scores, which are the only ones inserted, keep fp32 order in
    these bits. The local top-512 compares the full score.
    """
    bits = score.bitcast(Int32)
    ordered = bits ^ ((bits >> Int32(31)) & Int32(0x7FFFFFFF))
    return ordered >> Int32(16)


def _hist_add(histogram, row, score, delta):
    """Add ``delta`` into the first-level bin of ``score``'s key."""
    bin_id = _score_key16(score) >> Int32(8)
    histogram[row, bin_id] = histogram[row, bin_id] + delta


def build_qsa_indexer_fused_select_module(
    page_size: int,
    use_k32: bool,
    wide_cache: bool = False,
    k_strides: tuple[int, int, int] | None = None,
):
    """Build the fused scorer. One launch scores one column span."""
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    n_heads = _H
    block_n = _ROWS_BLOCK_N
    fold_heads = False
    dense_k_strides = _dense_indexer_strides(page_size)
    if k_strides is None:
        k_strides = dense_k_strides
    else:
        k_strides = tuple(int(s) for s in k_strides)
    strided_wide = wide_cache and k_strides != dense_k_strides

    block_m = 16
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
        raise ValueError("fused Q/K vector counts must divide block threads")
    if block_threads != HIST_BINS:
        raise ValueError(f"fused histogram init needs {HIST_BINS} threads")
    q_vectors_per_thread = q_vectors // block_threads
    k_vectors_per_thread = k_vectors // block_threads
    q_shape, q_stride = _q_lds_layout(block_m, n_heads, fold_heads)
    k_shape = (block_n, _D)
    k_stride = (_D + _LDS_ROW_PAD, 1)
    tile_score_elems = block_m * block_n
    tile_reps = tile_score_elems // block_threads

    @fx.struct
    class SharedStorage:
        q: fx.Array[BFloat16, _layout_elems(q_shape, q_stride), 16]
        k: fx.Array[BFloat16, _layout_elems(k_shape, k_stride), 16]
        visible: fx.Array[Int32, block_m, 16]
        req: fx.Array[Int32, block_m, 16]
        live: fx.Array[Int32, block_n, 16]
        # fp32 comparison state for the 16-row top-512. 32 KiB.
        cand: fx.Array[Float32, block_m * _K, 16]
        count: fx.Array[Int32, block_m, 16]
        tile_scores: fx.Array[Float32, tile_score_elems, 16]

    @flyc.kernel(
        name="qsa_indexer_fused_select_"
        + kernel_signature(
            ps=page_size,
            bm=block_m,
            bn=block_n,
            h=n_heads,
            d=_D,
            blk=block_threads,
            qkk=qk_k,
            wide=int(wide_cache),
            **({"kvs": "x".join(str(s) for s in k_strides)} if strided_wide else {}),
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def qsa_indexer_fused_select_kernel(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        candidates: fx.Tensor,
        histogram: fx.Tensor,
        n_columns: Int32,
        rows: Int32,
        n_req: Int32,
        score_scale: Float32,
        n_cache_blocks: Int32,
        split_begin: Int32,
        split_end: Int32,
    ):
        row_tile = fx.block_idx.x
        tile0 = div_nonneg(split_begin, Int32(block_n))
        # Last tile that overlaps the span, exclusive. ``split_begin`` may sit
        # mid-tile; columns before it are masked when the score is stored.
        tile_limit = div_nonneg(split_end + Int32(block_n - 1), Int32(block_n))
        tid = fx.thread_idx.x
        zero = Int32(0)
        one = Int32(1)
        page = Int32(page_size)
        wave = div_nonneg(tid, Int32(64))
        lane = tid - wave * Int32(64)
        vec_layout = fx.make_layout(vec, 1)
        g_copy = buf_copy_atom(16, BFloat16)
        k_copy = (
            fx.make_copy_atom(fx.UniversalCopy128b(), BFloat16)
            if const_expr(wide_cache)
            else g_copy
        )
        q_buf = fx.rocdl.make_buffer_tensor(q)
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
        q_lds = storage.q.view(fx.make_layout(q_shape, q_stride))
        k_lds = storage.k.view(fx.make_layout(k_shape, k_stride))
        visible_lds = storage.visible.view(fx.make_layout(block_m, 1))
        req_lds = storage.req.view(fx.make_layout(block_m, 1))
        live_lds = storage.live.view(fx.make_layout(block_n, 1))
        cand_lds = storage.cand.view(fx.make_layout((block_m, _K), (_K, 1)))
        count_lds = storage.count.view(fx.make_layout(block_m, 1))
        tile_scores = storage.tile_scores.view(
            fx.make_layout((block_m, block_n), (block_n, 1))
        )
        qk_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, qk_k, BFloat16))
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
        a_row = Int32(fx.get_scalar(a_m[0]))
        qk_a = fx.make_rmem_tensor(fx.make_layout(qk_vec, 1), BFloat16)
        qk_b = fx.make_rmem_tensor(fx.make_layout(qk_vec, 1), BFloat16)
        qk_c = fx.make_rmem_tensor(fx.make_layout(4, 1), Float32)
        neg = _neg_inf()

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
            for slot in range(Int32(_K)):
                cand_lds[tid, Int32(slot)] = neg
            count_lds[tid] = zero
        # Every live row starts empty. The histogram bins are exactly the
        # workgroup, so each thread clears one bin of every row in the tile.
        for r in range_constexpr(block_m):
            row_g = row_tile * Int32(block_m) + Int32(r)
            if row_g < rows:
                histogram[row_g, tid] = zero
                candidates[row_g, tid] = Int32(_INVALID_PACKED)
                candidates[row_g, tid + Int32(block_threads)] = Int32(_INVALID_PACKED)
        gpu.barrier()

        tile_visible = zero
        req_lo = n_req
        req_hi = Int32(-1)
        for r in range_constexpr(block_m):
            v = visible_lds[Int32(r)]
            rq = req_lds[Int32(r)]
            tile_visible = fx.max(v, tile_visible)
            req_lo = ((rq >= zero) & (rq < req_lo)).select(rq, req_lo)
            req_hi = fx.max(rq, req_hi)

        if tile0 * Int32(block_n) < tile_visible:
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
                    tile_end = fx.min(tile_end, tile_limit)
                    for cur_tile in range(tile0, tile_end, one):
                        tile_t = Int32(cur_tile)
                        for zpart in range_constexpr(tile_reps):
                            zlinear = tid + Int32(zpart * block_threads)
                            zrow = div_nonneg(zlinear, Int32(block_n))
                            zcol = zlinear - zrow * Int32(block_n)
                            tile_scores[zrow, zcol] = neg
                        for part_t in range_constexpr(k_vectors_per_thread):
                            linear_t = tid + Int32(part_t * block_threads)
                            col_t = div_nonneg(linear_t, Int32(vec_chunks))
                            d_chunk_t = linear_t - col_t * Int32(vec_chunks)
                            score_col_t = tile_t * Int32(block_n) + col_t
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
                            live_lds[col_t] = col_live_t.select(one, zero)
                        gpu.barrier()

                        n_row_t = wave * Int32(16) + k_col
                        out_n_t = wave * Int32(16) + atom_n
                        accs_t = []
                        for head_t in range_constexpr(n_heads):
                            acc4_t = fx.Vector.filled(4, 0.0, Float32)
                            for ks_t in range_constexpr(k_steps):
                                dk_t = Int32(ks_t * qk_k)
                                a_vec_t = fx.Vector.from_elements(
                                    [
                                        q_lds[
                                            a_row,
                                            head_t,
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
                        page_live_t = live_lds[n_row_t] != zero
                        in_split_t = (out_col_t >= split_begin) & (
                            out_col_t < split_end
                        )
                        for i_t in range_constexpr(4):
                            row_local_t = Int32(fx.get_scalar(c_m[i_t]))
                            row_t = row_tile * Int32(block_m) + row_local_t
                            score_t = Float32(0.0)
                            for head_t in range_constexpr(n_heads):
                                score_t = score_t + fx.max(
                                    accs_t[head_t][i_t], Float32(0.0)
                                )
                            mine_t = req_lds[row_local_t] == cur
                            kept_t = neg
                            if mine_t & (row_t < rows) & (out_col_t < n_columns):
                                live_t = (
                                    (out_col_t < visible_lds[row_local_t])
                                    & page_live_t
                                    & in_split_t
                                )
                                kept_t = live_t.select(score_t * score_scale, neg)
                            tile_scores[row_local_t, out_n_t] = kept_t
                        gpu.barrier()

                        # One thread per row inserts this tile. A smaller
                        # offset wins an exact fp32 tie, matching the oracle's
                        # stable argsort.
                        if tid < Int32(block_m):
                            row_i = row_tile * Int32(block_m) + tid
                            if row_i < rows:
                                for c in range(Int32(block_n)):
                                    c_i = Int32(c)
                                    sc = tile_scores[tid, c_i]
                                    finite = sc > Float32(-1.0e20)
                                    if finite:
                                        offset = (
                                            tile_t * Int32(block_n) + c_i - split_begin
                                        )
                                        cnt = count_lds[tid]
                                        if cnt < Int32(_K):
                                            cand_lds[tid, cnt] = sc
                                            candidates[row_i, cnt] = offset
                                            count_lds[tid] = cnt + one
                                            _hist_add(histogram, row_i, sc, one)
                                        else:
                                            worst_score = cand_lds[tid, zero]
                                            worst_off = candidates[row_i, zero]
                                            worst_slot = zero
                                            for slot in range(one, Int32(_K)):
                                                slot_i = Int32(slot)
                                                slot_score = cand_lds[tid, slot_i]
                                                slot_off = candidates[row_i, slot_i]
                                                worse = (slot_score < worst_score) | (
                                                    (slot_score == worst_score)
                                                    & (slot_off > worst_off)
                                                )
                                                worst_score = worse.select(
                                                    slot_score, worst_score
                                                )
                                                worst_off = worse.select(
                                                    slot_off, worst_off
                                                )
                                                worst_slot = worse.select(
                                                    slot_i, worst_slot
                                                )
                                            better = (sc > worst_score) | (
                                                (sc == worst_score)
                                                & (offset < worst_off)
                                            )
                                            if better:
                                                _hist_add(
                                                    histogram,
                                                    row_i,
                                                    worst_score,
                                                    Int32(-1),
                                                )
                                                cand_lds[tid, worst_slot] = sc
                                                candidates[row_i, worst_slot] = offset
                                                _hist_add(histogram, row_i, sc, one)
                        gpu.barrier()

        if tid < Int32(block_m):
            row_p = row_tile * Int32(block_m) + tid
            if row_p < rows:
                cnt_p = count_lds[tid]
                for slot_p in range(Int32(_K)):
                    slot_i = Int32(slot_p)
                    filled = slot_i < cnt_p
                    key = _score_key16(cand_lds[tid, slot_i])
                    offset_p = candidates[row_p, slot_i]
                    packed = (key << Int32(16)) | (offset_p & Int32(0xFFFF))
                    candidates[row_p, slot_i] = filled.select(
                        packed, Int32(_INVALID_PACKED)
                    )

    @flyc.jit
    def launch_qsa_indexer_fused_select(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        candidates: fx.Tensor,
        histogram: fx.Tensor,
        n_columns: Int32,
        rows: Int32,
        n_req: Int32,
        score_scale: Float32,
        n_cache_blocks: Int32,
        split_begin: Int32,
        split_end: Int32,
        row_tiles: Int32,
        stream: fx.Stream,
    ):
        qsa_indexer_fused_select_kernel(
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            candidates,
            histogram,
            n_columns,
            rows,
            n_req,
            score_scale,
            n_cache_blocks,
            split_begin,
            split_end,
        ).launch(
            grid=(row_tiles, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_qsa_indexer_fused_select
