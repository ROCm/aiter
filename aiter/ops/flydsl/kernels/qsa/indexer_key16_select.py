# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Top 512 from the 16-row scorer's int16 keys, with an fp32 boundary rescore.

One workgroup owns one row. It stages the row's visible keys in LDS while it
counts their high byte, counts the low byte inside the bin that holds the
512th key, and so finds the boundary key. Every key strictly above it is
kept. The boundary bucket is rescored from compressed K with the oracle's
formula, bf16 in and fp32 accumulate, and the best of it fills the rest.
A bucket that fits ``BUCKET_CAPACITY`` is ranked from LDS; a wider one runs
an fp32 radix select that recomputes each score per pass. Exact ties at the
512th score may resolve to either block. A row with at most 512 live keys
keeps all of them. Block order within a row is unspecified, and unused slots
are -1. gfx950 only: the key row alone is up to 128.6 KiB of LDS.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import (
    BFloat16,
    Float32,
    Int16,
    Int32,
    Int64,
    const_expr,
    gpu,
    range_constexpr,
)
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.kernels_common import (
    atomic_add_i32,
    div_nonneg,
    kernel_signature,
)
from aiter.ops.flydsl.kernels.qsa.indexer_select import (
    _D,
    _K,
    _compressed_span,
    _dense_indexer_strides,
    _k_page_row,
    _neg_inf,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_base_i64, buf_copy_atom

BUCKET_CAPACITY = 1024
_THREADS = 512
_WAVE = 64
_BINS = 256
_LDS_BYTES = 160 * 1024
# Metadata slots.
_M_BIN = 0
_M_HIGHER = 1
_M_COUNT = 2
_M_OUT = 3
_M_BUCKET = 4
_M_TIES = 5
_M_PREFIX = 6
_M_NEED = 7
_M_SLOTS = 8
_SIGN = -2147483648


def key16_columns_capacity(n_columns: int) -> int:
    """LDS key slots for a table of ``n_columns``, a whole number of vectors."""
    vec = _key_vec(n_columns)
    return (n_columns + vec - 1) // vec * vec


def _key_vec(n_columns: int) -> int:
    for vec in (8, 4, 2):
        if n_columns % vec == 0:
            return vec
    return 1


def key16_select_lds_bytes(n_columns: int, n_heads: int) -> int:
    """Static LDS of one workgroup, before the scan's scratch."""
    return (
        key16_columns_capacity(n_columns) * 2
        + 2 * _BINS * 4
        + _K * 4
        + BUCKET_CAPACITY * 8
        + n_heads * _D * 4
        + _M_SLOTS * 4
    )


def build_qsa_indexer_key16_select_module(
    page_size: int,
    n_columns: int,
    n_heads: int,
    wide_cache: bool = False,
    k_strides: tuple[int, int, int] | None = None,
):
    """Build the key select for one table width. One launch selects every row."""
    if page_size < 1 or n_columns < 1:
        raise ValueError(
            f"key select needs positive sizes, got page_size={page_size} "
            f"n_columns={n_columns}"
        )
    if n_heads not in (4, 8):
        raise ValueError(f"key select takes 4 or 8 heads, got {n_heads}")
    if _THREADS != _K:
        raise ValueError("one thread per output slot")
    lds_bytes = key16_select_lds_bytes(n_columns, n_heads)
    # The block scan's scratch is a few words per wave.
    if lds_bytes + 1024 > _LDS_BYTES:
        raise ValueError(
            f"{n_columns} columns need {lds_bytes} bytes of key-select LDS, "
            f"over {_LDS_BYTES}"
        )
    dense_k_strides = _dense_indexer_strides(page_size)
    if k_strides is None:
        k_strides = dense_k_strides
    else:
        k_strides = tuple(int(s) for s in k_strides)
    strided_wide = wide_cache and k_strides != dense_k_strides
    cap_columns = key16_columns_capacity(n_columns)
    vec = _key_vec(n_columns)
    k_vec = 8
    k_chunks = _D // k_vec
    q_vectors = n_heads * k_chunks
    if q_vectors > _THREADS:
        raise ValueError("one Q vector per thread must cover every head")
    block_scan = fx.coop.BlockScan[Int32, _THREADS]

    @fx.struct
    class SharedStorage:
        keys: fx.Array[Int16, cap_columns, 16]
        high: fx.Array[Int32, _BINS, 16]
        low: fx.Array[Int32, _BINS, 16]
        out: fx.Array[Int32, _K, 16]
        b_col: fx.Array[Int32, BUCKET_CAPACITY, 16]
        b_score: fx.Array[Float32, BUCKET_CAPACITY, 16]
        q: fx.Array[Float32, n_heads * _D, 16]
        meta: fx.Array[Int32, _M_SLOTS, 16]

    @flyc.kernel(
        name="qsa_indexer_key16_select_"
        + kernel_signature(
            ps=page_size,
            cols=cap_columns,
            h=n_heads,
            d=_D,
            blk=_THREADS,
            cap=BUCKET_CAPACITY,
            wide=int(wide_cache),
            **({"kvs": "x".join(str(s) for s in k_strides)} if strided_wide else {}),
        ),
        known_block_size=[_THREADS, 1, 1],
    )
    def qsa_indexer_key16_select_kernel(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        keys: fx.Tensor,
        block_ids: fx.Tensor,
        n_req: Int32,
        score_scale: Float32,
        n_cache_blocks: Int32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % Int32(_WAVE)
        zero = Int32(0)
        one = Int32(1)
        neg1 = Int32(-1)
        budget = Int32(_K)
        threads = Int32(_THREADS)
        page = Int32(page_size)
        vec_layout = fx.make_layout(vec, 1)
        k_layout = fx.make_layout(k_vec, 1)
        key_copy = buf_copy_atom(vec * 2, Int16)
        g_copy = buf_copy_atom(16, BFloat16)
        k_copy = (
            fx.make_copy_atom(fx.UniversalCopy128b(), BFloat16)
            if const_expr(wide_cache)
            else g_copy
        )
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
        q_buf = fx.rocdl.make_buffer_tensor(q)
        keys_buf = fx.rocdl.make_buffer_tensor(keys)

        smem = fx.SharedAllocator()
        storage = smem.allocate(SharedStorage).peek()
        scan = smem.allocate(block_scan.SharedStorage).peek()
        keys_lds = storage.keys.view(fx.make_layout(cap_columns, 1))
        high = storage.high.view(fx.make_layout(_BINS, 1))
        low = storage.low.view(fx.make_layout(_BINS, 1))
        out = storage.out.view(fx.make_layout(_K, 1))
        b_col = storage.b_col.view(fx.make_layout(BUCKET_CAPACITY, 1))
        b_score = storage.b_score.view(fx.make_layout(BUCKET_CAPACITY, 1))
        q_lds = storage.q.view(fx.make_layout(n_heads * _D, 1))
        meta = storage.meta.view(fx.make_layout(_M_SLOTS, 1))

        req = token_to_req[row]
        req_live = (req >= zero) & (req < n_req)
        safe_req = req_live.select(req, zero)
        slen = context_lens[safe_req]
        qpos = query_positions[row]
        n_vis = fx.min(_compressed_span(qpos, slen), Int32(n_columns))
        n_vis = req_live.select(n_vis, zero)
        n_vec = div_nonneg(n_vis + Int32(vec - 1), Int32(vec))

        def hist_add(histogram, bucket):
            """Add one to ``bucket``. Lanes on the wave's first bucket add once."""
            first = Int32(fx.rocdl.readfirstlane(T.i32, bucket))
            matching = fx.Uint64(fx.rocdl.ballot(T.i64, bucket == first))
            leader = Int32(fx.rocdl.readfirstlane(T.i32, lane))
            count = Int32(fx.ctpop(matching))
            if (bucket != first) | (lane == leader):
                atomic_add_i32(
                    histogram, (bucket == first).select(count, one), bucket, "workgroup"
                )

        def find_crossing(histogram, need):
            """Bin holding the ``need``-th entry, counted from the top.

            ``hit`` is true on the one thread that owns that bin. ``higher`` is
            the count strictly above it. The caller publishes both.
            """
            has_bin = tid < Int32(_BINS)
            bin_id = Int32(_BINS - 1) - tid
            count = has_bin.select(histogram[has_bin.select(bin_id, zero)], zero)
            higher, total = block_scan.exclusive_with_aggregate(
                count, fx.ReductionOp.ADD, storage=scan
            )
            hit = has_bin & (higher < need) & (higher + count >= need)
            return hit, bin_id, higher, count, total

        def row_key(col):
            return keys_lds[col].to(Int32)

        def rescore(col):
            """Oracle score of block ``col``: fp32 ReLU head sum of bf16 Q.K."""
            logical = div_nonneg(col, page)
            off = col - logical * page
            phys = page_table[safe_req, logical]
            phys_live = (phys >= zero) & (phys < n_cache_blocks)
            safe_phys = phys_live.select(phys, zero)
            if const_expr(wide_cache):
                k_row = _k_page_row(
                    k_base,
                    safe_phys,
                    off,
                    page_elems64,
                    token_elems64,
                    head_elems64,
                    zero,
                    row_ptr_ty,
                    k_layout,
                )
            else:
                k_row = fx.logical_divide(
                    fx.slice(k_buf, (safe_phys, off, zero, None)), k_layout
                )
            accs = [Float32(0.0) for _ in range_constexpr(n_heads)]
            for chunk in range_constexpr(k_chunks):
                k_src = fx.slice(k_row, (None, Int32(chunk)))
                k_frag = fx.make_fragment_like(k_src)
                fx.copy(k_copy, k_src, k_frag)
                k_vals = fx.Vector(fx.memref_load_vec(k_frag))
                for i in range_constexpr(k_vec):
                    kf = k_vals[i].to(Float32)
                    d = chunk * k_vec + i
                    for h in range_constexpr(n_heads):
                        accs[h] = accs[h] + q_lds[Int32(h * _D + d)] * kf
            score = Float32(0.0)
            for h in range_constexpr(n_heads):
                score = score + fx.max(accs[h], Float32(0.0))
            return phys_live.select(score * score_scale, _neg_inf())

        def ordered(score):
            """Signed int32 with the fp32 order of ``score``."""
            bits = score.bitcast(Int32)
            return bits ^ ((bits >> Int32(31)) & Int32(0x7FFFFFFF))

        if tid < Int32(_BINS):
            high[tid] = zero
            low[tid] = zero
        out[tid] = neg1
        if tid < Int32(_M_SLOTS):
            meta[tid] = zero
        if tid < Int32(q_vectors):
            q_head = div_nonneg(tid, Int32(k_chunks))
            q_chunk = tid - q_head * Int32(k_chunks)
            q_row = fx.logical_divide(fx.slice(q_buf, (row, q_head, None)), k_layout)
            q_src = fx.slice(q_row, (None, q_chunk))
            q_frag = fx.make_fragment_like(q_src)
            fx.copy(g_copy, q_src, q_frag)
            q_vals = fx.Vector(fx.memref_load_vec(q_frag))
            for i in range_constexpr(k_vec):
                q_lds[q_head * Int32(_D) + q_chunk * Int32(k_vec) + Int32(i)] = q_vals[
                    i
                ].to(Float32)
        gpu.barrier()

        # Stage the visible keys and count their high byte.
        keys_row = fx.logical_divide(fx.slice(keys_buf, (row, None)), vec_layout)
        for v in range(tid, n_vec, threads):
            v_i = Int32(v)
            src = fx.slice(keys_row, (None, v_i))
            frag = fx.make_fragment_like(src)
            fx.copy(key_copy, src, frag)
            key_vals = fx.Vector(fx.memref_load_vec(frag))
            for i in range_constexpr(vec):
                col = v_i * Int32(vec) + Int32(i)
                key_i = key_vals[i]
                keys_lds[col] = key_i
                key = key_i.to(Int32)
                if (key >= zero) & (col < n_vis):
                    hist_add(high, key >> Int32(8))
        gpu.barrier()

        hit, bin_id, higher, count, live_total = find_crossing(high, budget)
        if hit:
            meta[_M_BIN] = bin_id
            meta[_M_HIGHER] = higher
            meta[_M_COUNT] = count
        gpu.barrier()
        long_row = live_total > budget
        # A short row keeps every live key: the boundary is below them all.
        boundary = neg1
        if long_row:
            high_bin = meta[_M_BIN]
            above_high = meta[_M_HIGHER]
            for c in range(tid, n_vis, threads):
                key_c = row_key(Int32(c))
                if (key_c >> Int32(8)) == high_bin:
                    hist_add(low, key_c & Int32(0xFF))
            gpu.barrier()
            hit_l, bin_l, higher_l, count_l, _total_l = find_crossing(
                low, budget - above_high
            )
            if hit_l:
                meta[_M_BIN] = bin_l
                meta[_M_HIGHER] = higher_l
                meta[_M_COUNT] = count_l
            gpu.barrier()
            boundary = (high_bin << Int32(8)) | meta[_M_BIN]

        # Keys above the boundary go straight out; the bucket is collected.
        for c in range(tid, n_vis, threads):
            c_i = Int32(c)
            key_d = row_key(c_i)
            if key_d >= zero:
                if key_d > boundary:
                    slot = atomic_add_i32(meta, one, _M_OUT, "workgroup")
                    out[slot] = c_i
                elif key_d == boundary:
                    slot_b = atomic_add_i32(meta, one, _M_BUCKET, "workgroup")
                    if slot_b < Int32(BUCKET_CAPACITY):
                        b_col[slot_b] = c_i
        gpu.barrier()

        n_above = meta[_M_OUT]
        width = meta[_M_BUCKET]
        quota = budget - n_above
        if long_row & (width <= Int32(BUCKET_CAPACITY)):
            for i in range(tid, width, threads):
                b_score[Int32(i)] = rescore(b_col[Int32(i)])
            gpu.barrier()
            # Rank by score, then by the smaller block, and keep the first quota.
            for i in range(tid, width, threads):
                i_r = Int32(i)
                s_i = b_score[i_r]
                c_r = b_col[i_r]
                rank = zero
                for j in range(width):
                    j_r = Int32(j)
                    s_j = b_score[j_r]
                    before = (s_j > s_i) | ((s_j == s_i) & (b_col[j_r] < c_r))
                    rank = rank + before.select(one, zero)
                if rank < quota:
                    out[n_above + rank] = c_r
        if long_row & (width > Int32(BUCKET_CAPACITY)):
            # Four 8-bit passes over the ordered fp32 score, then a take pass.
            # Scores are recomputed each pass, so the bucket can be any size.
            if tid == zero:
                meta[_M_NEED] = quota
            gpu.barrier()
            for p in range(Int32(5)):
                p_i = Int32(p)
                if p_i < Int32(4):
                    if tid < Int32(_BINS):
                        high[tid] = zero
                    gpu.barrier()
                prefix = meta[_M_PREFIX]
                need = meta[_M_NEED]
                shift = Int32(24) - p_i * Int32(8)
                # Bits already fixed by earlier passes. Pass 0 fixes none.
                fixed = (p_i == zero).select(
                    zero, Int32(-1) << fx.min(shift + Int32(8), Int32(31))
                )
                for c in range(tid, n_vis, threads):
                    c_f = Int32(c)
                    if row_key(c_f) == boundary:
                        u = ordered(rescore(c_f)) ^ Int32(_SIGN)
                        if p_i < Int32(4):
                            if ((u ^ prefix) & fixed) == zero:
                                hist_add(high, (u >> shift) & Int32(0xFF))
                        else:
                            o = u ^ Int32(_SIGN)
                            o_t = prefix ^ Int32(_SIGN)
                            take = o > o_t
                            if o == o_t:
                                tie = atomic_add_i32(meta, one, _M_TIES, "workgroup")
                                take = tie < need
                            if take:
                                slot_f = atomic_add_i32(meta, one, _M_OUT, "workgroup")
                                out[slot_f] = c_f
                gpu.barrier()
                if p_i < Int32(4):
                    hit_p, bin_p, higher_p, count_p, _total_p = find_crossing(
                        high, need
                    )
                    if hit_p:
                        meta[_M_BIN] = bin_p
                        meta[_M_HIGHER] = higher_p
                        meta[_M_COUNT] = count_p
                    gpu.barrier()
                    if tid == zero:
                        meta[_M_PREFIX] = prefix | (meta[_M_BIN] << shift)
                        meta[_M_NEED] = need - meta[_M_HIGHER]
                    gpu.barrier()
        gpu.barrier()

        block_ids[row, tid] = out[tid]

    @flyc.jit
    def launch_qsa_indexer_key16_select(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        keys: fx.Tensor,
        block_ids: fx.Tensor,
        rows: Int32,
        n_req: Int32,
        score_scale: Float32,
        n_cache_blocks: Int32,
        stream: fx.Stream,
    ):
        qsa_indexer_key16_select_kernel(
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            keys,
            block_ids,
            n_req,
            score_scale,
            n_cache_blocks,
        ).launch(
            grid=(rows, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    return launch_qsa_indexer_key16_select
