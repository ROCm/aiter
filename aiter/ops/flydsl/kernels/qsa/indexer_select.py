# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL indexer-select kernels for QSA.

``build_qsa_indexer_emit_module`` writes block ids when every row fits in the
budget. ``build_qsa_indexer_decode_scores_module`` and
``build_qsa_indexer_prefill_scores_module`` are the one-row and 16-row BF16
MFMA scorers. ``build_qsa_indexer_active_columns_module`` reports the widest
visible column count. Host dispatch and top-512 selection live in
``aiter.ops.flydsl.qsa``.

Every shape these kernels serve shares one indexer contract. The scorers
accept 4 or 8 heads, and each head count is its own compile.
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

from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from aiter.ops.flydsl.kernels.tensor_shim import buf_base_i64, buf_copy_atom

# The indexer contract this module implements, rather than any one model's
# numbers: a budget of _K compressed blocks, _KV_HEADS head of _D elements,
# and _R raw tokens per compressed block. The emit kernel bakes _K into its
# launch shape and the scorers bake _D into their LDS tile, so these are the
# kernel's own constants. The test suite asserts they still cover every
# shape we validate against.
_BLOCK_THREADS = 512
_K = 512
_H = 4
_D = 128
_R = 4
_KV_HEADS = 1
_STREAM_SELECT_MIN_COLUMNS = 32768
# topk_select's decode gate for k=512 is rows <= 64. Above that, one
# workgroup per row loses to the streaming selector.
_DECODE_MAX_ROWS = 64
_SCORE_HEADS = (4, 8)
_SCORE_SCALE = _D**-0.5


def _idiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def _neg_inf():
    return Float32(float("-inf"))


def build_qsa_indexer_emit_module(page_size: int):
    """Build the short-context emit kernel.

    The kernel writes block ids from the query position and the context
    length. It does not take Q, the cache, the page table, or the score
    scale; those stay on the host wrapper for the long-row scorer.
    """
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if _K % _BLOCK_THREADS:
        raise ValueError("block budget must be a multiple of block threads")

    @flyc.kernel(
        name="qsa_indexer_emit_"
        + kernel_signature(ps=page_size, k=_K, blk=_BLOCK_THREADS),
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def qsa_indexer_emit_kernel(
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        block_ids: fx.Tensor,
        n_columns: Int32,
        n_req: Int32,
    ):
        row = Int32(gpu.block_id("x"))
        tid = Int32(gpu.thread_id("x"))
        zero = Int32(0)
        one = Int32(1)
        neg_one = Int32(-1)
        req = token_to_req[row]
        valid_req = (req >= zero) & (req < n_req)
        safe_req = valid_req.select(req, zero)
        qpos = query_positions[row]
        slen = valid_req.select(context_lens[safe_req], zero)
        vis_q = _idiv(qpos + one, Int32(_R))
        vis_s = _idiv(slen, Int32(_R))
        visible = (vis_q < vis_s).select(vis_q, vis_s)
        take = (tid < visible) & (tid < n_columns) & valid_req
        block_ids[row, tid] = take.select(tid, neg_one)

    @flyc.jit
    def launch_qsa_indexer_emit(
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        block_ids: fx.Tensor,
        n_columns: Int32,
        n_req: Int32,
        rows: Int32,
        stream: fx.Stream,
    ):
        qsa_indexer_emit_kernel(
            token_to_req,
            query_positions,
            context_lens,
            block_ids,
            n_columns,
            n_req,
        ).launch(
            grid=(rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_qsa_indexer_emit


def _dense_indexer_strides(page_size: int) -> tuple[int, int, int]:
    """Element strides of a packed ``[pages, page_size, 1, D]`` cache."""
    return (page_size * _KV_HEADS * _D, _KV_HEADS * _D, _D)


def _k_page_row(
    k_base,
    phys,
    page_off,
    page_elems64,
    token_elems64,
    head_elems64,
    zero,
    row_ptr_ty,
    vec_layout,
):
    """Wide indexer row: page, token, and head 0, then a unit-stride D view."""
    addr = k_base + (
        Int64(phys) * page_elems64
        + Int64(page_off) * token_elems64
        + Int64(zero) * head_elems64
    ) * Int64(2)
    flat = fx.Tensor(
        fx.make_view(
            fx.inttoptr(row_ptr_ty, addr),
            fx.make_layout((_D,), (1,)),
        )
    )
    return fx.logical_divide(flat, vec_layout)


def _qk_mfma(qk_mma, qk_a, qk_b, qk_c, a_vec, b_vec, c_vec):
    """One K1 QK atom. Fragments are the caller's, reused across K steps."""
    fx.memref_store_vec(a_vec, qk_a)
    fx.memref_store_vec(b_vec, qk_b)
    fx.memref_store_vec(c_vec, qk_c)
    fx.gemm(qk_mma, qk_c, qk_a, qk_b, qk_c)
    return fx.memref_load_vec(qk_c)


def build_qsa_indexer_decode_scores_module(
    page_size: int,
    use_k32: bool,
    block_n: int,
    n_heads: int = _H,
    wide_cache: bool = False,
    k_strides: tuple[int, int, int] | None = None,
):
    """Build a long-context paged MFMA scorer."""
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if block_n not in (16, 32):
        raise ValueError(f"score block_n must be 16 or 32, got {block_n}")
    if n_heads not in _SCORE_HEADS:
        raise ValueError(f"score heads must be {_SCORE_HEADS}, got {n_heads}")
    dense_k_strides = _dense_indexer_strides(page_size)
    if k_strides is None:
        k_strides = dense_k_strides
    else:
        k_strides = tuple(int(s) for s in k_strides)
    # Page, token, and head strides are constants only on the wide path.
    # The narrow descriptor reads them from the tensor layout.
    strided_wide = wide_cache and k_strides != dense_k_strides

    block_threads = 128
    head_pad = 16
    qk_k = 32 if use_k32 else 16
    qk_vec = qk_k // 4
    k_steps = 64 // qk_k
    n_subtiles = block_n // 16
    vec = 8
    vec_chunks = _D // vec
    q_chunks_per_thread = vec_chunks // (block_threads // head_pad)
    chunks_per_thread = vec_chunks // (block_threads // block_n)
    num_waves = block_threads // 64

    @fx.struct
    class SharedStorage:
        q: fx.Array[BFloat16, head_pad * _D, 16]
        k: fx.Array[BFloat16, block_n * _D, 16]
        live: fx.Array[Int32, block_n, 16]
        c: fx.Array[Float32, n_subtiles * num_waves * 64 * 4, 16]

    @flyc.kernel(
        name="qsa_indexer_scores_"
        + kernel_signature(
            ps=page_size,
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
    def qsa_indexer_scores_kernel(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        scores: fx.Tensor,
        row_lens: fx.Tensor,
        n_columns: Int32,
        n_req: Int32,
        score_scale: Float32,
        n_cache_blocks: Int32,
    ):
        tile = Int32(gpu.block_id("x"))
        row = Int32(gpu.block_id("y"))
        tid = Int32(gpu.thread_id("x"))
        zero = Int32(0)
        one = Int32(1)
        page = Int32(page_size)
        wave = _idiv(tid, Int32(64))
        lane = tid - wave * Int32(64)
        lane_m = lane % Int32(16)
        lane_kg = _idiv(lane, Int32(16))
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
        # A V# voffset is 32 bits. A cache that fits in 4 GiB keeps one
        # uniform descriptor. A larger cache is a separate compile: each
        # gathered row rebases its page in 64-bit and is read with a raw
        # 128-bit load. The two bodies are not both traced.
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
        q_lds = storage.q.view(fx.make_layout((head_pad, _D), (_D, 1)))
        k_lds = storage.k.view(fx.make_layout((block_n, _D), (_D, 1)))
        live_lds = storage.live.view(fx.make_layout(block_n, 1))
        c_lds = storage.c.view(
            fx.make_layout(
                (n_subtiles, num_waves, 64, 4),
                (num_waves * 64 * 4, 64 * 4, 4, 1),
            )
        )
        qk_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, qk_k, BFloat16))
        qk_a = fx.make_rmem_tensor(fx.make_layout(qk_vec, 1), BFloat16)
        qk_b = fx.make_rmem_tensor(fx.make_layout(qk_vec, 1), BFloat16)
        qk_c = fx.make_rmem_tensor(fx.make_layout(4, 1), Float32)

        req = token_to_req[row]
        valid_req = (req >= zero) & (req < n_req)
        safe_req = valid_req.select(req, zero)
        qpos = query_positions[row]
        slen = valid_req.select(context_lens[safe_req], zero)
        vis_q = _idiv(qpos + one, Int32(_R))
        vis_s = _idiv(slen, Int32(_R))
        visible = (vis_q < vis_s).select(vis_q, vis_s)
        visible = (visible < n_columns).select(visible, n_columns)

        if (tile == zero) & (tid == zero):
            row_lens[row] = valid_req.select(visible, zero)

        # Tiles past the row's visible blocks are never read: both selectors
        # stop at row_lens. A block table padded to max_model_len is mostly
        # such tiles, so they must not cost a full gather and MFMA each.
        if tile * Int32(block_n) < visible:
            qh = tid % Int32(head_pad)
            q_chunk = _idiv(tid, Int32(head_pad))
            q_live = qh < Int32(n_heads)
            safe_qh = q_live.select(qh, zero)
            q_row = fx.logical_divide(fx.slice(q_buf, (row, safe_qh, None)), vec_layout)
            for part in range_constexpr(q_chunks_per_thread):
                d_chunk = q_chunk + Int32(part * (block_threads // head_pad))
                q_src = fx.slice(q_row, (None, d_chunk))
                q_frag = fx.make_fragment_like(q_src)
                fx.copy(g_copy, q_src, q_frag)
                q_vec = fx.Vector(fx.memref_load_vec(q_frag))
                d0 = d_chunk * Int32(vec)
                for i in range_constexpr(vec):
                    qv = q_live.select(q_vec[i].to(Float32), Float32(0.0))
                    q_lds[qh, d0 + Int32(i)] = qv.to(BFloat16)

            col = tid % Int32(block_n)
            chunk = _idiv(tid, Int32(block_n))
            score_col = tile * Int32(block_n) + col
            col_live = (score_col < n_columns) & (score_col < visible) & valid_req
            safe_col = col_live.select(score_col, zero)
            logical_page = _idiv(safe_col, page)
            off = safe_col - logical_page * page
            phys = page_table[safe_req, logical_page]
            phys_live = (phys >= zero) & (phys < n_cache_blocks)
            safe_phys = phys_live.select(phys, zero)
            col_live = col_live & phys_live
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
                    vec_layout,
                )
            else:
                k_row = fx.logical_divide(
                    fx.slice(k_buf, (safe_phys, off, zero, None)), vec_layout
                )
            for part in range_constexpr(chunks_per_thread):
                d_chunk = chunk + Int32(part * (block_threads // block_n))
                k_src = fx.slice(k_row, (None, d_chunk))
                k_frag = fx.make_fragment_like(k_src)
                fx.copy(k_copy, k_src, k_frag)
                k_vec = fx.Vector(fx.memref_load_vec(k_frag))
                kd0 = d_chunk * Int32(vec)
                for i in range_constexpr(vec):
                    kv = col_live.select(k_vec[i].to(Float32), Float32(0.0))
                    k_lds[col, kd0 + Int32(i)] = kv.to(BFloat16)
            if chunk == zero:
                live_lds[col] = col_live.select(one, zero)
            gpu.barrier()

            for ng in range_constexpr(n_subtiles):
                n_row = Int32(ng * 16) + lane_m
                acc4 = fx.Vector.filled(4, 0.0, Float32)
                for ks in range_constexpr(k_steps):
                    md0 = wave * Int32(64) + Int32(ks * qk_k) + lane_kg * Int32(qk_vec)
                    a_vec = fx.Vector.from_elements(
                        [
                            q_lds[lane_m, md0 + Int32(i)]
                            for i in range_constexpr(qk_vec)
                        ],
                        BFloat16,
                    )
                    b_vec = fx.Vector.from_elements(
                        [k_lds[n_row, md0 + Int32(i)] for i in range_constexpr(qk_vec)],
                        BFloat16,
                    )
                    acc4 = fx.Vector(
                        _qk_mfma(qk_mma, qk_a, qk_b, qk_c, a_vec, b_vec, acc4)
                    )
                for i in range_constexpr(4):
                    c_lds[ng, wave, lane, i] = acc4[i]
            gpu.barrier()

            if (wave == zero) & (lane_kg == zero):
                for ng in range_constexpr(n_subtiles):
                    out_col = tile * Int32(block_n) + Int32(ng * 16) + lane_m
                    score = Float32(0.0)
                    for h in range_constexpr(n_heads):
                        # 16x16 C: n = lane%16, m = 4*(lane/16) + elem.
                        src_lane = lane_m + Int32(16 * (h // 4))
                        elem = Int32(h % 4)
                        dot = Float32(0.0)
                        for w in range_constexpr(num_waves):
                            dot = dot + c_lds[ng, w, src_lane, elem]
                        score = score + fx.max(dot, Float32(0.0))
                    if out_col < n_columns:
                        live = live_lds[Int32(ng * 16) + lane_m] != zero
                        # ``scores`` must keep a dynamic row stride: a static
                        # layout, use_32bit_stride=True, or an Int32 linear
                        # index would wrap.
                        scores[row, out_col] = live.select(
                            score * score_scale, _neg_inf()
                        )

    @flyc.jit
    def launch_qsa_indexer_scores(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        scores: fx.Tensor,
        row_lens: fx.Tensor,
        n_columns: Int32,
        n_req: Int32,
        score_scale: Float32,
        n_cache_blocks: Int32,
        rows: Int32,
        tiles: Int32,
        stream: fx.Stream,
    ):
        qsa_indexer_scores_kernel(
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            scores,
            row_lens,
            n_columns,
            n_req,
            score_scale,
            n_cache_blocks,
        ).launch(
            grid=(tiles, rows, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_qsa_indexer_scores


def build_qsa_indexer_prefill_scores_module(
    page_size: int,
    use_k32: bool,
    n_heads: int = _H,
    wide_cache: bool = False,
    k_strides: tuple[int, int, int] | None = None,
):
    """Build the single-request, 16-row by 32-column MFMA scorer."""
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if n_heads not in _SCORE_HEADS:
        raise ValueError(f"score heads must be {_SCORE_HEADS}, got {n_heads}")
    dense_k_strides = _dense_indexer_strides(page_size)
    if k_strides is None:
        k_strides = dense_k_strides
    else:
        k_strides = tuple(int(s) for s in k_strides)
    strided_wide = wide_cache and k_strides != dense_k_strides

    block_m = 16
    block_n = 32
    block_threads = 128
    qk_k = 32 if use_k32 else 16
    qk_vec = qk_k // 4
    k_steps = 64 // qk_k
    n_subtiles = block_n // 16
    vec = 8
    vec_chunks = _D // vec
    q_vectors = block_m * n_heads * vec_chunks
    k_vectors = block_n * vec_chunks
    if q_vectors % block_threads or k_vectors % block_threads:
        raise ValueError("prefill Q/K vector counts must divide block threads")
    q_vectors_per_thread = q_vectors // block_threads
    k_vectors_per_thread = k_vectors // block_threads
    num_waves = block_threads // 64

    @fx.struct
    class SharedStorage:
        q: fx.Array[BFloat16, block_m * n_heads * _D, 16]
        k: fx.Array[BFloat16, block_n * _D, 16]
        visible: fx.Array[Int32, block_m, 16]
        c: fx.Array[Float32, n_heads * n_subtiles * num_waves * 64 * 4, 16]

    @flyc.kernel(
        name="qsa_indexer_prefill_scores_"
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
        score_scale: Float32,
        n_cache_blocks: Int32,
    ):
        tile = Int32(gpu.block_id("x"))
        row_tile = Int32(gpu.block_id("y"))
        tid = Int32(gpu.thread_id("x"))
        zero = Int32(0)
        one = Int32(1)
        page = Int32(page_size)
        wave = _idiv(tid, Int32(64))
        lane = tid - wave * Int32(64)
        lane_m = lane % Int32(16)
        lane_kg = _idiv(lane, Int32(16))
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
        q_lds = storage.q.view(
            fx.make_layout((block_m, n_heads, _D), (n_heads * _D, _D, 1))
        )
        k_lds = storage.k.view(fx.make_layout((block_n, _D), (_D, 1)))
        visible_lds = storage.visible.view(fx.make_layout(block_m, 1))
        c_lds = storage.c.view(
            fx.make_layout(
                (n_heads, n_subtiles, num_waves, 64, 4),
                (
                    n_subtiles * num_waves * 64 * 4,
                    num_waves * 64 * 4,
                    64 * 4,
                    4,
                    1,
                ),
            )
        )
        qk_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, qk_k, BFloat16))
        qk_a = fx.make_rmem_tensor(fx.make_layout(qk_vec, 1), BFloat16)
        qk_b = fx.make_rmem_tensor(fx.make_layout(qk_vec, 1), BFloat16)
        qk_c = fx.make_rmem_tensor(fx.make_layout(4, 1), Float32)

        if tid < Int32(block_m):
            row = row_tile * Int32(block_m) + tid
            row_live = row < rows
            safe_row = row_live.select(row, zero)
            req_live = row_live & (token_to_req[safe_row] == zero)
            qpos = query_positions[safe_row]
            slen = context_lens[zero]
            vis_q = _idiv(qpos + one, Int32(_R))
            vis_s = _idiv(slen, Int32(_R))
            visible = (vis_q < vis_s).select(vis_q, vis_s)
            visible = (visible < n_columns).select(visible, n_columns)
            visible = req_live.select(visible, zero)
            visible_lds[tid] = visible
            if (tile == zero) & row_live:
                row_lens[row] = visible
        gpu.barrier()
        tile_visible = visible_lds[zero]
        for r in range_constexpr(1, block_m):
            v = visible_lds[Int32(r)]
            tile_visible = (v > tile_visible).select(v, tile_visible)

        # As in the one-row scorer: skip tiles no row of this block can see.
        if tile * Int32(block_n) < tile_visible:
            for part in range_constexpr(q_vectors_per_thread):
                linear = tid + Int32(part * block_threads)
                row_local = _idiv(linear, Int32(n_heads * vec_chunks))
                rem = linear - row_local * Int32(n_heads * vec_chunks)
                head = _idiv(rem, Int32(vec_chunks))
                d_chunk = rem - head * Int32(vec_chunks)
                row = row_tile * Int32(block_m) + row_local
                row_live = row < rows
                safe_row = row_live.select(row, zero)
                req_live = row_live & (token_to_req[safe_row] == zero)
                q_row = fx.logical_divide(
                    fx.slice(q_buf, (safe_row, head, None)), vec_layout
                )
                q_src = fx.slice(q_row, (None, d_chunk))
                q_frag = fx.make_fragment_like(q_src)
                fx.copy(g_copy, q_src, q_frag)
                q_vec = fx.Vector(fx.memref_load_vec(q_frag))
                d0 = d_chunk * Int32(vec)
                for i in range_constexpr(vec):
                    qv = req_live.select(q_vec[i].to(Float32), Float32(0.0))
                    q_lds[row_local, head, d0 + Int32(i)] = qv.to(BFloat16)

            context_cols = _idiv(context_lens[zero], Int32(_R))
            for part in range_constexpr(k_vectors_per_thread):
                linear = tid + Int32(part * block_threads)
                col = _idiv(linear, Int32(vec_chunks))
                d_chunk = linear - col * Int32(vec_chunks)
                score_col = tile * Int32(block_n) + col
                col_live = (score_col < n_columns) & (score_col < context_cols)
                safe_col = col_live.select(score_col, zero)
                logical_page = _idiv(safe_col, page)
                off = safe_col - logical_page * page
                phys = page_table[zero, logical_page]
                phys_live = (phys >= zero) & (phys < n_cache_blocks)
                safe_phys = phys_live.select(phys, zero)
                col_live = col_live & phys_live
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
                        vec_layout,
                    )
                else:
                    k_row = fx.logical_divide(
                        fx.slice(k_buf, (safe_phys, off, zero, None)), vec_layout
                    )
                k_src = fx.slice(k_row, (None, d_chunk))
                k_frag = fx.make_fragment_like(k_src)
                fx.copy(k_copy, k_src, k_frag)
                k_vec = fx.Vector(fx.memref_load_vec(k_frag))
                d0 = d_chunk * Int32(vec)
                for i in range_constexpr(vec):
                    kv = col_live.select(k_vec[i].to(Float32), Float32(0.0))
                    k_lds[col, d0 + Int32(i)] = kv.to(BFloat16)
            gpu.barrier()

            for head in range_constexpr(n_heads):
                for ng in range_constexpr(n_subtiles):
                    n_row = Int32(ng * 16) + lane_m
                    acc4 = fx.Vector.filled(4, 0.0, Float32)
                    for ks in range_constexpr(k_steps):
                        d0 = (
                            wave * Int32(64)
                            + Int32(ks * qk_k)
                            + lane_kg * Int32(qk_vec)
                        )
                        a_vec = fx.Vector.from_elements(
                            [
                                q_lds[lane_m, head, d0 + Int32(i)]
                                for i in range_constexpr(qk_vec)
                            ],
                            BFloat16,
                        )
                        b_vec = fx.Vector.from_elements(
                            [
                                k_lds[n_row, d0 + Int32(i)]
                                for i in range_constexpr(qk_vec)
                            ],
                            BFloat16,
                        )
                        acc4 = fx.Vector(
                            _qk_mfma(qk_mma, qk_a, qk_b, qk_c, a_vec, b_vec, acc4)
                        )
                    for i in range_constexpr(4):
                        c_lds[head, ng, wave, lane, i] = acc4[i]
            gpu.barrier()

            if wave == zero:
                for ng in range_constexpr(n_subtiles):
                    out_col = tile * Int32(block_n) + Int32(ng * 16) + lane_m
                    for i in range_constexpr(4):
                        row_local = lane_kg * Int32(4) + Int32(i)
                        row = row_tile * Int32(block_m) + row_local
                        score = Float32(0.0)
                        for head in range_constexpr(n_heads):
                            dot = Float32(0.0)
                            for w in range_constexpr(num_waves):
                                dot = dot + c_lds[head, ng, w, lane, i]
                            score = score + fx.max(dot, Float32(0.0))
                        if (row < rows) & (out_col < n_columns):
                            live = out_col < visible_lds[row_local]
                            # ``scores`` must keep a dynamic row stride: a static
                            # layout, use_32bit_stride=True, or an Int32 linear
                            # index would wrap.
                            scores[row, out_col] = live.select(
                                score * score_scale, _neg_inf()
                            )

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
        score_scale: Float32,
        n_cache_blocks: Int32,
        row_tiles: Int32,
        tiles: Int32,
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
            score_scale,
            n_cache_blocks,
        ).launch(
            grid=(tiles, row_tiles, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_qsa_indexer_prefill_scores


def build_qsa_indexer_active_columns_module():
    """One thread, the widest visible column count, matching the scorers."""

    @flyc.kernel(
        name="qsa_indexer_active_columns",
        known_block_size=[1, 1, 1],
    )
    def qsa_indexer_active_columns_kernel(
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        out: fx.Tensor,
        n_columns: Int32,
        n_req: Int32,
        rows: Int32,
    ):
        zero = Int32(0)
        one = Int32(1)
        for _row, state in range(zero, rows, one, init=[zero]):
            local = state[0]
            row = Int32(_row)
            req = token_to_req[row]
            valid_req = (req >= zero) & (req < n_req)
            safe_req = valid_req.select(req, zero)
            qpos = query_positions[row]
            slen = valid_req.select(context_lens[safe_req], zero)
            vis_q = _idiv(qpos + one, Int32(_R))
            vis_s = _idiv(slen, Int32(_R))
            visible = (vis_q < vis_s).select(vis_q, vis_s)
            visible = (visible < n_columns).select(visible, n_columns)
            visible = valid_req.select(visible, zero)
            local = (visible > local).select(visible, local)
            results = yield [local]
        out[zero] = results

    @flyc.jit
    def launch_qsa_indexer_active_columns(
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        out: fx.Tensor,
        n_columns: Int32,
        n_req: Int32,
        rows: Int32,
        stream: fx.Stream,
    ):
        qsa_indexer_active_columns_kernel(
            token_to_req,
            query_positions,
            context_lens,
            out,
            n_columns,
            n_req,
            rows,
        ).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    return launch_qsa_indexer_active_columns
