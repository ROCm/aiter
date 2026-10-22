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
# Columns per workgroup of the 16-row scorer.
_ROWS_BLOCK_N = 64
# gfx9 LDS is 64 banks of 4 bytes. A row of 128 bf16 is 256 bytes, one
# full pass of the banks, so the 16 lanes that reload one MFMA column
# (lane_m 0..15, the same d) all hit one bank. Eight bf16 of padding is
# 16 bytes: the bank advances by 4, those lanes take distinct banks, and
# each row stays 16-byte aligned for the 8-wide stores.
_LDS_ROW_PAD = 8
# One-row partial scores. A lane stride of 4 fp32 puts lanes 16 apart in
# the same bank, so the 64-lane store conflicts 4 ways. Stride 5 is
# coprime with 64, so that store uses every bank once. The 4 accumulators
# stay contiguous. The 16-row scorer sums heads in registers and has no
# such buffer.
_LDS_C_LANE_STRIDE = 5


def _layout_elems(shape: tuple[int, ...], stride: tuple[int, ...]) -> int:
    """Elements a layout addresses, including the padding in its strides."""
    return 1 + sum((n - 1) * s for n, s in zip(shape, stride))


def _q_lds_layout(
    block_m: int, n_heads: int, fold_heads: bool
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """``(shape, stride)`` for the 16-row scorer's Q tile.

    Non-fold reads 16 rows of one head, so only the row stride has to move
    the bank. Fold reads every (row, head) pair on the 16-wide M axis, so
    each head is padded and the row stride is that many heads: 4 rows by 4
    heads then land on 16 distinct banks.
    """
    if fold_heads:
        head_stride = _D + _LDS_ROW_PAD
        row_stride = n_heads * head_stride
    else:
        head_stride = _D
        row_stride = n_heads * _D + _LDS_ROW_PAD
    return (block_m, n_heads, _D), (row_stride, head_stride, 1)


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
    q_shape = (head_pad, _D)
    q_stride = (_D + _LDS_ROW_PAD, 1)
    k_shape = (block_n, _D)
    k_stride = (_D + _LDS_ROW_PAD, 1)
    c_shape = (n_subtiles, num_waves, 64, 4)
    c_stride = (
        num_waves * 64 * _LDS_C_LANE_STRIDE,
        64 * _LDS_C_LANE_STRIDE,
        _LDS_C_LANE_STRIDE,
        1,
    )

    @fx.struct
    class SharedStorage:
        q: fx.Array[BFloat16, _layout_elems(q_shape, q_stride), 16]
        k: fx.Array[BFloat16, _layout_elems(k_shape, k_stride), 16]
        live: fx.Array[Int32, block_n, 16]
        c: fx.Array[Float32, _layout_elems(c_shape, c_stride), 16]

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
        q_lds = storage.q.view(fx.make_layout(q_shape, q_stride))
        k_lds = storage.k.view(fx.make_layout(k_shape, k_stride))
        live_lds = storage.live.view(fx.make_layout(block_n, 1))
        c_lds = storage.c.view(fx.make_layout(c_shape, c_stride))
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
    block_n: int = _ROWS_BLOCK_N,
    fold_heads: bool = False,
):
    """Build the 16-row MFMA scorer, ``block_n`` columns per workgroup.

    One wave per 16 columns, each over the whole head dim, so the four
    heads accumulate in registers and the wave writes its scores directly.
    A row tile may span requests. Q is staged once per request; each
    request in the tile gathers its own K tiles and writes only its own
    rows, so a request-major batch (vLLM's layout) gathers each K tile once
    per request it touches. A workgroup walks ``tiles_per_wg`` consecutive
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

    @fx.struct
    class SharedStorage:
        q: fx.Array[BFloat16, _layout_elems(q_shape, q_stride), 16]
        k: fx.Array[BFloat16, _layout_elems(k_shape, k_stride), 16]
        visible: fx.Array[Int32, block_m, 16]
        req: fx.Array[Int32, block_m, 16]
        live: fx.Array[Int32, block_n, 16]

    @flyc.kernel(
        name="qsa_indexer_rows_scores_"
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
    ):
        # Row tiles vary fastest, so workgroups resident together share
        # column tiles: the same K, many Q tiles. Each workgroup scores
        # tiles_per_wg consecutive column tiles with one Q staging.
        row_tile = Int32(gpu.block_id("x"))
        tile0 = Int32(gpu.block_id("y")) * tiles_per_wg
        tid = Int32(gpu.thread_id("x"))
        zero = Int32(0)
        one = Int32(1)
        page = Int32(page_size)
        wave = _idiv(tid, Int32(64))
        lane = tid - wave * Int32(64)
        lane_m = lane % Int32(16)
        lane_kg = _idiv(lane, Int32(16))
        # The A row this lane feeds: row lane_m, or (row, head) when folded.
        if const_expr(fold_heads):
            a_row = _idiv(lane_m, Int32(n_heads))
            a_head = lane_m - a_row * Int32(n_heads)
        else:
            a_row = lane_m
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
        q_lds = storage.q.view(fx.make_layout(q_shape, q_stride))
        k_lds = storage.k.view(fx.make_layout(k_shape, k_stride))
        visible_lds = storage.visible.view(fx.make_layout(block_m, 1))
        req_lds = storage.req.view(fx.make_layout(block_m, 1))
        live_lds = storage.live.view(fx.make_layout(block_n, 1))
        qk_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, qk_k, BFloat16))
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
            vis_q = _idiv(qpos + one, Int32(_R))
            vis_s = _idiv(slen, Int32(_R))
            visible = (vis_q < vis_s).select(vis_q, vis_s)
            visible = (visible < n_columns).select(visible, n_columns)
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
            tile_visible = (v > tile_visible).select(v, tile_visible)
            req_lo = ((rq >= zero) & (rq < req_lo)).select(rq, req_lo)
            req_hi = (rq > req_hi).select(rq, req_hi)

        # As in the one-row scorer: skip tiles no row of this block can see.
        if tile0 * Int32(block_n) < tile_visible:
            # Names assigned in this loop are its own: a name it shares with
            # the code above would become an scf.for iter_arg.
            for cur_req in range(req_lo, req_hi + one, one):
                cur = Int32(cur_req)
                req_visible = zero
                for r_r in range_constexpr(block_m):
                    v_r = (req_lds[Int32(r_r)] == cur).select(
                        visible_lds[Int32(r_r)], zero
                    )
                    req_visible = (v_r > req_visible).select(v_r, req_visible)
                if tile0 * Int32(block_n) < req_visible:
                    # Q is staged once per request and reused by every tile
                    # this workgroup scores; the first K gather's barrier
                    # publishes it.
                    for qpart_r in range_constexpr(q_vectors_per_thread):
                        qlinear_r = tid + Int32(qpart_r * block_threads)
                        qrow_local_r = _idiv(qlinear_r, Int32(n_heads * vec_chunks))
                        qrem_r = qlinear_r - qrow_local_r * Int32(n_heads * vec_chunks)
                        qhead_r = _idiv(qrem_r, Int32(vec_chunks))
                        qd_chunk_r = qrem_r - qhead_r * Int32(vec_chunks)
                        qrow_r = row_tile * Int32(block_m) + qrow_local_r
                        qrow_live_r = qrow_r < rows
                        qsafe_row_r = qrow_live_r.select(qrow_r, zero)
                        qreq_live_r = req_lds[qrow_local_r] >= zero
                        qs_row_r = fx.logical_divide(
                            fx.slice(q_buf, (qsafe_row_r, qhead_r, None)), vec_layout
                        )
                        qs_src_r = fx.slice(qs_row_r, (None, qd_chunk_r))
                        qs_frag_r = fx.make_fragment_like(qs_src_r)
                        fx.copy(g_copy, qs_src_r, qs_frag_r)
                        qs_vec_r = fx.Vector(fx.memref_load_vec(qs_frag_r))
                        qd0_r = qd_chunk_r * Int32(vec)
                        for iq_r in range_constexpr(vec):
                            qv_r = qreq_live_r.select(
                                qs_vec_r[iq_r].to(Float32), Float32(0.0)
                            )
                            q_lds[qrow_local_r, qhead_r, qd0_r + Int32(iq_r)] = qv_r.to(
                                BFloat16
                            )
                    tile_end = _idiv(req_visible + Int32(block_n - 1), Int32(block_n))
                    tile_end = (tile_end < tile0 + tiles_per_wg).select(
                        tile_end, tile0 + tiles_per_wg
                    )
                    # The tile loop's names carry _t, for the same reason.
                    for cur_tile in range(tile0, tile_end, one):
                        tile_t = Int32(cur_tile)
                        for part_t in range_constexpr(k_vectors_per_thread):
                            linear_t = tid + Int32(part_t * block_threads)
                            col_t = _idiv(linear_t, Int32(vec_chunks))
                            d_chunk_t = linear_t - col_t * Int32(vec_chunks)
                            score_col_t = tile_t * Int32(block_n) + col_t
                            # req_visible is within n_columns and the context.
                            col_live_t = score_col_t < req_visible
                            safe_col_t = col_live_t.select(score_col_t, zero)
                            logical_page_t = _idiv(safe_col_t, page)
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
                        n_row_t = wave * Int32(16) + lane_m
                        accs_t = []
                        for head_t in range_constexpr(mfma_chains):
                            acc4_t = fx.Vector.filled(4, 0.0, Float32)
                            a_head_t = a_head if const_expr(fold_heads) else head_t
                            for ks_t in range_constexpr(k_steps):
                                dk_t = Int32(ks_t * qk_k) + lane_kg * Int32(qk_vec)
                                a_vec_t = fx.Vector.from_elements(
                                    [
                                        q_lds[a_row, a_head_t, dk_t + Int32(i_t)]
                                        for i_t in range_constexpr(qk_vec)
                                    ],
                                    BFloat16,
                                )
                                b_vec_t = fx.Vector.from_elements(
                                    [
                                        k_lds[n_row_t, dk_t + Int32(i_t)]
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

                        # 16x16 C: n = lane%16 (column), m = 4*(lane/16) + elem:
                        # the row, or 4*row + head when folded.
                        out_col_t = tile_t * Int32(block_n) + n_row_t
                        # A visible column on an invalid page scored a zeroed K;
                        # it must not tie into the top-512.
                        page_live_t = live_lds[n_row_t] != zero
                        if const_expr(fold_heads):
                            # m = 4*(lane/16) + elem = 4*row + head.
                            row_local_t = lane_kg
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
                                scores[row_t, out_col_t] = live_t.select(
                                    score_t * score_scale, _neg_inf()
                                )
                        else:
                            for i_t in range_constexpr(4):
                                row_local_t = lane_kg * Int32(4) + Int32(i_t)
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
        ).launch(
            grid=(row_tiles, tile_groups, 1),
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
