# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL QSA K1: short-context emit or unfused score + top-512.

When ``n_columns <= 512``, every visible block is selected and the emit
kernel writes its id without scoring. Longer rows use independent
16/32-column BF16 MFMA scorer workgroups, an fp32 ``[M, n_columns]`` score
buffer, and a per-row selector. Rows narrower than 32768 columns use the
stable decode radix; wider rows use streaming radix with ``tie='low'``.
Single-request prefill batches 16 rows per scorer workgroup; decode and
multi-request inputs keep the one-row scorer. BLOCK_N=32 is the measured
default for both.

Every shape this serves shares one indexer contract, so the only thing that
varies is the accepted head count. Callers pin it through ``heads``: pass
``(4,)`` for the narrow contract, or take the ``(4, 8)`` default. ``H=8``
is a second compile of the same scorers.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import BFloat16, Float32, Int32, gpu, range_constexpr

from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, buf_copy_atom
from aiter.ops.flydsl.topk.topk_per_row import flydsl_top_k_per_row_decode
from aiter.ops.topk_select import topk_select

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
_SCORE_HEADS = (4, 8)
_SCORE_SCALE = _D**-0.5


def _idiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def _neg_inf():
    return Float32(float("-inf"))


def build_qsa_k1_emit_module(page_size: int):
    """Build the short-context emit kernel. It never reads Q or K."""
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if _K % _BLOCK_THREADS:
        raise ValueError("block budget must be a multiple of block threads")

    @flyc.kernel(
        name="qsa_k1_emit_" + kernel_signature(ps=page_size, k=_K, blk=_BLOCK_THREADS),
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def qsa_k1_emit_kernel(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        block_ids: fx.Tensor,
        n_columns: Int32,
        n_req: Int32,
        score_scale: Float32,
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
    def launch_qsa_k1_emit(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        page_table: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        block_ids: fx.Tensor,
        n_columns: Int32,
        n_req: Int32,
        score_scale: Float32,
        rows: Int32,
        stream: fx.Stream,
    ):
        qsa_k1_emit_kernel(
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            block_ids,
            n_columns,
            n_req,
            score_scale,
        ).launch(
            grid=(rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_qsa_k1_emit


def build_qsa_k1_scores_module(
    page_size: int,
    use_k32: bool,
    block_n: int,
    n_heads: int = _H,
):
    """Build a long-context paged MFMA scorer."""
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if block_n not in (16, 32):
        raise ValueError(f"score block_n must be 16 or 32, got {block_n}")
    if n_heads not in _SCORE_HEADS:
        raise ValueError(f"score heads must be {_SCORE_HEADS}, got {n_heads}")

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
        name="qsa_k1_scores_"
        + kernel_signature(
            ps=page_size,
            bn=block_n,
            h=n_heads,
            d=_D,
            blk=block_threads,
            qkk=qk_k,
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def qsa_k1_scores_kernel(
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
        q_buf = fx.rocdl.make_buffer_tensor(q)
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

        def qk_mfma(a_vec, b_vec, c_vec):
            fx.memref_store_vec(a_vec, qk_a)
            fx.memref_store_vec(b_vec, qk_b)
            fx.memref_store_vec(c_vec, qk_c)
            fx.mma_atom_call(qk_mma, qk_c, qk_a, qk_b, qk_c)
            return fx.memref_load_vec(qk_c)

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
        k_row = fx.logical_divide(fx.slice(k_buf, (phys, off, zero, None)), vec_layout)
        for part in range_constexpr(chunks_per_thread):
            d_chunk = chunk + Int32(part * (block_threads // block_n))
            k_src = fx.slice(k_row, (None, d_chunk))
            k_frag = fx.make_fragment_like(k_src)
            fx.copy(g_copy, k_src, k_frag)
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
                    [q_lds[lane_m, md0 + Int32(i)] for i in range_constexpr(qk_vec)],
                    BFloat16,
                )
                b_vec = fx.Vector.from_elements(
                    [k_lds[n_row, md0 + Int32(i)] for i in range_constexpr(qk_vec)],
                    BFloat16,
                )
                acc4 = fx.Vector(qk_mfma(a_vec, b_vec, acc4))
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
                    scores[row, out_col] = live.select(score * score_scale, _neg_inf())

    @flyc.jit
    def launch_qsa_k1_scores(
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
        rows: Int32,
        tiles: Int32,
        stream: fx.Stream,
    ):
        qsa_k1_scores_kernel(
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
        ).launch(
            grid=(tiles, rows, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_qsa_k1_scores


def build_qsa_k1_prefill_scores_module(
    page_size: int,
    use_k32: bool,
    n_heads: int = _H,
):
    """Build the single-request, 16-row by 32-column MFMA scorer."""
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if n_heads not in _SCORE_HEADS:
        raise ValueError(f"score heads must be {_SCORE_HEADS}, got {n_heads}")

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
        name="qsa_k1_prefill_scores_"
        + kernel_signature(
            ps=page_size,
            bm=block_m,
            bn=block_n,
            h=n_heads,
            d=_D,
            blk=block_threads,
            qkk=qk_k,
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def qsa_k1_prefill_scores_kernel(
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
        q_buf = fx.rocdl.make_buffer_tensor(q)
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

        def qk_mfma(a_vec, b_vec, c_vec):
            fx.memref_store_vec(a_vec, qk_a)
            fx.memref_store_vec(b_vec, qk_b)
            fx.memref_store_vec(c_vec, qk_c)
            fx.mma_atom_call(qk_mma, qk_c, qk_a, qk_b, qk_c)
            return fx.memref_load_vec(qk_c)

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

        for part in range_constexpr(k_vectors_per_thread):
            linear = tid + Int32(part * block_threads)
            col = _idiv(linear, Int32(vec_chunks))
            d_chunk = linear - col * Int32(vec_chunks)
            score_col = tile * Int32(block_n) + col
            col_live = score_col < n_columns
            safe_col = col_live.select(score_col, zero)
            logical_page = _idiv(safe_col, page)
            off = safe_col - logical_page * page
            phys = page_table[zero, logical_page]
            k_row = fx.logical_divide(
                fx.slice(k_buf, (phys, off, zero, None)), vec_layout
            )
            k_src = fx.slice(k_row, (None, d_chunk))
            k_frag = fx.make_fragment_like(k_src)
            fx.copy(g_copy, k_src, k_frag)
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
                    d0 = wave * Int32(64) + Int32(ks * qk_k) + lane_kg * Int32(qk_vec)
                    a_vec = fx.Vector.from_elements(
                        [
                            q_lds[lane_m, head, d0 + Int32(i)]
                            for i in range_constexpr(qk_vec)
                        ],
                        BFloat16,
                    )
                    b_vec = fx.Vector.from_elements(
                        [k_lds[n_row, d0 + Int32(i)] for i in range_constexpr(qk_vec)],
                        BFloat16,
                    )
                    acc4 = fx.Vector(qk_mfma(a_vec, b_vec, acc4))
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
                        scores[row, out_col] = live.select(
                            score * score_scale, _neg_inf()
                        )

    @flyc.jit
    def launch_qsa_k1_prefill_scores(
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
        row_tiles: Int32,
        tiles: Int32,
        stream: fx.Stream,
    ):
        qsa_k1_prefill_scores_kernel(
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
        ).launch(
            grid=(tiles, row_tiles, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_qsa_k1_prefill_scores


@lru_cache(maxsize=8)
def _emit_plan(page_size: int):
    return build_qsa_k1_emit_module(page_size)


@lru_cache(maxsize=16)
def _scores_plan(page_size: int, use_k32: bool, block_n: int, n_heads: int = _H):
    return build_qsa_k1_scores_module(page_size, use_k32, block_n, n_heads)


@lru_cache(maxsize=8)
def _prefill_scores_plan(page_size: int, use_k32: bool, n_heads: int = _H):
    return build_qsa_k1_prefill_scores_module(page_size, use_k32, n_heads)


def qsa_k1_score_and_select(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    n_columns: int,
    score_scale: float,
    n_heads: int,
) -> torch.Tensor:
    """Score long rows into ``[M, n_columns]`` and write top-512 ids into ``out``.

    ``n_heads`` is 4 or 8, each a separate compile. Selection is the
    stable decode radix below 32768 columns and streaming radix
    (``tie='low'``) at or above that width.
    """
    if n_heads not in _SCORE_HEADS:
        raise ValueError(f"score heads must be {_SCORE_HEADS}, got {n_heads}")
    if q.shape[1] != n_heads:
        raise ValueError(f"q must have {n_heads} heads, got {tuple(q.shape)}")
    m = q.shape[0]
    page_size = k_cache.shape[1]
    score_block_n = 32
    scores = torch.empty(m, n_columns, dtype=torch.float32, device=q.device)
    row_lens = torch.empty(m, dtype=torch.int32, device=q.device)
    use_k32 = torch.cuda.get_device_properties(q.device).gcnArchName.startswith(
        "gfx950"
    )
    score_tiles = (n_columns + score_block_n - 1) // score_block_n
    if context_lens.shape[0] == 1 and m >= 16:
        _run_compiled(
            _prefill_scores_plan(page_size, use_k32, n_heads),
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            scores,
            row_lens,
            int(n_columns),
            m,
            float(score_scale),
            (m + 15) // 16,
            score_tiles,
            torch.cuda.current_stream(q.device),
        )
    else:
        _run_compiled(
            _scores_plan(page_size, use_k32, score_block_n, n_heads),
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            scores,
            row_lens,
            int(n_columns),
            int(context_lens.shape[0]),
            float(score_scale),
            m,
            score_tiles,
            torch.cuda.current_stream(q.device),
        )
    if n_columns >= _STREAM_SELECT_MIN_COLUMNS:
        topk_select(
            scores,
            _K,
            end=row_lens,
            output_idx=out,
            tie="low",
        )
    else:
        flydsl_top_k_per_row_decode(
            scores,
            1,
            row_lens,
            out,
            m,
            scores.stride(0),
            scores.stride(1),
            k=_K,
            stable=True,
        )
    return out


def qsa_k1_serves(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    heads: tuple[int, ...] = _SCORE_HEADS,
) -> str | None:
    """Why this K1 kernel cannot serve these tensors, or None if it can.

    ``heads`` narrows the accepted indexer head count; pass ``(4,)`` to
    reject the 8-head variant. Every other check is the same either way.
    """
    if not heads or any(h not in _SCORE_HEADS for h in heads):
        raise ValueError(f"heads must be a non-empty subset of {_SCORE_HEADS}")
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        return f"q and k_cache must be bfloat16, got {q.dtype} and {k_cache.dtype}"
    if q.dim() != 3 or q.shape[1] not in heads or q.shape[2] != _D:
        allowed = "|".join(str(h) for h in heads)
        return f"q must be [M, {allowed}, {_D}], got {tuple(q.shape)}"
    if k_cache.dim() != 4:
        return f"k_cache must be [pages, page_size, H, D], got {tuple(k_cache.shape)}"
    if k_cache.shape[2] != _KV_HEADS or k_cache.shape[3] != _D:
        return f"k_cache KV/D must be ({_KV_HEADS}, {_D}), got {k_cache.shape[2:]}"
    if page_table.dim() != 2 or page_table.dtype != torch.int32:
        return (
            f"page_table must be int32 [n_req, n_pages], got {tuple(page_table.shape)}"
        )
    return None


def qsa_k1_block_ids(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor | None = None,
    score_scale: float = _SCORE_SCALE,
    heads: tuple[int, ...] = _SCORE_HEADS,
) -> torch.Tensor:
    """Write indexer ``block_ids [M, 512]`` from paged compressed K.

    Rows no wider than 512 use the fused emit path. Longer rows materialize
    scores with a BLOCK_N=32 MFMA writer; single-request prefill batches 16
    query rows while other inputs use one row per workgroup. Selection is the
    stable decode radix below 32768 columns and streaming radix
    (``tie='low'``) at or above that width. Expand+tail is still separate.

    ``heads`` is the accepted head count. Pass ``(4,)`` to keep the contract
    narrow; the ``(4, 8)`` default also admits the 8-head indexer.
    """
    reason = qsa_k1_serves(q, k_cache, page_table, heads)
    if reason is not None:
        raise ValueError(f"[FlyDSL qsa_k1] {reason}")
    m = q.shape[0]
    if token_to_req.shape != (m,) or token_to_req.dtype != torch.int32:
        raise ValueError(f"token_to_req must be int32 [{m}]")
    if query_positions.shape != (m,) or query_positions.dtype != torch.int32:
        raise ValueError(f"query_positions must be int32 [{m}]")
    if context_lens.dim() != 1 or context_lens.dtype != torch.int32:
        raise ValueError("context_lens must be 1-D int32")
    if out is None:
        out = torch.empty(m, _K, dtype=torch.int32, device=q.device)
    elif out.shape != (m, _K) or out.dtype != torch.int32:
        raise ValueError(f"out must be int32 [{m}, {_K}], got {tuple(out.shape)}")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    tensors = (q, k_cache, page_table, token_to_req, query_positions, context_lens, out)
    if any(not t.is_cuda for t in tensors):
        raise ValueError("every tensor must be on the GPU")
    if any(t.device != q.device for t in tensors[1:]):
        raise ValueError("every tensor must be on the same GPU")
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    page_table = page_table.contiguous()
    token_to_req = token_to_req.contiguous()
    query_positions = query_positions.contiguous()
    context_lens = context_lens.contiguous()
    page_size = k_cache.shape[1]
    n_columns = page_table.shape[1] * page_size
    if n_columns <= _K:
        _run_compiled(
            _emit_plan(page_size),
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            out,
            int(n_columns),
            int(context_lens.shape[0]),
            float(score_scale),
            m,
            torch.cuda.current_stream(q.device),
        )
    else:
        qsa_k1_score_and_select(
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            context_lens,
            out,
            int(n_columns),
            float(score_scale),
            int(q.shape[1]),
        )
    return out
