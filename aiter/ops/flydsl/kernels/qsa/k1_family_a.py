# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Family A FlyDSL QSA K1 (SILOTIGER-1047 2b): paged ReLU-sum + tiled top-512.

Streams compressed index-K from the paged cache in 512-slot tiles, scores
complete causal blocks, and merges a running LDS top-512. Writes
``block_ids [M, 512]``. Scores never land in a global ``[M, n_blocks]`` buffer.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import BFloat16, Float32, Int32, gpu, range_constexpr

from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from aiter.ops.flydsl.kernels.qsa.shapes import FAMILY_A_INDEXER, FAMILY_A_SCORE_SCALE
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, buf_copy_atom

_BLOCK_THREADS = 64
_TILE = 512
_K = FAMILY_A_INDEXER.block_budget
_CANDIDATES = _K + _TILE
_H = FAMILY_A_INDEXER.n_heads
_D = FAMILY_A_INDEXER.head_dim
_R = FAMILY_A_INDEXER.compress_ratio
_VEC = 8
_BITONIC_STAGES = tuple(
    (span, stride)
    for span in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    for stride in tuple(1 << shift for shift in range(span.bit_length() - 2, -1, -1))
)


def _idiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def _neg_inf():
    return Float32(float("-inf"))


def build_qsa_k1_family_a_module(page_size: int):
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if _TILE % _BLOCK_THREADS:
        raise ValueError("tile size must be a multiple of block threads")
    if _K != _TILE:
        raise ValueError("family A local heap is one tile (k=512)")
    if _D % _VEC:
        raise ValueError("head dimension must be a multiple of vector width")
    tile_steps = _TILE // _BLOCK_THREADS
    candidate_steps = _CANDIDATES // _BLOCK_THREADS

    @fx.struct
    class SharedStorage:
        q: fx.Array[BFloat16, _H * _D, 16]
        cand_s: fx.Array[Float32, _CANDIDATES, 16]
        cand_c: fx.Array[Int32, _CANDIDATES, 16]

    @flyc.kernel(
        name="qsa_k1_family_a_"
        + kernel_signature(ps=page_size, tile=_TILE, k=_K, h=_H, d=_D),
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def qsa_k1_family_a_kernel(
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
        page = Int32(page_size)
        n_col = n_columns
        vec_layout = fx.make_layout(_VEC, 1)
        k_copy = buf_copy_atom(16, BFloat16)
        q_load = buf_copy_atom(16, BFloat16)
        q_store = fx.make_copy_atom(fx.UniversalCopy128b(), BFloat16)
        q_buf = fx.rocdl.make_buffer_tensor(q)
        k_buf = fx.rocdl.make_buffer_tensor(k_cache)

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        smem_q = storage.q.view(fx.make_layout((_H, _D), (_D, 1)))
        cand_s = storage.cand_s.view(fx.make_layout(_CANDIDATES, 1))
        cand_c = storage.cand_c.view(fx.make_layout(_CANDIDATES, 1))

        req = token_to_req[row]
        valid_req = (req >= zero) & (req < n_req)
        safe_req = valid_req.select(req, zero)
        qpos = query_positions[row]
        slen = valid_req.select(context_lens[safe_req], zero)
        vis_q = _idiv(qpos + one, Int32(_R))
        vis_s = _idiv(slen, Int32(_R))
        visible = (vis_q < vis_s).select(vis_q, vis_s)
        n_tiles = fx.ceildiv(n_col, Int32(_TILE))

        def better(s, c, bs, bc):
            return (c >= zero) & ((s > bs) | ((s == bs) & ((bc < zero) | (c < bc))))

        # One thread owns one contiguous BF16x8 Q vector. The TV layout maps
        # 4x16 threads onto the exact [H=4,D=128] row.
        q_tile, q_tv = fx.make_layout_tv(
            fx.make_layout((_H, _D // _VEC), (_D // _VEC, 1)),
            fx.make_layout((1, _VEC), (_VEC, 1)),
        )
        q_thr = fx.make_tiled_copy(q_load, q_tv, q_tile).get_slice(tid)
        q_row = fx.slice(q_buf, (row, None, None))
        q_block = fx.slice(fx.zipped_divide(q_row, q_tile), (None, (0, 0)))
        q_src = q_thr.partition_S(q_block)
        q_dst = q_thr.partition_D(smem_q)
        q_frag = fx.make_fragment_like(q_src)
        fx.copy(q_load, q_src, q_frag)
        fx.copy(q_store, q_frag, q_dst)
        for t in range_constexpr(tile_steps):
            j = tid + Int32(t * _BLOCK_THREADS)
            cand_s[j] = _neg_inf()
            cand_c[j] = neg_one
        gpu.barrier()

        def score_col(col):
            logical_page = _idiv(col, page)
            off = col - logical_page * page
            phys = page_table[safe_req, logical_page]
            k_row = fx.slice(k_buf, (phys, off, zero, None))
            k_chunks = fx.logical_divide(k_row, vec_layout)
            total = Float32(0.0)
            for h in range_constexpr(_H):
                q_chunks = fx.logical_divide(fx.slice(smem_q, (h, None)), vec_layout)
                acc = Float32(0.0)
                for chunk in range_constexpr(_D // _VEC):
                    k_src = fx.slice(k_chunks, (None, chunk))
                    q_src = fx.slice(q_chunks, (None, chunk))
                    k_frag = fx.make_fragment_like(k_src)
                    q_frag = fx.make_fragment_like(q_src)
                    fx.copy(k_copy, k_src, k_frag)
                    fx.copy(q_store, q_src, q_frag)
                    k_vec = fx.Vector(fx.memref_load_vec(k_frag))
                    q_vec = fx.Vector(fx.memref_load_vec(q_frag))
                    for j in range_constexpr(_VEC):
                        acc = acc + q_vec[j].to(Float32) * k_vec[j].to(Float32)
                total = total + fx.max(acc, Float32(0.0))
            return total * score_scale

        for tile in range(zero, n_tiles, one):
            tile_base = tile * Int32(_TILE)
            for t in range_constexpr(tile_steps):
                local = tid + Int32(t * _BLOCK_THREADS)
                blk = tile_base + local
                candidate = Int32(_K) + local
                live = (blk < n_col) & (blk < visible) & valid_req
                if live:
                    cand_s[candidate] = score_col(blk)
                    cand_c[candidate] = blk
                else:
                    cand_s[candidate] = _neg_inf()
                    cand_c[candidate] = neg_one
            gpu.barrier()

            # Sort the running top-512 plus this 512-slot tile in-place. The
            # final order is best-first, so the lower half becomes the next heap.
            for span, stride in _BITONIC_STAGES:
                for t in range_constexpr(candidate_steps):
                    j = tid + Int32(t * _BLOCK_THREADS)
                    peer = j ^ Int32(stride)
                    if j < peer:
                        s0 = cand_s[j]
                        c0 = cand_c[j]
                        s1 = cand_s[peer]
                        c1 = cand_c[peer]
                        best_first = (j & Int32(span)) == zero
                        swap = best_first.select(
                            better(s1, c1, s0, c0),
                            better(s0, c0, s1, c1),
                        )
                        cand_s[j] = swap.select(s1, s0)
                        cand_c[j] = swap.select(c1, c0)
                        cand_s[peer] = swap.select(s0, s1)
                        cand_c[peer] = swap.select(c0, c1)
                gpu.barrier()

        for t in range_constexpr(tile_steps):
            j = tid + Int32(t * _BLOCK_THREADS)
            block_ids[row, j] = cand_c[j]

    @flyc.jit
    def launch_qsa_k1_family_a(
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
        qsa_k1_family_a_kernel(
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

    return launch_qsa_k1_family_a


@lru_cache(maxsize=8)
def _plan(page_size: int):
    return build_qsa_k1_family_a_module(page_size)


def qsa_k1_family_a_serves(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
) -> str | None:
    """Why this K1 kernel cannot serve these tensors, or None if it can."""
    idx = FAMILY_A_INDEXER
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        return f"q and k_cache must be bfloat16, got {q.dtype} and {k_cache.dtype}"
    if q.dim() != 3 or q.shape[1] != idx.n_heads or q.shape[2] != idx.head_dim:
        return f"q must be [M, {idx.n_heads}, {idx.head_dim}], got {tuple(q.shape)}"
    if k_cache.dim() != 4:
        return f"k_cache must be [pages, page_size, H, D], got {tuple(k_cache.shape)}"
    if k_cache.shape[2] != idx.kv_heads or k_cache.shape[3] != idx.head_dim:
        return (
            f"k_cache KV/D must be ({idx.kv_heads}, {idx.head_dim}), "
            f"got {k_cache.shape[2:]}"
        )
    if page_table.dim() != 2 or page_table.dtype != torch.int32:
        return (
            f"page_table must be int32 [n_req, n_pages], got {tuple(page_table.shape)}"
        )
    return None


def qsa_k1_family_a_block_ids(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor | None = None,
    score_scale: float = FAMILY_A_SCORE_SCALE,
) -> torch.Tensor:
    """Write family A indexer ``block_ids [M, 512]`` from paged compressed K.

    Streams 512-slot tiles and merges a running LDS top-512. Does not allocate
    a score matrix. Expand+tail is still a separate launch.
    """
    reason = qsa_k1_family_a_serves(q, k_cache, page_table)
    if reason is not None:
        raise ValueError(f"[FlyDSL qsa_k1_family_a] {reason}")
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
    _run_compiled(
        _plan(page_size),
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
    return out
