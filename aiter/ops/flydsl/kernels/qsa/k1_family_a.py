# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Family A FlyDSL QSA K1 (SILOTIGER-1047 2a): paged ReLU-sum + local top-512.

Streams compressed index-K from the paged cache, scores complete causal blocks,
and writes ``block_ids [M, 512]``. Scores stay in LDS; there is no global
``[M, n_blocks]`` buffer.

2a is decode-correctness only: the compile-time row bound is 512 page-aligned
slots (``L <= 2048`` at ``r=4``). Longer contexts are 2b (tile + merge).
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
_N_MAX = 512
_K = FAMILY_A_INDEXER.block_budget
_H = FAMILY_A_INDEXER.n_heads
_D = FAMILY_A_INDEXER.head_dim
_R = FAMILY_A_INDEXER.compress_ratio
_VEC = 8


def _idiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def _neg_inf():
    return Float32(float("-inf"))


def build_qsa_k1_family_a_module(page_size: int):
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if _N_MAX % _BLOCK_THREADS:
        raise ValueError("n_max must be a multiple of block threads")
    if _D % _VEC:
        raise ValueError("head dimension must be a multiple of vector width")
    steps = _N_MAX // _BLOCK_THREADS

    @fx.struct
    class SharedStorage:
        q: fx.Array[BFloat16, _H * _D, 16]
        scores: fx.Array[Float32, _N_MAX, 16]
        cols: fx.Array[Int32, _N_MAX, 16]

    @flyc.kernel(
        name="qsa_k1_family_a_"
        + kernel_signature(ps=page_size, n=_N_MAX, k=_K, h=_H, d=_D),
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
        q_copy = fx.make_copy_atom(fx.UniversalCopy128b(), BFloat16)
        k_buf = fx.rocdl.make_buffer_tensor(k_cache)

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        smem_q = storage.q.view(fx.make_layout((_H, _D), (_D, 1)))
        smem_s = storage.scores.view(fx.make_layout(_N_MAX, 1))
        smem_c = storage.cols.view(fx.make_layout(_N_MAX, 1))

        req = token_to_req[row]
        valid_req = (req >= zero) & (req < n_req)
        safe_req = valid_req.select(req, zero)
        qpos = query_positions[row]
        slen = valid_req.select(context_lens[safe_req], zero)
        vis_q = _idiv(qpos + one, Int32(_R))
        vis_s = _idiv(slen, Int32(_R))
        visible = (vis_q < vis_s).select(vis_q, vis_s)
        n_live = valid_req.select(fx.min(visible, n_col), zero)
        n_select = fx.min(n_live, Int32(_K))

        # Q is reused by all 512 candidate columns. Stage it once in LDS instead
        # of issuing the same global scalar load from every scoring thread.
        for t in range_constexpr(steps):
            flat = tid + Int32(t * _BLOCK_THREADS)
            h = _idiv(flat, Int32(_D))
            d = flat - h * Int32(_D)
            smem_q[h, d] = q[row, h, d]
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
                    fx.copy(q_copy, q_src, q_frag)
                    k_vec = fx.Vector(fx.memref_load_vec(k_frag))
                    q_vec = fx.Vector(fx.memref_load_vec(q_frag))
                    for j in range_constexpr(_VEC):
                        acc = acc + q_vec[j].to(Float32) * k_vec[j].to(Float32)
                total = total + fx.max(acc, Float32(0.0))
            return total * score_scale

        for t in range_constexpr(steps):
            blk = tid + Int32(t * _BLOCK_THREADS)
            live = (blk < n_col) & (blk < visible) & valid_req
            if live:
                smem_s[blk] = score_col(blk)
                smem_c[blk] = blk
            else:
                smem_s[blk] = _neg_inf()
                smem_c[blk] = neg_one
        gpu.barrier()

        # Each lane scans its eight candidates, then a wave64 butterfly selects
        # one winner. This preserves the smaller-column tie-break without the
        # former lane-0 512x512 serial scan.
        for slot in range(zero, n_select, one):
            best_s = _neg_inf()
            best_c = neg_one
            best_j = neg_one
            for t in range_constexpr(steps):
                j = tid + Int32(t * _BLOCK_THREADS)
                cand = smem_c[j]
                score = smem_s[j]
                take = (cand >= zero) & (
                    (score > best_s)
                    | ((score == best_s) & ((best_c < zero) | (cand < best_c)))
                )
                best_s = take.select(score, best_s)
                best_c = take.select(cand, best_c)
                best_j = take.select(j, best_j)
            for shift in (32, 16, 8, 4, 2, 1):
                peer_s = best_s.shuffle_xor(Int32(shift), Int32(_BLOCK_THREADS))
                peer_c = best_c.shuffle_xor(Int32(shift), Int32(_BLOCK_THREADS))
                peer_j = best_j.shuffle_xor(Int32(shift), Int32(_BLOCK_THREADS))
                take = (peer_c >= zero) & (
                    (peer_s > best_s)
                    | ((peer_s == best_s) & ((best_c < zero) | (peer_c < best_c)))
                )
                best_s = take.select(peer_s, best_s)
                best_c = take.select(peer_c, best_c)
                best_j = take.select(peer_j, best_j)
            if tid == zero:
                block_ids[row, slot] = best_c
            for t in range_constexpr(steps):
                j = tid + Int32(t * _BLOCK_THREADS)
                selected = (best_j >= zero) & (j == best_j)
                smem_c[j] = selected.select(neg_one, smem_c[j])
                smem_s[j] = selected.select(_neg_inf(), smem_s[j])
            gpu.barrier()
        for slot in range(n_select + tid, Int32(_K), Int32(_BLOCK_THREADS)):
            block_ids[row, slot] = neg_one

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
    """Why this 2a kernel cannot serve these tensors, or None if it can."""
    idx = FAMILY_A_INDEXER
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        return "q and k_cache must be bfloat16, got " f"{q.dtype} and {k_cache.dtype}"
    if q.dim() != 3 or q.shape[1] != idx.n_heads or q.shape[2] != idx.head_dim:
        return f"q must be [M, {idx.n_heads}, {idx.head_dim}], got {tuple(q.shape)}"
    if k_cache.dim() != 4:
        return f"k_cache must be [pages, page_size, H, D], got {tuple(k_cache.shape)}"
    if k_cache.shape[2] != idx.kv_heads or k_cache.shape[3] != idx.head_dim:
        return (
            f"k_cache KV/D must be ({idx.kv_heads}, {idx.head_dim}), "
            f"got {k_cache.shape[2:]}"
        )
    page_size = k_cache.shape[1]
    if page_table.dim() != 2 or page_table.dtype != torch.int32:
        return (
            f"page_table must be int32 [n_req, n_pages], got {tuple(page_table.shape)}"
        )
    n_columns = page_table.shape[1] * page_size
    if n_columns > _N_MAX:
        return (
            f"2a K1 row bound is {_N_MAX} page-aligned slots, got {n_columns} "
            f"(seq_len/r padded to pages)"
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

    Does not allocate a score matrix. Expand+tail is still a separate launch.
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
