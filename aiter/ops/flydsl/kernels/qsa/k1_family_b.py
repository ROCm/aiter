# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Family B FlyDSL QSA K1: emit for short rows, family A scorer for long rows.

Gluon-parity indexer: ``H`` is 4 or 8, ``D=128``, ``k=512``. Writes
``block_ids [M, 512]``. Short rows (``n_columns <= 512``) use the same emit
kernel as family A. Longer rows reuse family A's BLOCK_N=32 MFMA scorer plus
decode/streaming radix; ``H=8`` is a second compile of those scorers.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import Float32, Int32, gpu

from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from aiter.ops.flydsl.kernels.qsa.k1_family_a import qsa_k1_score_and_select
from aiter.ops.flydsl.kernels.qsa.shapes import FAMILY_B_INDEXER, FAMILY_B_INDEXER_H8
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

_BLOCK_THREADS = 512
_K = FAMILY_B_INDEXER.block_budget
_D = FAMILY_B_INDEXER.head_dim
_R = FAMILY_B_INDEXER.compress_ratio
_FAMILY_B_SCORE_SCALE = FAMILY_B_INDEXER.head_dim**-0.5


def _idiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def build_qsa_k1_family_b_module(page_size: int):
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if _K % _BLOCK_THREADS:
        raise ValueError("block budget must be a multiple of block threads")

    @flyc.kernel(
        name="qsa_k1_family_b_emit_"
        + kernel_signature(ps=page_size, k=_K, blk=_BLOCK_THREADS),
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def qsa_k1_family_b_kernel(
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
    def launch_qsa_k1_family_b(
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
        qsa_k1_family_b_kernel(
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

    return launch_qsa_k1_family_b


@lru_cache(maxsize=8)
def _plan(page_size: int):
    return build_qsa_k1_family_b_module(page_size)


def qsa_k1_family_b_serves(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
) -> str | None:
    """Why this K1 kernel cannot serve these tensors, or None if it can."""
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        return f"q and k_cache must be bfloat16, got {q.dtype} and {k_cache.dtype}"
    if q.dim() != 3 or q.shape[1] not in (4, 8) or q.shape[2] != _D:
        return f"q must be [M, 4|8, {_D}], got {tuple(q.shape)}"
    if k_cache.dim() != 4:
        return f"k_cache must be [pages, page_size, H, D], got {tuple(k_cache.shape)}"
    idx = FAMILY_B_INDEXER if q.shape[1] == 4 else FAMILY_B_INDEXER_H8
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


def qsa_k1_family_b_block_ids(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor | None = None,
    score_scale: float = _FAMILY_B_SCORE_SCALE,
) -> torch.Tensor:
    """Write family B indexer ``block_ids [M, 512]`` from paged compressed K.

    ``H`` is 4 or 8. Emit when every complete block fits in the budget.
    Longer rows use family A's MFMA scorer (``H=8`` is a second compile)
    plus the same radix selectors. Expand+tail is still a separate launch.
    """
    reason = qsa_k1_family_b_serves(q, k_cache, page_table)
    if reason is not None:
        raise ValueError(f"[FlyDSL qsa_k1_family_b] {reason}")
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
    n_heads = int(q.shape[1])
    if n_columns <= _K:
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
            n_heads,
        )
    return out
