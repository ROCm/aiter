# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL absorbed sparse MLA host wrappers. bf16, H=16, D=512."""

from __future__ import annotations

import math

import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.sparse_mla_qblock import (
    HEAD_DIM,
    NUM_HEADS,
    compile_sparse_mla_qblock,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

# slot * D and token * H * D are int32 element offsets. Larger shapes fall back.
_MAX_TOKENS = (1 << 31) // (NUM_HEADS * HEAD_DIM)
_MAX_KV_ROWS = (1 << 31) // HEAD_DIM

__all__ = [
    "sparse_mla_one_query_fwd_flydsl",
    "sparse_mla_qblock_fwd_flydsl",
]


def _require_gfx942():
    gfx = get_gfx()
    if gfx != "gfx942":
        raise RuntimeError(
            f"FlyDSL sparse_mla_qblock is gfx942-only (64 KiB LDS), got {gfx}"
        )


def _require_contract(q: torch.Tensor, kv: torch.Tensor, out: torch.Tensor):
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("FlyDSL sparse_mla_qblock is bf16-only in this ABI")
    if q.ndim != 3 or kv.ndim != 2:
        raise ValueError(
            f"expected q[T,H,D] kv[N,D], got {tuple(q.shape)} {tuple(kv.shape)}"
        )
    n_tok, num_heads, head_dim = q.shape
    if num_heads != NUM_HEADS or head_dim != HEAD_DIM:
        raise ValueError(
            f"FlyDSL sparse_mla_qblock is specialized for H={NUM_HEADS} D={HEAD_DIM}, "
            f"got H={num_heads} D={head_dim}"
        )
    if kv.shape[1] != HEAD_DIM:
        raise ValueError(f"kv last dim must be {HEAD_DIM}, got {kv.shape[1]}")
    if out.shape != q.shape or out.dtype != q.dtype:
        raise ValueError("out must match q shape and dtype")
    if not q.is_contiguous() or not kv.is_contiguous() or not out.is_contiguous():
        raise ValueError("q, kv, and out must be contiguous")
    if n_tok >= _MAX_TOKENS or kv.shape[0] >= _MAX_KV_ROWS:
        raise ValueError(
            f"FlyDSL sparse MLA int32 offsets support T < {_MAX_TOKENS} "
            f"and KV rows < {_MAX_KV_ROWS}"
        )
    return n_tok, kv.shape[0]


def _packed_csr(indptr: torch.Tensor, n_tok: int, topk: int) -> bool:
    expected = torch.arange(n_tok + 1, device=indptr.device, dtype=indptr.dtype) * topk
    return bool(torch.equal(indptr, expected))


def _uniform_topk(
    indices: torch.Tensor,
    n_tok: int,
    indptr: torch.Tensor | None,
    topk_hint: int | None = None,
):
    if n_tok <= 0:
        raise ValueError(f"T must be positive, got {n_tok}")
    if topk_hint is not None:
        topk = int(topk_hint)
    elif indptr is None:
        if indices.numel() % n_tok != 0:
            raise ValueError(
                f"indices length {indices.numel()} is not divisible by T={n_tok}"
            )
        topk = indices.numel() // n_tok
    else:
        if indptr.numel() != n_tok + 1:
            raise ValueError(f"kv_indptr must have T+1 entries, got {indptr.numel()}")
        topk = int((indptr[1] - indptr[0]).item())
    if topk <= 0 or indices.numel() != n_tok * topk:
        raise ValueError(
            f"packed topk={topk} requires {n_tok * topk} indices, got {indices.numel()}"
        )
    if indptr is not None and (
        indptr.numel() != n_tok + 1 or not _packed_csr(indptr, n_tok, topk)
    ):
        raise ValueError("FlyDSL sparse MLA requires kv_indptr[t] == t * topk")
    return indices.reshape(-1).contiguous(), topk


def suggest_block_q(indices: torch.Tensor, topk: int) -> int:
    """BQ=4 when the first two queries share KV; otherwise BQ=2."""
    if indices.numel() < 2 * topk:
        return 2
    union = torch.unique(torch.cat([indices[:topk], indices[topk : 2 * topk]])).numel()
    return 4 if union <= int(1.25 * topk) else 2


def pack_qblock_union(
    indices: torch.Tensor,
    num_queries: int,
    topk: int,
    block_q: int,
    block_k: int,
) -> tuple[torch.Tensor | None, int, bool]:
    """Return the shared slot list when every query in each tile matches.

    Pads T up to a multiple of `block_q` by repeating the last query. The
    kernel masks those rows with `n_tok`. A partial union returns skip=False;
    the caller uses the one-query kernel.
    """
    flat = indices.reshape(-1)
    pad = (block_q - num_queries % block_q) % block_q
    if pad:
        tail = flat.view(num_queries, topk)[-1:].expand(pad, topk).reshape(-1)
        flat = torch.cat([flat, tail])
    n_tiles = (num_queries + pad) // block_q
    x = flat.view(n_tiles, block_q, topk)
    if block_q > 1 and torch.equal(x, x[:, :1, :].expand_as(x)):
        if topk % block_k:
            raise ValueError(f"exact Q-block reuse requires topk % {block_k} == 0")
        return x[:, 0, :].contiguous(), topk, True
    return None, 0, False


def _launch(
    q,
    kv,
    idx,
    out,
    n_tok,
    n_unique,
    n_tiles,
    block_q,
    attn_sink,
    softmax_scale,
):
    has_sink = attn_sink is not None
    kernel = compile_sparse_mla_qblock(block_q, has_sink)
    scale = 1.0 / math.sqrt(HEAD_DIM) if softmax_scale is None else float(softmax_scale)
    if has_sink:
        attn_sink = attn_sink.reshape(-1).to(torch.float32).contiguous()
        if attn_sink.numel() != NUM_HEADS:
            raise ValueError(f"attn_sink must have {NUM_HEADS} elements")
    else:
        attn_sink = torch.empty(1, dtype=torch.float32, device=q.device)
    stream = torch.cuda.current_stream(q.device)
    _run_compiled(
        kernel,
        ptr_arg(q, fx.BFloat16),
        ptr_arg(kv, fx.BFloat16),
        ptr_arg(idx, fx.Int32),
        ptr_arg(out, fx.BFloat16),
        int(n_tok),
        int(n_unique),
        float(scale),
        ptr_arg(attn_sink, fx.Float32),
        int(n_tiles),
        fx.Stream(stream),
    )


def sparse_mla_one_query_fwd_flydsl(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_indices: torch.Tensor,
    out: torch.Tensor,
    kv_indptr: torch.Tensor | None = None,
    *,
    attn_sink: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    uniform_topk: int | None = None,
):
    """One CTA per query. `uniform_topk` must describe a packed CSR."""
    _require_gfx942()
    n_tok, _n_kv = _require_contract(q, kv, out)
    flat, topk = _uniform_topk(kv_indices, n_tok, kv_indptr, uniform_topk)
    if topk % 16:
        raise ValueError("FlyDSL fast path requires topk divisible by 16")
    _launch(
        q,
        kv,
        flat.to(dtype=torch.int32),
        out,
        n_tok,
        topk,
        n_tok,
        block_q=1,
        attn_sink=attn_sink,
        softmax_scale=softmax_scale,
    )
    return out


def sparse_mla_qblock_fwd_flydsl(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_indices: torch.Tensor,
    out: torch.Tensor,
    kv_indptr: torch.Tensor | None = None,
    block_q: int | None = None,
    packed: tuple | None = None,
    attn_sink: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    uniform_topk: int | None = None,
):
    """Exact shared-list reuse. Partial overlap falls back to one-query."""
    _require_gfx942()
    n_tok, _n_kv = _require_contract(q, kv, out)
    flat, topk = _uniform_topk(kv_indices, n_tok, kv_indptr, uniform_topk)
    if topk % 16:
        raise ValueError("FlyDSL fast path requires topk divisible by 16")
    if block_q is None:
        block_q = suggest_block_q(flat, topk)
    if block_q not in (2, 4):
        raise ValueError(f"Q-block FlyDSL BQ must be 2 or 4, got {block_q}")
    if packed is None:
        unique, n_unique, skip_mask = pack_qblock_union(flat, n_tok, topk, block_q, 16)
    else:
        unique, n_unique, skip_mask = packed
    if not skip_mask:
        # Partial unions do not beat the one-query kernel once packing is paid.
        return sparse_mla_one_query_fwd_flydsl(
            q,
            kv,
            flat,
            out,
            kv_indptr=kv_indptr,
            attn_sink=attn_sink,
            softmax_scale=softmax_scale,
            uniform_topk=topk,
        )
    _launch(
        q,
        kv,
        unique.reshape(-1).to(dtype=torch.int32).contiguous(),
        out,
        n_tok,
        n_unique,
        unique.shape[0],
        block_q=block_q,
        attn_sink=attn_sink,
        softmax_scale=softmax_scale,
    )
    return out
