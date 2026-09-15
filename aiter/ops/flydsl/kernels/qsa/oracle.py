# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""fp32 QSA oracle: block-causal ReLU-sum -> top-k -> expand+tail -> sparse GQA.

Authoritative math: Qwen3.8-Next tech report ?2.1 (QSA). This path *may*
materialize ``[M, n_blocks]`` scores; FlyDSL K1 must not.

Tie-break (locked to the live AMD HIP selector ``top_k_per_row_decode``): among
equal finite scores, keep the **smaller block index**. Invalid / incomplete
blocks are ``-inf`` and never win. Selected ids are written in that sort order
(higher score first; ties already smaller-index-first). Remaining slots are
``-1``.

Expand+tail matches vLLM ``_expand_qsa_indices_kernel`` (PR 53896 / vLLM main):
complete blocks expand to ``r`` tokens; the open group contributes at most
``r-1`` tail tokens after the expanded prefix; padding is ``-1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .shapes import QsaGqaSpec, QsaIndexerSpec


def qsa_visible_blocks(
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """Number of complete ``r``-token blocks visible to each query row.

    Block ``b`` (tokens ``[r*b, r*b+r)``) is complete for query position ``i``
    iff ``r*b + r - 1 <= i``, i.e. ``b < (i + 1) // r``, and also
    ``b < seq_len // r``.
    """
    n_req = sequence_lengths.shape[0]
    req = token_to_req.clamp(0, n_req - 1)
    slen = sequence_lengths[req]
    valid_req = (token_to_req >= 0) & (token_to_req < n_req)
    slen = torch.where(valid_req, slen, slen.new_zeros(()))
    r = compress_ratio
    return torch.minimum((query_positions + 1) // r, slen // r)


def qsa_indexer_scores(
    q: torch.Tensor,
    k_bar: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    score_scale: float = 1.0,
) -> torch.Tensor:
    """``I_ib = score_scale * sum_h ReLU(dot(q[h], k_bar[b]))`` for complete blocks.

    Args:
        q: ``[M, H, D]`` (indexer heads; family A is 4 x 128).
        k_bar: ``[n_blocks, D]`` mean-pooled compressed keys (one KV head).
        score_scale: optional ``1/sqrt(D)``; must not change argmax.
    """
    q_f = q.to(torch.float32)
    k_f = k_bar.to(torch.float32)
    # [M, H, n_blocks]
    dots = torch.einsum("mhd,nd->mhn", q_f, k_f)
    scores = F.relu(dots).sum(dim=1) * float(score_scale)
    visible = qsa_visible_blocks(
        query_positions, sequence_lengths, token_to_req, compress_ratio
    )
    n_blocks = k_bar.shape[0]
    block_ids = torch.arange(n_blocks, device=q.device)
    complete = block_ids.unsqueeze(0) < visible.unsqueeze(1)
    return torch.where(complete, scores, torch.full_like(scores, float("-inf")))


def qsa_topk_blocks(
    scores: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Per-row top-``k`` block ids. Smaller index wins exact-score ties.

    Incomplete / padded columns are ``-inf`` and are **not** selected (live AMD
    ``top_k_per_row_decode`` only reads the first ``visible`` columns). Slots
    past the number of finite scores are ``-1``.
    """
    n_blocks = scores.shape[1]
    kk = min(k, n_blocks)
    # Stable descending sort keeps original (increasing) index on ties.
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :kk]
    pick = torch.gather(scores, 1, order)
    chosen = torch.where(torch.isfinite(pick), order, order.new_full((), -1))
    chosen = chosen.to(torch.int32)
    if kk == k:
        return chosen
    pad = torch.full(
        (scores.shape[0], k - kk),
        -1,
        dtype=torch.int32,
        device=scores.device,
    )
    return torch.cat([chosen, pad], dim=1)


def qsa_expand_tail(
    block_ids: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Expand selected blocks to token ids and append the incomplete tail.

    Output width is ``token_topk + compress_ratio - 1`` (family A: 2051).
    """
    if token_topk % compress_ratio:
        raise ValueError("token_topk must be divisible by compress_ratio")
    block_topk = token_topk // compress_ratio
    if block_ids.shape[1] != block_topk:
        raise ValueError(
            f"block_ids width {block_ids.shape[1]} != token_topk/r {block_topk}"
        )
    r = compress_ratio
    output_width = token_topk + r - 1
    device = block_ids.device
    rows = block_ids.shape[0]
    columns = torch.arange(output_width, device=device)
    visible = qsa_visible_blocks(query_positions, sequence_lengths, token_to_req, r)
    complete_blocks = torch.minimum(visible, torch.full_like(visible, block_topk))
    expanded_count = complete_blocks * r
    tail_start = ((query_positions + 1) // r) * r
    tail_count = (query_positions + 1) - tail_start

    is_expanded = columns.unsqueeze(0) < expanded_count.unsqueeze(1)
    block_rank = (columns // r).clamp(max=block_topk - 1)
    rank = block_rank.unsqueeze(0).expand(rows, output_width)
    block = torch.gather(block_ids, 1, rank)
    offset = columns % r
    expanded = block.to(torch.int64) * r + offset
    tail_offset = columns.unsqueeze(0) - expanded_count.unsqueeze(1)
    is_tail = (
        (~is_expanded)
        & (tail_offset < tail_count.unsqueeze(1))
        & (tail_offset < (r - 1))
    )
    token = torch.where(is_expanded, expanded, tail_start.unsqueeze(1) + tail_offset)
    n_req = sequence_lengths.shape[0]
    req = token_to_req.clamp(0, n_req - 1)
    slen = sequence_lengths[req]
    valid_req = (token_to_req >= 0) & (token_to_req < n_req)
    slen = torch.where(valid_req, slen, slen.new_zeros(()))
    valid = (is_expanded | is_tail) & (token >= 0) & (token < slen.unsqueeze(1))
    return torch.where(valid, token, token.new_full((), -1)).to(torch.int32)


def qsa_sparse_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Standard GQA on the selected token positions (``-1`` padded).

    Args:
        q: ``[M, Hq, D]``
        k, v: ``[S, Hk, D]`` contiguous (unpaged) keys/values.
        indices: ``[M, W]`` token ids, ``-1`` empty.
    """
    m, hq, d = q.shape
    _s, hk, dk = k.shape
    if dk != d or v.shape != k.shape:
        raise ValueError("QSA GQA Q/K/V head_dim or K/V shape mismatch")
    if hq % hk:
        raise ValueError("QSA GQA requires Hq divisible by Hk")
    if softmax_scale is None:
        softmax_scale = d**-0.5
    group = hq // hk
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    valid = indices >= 0
    safe = indices.clamp(min=0).to(torch.int64)
    k_sel = k_f[safe]  # [M, W, Hk, D]
    v_sel = v_f[safe]
    k_sel = k_sel * valid.unsqueeze(-1).unsqueeze(-1)
    v_sel = v_sel * valid.unsqueeze(-1).unsqueeze(-1)
    qg = q_f.view(m, hk, group, d)
    scores = torch.matmul(qg, k_sel.permute(0, 2, 3, 1)) * float(softmax_scale)
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.matmul(attn, v_sel.permute(0, 2, 1, 3))
    return out.reshape(m, hq, d)


@dataclass
class QsaOracleResult:
    scores: torch.Tensor
    block_ids: torch.Tensor
    indices: torch.Tensor
    output: torch.Tensor


def qsa_oracle(
    q_indexer: torch.Tensor,
    k_bar: torch.Tensor,
    q_gqa: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    indexer: QsaIndexerSpec,
    gqa: QsaGqaSpec,
    score_scale: float = 1.0,
    softmax_scale: float | None = None,
    out_dtype: torch.dtype | None = None,
) -> QsaOracleResult:
    """Full QSA layer reference (indexer through sparse GQA)."""
    if q_indexer.shape[-2:] != (indexer.n_heads, indexer.head_dim):
        raise ValueError(
            f"indexer q shape {tuple(q_indexer.shape[-2:])} != "
            f"{(indexer.n_heads, indexer.head_dim)}"
        )
    if q_gqa.shape[-2:] != (gqa.n_heads, gqa.head_dim):
        raise ValueError(
            f"gqa q shape {tuple(q_gqa.shape[-2:])} != "
            f"{(gqa.n_heads, gqa.head_dim)}"
        )
    scores = qsa_indexer_scores(
        q_indexer,
        k_bar,
        query_positions,
        sequence_lengths,
        token_to_req,
        indexer.compress_ratio,
        score_scale=score_scale,
    )
    block_ids = qsa_topk_blocks(scores, indexer.block_budget)
    indices = qsa_expand_tail(
        block_ids,
        query_positions,
        sequence_lengths,
        token_to_req,
        indexer.compress_ratio,
        indexer.token_budget,
    )
    output = qsa_sparse_gqa(q_gqa, k, v, indices, softmax_scale=softmax_scale)
    if out_dtype is None:
        out_dtype = q_gqa.dtype
    return QsaOracleResult(
        scores=scores,
        block_ids=block_ids,
        indices=indices,
        output=output.to(out_dtype),
    )
