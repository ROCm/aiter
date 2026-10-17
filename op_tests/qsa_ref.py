# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""fp32 QSA reference and the paged-cache fixtures the tests build from it.

The kernels do not import this module. The Flash-Next and Gluon-parity
specs below are the shapes the tests build from.

Authoritative math: Qwen3.8-Next tech report ?2.1 (QSA). The oracle may
materialize ``[M, n_blocks]`` scores. FlyDSL K1 does the same on every
path except short-row emit: a private fp32 ``[M, n_columns]`` buffer,
then top-512 block ids, and the scores are dropped. ``n_columns`` is
the page-table capacity (``page_table.shape[1] * page_size``), so a
padded ``max_model_len`` table pays full width. At compress ratio 4
that buffer is 1 GiB for 128k tokens at M=8192 and 8 GiB for 1M tokens
at the same M. The allocation is capacity-sized. Score traffic is
live-bounded: both scorers skip tiles past the visible prefix. Fusing
that buffer waits on 128k and 1M measurements. A caller may pass the
buffer in; it is not a returned intermediate.

Tie-break (locked to the live AMD HIP selector ``top_k_per_row_decode``): among
equal finite scores, keep the **smaller block index**. Invalid / incomplete
blocks are ``-inf`` and never win. Selected ids are written in that sort order
(higher score first; ties already smaller-index-first). Remaining slots are
``-1``.

Expand+tail matches vLLM ``_expand_qsa_indices_kernel`` (PR 53896 / vLLM main):
complete blocks expand to ``r`` tokens; the open group contributes at most
``r-1`` tail tokens after the expanded prefix; padding is ``-1``.

Packed caches match live AMD ``qwen4_exp``: ``cache [n_pages, page_size, H, D]``
and ``block_table [n_req, n_pages]``. Indexer K is one compressed block per
logical slot; GQA K/V use uncompressed tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QsaIndexerSpec:
    """Paged mean-pooled indexer (ReLU-sum over heads, no per-head ``w_h``)."""

    n_heads: int
    kv_heads: int
    head_dim: int
    compress_ratio: int
    block_budget: int
    dtype: torch.dtype = torch.bfloat16

    @property
    def token_budget(self) -> int:
        return self.block_budget * self.compress_ratio

    @property
    def index_width(self) -> int:
        # 512 blocks * 4 tokens + incomplete tail of 0..3.
        return self.token_budget + self.compress_ratio - 1


@dataclass(frozen=True)
class QsaGqaSpec:
    """Sparse GQA over uncompressed paged K/V at the expanded token ids."""

    n_heads: int
    kv_heads: int
    head_dim: int
    rope_dim: int
    dtype: torch.dtype = torch.bfloat16
    # The oracle does not apply the sigmoid output gate. Callers may
    # multiply ``o * sigmoid(gate)`` outside.
    sigmoid_gate: bool = True

    @property
    def group_size(self) -> int:
        return self.n_heads // self.kv_heads


# Test fixtures. The kernels declare their own structural constants, so they
# do not depend on a model registry. Flash-Next is the production shape.
# Gluon-parity is the #4882 shape; its sparse GQA does not auto-dispatch on
# Flash-Next (group 12, D=256).

# Flash-Next / Qwen3.8-Flash-Next / qwen4_exp (SILOTIGER-1047).
# Indexer in:  q [M, 4, 128] BF16, paged compressed k (1 KV head, D=128).
# Indexer out: block_ids [M, 512], token indices [M, <=2051].
# GQA in:      q [M, 24, 256] BF16, paged k/v [..., 2, 256], indices.
# GQA out:     o [M, 24, 256] BF16 (pre-o_proj).
FLASH_NEXT_INDEXER = QsaIndexerSpec(
    n_heads=4,
    kv_heads=1,
    head_dim=128,
    compress_ratio=4,
    block_budget=512,
)
FLASH_NEXT_GQA = QsaGqaSpec(
    n_heads=24,
    kv_heads=2,
    head_dim=256,
    rope_dim=64,
)

# #4882 Gluon-parity shapes. Indexer H is 4 or 8, D=128.
# Sparse GQA: D=128, group size 5, selection_width=2051 (Hq=10, Hk=2).
GLUON_PARITY_INDEXER = QsaIndexerSpec(
    n_heads=4,
    kv_heads=1,
    head_dim=128,
    compress_ratio=4,
    block_budget=512,
)
GLUON_PARITY_INDEXER_H8 = QsaIndexerSpec(
    n_heads=8,
    kv_heads=1,
    head_dim=128,
    compress_ratio=4,
    block_budget=512,
)
GLUON_PARITY_GQA = QsaGqaSpec(
    n_heads=10,
    kv_heads=2,
    head_dim=128,
    rope_dim=0,
    sigmoid_gate=False,
)

# Optional serving scale applied after the ReLU-sum. A positive constant
# cannot change top-k argmax (ties stay ties).
FLASH_NEXT_SCORE_SCALE = FLASH_NEXT_INDEXER.head_dim**-0.5  # 1/sqrt(128)


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
        q: ``[M, H, D]`` (indexer heads; typically 4 x 128).
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

    Output width is ``token_topk + compress_ratio - 1`` (2051 at k=512, r=4).
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


def pack_paged_cache(
    dense: torch.Tensor,
    page_size: int,
    permute_pages: bool = True,
    generator: torch.Generator | None = None,
    physical: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``dense [S, H, D]`` into ``[n_pages, page_size, H, D]`` + table.

    One request. Physical pages are shuffled when ``permute_pages`` is true so
    a gather that ignores ``block_table`` cannot pass. Pass ``physical`` to
    reuse an existing logical->physical map (GQA K and V must share a table).
    """
    if dense.ndim != 3:
        raise ValueError(f"dense cache must be [S, H, D], got {tuple(dense.shape)}")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    n_logical, n_heads, head_dim = dense.shape
    n_pages = (n_logical + page_size - 1) // page_size
    pad = n_pages * page_size - n_logical
    if pad:
        dense = torch.nn.functional.pad(dense, (0, 0, 0, 0, 0, pad))
    packed = dense.view(n_pages, page_size, n_heads, head_dim).contiguous()
    if n_pages == 0:
        table = torch.empty(1, 0, dtype=torch.int32, device=dense.device)
        return packed, table
    if physical is None:
        if permute_pages and n_pages > 1:
            physical = torch.randperm(
                n_pages, device=dense.device, generator=generator
            ).to(torch.int32)
        else:
            physical = torch.arange(n_pages, dtype=torch.int32, device=dense.device)
    elif physical.numel() != n_pages:
        raise ValueError(f"physical map length {physical.numel()} != n_pages {n_pages}")
    cache = packed.new_empty(packed.shape)
    cache[physical.to(torch.int64)] = packed
    table = physical.reshape(1, n_pages).contiguous()
    return cache, table


def gather_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    n_logical: int,
    req: int = 0,
) -> torch.Tensor:
    """Gather ``n_logical`` rows back to dense ``[S, H, D]`` using the table."""
    if cache.ndim != 4:
        raise ValueError(
            f"paged cache must be [n_pages, page_size, H, D], got {tuple(cache.shape)}"
        )
    page_size = cache.shape[1]
    device = cache.device
    logical = torch.arange(n_logical, device=device)
    logical_page = logical // page_size
    offset = logical % page_size
    physical = block_table[req].index_select(0, logical_page).to(torch.int64)
    return cache[physical, offset]


def gather_qsa_caches(
    index_cache: torch.Tensor,
    index_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_table: torch.Tensor,
    n_blocks: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack paged indexer K and GQA K/V to dense tensors for the oracle."""
    k_bar = gather_paged_cache(index_cache, index_table, n_blocks)[:, 0, :]
    k = gather_paged_cache(k_cache, kv_table, seq_len)
    v = gather_paged_cache(v_cache, kv_table, seq_len)
    return k_bar, k, v
