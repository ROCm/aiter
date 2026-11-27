# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Flash-Next and Gluon-parity QSA sweep, plus untimed edge checks."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import qsa_paged
from aiter.ops.flydsl.qsa_paged import (
    flydsl_qsa_paged_gqa,
    flydsl_qsa_paged_gqa_supported,
    qsa_indexer_block_ids,
    span_bytes,
)
from aiter.ops.qsa import qsa_expand_block_indices
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)


@dataclass(frozen=True)
class QsaIndexerSpec:
    """Paged mean-pooled indexer (ReLU-sum over heads, no per-head ``w_h``)."""

    n_heads: int
    head_dim: int
    compress_ratio: int
    block_budget: int

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


# Test fixtures. The kernels declare their own structural constants, so they
# do not depend on a model registry. Flash-Next is the production shape:
# sparse GQA group 12 at D=256. Gluon-parity is the other shape: sparse GQA
# group 5 at D=128.

# Flash-Next.
# Indexer in:  q [M, 4, 128] BF16, paged compressed k (1 KV head, D=128).
# Indexer out: block_ids [M, 512], token indices [M, <=2051].
# GQA in:      q [M, 24, 256] BF16, paged k/v [..., 2, 256], indices.
# GQA out:     o [M, 24, 256] BF16 (pre-o_proj).
FLASH_NEXT_INDEXER = QsaIndexerSpec(
    n_heads=4,
    head_dim=128,
    compress_ratio=4,
    block_budget=512,
)
FLASH_NEXT_GQA = QsaGqaSpec(
    n_heads=24,
    kv_heads=2,
    head_dim=256,
)

# Gluon-parity. Indexer H is 4 or 8, D=128.
# Sparse GQA: D=128, group size 5, selection_width=2051 (Hq=10, Hk=2).
GLUON_PARITY_INDEXER = QsaIndexerSpec(
    n_heads=4,
    head_dim=128,
    compress_ratio=4,
    block_budget=512,
)
GLUON_PARITY_INDEXER_H8 = QsaIndexerSpec(
    n_heads=8,
    head_dim=128,
    compress_ratio=4,
    block_budget=512,
)
GLUON_PARITY_GQA = QsaGqaSpec(
    n_heads=10,
    kv_heads=2,
    head_dim=128,
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
    """Complete compress-ratio blocks visible to each query row."""
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
    """ReLU-sum indexer scores for complete blocks, and -inf elsewhere."""
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
    """Per-row top-k block ids. The smaller index wins an exact tie."""
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
    """Expand selected blocks to token ids and append the incomplete tail."""
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
) -> torch.Tensor:
    """GQA on the selected token positions, with -1 slots masked."""
    m, hq, d = q.shape
    hk = k.shape[1]
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
    scores = torch.matmul(qg, k_sel.permute(0, 2, 3, 1)) * d**-0.5
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.matmul(attn, v_sel.permute(0, 2, 1, 3))
    return out.reshape(m, hq, d)


@dataclass
class QsaOracleResult:
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
) -> QsaOracleResult:
    """Full QSA layer reference (indexer through sparse GQA), in fp32."""
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
    output = qsa_sparse_gqa(q_gqa, k, v, indices)
    return QsaOracleResult(indices=indices, output=output)


def pack_paged_cache(
    dense: torch.Tensor,
    page_size: int,
    generator: torch.Generator | None = None,
    physical: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one request into a paged cache and a shuffled page table."""
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
        if n_pages > 1:
            physical = torch.randperm(
                n_pages, device=dense.device, generator=generator
            ).to(torch.int32)
        else:
            physical = torch.arange(n_pages, dtype=torch.int32, device=dense.device)
    cache = packed.new_empty(packed.shape)
    cache[physical.to(torch.int64)] = packed
    table = physical.reshape(1, n_pages).contiguous()
    return cache, table


SUPPORTED_GFX = ["gfx942", "gfx950"]
# One tensor-parallel rank of Flash-Next sparse GQA. The indexer stays at 4
# heads on every rank; only the GQA query and KV heads split.
_FLASH_NEXT_GQA_RANKS = ((24, 2), (12, 1), (6, 1), (3, 1))
# Exactly 4 GiB still fits in a V# offset. The wide compile starts past that.
_FOUR_GIB = 1 << 32
# Every bench times both. Packed is contiguous. Wide is the serving span.
_CACHE_LAYOUTS = ("packed", "wide")
# MTP decode puts four query rows in one 16-row indexer tile.
_MTP_ROWS_PER_REQUEST = 4
# The indexer scores past this matrix in 128 MiB chunks.
_SCORE_WORKSPACE_BYTES = 128 * 1024 * 1024


def _wide_parent(pages, tail_elems):
    """Place packed pages on a stride that puts the tail past 4 GiB."""
    n_pages, page_elems = (int(size) for size in pages.shape)
    view_pages = max(n_pages, 2)
    gaps = view_pages - 1
    need = (_FOUR_GIB // pages.element_size()) + 1
    page_stride = max(page_elems, (need - tail_elems + gaps - 1) // gaps)
    if page_stride % 8:
        page_stride += 8 - (page_stride % 8)
    parent = pages.new_empty(gaps * page_stride + page_elems)
    parent.zero_()
    for i in range(n_pages):
        parent[i * page_stride : i * page_stride + page_elems] = pages[i]
    return parent, page_stride, view_pages


def _widen_cache(cache: torch.Tensor) -> torch.Tensor:
    """The same pages, with a page stride that selects the wide compile."""
    n_pages, page_size, n_heads, head_dim = (int(s) for s in cache.shape)
    if n_pages < 1:
        return cache
    page_elems = page_size * n_heads * head_dim
    parent, page_stride, view_pages = _wide_parent(
        cache.contiguous().reshape(n_pages, page_elems), page_elems
    )
    view = parent.as_strided(
        (view_pages, page_size, n_heads, head_dim),
        (page_stride, n_heads * head_dim, head_dim, 1),
    )
    if span_bytes(view) <= _FOUR_GIB:
        raise RuntimeError(f"wide cache span {span_bytes(view)} did not pass 4 GiB")
    return view


def _apply_cache_layout(cache: torch.Tensor, layout: str) -> torch.Tensor:
    """``packed`` leaves the cache. ``wide`` is the serving span."""
    if layout == "packed":
        return cache
    if layout != "wide":
        raise ValueError(f"cache_layout must be packed or wide, got {layout}")
    return _widen_cache(cache)


def _span_gib(cache: torch.Tensor) -> float:
    return span_bytes(cache) / (1 << 30)


class _WideRotation(torch.Tensor):
    """Wide cache that stays on the wide compile while run_perftest rotates it."""

    iters = 101

    @staticmethod
    def __new__(cls, view, copies):
        replicas = cls._replicas(view, copies)
        obj = torch.Tensor._make_subclass(cls, replicas[0])
        # Tensor.__setattr__ does not keep a plain list, and deepcopy has to
        # walk these replicas in order.
        object.__setattr__(obj, "_gap_replicas", replicas[1:])
        object.__setattr__(obj, "_gap_cursor", 0)
        return obj

    def __init__(self, view, copies):
        pass

    def __deepcopy__(self, memo):
        replicas = object.__getattribute__(self, "_gap_replicas")
        cursor = object.__getattribute__(self, "_gap_cursor")
        if cursor >= len(replicas):
            raise RuntimeError(f"wide rotation deepcopy past {len(replicas)} replicas")
        replica = replicas[cursor]
        object.__setattr__(self, "_gap_cursor", cursor + 1)
        memo[id(self)] = replica
        return replica

    @staticmethod
    def _replicas(view, copies):
        """Wide views of one cache whose live pages sit at distinct addresses."""
        if copies < 2:
            raise ValueError(f"wide rotation wants at least 2 copies, got {copies}")
        n_pages, page_size, n_heads, head_dim = (int(size) for size in view.shape)
        page_elems = page_size * n_heads * head_dim
        token_stride = int(view.stride(1))
        head_stride = int(view.stride(2))
        if (
            token_stride != n_heads * head_dim
            or head_stride != head_dim
            or int(view.stride(3)) != 1
        ):
            raise RuntimeError(
                f"wide rotation expects packed pages, got strides {tuple(view.stride())}"
            )
        page_stride = int(view.stride(0))
        pitch = page_elems if page_elems % 8 == 0 else page_elems + (8 - page_elems % 8)
        if copies * pitch > page_stride:
            page_stride = copies * pitch
            if page_stride % 8:
                page_stride += 8 - page_stride % 8
        span_elems = (n_pages - 1) * page_stride + (copies - 1) * pitch + page_elems
        parent = view.new_empty(span_elems)
        packed = view.contiguous().reshape(n_pages, page_elems)
        for replica in range(copies):
            slot = parent.as_strided(
                (n_pages, page_elems),
                (page_stride, 1),
                storage_offset=replica * pitch,
            )
            slot.copy_(packed)
        views = []
        for replica in range(copies):
            rotated = parent.as_strided(
                (n_pages, page_size, n_heads, head_dim),
                (page_stride, token_stride, head_stride, 1),
                storage_offset=replica * pitch,
            )
            if span_bytes(rotated) <= _FOUR_GIB:
                raise RuntimeError(
                    f"rotated wide cache spans {span_bytes(rotated)} bytes"
                )
            views.append(rotated)
        return views

    @classmethod
    def wrap(cls, rotate, *args, num_iters=None):
        """Replace each wide positional tensor with a rotating view."""
        if num_iters is None:
            num_iters = cls.iters
        copies = num_iters if int(rotate) < 1 else max(1, min(int(rotate), num_iters))
        if copies < 2:
            return args
        wrapped = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and span_bytes(arg) > _FOUR_GIB:
                wrapped.append(cls(arg, copies))
            else:
                wrapped.append(arg)
        return tuple(wrapped)


def _query_positions(m: int, seq_len: int, device) -> torch.Tensor:
    return torch.arange(seq_len - m, seq_len, device=device, dtype=dtypes.i32)


def _paged_gqa_live_n_sel(
    m: int, seq_len: int, token_budget: int, compress_ratio: int
) -> int:
    """Widest packed prefix expand fills, the max over the batch."""
    if compress_ratio < 1 or token_budget % compress_ratio:
        raise ValueError(
            f"token_budget {token_budget} must be a multiple of "
            f"compress_ratio {compress_ratio}"
        )
    if m < 1 or m > seq_len:
        raise ValueError(f"m must fit in seq_len, got m={m} seq_len={seq_len}")
    block_topk = token_budget // compress_ratio
    widest = 0
    for qpos in range(seq_len - m, seq_len):
        visible = min((qpos + 1) // compress_ratio, seq_len // compress_ratio)
        complete = min(visible, block_topk)
        widest = max(widest, complete * compress_ratio + (qpos + 1) % compress_ratio)
    return widest


def _selected_width(indices: torch.Tensor) -> float:
    """Mean live selection slots per row, ignoring -1 padding."""
    return float((indices >= 0).sum().item()) / indices.shape[0]


def _indexer_inputs(idx, m, seq_len, dtype, device):
    """Seeded single-request indexer inputs: q, k_bar, qpos, slen, token_to_req."""
    torch.manual_seed(0)
    q = torch.randn(
        m, idx.n_heads, idx.head_dim, dtype=dtype, device=device
    ).contiguous()
    k_bar = torch.randn(
        seq_len // idx.compress_ratio, idx.head_dim, dtype=dtype, device=device
    )
    qpos = _query_positions(m, seq_len, device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    return q, k_bar, qpos, slen, token_to_req


def _pack_index_cache(k_bar, page_size, generator=None):
    """Pack one request's compressed K ([n_blocks, D]) into a shuffled paged cache."""
    if generator is None:
        generator = torch.Generator(device=k_bar.device)
        generator.manual_seed(1)
    return pack_paged_cache(k_bar.unsqueeze(1), page_size, generator=generator)


def _indexer_scores(idx, q, k_bar, qpos, slen, token_to_req):
    """Oracle indexer scores at the spec's serving scale."""
    return qsa_indexer_scores(
        q,
        k_bar,
        qpos,
        slen,
        token_to_req,
        idx.compress_ratio,
        score_scale=idx.head_dim**-0.5,
    )


def _pack_flash_next(k_bar, k, v, page_size):
    index_cache, index_table = _pack_index_cache(k_bar, page_size)
    gen_kv = torch.Generator(device=k.device)
    gen_kv.manual_seed(2)
    k_cache, kv_table = pack_paged_cache(k, page_size, generator=gen_kv)
    v_cache, kv_table_v = pack_paged_cache(v, page_size, physical=kv_table[0])
    if not torch.equal(kv_table, kv_table_v):
        raise RuntimeError("K and V page tables diverged")
    return index_cache, index_table, k_cache, v_cache, kv_table


def _set_mismatch_ratio(
    ref: torch.Tensor,
    got: torch.Tensor,
    scores: torch.Tensor | None = None,
) -> float:
    """Fraction of rows whose selected ids miss the oracle top-k."""
    if scores is None:
        miss = 0
        rows = ref.shape[0]
        for i in range(rows):
            a = set(ref[i].tolist()) - {-1}
            b = set(got[i].tolist()) - {-1}
            if a != b:
                miss += 1
        return miss / rows if rows else 0.0

    rows, n_blocks = scores.shape
    if rows == 0:
        return 0.0
    in_range = ((ref < 0) | (ref < n_blocks)).all(dim=1) & (
        (got < 0) | (got < n_blocks)
    ).all(dim=1)
    last = n_blocks - 1
    ref_idx = ref.clamp(0, last)
    got_idx = got.clamp(0, last)
    ref_scores = scores.gather(1, ref_idx).masked_fill(ref < 0, float("inf"))
    kth = ref_scores.min(dim=1).values
    floor = torch.nextafter(kth, torch.full_like(kth, float("-inf")))
    got_scores = scores.gather(1, got_idx)
    got_valid = got >= 0
    kth_col = kth.unsqueeze(1)
    # Equal scores keep the smaller index, so a tie id has to be one the
    # oracle kept. A different id is only the one-ulp boundary below kth.
    ref_kept = torch.zeros(rows, n_blocks, dtype=torch.bool, device=scores.device)
    ref_safe = (ref >= 0) & (ref < n_blocks)
    ref_rows = torch.arange(rows, device=scores.device).unsqueeze(1).expand_as(ref)
    ref_kept[ref_rows[ref_safe], ref[ref_safe]] = True
    equal_tie = got_valid & (got_scores == kth_col)
    bad_tie = equal_tie & ~ref_kept.gather(1, got_idx)
    boundary = got_valid & (got_scores != kth_col)
    bad_score = boundary & (
        ~torch.isfinite(got_scores) | (got_scores < floor.unsqueeze(1))
    )
    selected = torch.zeros(rows, n_blocks, dtype=torch.bool, device=scores.device)
    # A padded slot clamps to column 0. Scatter that False and it wipes a
    # real selection of block 0, so only the in-range ids are marked.
    safe = got_valid & (got < n_blocks)
    row_idx = torch.arange(rows, device=scores.device).unsqueeze(1).expand_as(got)
    selected[row_idx[safe], got[safe]] = True
    n_got = got_valid.sum(dim=1)
    n_ref = (ref >= 0).sum(dim=1)
    count_bad = (selected.sum(dim=1) != n_got) | (n_got != n_ref)
    required = torch.isfinite(scores) & (scores > kth.unsqueeze(1))
    missing = (required & ~selected).any(dim=1)
    miss = bad_score.any(dim=1) | bad_tie.any(dim=1) | ~in_range | count_bad | missing
    return float(miss.sum().item()) / rows


def _indexer_row_is_packed(row: torch.Tensor) -> str | None:
    """Valid ids, each once, then only ``-1``. ``None`` when the row is packed."""
    seen = set()
    padding = False
    for value in row.tolist():
        if value == -1:
            padding = True
            continue
        if padding:
            return "valid id after -1"
        if value in seen:
            return "duplicate id"
        seen.add(value)
    return None


def _assert_indexer_block_ids(ref: torch.Tensor, got: torch.Tensor, what: str) -> None:
    """Set equality, plus no duplicates and a compact valid prefix on ``got``."""
    for i in range(got.shape[0]):
        reason = _indexer_row_is_packed(got[i])
        if reason is not None:
            raise AssertionError(f"{what}: row {i} {reason}")
    if _set_mismatch_ratio(ref, got) != 0.0:
        raise AssertionError(f"{what}: set mismatch")


def _interleave_kv(k_cache, v_cache, *, wide=False):
    """K and V as views of one buffer, packed or with a span past 4 GiB."""
    d = int(k_cache.shape[-1])
    if not wide:
        kv = torch.cat((k_cache, v_cache), dim=-1)
        return kv[..., :d], kv[..., d:]
    n_pages, page_size, n_heads, head_dim = (int(s) for s in k_cache.shape)
    # One token is K then V. Head stride 2*D, D stride 1, so a row of D is
    # contiguous and the gather's 128-bit load stays aligned. The span counts
    # through the last K element; V sits head_dim elements later.
    token_stride = n_heads * 2 * head_dim
    head_stride = 2 * head_dim
    kv_page_elems = page_size * token_stride
    k_tail = (page_size - 1) * token_stride + (n_heads - 1) * head_stride + head_dim
    pages = torch.cat((k_cache.contiguous(), v_cache.contiguous()), dim=-1)
    parent, page_stride, view_pages = _wide_parent(
        pages.reshape(n_pages, kv_page_elems), k_tail
    )
    shape = (view_pages, page_size, n_heads, head_dim)
    strides = (page_stride, token_stride, head_stride, 1)
    k_view = parent.as_strided(shape, strides)
    v_view = parent.as_strided(shape, strides, storage_offset=head_dim)
    if span_bytes(k_view) <= _FOUR_GIB:
        raise RuntimeError(
            f"wide interleaved span {span_bytes(k_view)} did not pass 4 GiB"
        )
    return k_view, v_view


def _paged_gqa_one_live_token(m, width, live_col):
    """Q/K zero, V one, every index -1 except ``live_col``."""
    gqa = FLASH_NEXT_GQA
    device = torch.device("cuda")
    q = torch.zeros(m, gqa.n_heads, gqa.head_dim, dtype=dtypes.bf16, device=device)
    k_cache = torch.zeros(
        1, 16, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    v_cache = torch.ones_like(k_cache)
    page_table = torch.zeros(1, 1, dtype=dtypes.i32, device=device)
    indices = torch.full((m, width), -1, dtype=dtypes.i32, device=device)
    indices[:, live_col] = 0
    token_to_req = torch.zeros(m, dtype=dtypes.i32, device=device)
    return flydsl_qsa_paged_gqa(q, k_cache, v_cache, indices, page_table, token_to_req)


def _indexer_select(
    q,
    k_cache,
    page_table,
    token_to_req,
    query_positions,
    sequence_lengths,
    token_topk,
    compress_ratio,
    heads,
):
    """Block ids and the expanded token ids from the public indexer."""
    block_ids = qsa_indexer_block_ids(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        heads=heads,
    )
    indices = qsa_expand_block_indices(
        block_ids,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio,
        token_topk,
    )
    return indices, block_ids


def _pad_index_table(index_table, pad_pages):
    """Zero-fill a one-request block table out to pad_pages."""
    if pad_pages <= 0:
        return index_table
    n_pages = index_table.shape[1]
    if pad_pages < n_pages:
        raise ValueError(
            f"pad_pages={pad_pages} is narrower than the packed table ({n_pages} pages)"
        )
    if pad_pages == n_pages:
        return index_table
    wide = torch.zeros(
        index_table.shape[0],
        pad_pages,
        dtype=index_table.dtype,
        device=index_table.device,
    )
    wide[:, : index_table.shape[1]] = index_table
    return wide


def _expect_value_error(fn, needle, what):
    try:
        fn()
    except ValueError as exc:
        if needle not in str(exc):
            raise AssertionError(f"{what}: expected {needle!r} in {exc}") from exc
    else:
        raise AssertionError(f"{what} was accepted")


def _enough_memory(need, what):
    free, _total = torch.cuda.mem_get_info()
    if free >= need:
        return True
    aiter.logger.warning("skip %s: need %s bytes, %s free", what, need, free)
    return False


def _apply_kv_layout(k_cache, v_cache, layout):
    """``packed`` leaves K and V. ``interleaved`` is one K|V buffer."""
    if layout == "packed":
        return k_cache, v_cache
    if layout != "interleaved":
        raise ValueError(f"kv_layout must be packed or interleaved, got {layout}")
    k_view, v_view = _interleave_kv(k_cache, v_cache)
    if k_view.is_contiguous() or v_view.is_contiguous():
        raise RuntimeError("interleaved K|V views are contiguous")
    return k_view, v_view


def _indexer_case(
    idx, m, seq_len, page_size, dtype, pad_pages, cache_layout, requests, device
):
    """Indexer tensors and the oracle block ids for one spec."""
    n_blocks = seq_len // idx.compress_ratio
    if requests == 1:
        q_indexer, k_bar, qpos, slen, token_to_req = _indexer_inputs(
            idx, m, seq_len, dtype, device
        )
        index_cache, index_table = _pack_index_cache(k_bar, page_size)
        ref_scores = _indexer_scores(idx, q_indexer, k_bar, qpos, slen, token_to_req)
    else:
        rows_per = _MTP_ROWS_PER_REQUEST
        if m != rows_per * requests:
            raise ValueError(
                f"multi-request indexer wants {rows_per} rows per request, "
                f"got M={m} requests={requests}"
            )
        gen_i = torch.Generator(device=device)
        gen_i.manual_seed(1)
        qs, k_bars, caches, tables = [], [], [], []
        base = 0
        for req in range(requests):
            torch.manual_seed(req + 1)
            qs.append(
                torch.randn(
                    rows_per, idx.n_heads, idx.head_dim, dtype=dtype, device=device
                )
            )
            k_bar = torch.randn(n_blocks, idx.head_dim, dtype=dtype, device=device)
            cache, table = _pack_index_cache(k_bar, page_size, gen_i)
            k_bars.append(k_bar)
            caches.append(cache)
            tables.append(table[0] + base)
            base += int(cache.shape[0])
        q_indexer = torch.cat(qs).contiguous()
        index_cache = torch.cat(caches)
        width = max(int(table.numel()) for table in tables)
        index_table = torch.zeros(requests, width, dtype=dtypes.i32, device=device)
        for req, physical in enumerate(tables):
            index_table[req, : physical.numel()] = physical
        slen = torch.full((requests,), seq_len, dtype=dtypes.i32, device=device)
        token_to_req = torch.arange(
            requests, device=device, dtype=dtypes.i32
        ).repeat_interleave(rows_per)
        qpos = torch.cat(
            [_query_positions(rows_per, seq_len, device) for _req in range(requests)]
        )
        score_rows = []
        zeros = torch.zeros(rows_per, dtype=dtypes.i32, device=device)
        for req in range(requests):
            rows = slice(req * rows_per, (req + 1) * rows_per)
            score_rows.append(
                _indexer_scores(
                    idx,
                    q_indexer[rows],
                    k_bars[req],
                    qpos[rows],
                    slen[req : req + 1],
                    zeros,
                )
            )
        ref_scores = torch.cat(score_rows)
    index_cache = _apply_cache_layout(index_cache.contiguous(), cache_layout)
    index_table = _pad_index_table(index_table.contiguous(), pad_pages)
    return {
        "idx": idx,
        "q": q_indexer,
        "cache": index_cache,
        "table": index_table,
        "token_to_req": token_to_req,
        "qpos": qpos,
        "slen": slen,
        "ref_scores": ref_scores,
        "ref_ids": qsa_topk_blocks(ref_scores, idx.block_budget),
        "n_blocks": n_blocks,
        "requests": requests,
    }


def _flash_next_gqa_case(
    m, seq_len, page_size, dtype, n_heads, kv_heads, cache_layout, kv_layout
):
    """Flash-Next paged GQA tensors. The oracle stays on dense K and V."""
    if cache_layout == "wide" and kv_layout == "interleaved":
        # cat would pack the pages. The untimed check builds this view.
        raise ValueError("wide interleaved KV is not a timed layout")
    gqa = _flash_next_gqa(n_heads, kv_heads)
    # The GQA kernel never reads the index cache. Widening it would allocate a
    # second span past 4 GiB per wide row.
    case = _prepare_qsa_layer(
        FLASH_NEXT_INDEXER,
        gqa,
        m,
        seq_len,
        page_size,
        dtype,
        cache_layout,
        kv_layout,
        widen_index=False,
    )
    return {
        "gqa": gqa,
        "q": case["q_gqa"],
        "k_cache": case["k_cache"],
        "v_cache": case["v_cache"],
        "indices": case["ref"].indices.contiguous(),
        "kv_table": case["kv_table"],
        "token_to_req": case["token_to_req"],
        "ref": case["ref"],
        "n_blocks": case["n_blocks"],
    }


def _assert_indexer_caller_workspace(
    q, cache, table, token_to_req, qpos, slen, ref_ids, page_size
):
    """A caller score buffer selects the same ids as the private buffer."""
    m = q.shape[0]
    n_columns = int(table.shape[1]) * int(page_size)
    workspace = torch.full((m, n_columns), float("nan"), device=q.device)
    got = qsa_indexer_block_ids(
        q,
        cache,
        table,
        token_to_req,
        qpos,
        slen,
        heads=(int(q.shape[1]),),
        scores=workspace,
    )
    _assert_indexer_block_ids(ref_ids, got, "caller score workspace")
    # Emit does not read the buffer. Longer rows write it.
    if n_columns > 512 and not torch.isfinite(workspace).any():
        raise AssertionError("caller score workspace was not written")


def _assert_gqa_caller_workspace(
    q, k_cache, v_cache, indices, kv_table, token_to_req, ref_out, n_sel=None
):
    """A caller split workspace matches the default launch at the same n_sel."""
    m = q.shape[0]
    n_heads = q.shape[1]
    head_dim = q.shape[2]
    device = q.device
    out = torch.empty_like(q)
    accepted = None
    partial_lse = None
    for n_splits in range(2, 129):
        partial_out = torch.full(
            (n_splits, m, n_heads, head_dim), 7, dtype=torch.float32, device=device
        )
        partial_lse = torch.full(
            (n_splits, m, n_heads), 3, dtype=torch.float32, device=device
        )
        try:
            got = flydsl_qsa_paged_gqa(
                q,
                k_cache,
                v_cache,
                indices,
                kv_table,
                token_to_req,
                out=out,
                n_sel=n_sel,
                workspace=(partial_out, partial_lse),
            )
        except ValueError:
            continue
        if torch.all(partial_out == 7):
            continue
        accepted = (got, n_splits)
        break
    if accepted is None:
        dummy_out = torch.empty(
            (2, m, n_heads, head_dim), dtype=torch.float32, device=device
        )
        dummy_lse = torch.empty((2, m, n_heads), dtype=torch.float32, device=device)
        try:
            got = flydsl_qsa_paged_gqa(
                q,
                k_cache,
                v_cache,
                indices,
                kv_table,
                token_to_req,
                out=out,
                n_sel=n_sel,
                workspace=(dummy_out, dummy_lse),
            )
        except ValueError as exc:
            raise AssertionError(
                "caller workspace was not accepted for any split count"
            ) from exc
        if not torch.equal(got, ref_out):
            raise AssertionError("one-split workspace launch diverged")
        return
    got, n_splits = accepted
    if not torch.equal(got, ref_out):
        raise AssertionError("workspace launch diverged from the default launch")
    bad = torch.empty(
        (n_splits + 1, m, n_heads, head_dim), dtype=torch.float32, device=device
    )
    _expect_value_error(
        lambda: flydsl_qsa_paged_gqa(
            q,
            k_cache,
            v_cache,
            indices,
            kv_table,
            token_to_req,
            out=out,
            n_sel=n_sel,
            workspace=(bad, partial_lse),
        ),
        "partial_out",
        "wrong split count",
    )


def _indexer_with_missing_pages(m, n_blocks, bad_pages, what):
    idx = FLASH_NEXT_INDEXER
    page_size = 16
    context = n_blocks * idx.compress_ratio
    q, k_bar, qpos, slen, token_to_req = _indexer_inputs(
        idx, m, context, dtypes.bf16, torch.device("cuda")
    )
    cache, table = _pack_index_cache(k_bar, page_size)
    table = table.clone()
    for page in bad_pages:
        table[0, page] = -1
    ref_scores = _indexer_scores(idx, q, k_bar, qpos, slen, token_to_req)
    for page in bad_pages:
        block0 = page * page_size
        ref_scores[:, block0 : block0 + page_size] = float("-inf")
    ref_ids = qsa_topk_blocks(ref_scores, idx.block_budget)
    got = qsa_indexer_block_ids(
        q,
        cache,
        table,
        token_to_req,
        qpos,
        slen,
        heads=(4,),
        score_scale=FLASH_NEXT_SCORE_SCALE,
    )
    _assert_indexer_block_ids(ref_ids, got, what)


def check_cache_past_4gib():
    """One wide indexer gather, one wide paged-GQA gather, and one wide K|V view."""
    device = torch.device("cuda")
    # The wide parent is the span, just past 4 GiB, plus the live pages.
    if not _enough_memory((5 << 30), "indexer cache past 4 GiB"):
        return
    # 528 blocks is 33 pages, past the 512-column emit path.
    case = _indexer_case(
        FLASH_NEXT_INDEXER, 2, 2112, 16, dtypes.bf16, 0, "wide", 1, device
    )
    got = qsa_indexer_block_ids(
        case["q"],
        case["cache"],
        case["table"],
        case["token_to_req"],
        case["qpos"],
        case["slen"],
        heads=(4,),
    )
    _assert_indexer_block_ids(case["ref_ids"], got, "indexer cache past 4 GiB")
    del case
    if not _enough_memory((5 << 30), "paged GQA cache past 4 GiB"):
        return
    gqa = _flash_next_gqa_case(2, 64, 16, dtypes.bf16, 24, 2, "wide", "packed")
    out = flydsl_qsa_paged_gqa(
        gqa["q"],
        gqa["k_cache"],
        gqa["v_cache"],
        gqa["indices"],
        gqa["kv_table"],
        gqa["token_to_req"],
    )
    err = checkAllclose(
        gqa["ref"].output,
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="paged GQA cache past 4 GiB",
    )
    if err != 0:
        raise AssertionError(f"paged GQA cache past 4 GiB err={err}")
    del gqa
    if not _enough_memory((5 << 30), "interleaved paged GQA cache past 4 GiB"):
        return
    packed = _flash_next_gqa_case(2, 64, 16, dtypes.bf16, 24, 2, "packed", "packed")
    k_view, v_view = _interleave_kv(packed["k_cache"], packed["v_cache"], wide=True)
    live_bytes = k_view.numel() * k_view.element_size()
    if live_bytes > _FOUR_GIB:
        raise AssertionError(
            f"wide interleaved K numel spans {live_bytes} bytes, past 4 GiB"
        )
    head_dim = int(k_view.shape[-1])
    if k_view.stride(-1) != 1 or k_view.stride(-2) != 2 * head_dim:
        raise AssertionError(f"wide interleaved strides {k_view.stride()}")
    if k_view.untyped_storage().data_ptr() != v_view.untyped_storage().data_ptr():
        raise AssertionError("wide interleaved K and V do not share a buffer")
    out = flydsl_qsa_paged_gqa(
        packed["q"],
        k_view,
        v_view,
        packed["indices"],
        packed["kv_table"],
        packed["token_to_req"],
    )
    err = checkAllclose(
        packed["ref"].output,
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="interleaved paged GQA cache past 4 GiB",
    )
    if err != 0:
        raise AssertionError(f"interleaved paged GQA cache past 4 GiB err={err}")
    del packed, k_view, v_view


def check_invalid_pages_match_oracle():
    """A missing page stays out of the top-k when the row is under budget."""
    _indexer_with_missing_pages(
        1, 512, (0,), "indexer emit kept a block on a missing page"
    )
    missing = tuple(range(1, 7))
    _indexer_with_missing_pages(
        4, 600, missing, "indexer decode filled the budget from a missing page"
    )
    _indexer_with_missing_pages(
        4,
        20480,
        tuple(range(30, 1280)),
        "indexer row-bounded radix filled the budget from a missing page",
    )


def check_empty_first_tile_keeps_later_token():
    """An empty first tile still attends the later live token."""
    for m, width, live_col in ((1, 2048, 16), (512, 128, 64)):
        out = _paged_gqa_one_live_token(m, width, live_col)
        err = checkAllclose(
            torch.ones_like(out, dtype=torch.float32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"paged GQA empty-to-valid M={m} W={width} column {live_col}",
        )
        if err != 0:
            raise AssertionError(
                f"paged GQA empty-to-valid M={m} W={width} column {live_col} err={err}"
            )


def check_large_negative_score_keeps_weight():
    """A large negative score stays finite and matches the oracle."""
    gqa = FLASH_NEXT_GQA
    device = torch.device("cuda")
    page_size, seq_len = 16, 64
    q_bit = torch.tensor(2**36, dtype=dtypes.bf16, device=device)
    k_bit = torch.tensor(-(2**36), dtype=dtypes.bf16, device=device)
    if q_bit.float().item() != 2**36 or k_bit.float().item() != -(2**36):
        raise AssertionError("2**36 is not exact bf16")
    q = torch.empty(1, gqa.n_heads, gqa.head_dim, dtype=dtypes.bf16, device=device)
    q.fill_(q_bit)
    k = torch.empty(
        seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    k.fill_(k_bit)
    v = torch.ones_like(k)
    token_to_req = torch.zeros(1, dtype=dtypes.i32, device=device)
    k_cache, kv_table = pack_paged_cache(k, page_size)
    v_cache, kv_table_v = pack_paged_cache(v, page_size, physical=kv_table[0])
    if not torch.equal(kv_table, kv_table_v):
        raise AssertionError("K and V page tables diverged")
    for width, masked in ((16, False), (2048, False), (16, True), (2048, True)):
        indices = torch.full((1, width), -1, dtype=dtypes.i32, device=device)
        if not masked:
            indices[0, 0] = 0
        ref = qsa_sparse_gqa(q, k, v, indices)
        out = flydsl_qsa_paged_gqa(q, k_cache, v_cache, indices, kv_table, token_to_req)
        if torch.isnan(out).any():
            raise AssertionError(
                f"paged GQA width {width} masked {masked} produced NaN"
            )
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"paged GQA large-negative score width {width} masked {masked}",
        )
        if err != 0:
            raise AssertionError(
                f"paged GQA width {width} masked {masked} diverged (err={err})"
            )


def _hand_built_indexer(q, k_bar, qpos, slen):
    """Run the public indexer on a packed one-page cache and return ids plus scores."""
    idx = FLASH_NEXT_INDEXER
    token_to_req = torch.zeros(q.shape[0], dtype=dtypes.i32, device=q.device)
    cache, table = _pack_index_cache(k_bar, 16)
    scores = _indexer_scores(idx, q, k_bar, qpos, slen, token_to_req)
    got = qsa_indexer_block_ids(
        q,
        cache,
        table,
        token_to_req,
        qpos,
        slen,
        heads=(4,),
        score_scale=FLASH_NEXT_SCORE_SCALE,
    )
    return scores, got


def check_tie_keeps_smaller_block_id():
    """Equal finite scores keep the smaller block id on the public indexer."""
    idx = FLASH_NEXT_INDEXER
    device = torch.device("cuda")
    q = torch.zeros(1, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device)
    q[0, 0, 0] = 1
    k_bar = torch.zeros(3, idx.head_dim, dtype=dtypes.bf16, device=device)
    k_bar[0, 0] = 1
    k_bar[1, 0] = 1
    qpos = torch.tensor([11], dtype=dtypes.i32, device=device)
    slen = torch.tensor([12], dtype=dtypes.i32, device=device)
    scores, got = _hand_built_indexer(q, k_bar, qpos, slen)
    if scores[0, 0] != scores[0, 1] or scores[0, 0] <= scores[0, 2]:
        raise AssertionError("tie fixture did not score blocks 0 and 1 equally")
    ref = qsa_topk_blocks(scores, idx.block_budget)
    if ref[0, 0].item() != 0 or ref[0, 1].item() != 1:
        raise AssertionError(f"oracle tie break was {ref[0, :2].tolist()}")
    _assert_indexer_block_ids(ref, got, "tie keeps the smaller block id")


def check_partial_block_not_selected():
    """An incomplete block with a huge score is not selected."""
    idx = FLASH_NEXT_INDEXER
    device = torch.device("cuda")
    q = torch.ones(1, idx.n_heads, idx.head_dim, dtype=dtypes.bf16, device=device)
    k_bar = torch.zeros(2, idx.head_dim, dtype=dtypes.bf16, device=device)
    k_bar[0] = 1
    k_bar[1] = 10
    qpos = torch.tensor([3], dtype=dtypes.i32, device=device)
    slen = torch.tensor([8], dtype=dtypes.i32, device=device)
    scores, got = _hand_built_indexer(q, k_bar, qpos, slen)
    if torch.isfinite(scores[0, 1]):
        raise AssertionError("partial block fixture scored the incomplete block")
    ref = qsa_topk_blocks(scores, idx.block_budget)
    if ref[0, 0].item() != 0 or ref[0, 1].item() != -1:
        raise AssertionError(f"oracle partial block ids were {ref[0, :2].tolist()}")
    _assert_indexer_block_ids(ref, got, "partial block was selected")


def check_expand_rejects_strided_metadata():
    """Expand rejects a strided metadata view."""
    device = torch.device("cuda")
    requests = torch.tensor([0, 9, 1, 9], dtype=dtypes.i32, device=device)[::2]
    positions = torch.tensor([7, 0, 3, 0], dtype=dtypes.i32, device=device)[::2]
    lengths = torch.tensor([20, 1, 8, 1], dtype=dtypes.i32, device=device)[::2]
    blocks = torch.zeros(2, 1, dtype=dtypes.i32, device=device)
    if requests.is_contiguous():
        raise AssertionError("the metadata fixture is already packed")
    _expect_value_error(
        lambda: qsa_expand_block_indices(blocks, positions, lengths, requests, 4, 4),
        "contiguous",
        "strided metadata",
    )


def check_score_chunk_matches_oracle():
    """A public prefill whose score matrix exceeds 128 MiB matches the oracle."""
    idx = FLASH_NEXT_INDEXER
    device = torch.device("cuda")
    page_size = 16
    n_columns = 16384
    rows_per_chunk = max(1, _SCORE_WORKSPACE_BYTES // (n_columns * 4))
    if rows_per_chunk <= 64:
        raise AssertionError(f"a chunk of {rows_per_chunk} rows is already decode")
    m = rows_per_chunk + 1
    if not _enough_memory(_SCORE_WORKSPACE_BYTES + (256 << 20), "indexer score chunk"):
        return
    seq_len = n_columns * idx.compress_ratio
    q, k_bar, _qpos, slen, token_to_req = _indexer_inputs(
        idx, m, seq_len, dtypes.bf16, device
    )
    cache, table = _pack_index_cache(k_bar, page_size)
    if int(table.shape[1]) * page_size != n_columns:
        raise AssertionError(
            f"table width {int(table.shape[1]) * page_size} != {n_columns}"
        )
    qpos = torch.full((m,), seq_len - 1, dtype=dtypes.i32, device=device)
    refs = []
    for start in range(0, m, 64):
        end = min(m, start + 64)
        refs.append(
            qsa_topk_blocks(
                _indexer_scores(
                    idx,
                    q[start:end],
                    k_bar,
                    qpos[start:end],
                    slen,
                    token_to_req[start:end],
                ),
                idx.block_budget,
            )
        )
    got = qsa_indexer_block_ids(
        q, cache, table, token_to_req, qpos, slen, heads=(idx.n_heads,)
    )
    _assert_indexer_block_ids(torch.cat(refs), got, "chunked indexer score workspace")


def check_paged_gqa_live_prefix_matches_full_width():
    """A packed live prefix matches the allocated row and the oracle."""
    idx = FLASH_NEXT_INDEXER
    gqa = FLASH_NEXT_GQA
    for m, seq_len in ((1, 512), (1, 2048), (1, 8192), (8, 512), (8, 8192)):
        case = _flash_next_gqa_case(
            m,
            seq_len,
            16,
            dtypes.bf16,
            gqa.n_heads,
            gqa.kv_heads,
            "packed",
            "packed",
        )
        q = case["q"]
        k_cache = case["k_cache"]
        v_cache = case["v_cache"]
        indices = case["indices"]
        kv_table = case["kv_table"]
        token_to_req = case["token_to_req"]
        live = _paged_gqa_live_n_sel(m, seq_len, idx.token_budget, idx.compress_ratio)
        if m == 1 and seq_len == 512:
            try:
                flydsl_qsa_paged_gqa(
                    q, k_cache, v_cache, indices, kv_table, token_to_req, n_sel=True
                )
            except TypeError:
                pass
            else:
                raise AssertionError("bool n_sel was accepted")
            past = int(indices.shape[1]) + 1
            _expect_value_error(
                lambda q=q, k_cache=k_cache, v_cache=v_cache, indices=indices, kv_table=kv_table, token_to_req=token_to_req, past=past: flydsl_qsa_paged_gqa(
                    q,
                    k_cache,
                    v_cache,
                    indices,
                    kv_table,
                    token_to_req,
                    n_sel=past,
                ),
                "n_sel",
                "n_sel past the row",
            )
        if live <= 0 or live > indices.shape[1]:
            raise AssertionError(f"live n_sel {live} outside 1..{indices.shape[1]}")
        if int((indices[:, live:] >= 0).any().item()):
            raise AssertionError(f"live token past n_sel={live} at M={m} L={seq_len}")
        if int((indices >= 0).sum(dim=1).max().item()) > live:
            raise AssertionError(
                f"a row is wider than n_sel={live} at M={m} L={seq_len}"
            )
        full = flydsl_qsa_paged_gqa(
            q, k_cache, v_cache, indices, kv_table, token_to_req
        )
        short = flydsl_qsa_paged_gqa(
            q, k_cache, v_cache, indices, kv_table, token_to_req, n_sel=live
        )
        err = checkAllclose(
            full.to(dtypes.fp32),
            short.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"live prefix vs full width M={m} L={seq_len}",
        )
        if err != 0:
            raise AssertionError(
                f"live prefix diverged from the full row at M={m} L={seq_len} (err={err})"
            )
        err = checkAllclose(
            case["ref"].output,
            short.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"live prefix vs oracle M={m} L={seq_len}",
        )
        if err != 0:
            raise AssertionError(
                f"live prefix diverged from the oracle at M={m} L={seq_len} (err={err})"
            )


def check_qsa_edges():
    """Run the untimed edge checks."""
    check_cache_past_4gib()
    check_invalid_pages_match_oracle()
    check_empty_first_tile_keeps_later_token()
    check_large_negative_score_keeps_weight()
    check_paged_gqa_live_prefix_matches_full_width()
    check_tie_keeps_smaller_block_id()
    check_partial_block_not_selected()
    check_expand_rejects_strided_metadata()
    check_score_chunk_matches_oracle()


def _check_unmeasured_gqa_warns_once(device):
    """An unswept rank-3 GQA query logs once and still returns."""
    import logging

    def launch(hq, head_dim, kv_heads):
        q = torch.zeros(0, hq, head_dim, dtype=torch.bfloat16, device=device)
        cache = torch.zeros(
            1, 16, kv_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        indices = torch.zeros(0, 1, dtype=torch.int32, device=device)
        table = torch.zeros(1, 1, dtype=torch.int32, device=device)
        token = torch.zeros(0, dtype=torch.int32, device=device)
        return flydsl_qsa_paged_gqa(q, cache, cache, indices, table, token)

    messages = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    aiter.logger.addHandler(handler)
    try:
        out = launch(8, 128, 1)
        if tuple(out.shape) != (0, 8, 128):
            raise AssertionError(f"unswept GQA returned {tuple(out.shape)}")
        launch(8, 128, 1)
        expected = "QSA: GQA query (8, 128) is not in the measured table"
        if messages != [expected]:
            raise AssertionError(f"unswept GQA warning was {messages}")
        out = launch(24, 256, 2)
        if tuple(out.shape) != (0, 24, 256):
            raise AssertionError(f"measured GQA returned {tuple(out.shape)}")
        if messages != [expected]:
            raise AssertionError(f"measured GQA query logged {messages}")
        rank2 = torch.zeros(1, 128, dtype=torch.bfloat16, device=device)
        cache = torch.zeros(1, 16, 1, 128, dtype=torch.bfloat16, device=device)
        try:
            flydsl_qsa_paged_gqa(
                rank2,
                cache,
                cache,
                torch.zeros(1, 1, dtype=torch.int32, device=device),
                torch.zeros(1, 1, dtype=torch.int32, device=device),
                torch.zeros(1, dtype=torch.int32, device=device),
            )
        except ValueError as exc:
            if "must be [M, Hq, D]" not in str(exc):
                raise
        else:
            raise AssertionError("rank-2 GQA was accepted")
        if messages != [expected]:
            raise AssertionError(f"rank-2 GQA logged {messages}")
    finally:
        aiter.logger.removeHandler(handler)


def check_qsa_rejections():
    """Public contract: bad inputs raise, empty geometry does not launch a gather."""
    q = torch.zeros(1, 4, 128, dtype=torch.bfloat16)
    cache = torch.zeros(1, 16, 1, 128, dtype=torch.bfloat16)
    table = torch.zeros(1, 8, dtype=torch.int32)
    rows = torch.zeros(1, dtype=torch.int32)
    n_columns = int(table.shape[1]) * int(cache.shape[1])
    for scores in (
        torch.empty(1, n_columns - 1),
        torch.empty(1, n_columns, dtype=torch.float16),
        torch.empty(1, n_columns * 2)[:, ::2],
    ):
        _expect_value_error(
            lambda scores=scores: qsa_indexer_block_ids(
                q, cache, table, rows, rows, rows, scores=scores
            ),
            "scores",
            f"scores {tuple(scores.shape)} {scores.dtype}",
        )
    q2 = torch.zeros(2, 4, 128, dtype=torch.bfloat16)
    rows2 = torch.zeros(2, dtype=torch.int32)
    lengths = torch.zeros(1, dtype=torch.int32)
    overlap = torch.empty(n_columns).expand(2, n_columns)
    _expect_value_error(
        lambda: qsa_indexer_block_ids(
            q2, cache, table, rows2, rows2, lengths, scores=overlap
        ),
        "scores rows overlap",
        "overlapping score rows",
    )

    short_table = torch.zeros(1, 1, dtype=torch.int32)
    context = torch.zeros(2, dtype=torch.int32)
    _expect_value_error(
        lambda: qsa_indexer_block_ids(q, cache, short_table, rows, rows, context),
        "context lengths",
        "page table shorter than context_lens",
    )
    _expect_value_error(
        lambda: qsa_indexer_block_ids(
            q, cache, torch.empty(0, 1, dtype=torch.int32), rows, rows, lengths
        ),
        "context lengths",
        "page table with no request rows",
    )

    page_size, head_dim = 16, 128
    n_pages = 3
    page_elems = page_size * head_dim
    dense = torch.zeros(n_pages, page_size, 1, head_dim, dtype=torch.bfloat16)
    one_table = torch.zeros(1, 1, dtype=torch.int32)
    if qsa_paged.qsa_indexer_supported(q, dense, one_table, (4,)) is not None:
        raise AssertionError("a packed indexer cache was rejected")
    page_stride = page_elems + 256
    parent = torch.empty(n_pages, page_stride, dtype=torch.bfloat16)
    view = parent.as_strided(
        (n_pages, page_size, 1, head_dim), (page_stride, head_dim, head_dim, 1)
    )
    if qsa_paged.qsa_indexer_supported(q, view, one_table, (4,)) is not None:
        raise AssertionError(f"padded page stride was rejected: {view.stride()}")
    bad = parent.as_strided(
        (n_pages, page_size, 1, head_dim), (page_elems + 4, head_dim, head_dim, 1)
    )
    reason = qsa_paged.qsa_indexer_supported(q, bad, one_table, (4,))
    if reason is None or "multiples of 8" not in reason:
        raise AssertionError(f"misaligned page stride served: {reason}")
    wide = torch.empty(n_pages, page_size, 1, head_dim * 2, dtype=torch.bfloat16)
    strided_d = wide.as_strided(
        dense.shape, (wide.stride(0), wide.stride(1), wide.stride(2), 2)
    )
    reason = qsa_paged.qsa_indexer_supported(q, strided_d, one_table, (4,))
    if reason is None or "unit D" not in reason:
        raise AssertionError(f"strided D was served: {reason}")
    reason = qsa_paged.qsa_indexer_supported(
        q, torch.zeros(1, 0, 1, head_dim, dtype=torch.bfloat16), one_table, (4,)
    )
    if reason is None or "page_size" not in reason:
        raise AssertionError(f"zero page size was served: {reason}")

    def gqa_tensors(hq, hkv, d, page, gqa_rows=1, gqa_width=8):
        gq = torch.zeros(gqa_rows, hq, d, dtype=torch.bfloat16)
        gk = torch.zeros(1, page, hkv, d, dtype=torch.bfloat16)
        indices = torch.zeros(gqa_rows, gqa_width, dtype=torch.int32)
        gtable = torch.zeros(1, 1, dtype=torch.int32)
        return gq, gk, torch.zeros_like(gk), indices, gtable

    if flydsl_qsa_paged_gqa_supported(*gqa_tensors(24, 2, 256, 16)) is not None:
        raise AssertionError("Flash-Next GQA was rejected")
    rejected = (
        (24, 5, 256, 16, 1, 8, "do not group"),
        (24, 0, 256, 16, 1, 8, "do not group"),
        (24, 1, 256, 16, 1, 8, "MFMA"),
        (17, 1, 128, 16, 1, 8, "MFMA"),
        (24, 2, 80, 16, 1, 8, "multiple of 32"),
        (24, 2, 256, 0, 1, 8, "page_size"),
        (16, 1, 32, 16, 1, 8, "gather"),
        (16, 1, 96, 16, 1, 8, "gather"),
        (24, 2, 192, 16, 32, 8, "K LDS"),
    )
    for hq, hkv, d, page, gqa_rows, gqa_width, needle in rejected:
        reason = flydsl_qsa_paged_gqa_supported(
            *gqa_tensors(hq, hkv, d, page, gqa_rows, gqa_width)
        )
        if reason is None or needle not in reason:
            raise AssertionError(
                f"Hq={hq} Hkv={hkv} D={d} page={page} rows={gqa_rows} "
                f"width={gqa_width} -> {reason!r}"
            )

    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    q = q.to(device)
    cache = cache.to(device)
    rows = rows.to(device)
    one_context = torch.zeros(1, dtype=torch.int32, device=device)
    for name, k_cache_e, page_table, context_lens in (
        (
            "empty context",
            cache,
            torch.zeros(1, 1, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
        ),
        (
            "empty cache",
            torch.empty(0, 16, 1, 128, dtype=torch.bfloat16, device=device),
            torch.zeros(1, 1, dtype=torch.int32, device=device),
            one_context,
        ),
        (
            "no logical pages",
            cache,
            torch.empty(1, 0, dtype=torch.int32, device=device),
            one_context,
        ),
    ):
        out = qsa_indexer_block_ids(q, k_cache_e, page_table, rows, rows, context_lens)
        if not torch.equal(out, torch.full_like(out, -1)):
            raise AssertionError(f"indexer {name} output was not -1")

    gqa = FLASH_NEXT_GQA
    gq = torch.randn(1, gqa.n_heads, gqa.head_dim, dtype=dtypes.bf16, device=device)
    indices = torch.zeros(1, 4, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(1, dtype=dtypes.i32, device=device)
    pages = torch.zeros(
        1, 16, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    empty_cache = torch.empty(
        0, 16, gqa.kv_heads, gqa.head_dim, dtype=dtypes.bf16, device=device
    )
    for name, gk, gv, page_table in (
        (
            "empty cache",
            empty_cache,
            empty_cache,
            torch.zeros(1, 1, dtype=dtypes.i32, device=device),
        ),
        (
            "empty table",
            pages,
            pages,
            torch.empty(1, 0, dtype=dtypes.i32, device=device),
        ),
        (
            "no requests",
            pages,
            pages,
            torch.empty(0, 1, dtype=dtypes.i32, device=device),
        ),
    ):
        out = flydsl_qsa_paged_gqa(gq, gk, gv, indices, page_table, token_to_req)
        if not torch.equal(out, torch.zeros_like(out)):
            raise AssertionError(f"paged GQA {name} output was not zeros")

    if torch.cuda.is_available():
        rows_e, compress_ratio, token_topk = 2, 4, 8
        output_width = token_topk + compress_ratio - 1
        block_indices = torch.zeros(
            (rows_e, token_topk // compress_ratio), dtype=torch.int32, device=device
        )
        query_positions = torch.zeros(rows_e, dtype=torch.int32, device=device)
        sequence_lengths = torch.zeros(1, dtype=torch.int32, device=device)
        token_to_req_e = torch.zeros(rows_e, dtype=torch.int32, device=device)
        for out in (
            torch.empty((rows_e, 1), dtype=torch.int32, device=device),
            torch.empty((rows_e, output_width), dtype=torch.int64, device=device),
        ):
            _expect_value_error(
                lambda out=out: qsa_expand_block_indices(
                    block_indices,
                    query_positions,
                    sequence_lengths,
                    token_to_req_e,
                    compress_ratio,
                    token_topk,
                    out,
                ),
                "out has an invalid shape",
                f"expand out {out.dtype} {tuple(out.shape)}",
            )
        empty_out = torch.empty((0, output_width), dtype=torch.int32, device=device)
        got = qsa_expand_block_indices(
            torch.zeros(
                (0, token_topk // compress_ratio), dtype=torch.int32, device=device
            ),
            query_positions[:0],
            sequence_lengths,
            token_to_req_e[:0],
            compress_ratio,
            token_topk,
            empty_out,
        )
        if got.data_ptr() != empty_out.data_ptr():
            raise AssertionError("a matching caller out was replaced")
        padded = torch.empty(
            (rows_e, output_width + 8), dtype=torch.int32, device=device
        )
        wider = padded[:, :output_width]
        if int(wider.stride(0)) <= output_width:
            raise AssertionError("padded expand fixture does not keep a row gap")
        got = qsa_expand_block_indices(
            block_indices,
            query_positions,
            sequence_lengths,
            token_to_req_e,
            compress_ratio,
            token_topk,
            wider,
        )
        if got.data_ptr() != wider.data_ptr():
            raise AssertionError("a padded expand out was replaced")
        _check_unmeasured_gqa_warns_once(device)


@benchmark()
def bench_qsa_flash_next_indexer(
    m,
    seq_len,
    page_size,
    dtype,
    rotate=0,
    pad_pages=0,
    cache_layout="packed",
    requests=1,
):
    """Flash-Next indexer versus oracle set equality."""
    device = torch.device("cuda")
    case = _indexer_case(
        FLASH_NEXT_INDEXER,
        m,
        seq_len,
        page_size,
        dtype,
        pad_pages,
        cache_layout,
        requests,
        device,
    )
    idx = case["idx"]
    q_indexer = case["q"]
    index_cache = case["cache"]
    index_table = case["table"]
    token_to_req = case["token_to_req"]
    qpos = case["qpos"]
    slen = case["slen"]
    ref_scores = case["ref_scores"]
    ref_ids = case["ref_ids"]
    n_blocks = case["n_blocks"]

    flops = 2 * m * idx.n_heads * idx.head_dim * n_blocks
    nbytes = (
        m * idx.n_heads * idx.head_dim + n_blocks * requests * idx.head_dim
    ) * dtype.itemsize
    select_args = (
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        idx.token_budget,
        idx.compress_ratio,
    )
    # Paged caches stay positional so ``run_perftest`` can rotate them.
    candidates = {
        "indexer": (_indexer_select, {"heads": (4,)}),
    }
    ret = {
        "gfx": get_gfx(),
        "n_blocks": n_blocks,
        "n_pages": int(index_table.shape[1]),
        "cache_span_gib": _span_gib(index_cache),
    }
    for name, (fn, extra) in candidates.items():
        got, us = run_perftest(
            fn,
            *_WideRotation.wrap(rotate, *select_args),
            num_rotate_args=rotate,
            num_iters=_WideRotation.iters,
            **extra,
        )
        err = _set_mismatch_ratio(ref_ids, got[1], ref_scores)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    # Padded decode uses the row-bounded radix on the full width, eager
    # and captured. The graph column times replay of that same selector.
    n_columns = int(index_table.shape[1]) * int(page_size)
    if qsa_paged.qsa_indexer_row_bounded(m, n_columns):
        block_ids_g = torch.empty(m, idx.block_budget, dtype=dtypes.i32, device=device)
        indices_g = torch.empty(m, idx.index_width, dtype=dtypes.i32, device=device)

        def flydsl_graph():
            qsa_indexer_block_ids(
                q_indexer,
                index_cache,
                index_table,
                token_to_req,
                qpos,
                slen,
                out=block_ids_g,
                heads=(4,),
            )
            qsa_expand_block_indices(
                block_ids_g,
                qpos,
                slen,
                token_to_req,
                idx.compress_ratio,
                idx.token_budget,
                out=indices_g,
            )

        graph_candidates = {
            "indexer_graph": (flydsl_graph, lambda: block_ids_g),
        }
        for name, (launch, ids) in graph_candidates.items():
            us = _capture_replay_us(launch)
            err = _set_mismatch_ratio(ref_ids, ids(), ref_scores)
            ret[f"{name} us"] = us
            ret[f"{name} TFLOPS"] = flops / us / 1e6
            ret[f"{name} TB/s"] = nbytes / us / 1e6
            ret[f"{name} err"] = err
    _assert_indexer_caller_workspace(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        ref_ids,
        page_size,
    )
    return ret


def _flash_next_gqa(n_heads: int, kv_heads: int) -> QsaGqaSpec:
    """Flash-Next sparse GQA at one tensor-parallel rank."""
    if kv_heads < 1 or n_heads % kv_heads != 0:
        raise ValueError(f"query heads {n_heads} do not group over {kv_heads} KV heads")
    group = n_heads // kv_heads
    if group > 16:
        raise ValueError(f"group {group} does not fit one MFMA M dim")
    if (n_heads, kv_heads) == (FLASH_NEXT_GQA.n_heads, FLASH_NEXT_GQA.kv_heads):
        return FLASH_NEXT_GQA
    return QsaGqaSpec(
        n_heads=n_heads,
        kv_heads=kv_heads,
        head_dim=FLASH_NEXT_GQA.head_dim,
    )


@benchmark()
def bench_qsa_flash_next_paged_gqa(
    m,
    seq_len,
    page_size,
    dtype,
    n_heads,
    kv_heads,
    rotate=0,
    cache_layout="packed",
    kv_layout="packed",
):
    """Flash-Next sparse GQA versus the oracle."""
    case = _flash_next_gqa_case(
        m, seq_len, page_size, dtype, n_heads, kv_heads, cache_layout, kv_layout
    )
    gqa = case["gqa"]
    q_gqa = case["q"]
    k_cache = case["k_cache"]
    v_cache = case["v_cache"]
    indices = case["indices"]
    kv_table = case["kv_table"]
    token_to_req = case["token_to_req"]
    ref = case["ref"]
    n_blocks = case["n_blocks"]

    w_alloc = indices.shape[1]
    w = _selected_width(indices)
    flops = 4 * m * gqa.n_heads * gqa.head_dim * w
    nbytes = (
        m * gqa.n_heads * gqa.head_dim * 2 + 2 * w * gqa.kv_heads * gqa.head_dim
    ) * dtype.itemsize
    gqa_args = (q_gqa, k_cache, v_cache, indices, kv_table, token_to_req)
    candidates = {
        "paged_gqa": (flydsl_qsa_paged_gqa, "paged GQA vs oracle GQA"),
    }
    ret = {
        "gfx": get_gfx(),
        "n_blocks": n_blocks,
        "width": w_alloc,
        "valid%": 100.0 * w / w_alloc,
        "cache_span_gib": _span_gib(k_cache),
    }
    live_n_sel = _paged_gqa_live_n_sel(
        m, seq_len, FLASH_NEXT_INDEXER.token_budget, FLASH_NEXT_INDEXER.compress_ratio
    )
    for name, (fn, msg) in candidates.items():
        extra = {"n_sel": live_n_sel} if fn is flydsl_qsa_paged_gqa else {}
        out, us = run_perftest(
            fn,
            *_WideRotation.wrap(rotate, *gqa_args),
            num_rotate_args=rotate,
            num_iters=_WideRotation.iters,
            **extra,
        )
        err = checkAllclose(
            ref.output,
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=msg,
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    _assert_gqa_caller_workspace(
        q_gqa,
        k_cache,
        v_cache,
        indices,
        kv_table,
        token_to_req,
        out,
        n_sel=live_n_sel,
    )
    return ret


@benchmark()
def bench_qsa_gluon_parity_indexer(
    m,
    seq_len,
    page_size,
    dtype,
    index_heads,
    rotate=0,
    pad_pages=0,
    cache_layout="packed",
):
    """Gluon-parity indexer versus oracle set equality."""
    idx = _gluon_parity_indexer(index_heads)
    device = torch.device("cuda")
    case = _indexer_case(
        idx, m, seq_len, page_size, dtype, pad_pages, cache_layout, 1, device
    )
    q_indexer = case["q"]
    index_cache = case["cache"]
    index_table = case["table"]
    token_to_req = case["token_to_req"]
    qpos = case["qpos"]
    slen = case["slen"]
    ref_scores = case["ref_scores"]
    ref_ids = case["ref_ids"]
    n_blocks = case["n_blocks"]
    score_scale = idx.head_dim**-0.5

    flops = 2 * m * idx.n_heads * idx.head_dim * n_blocks
    nbytes = (m * idx.n_heads * idx.head_dim + n_blocks * idx.head_dim) * dtype.itemsize
    select_args = (
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
    )
    candidates = {
        "indexer": qsa_indexer_block_ids,
    }
    ret = {
        "gfx": get_gfx(),
        "index_heads": idx.n_heads,
        "n_blocks": n_blocks,
        "n_pages": int(index_table.shape[1]),
        "cache_span_gib": _span_gib(index_cache),
    }
    for name, fn in candidates.items():
        block_ids, us = run_perftest(
            fn,
            *_WideRotation.wrap(rotate, *select_args),
            num_rotate_args=rotate,
            num_iters=_WideRotation.iters,
            score_scale=score_scale,
        )
        err = _set_mismatch_ratio(ref_ids, block_ids, ref_scores)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def _gluon_parity_indexer(index_heads):
    if index_heads == GLUON_PARITY_INDEXER.n_heads:
        return GLUON_PARITY_INDEXER
    if index_heads == GLUON_PARITY_INDEXER_H8.n_heads:
        return GLUON_PARITY_INDEXER_H8
    raise ValueError(f"Gluon-parity indexer heads must be 4 or 8, got {index_heads}")


def _prepare_qsa_layer(
    idx,
    gqa,
    m,
    seq_len,
    page_size,
    dtype,
    cache_layout="packed",
    kv_layout="packed",
    widen_index=True,
):
    """Dense and paged tensors for one QSA layer, plus the oracle."""
    device = torch.device("cuda")
    q_indexer, k_bar, qpos, slen, token_to_req = _indexer_inputs(
        idx, m, seq_len, dtype, device
    )
    q_gqa = torch.randn(
        m, gqa.n_heads, gqa.head_dim, dtype=dtype, device=device
    ).contiguous()
    k = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    v = torch.randn(seq_len, gqa.kv_heads, gqa.head_dim, dtype=dtype, device=device)
    index_cache, index_table, k_cache, v_cache, kv_table = _pack_flash_next(
        k_bar, k, v, page_size
    )
    if widen_index:
        index_cache = _apply_cache_layout(index_cache.contiguous(), cache_layout)
    index_table = index_table.contiguous()
    k_cache = _apply_cache_layout(k_cache.contiguous(), cache_layout)
    v_cache = _apply_cache_layout(v_cache.contiguous(), cache_layout)
    k_cache, v_cache = _apply_kv_layout(k_cache, v_cache, kv_layout)
    if k_cache.stride() != v_cache.stride():
        raise RuntimeError(
            f"K and V strides diverged: {k_cache.stride()} vs {v_cache.stride()}"
        )
    kv_table = kv_table.contiguous()
    ref = qsa_oracle(
        q_indexer,
        k_bar,
        q_gqa,
        k,
        v,
        qpos,
        slen,
        token_to_req,
        idx,
        gqa,
        score_scale=idx.head_dim**-0.5,
    )
    return {
        "n_blocks": seq_len // idx.compress_ratio,
        "q_indexer": q_indexer,
        "index_cache": index_cache,
        "index_table": index_table,
        "q_gqa": q_gqa,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "kv_table": kv_table,
        "token_to_req": token_to_req,
        "qpos": qpos,
        "slen": slen,
        "ref": ref,
    }


def _layer_args(case):
    return (
        case["q_indexer"],
        case["index_cache"],
        case["index_table"],
        case["q_gqa"],
        case["k_cache"],
        case["v_cache"],
        case["kv_table"],
        case["token_to_req"],
        case["qpos"],
        case["slen"],
    )


def _run_qsa(
    q_indexer,
    index_cache,
    index_table,
    q_gqa,
    k_cache,
    v_cache,
    kv_table,
    token_to_req,
    qpos,
    slen,
    *,
    indices,
    block_ids,
    out,
    n_heads,
    token_topk,
    compress_ratio,
    score_scale,
    scores=None,
):
    """Indexer, expand, and sparse GQA on buffers the caller already owns."""
    qsa_indexer_block_ids(
        q_indexer,
        index_cache,
        index_table,
        token_to_req,
        qpos,
        slen,
        out=block_ids,
        score_scale=score_scale,
        heads=(n_heads,),
        scores=scores,
    )
    qsa_expand_block_indices(
        block_ids,
        qpos,
        slen,
        token_to_req,
        compress_ratio,
        token_topk,
        indices,
    )
    return flydsl_qsa_paged_gqa(
        q_gqa,
        k_cache,
        v_cache,
        indices,
        kv_table,
        token_to_req,
        out=out,
    )


def _capture_replay_us(fn) -> float:
    """Warm up, capture one HIP graph, and time ``replay``. Replay stays hot."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            graph.capture_begin()
            fn()
            graph.capture_end()
    except RuntimeError:
        torch.cuda.current_stream().wait_stream(stream)
        raise
    torch.cuda.current_stream().wait_stream(stream)
    _ignored, us = run_perftest(graph.replay, num_rotate_args=1)
    graph.replay()
    torch.cuda.synchronize()
    return us


def _e2e_counts(case, idx, gqa, dtype):
    indices = case["ref"].indices
    w_alloc = indices.shape[1]
    w = _selected_width(indices)
    m = case["q_gqa"].shape[0]
    flops = (
        2 * m * idx.n_heads * idx.head_dim * case["n_blocks"]
        + 4 * m * gqa.n_heads * gqa.head_dim * w
    )
    nbytes = (
        m * idx.n_heads * idx.head_dim
        + case["n_blocks"] * idx.head_dim
        + m * gqa.n_heads * gqa.head_dim * 2
        + 2 * w * gqa.kv_heads * gqa.head_dim
    ) * dtype.itemsize
    return w_alloc, w, flops, nbytes


def _e2e_cells(name, us, err, flops, nbytes):
    return {
        f"{name} us": us,
        f"{name} TFLOPS": flops / us / 1e6,
        f"{name} TB/s": nbytes / us / 1e6,
        f"{name} err": err,
    }


def _raise_if_row_mismatch(row, m, seq_len):
    """Fail when a candidate error fraction is nonzero."""
    for key, err in row.items():
        if key.endswith(" err") and err != 0:
            raise AssertionError(
                f"{key}={err} at M={m} L={seq_len} is above that column's tolerance"
            )


def _time_layer(case, idx, rotate):
    """Time indexer, expand, and sparse GQA on buffers the caller already owns."""
    m = case["q_gqa"].shape[0]
    device = case["q_gqa"].device
    indices = torch.empty((m, idx.index_width), dtype=dtypes.i32, device=device)
    block_ids = torch.empty((m, idx.block_budget), dtype=dtypes.i32, device=device)
    out = torch.empty_like(case["q_gqa"])
    kwargs = {
        "indices": indices,
        "block_ids": block_ids,
        "out": out,
        "n_heads": idx.n_heads,
        "token_topk": idx.token_budget,
        "compress_ratio": idx.compress_ratio,
        "score_scale": idx.head_dim**-0.5,
    }
    n_columns = int(case["index_table"].shape[1]) * int(case["index_cache"].shape[1])
    if n_columns > idx.block_budget:
        kwargs["scores"] = torch.empty((m, n_columns), dtype=dtypes.fp32, device=device)
    timed, us = run_perftest(
        _run_qsa,
        *_WideRotation.wrap(rotate, *_layer_args(case)),
        num_rotate_args=rotate,
        num_iters=_WideRotation.iters,
        **kwargs,
    )
    return timed, us


@benchmark()
def bench_qsa_flash_next_e2e(
    m,
    seq_len,
    page_size,
    dtype,
    n_heads,
    kv_heads,
    rotate=0,
    cache_layout="packed",
):
    """One Flash-Next QSA layer: indexer, expand, and sparse GQA."""
    idx = FLASH_NEXT_INDEXER
    gqa = _flash_next_gqa(n_heads, kv_heads)
    case = _prepare_qsa_layer(idx, gqa, m, seq_len, page_size, dtype, cache_layout)
    ref = case["ref"].output
    w_alloc, w, flops, nbytes = _e2e_counts(case, idx, gqa, dtype)
    ret = {
        "gfx": get_gfx(),
        "n_blocks": case["n_blocks"],
        "width": w_alloc,
        "valid%": 100.0 * w / w_alloc,
        "cache_span_gib": _span_gib(case["k_cache"]),
    }
    out, us = _time_layer(case, idx, rotate)
    err = checkAllclose(
        ref, out.to(dtypes.fp32), rtol=1e-2, atol=1e-2, msg="e2e vs oracle"
    )
    ret.update(_e2e_cells("flydsl_e2e", us, err, flops, nbytes))
    _raise_if_row_mismatch(ret, m, seq_len)
    return ret


@benchmark()
def bench_qsa_gluon_parity_e2e(
    m, seq_len, page_size, dtype, index_heads, rotate=0, cache_layout="packed"
):
    """One Gluon-parity QSA layer."""
    idx = _gluon_parity_indexer(index_heads)
    gqa = GLUON_PARITY_GQA
    case = _prepare_qsa_layer(idx, gqa, m, seq_len, page_size, dtype, cache_layout)
    ref = case["ref"].output
    w_alloc, w, flops, nbytes = _e2e_counts(case, idx, gqa, dtype)
    ret = {
        "gfx": get_gfx(),
        "index_heads": idx.n_heads,
        "n_blocks": case["n_blocks"],
        "width": w_alloc,
        "valid%": 100.0 * w / w_alloc,
        "cache_span_gib": _span_gib(case["k_cache"]),
    }
    out, us = _time_layer(case, idx, rotate)
    err = checkAllclose(
        ref, out.to(dtypes.fp32), rtol=1e-2, atol=1e-2, msg="Gluon-parity e2e"
    )
    ret.update(_e2e_cells("flydsl_e2e", us, err, flops, nbytes))
    _raise_if_row_mismatch(ret, m, seq_len)
    return ret


@benchmark()
def bench_qsa_flash_next_e2e_graph(
    m, seq_len, page_size, dtype, n_heads, kv_heads, cache_layout="packed"
):
    """HIP-graph replay of one Flash-Next decode layer."""
    idx = FLASH_NEXT_INDEXER
    gqa = _flash_next_gqa(n_heads, kv_heads)
    case = _prepare_qsa_layer(idx, gqa, m, seq_len, page_size, dtype, cache_layout)
    ref = case["ref"].output
    _w_alloc, w, flops, nbytes = _e2e_counts(case, idx, gqa, dtype)
    device = case["q_gqa"].device
    width = idx.index_width

    def _replay(out):
        indices = torch.empty((m, width), dtype=dtypes.i32, device=device)
        block_ids = torch.empty((m, idx.block_budget), dtype=dtypes.i32, device=device)

        def launch():
            return _run_qsa(
                *_layer_args(case),
                indices=indices,
                block_ids=block_ids,
                out=out,
                n_heads=idx.n_heads,
                token_topk=idx.token_budget,
                compress_ratio=idx.compress_ratio,
                score_scale=idx.head_dim**-0.5,
            )

        us = _capture_replay_us(launch)
        err = checkAllclose(
            ref,
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg="graph replay vs oracle",
        )
        return us, err

    us, err = _replay(torch.empty_like(case["q_gqa"]))
    ret = {
        "gfx": get_gfx(),
        "n_blocks": case["n_blocks"],
        "valid%": 100.0 * w / width,
        "cache_span_gib": _span_gib(case["k_cache"]),
        **_e2e_cells("flydsl_graph", us, err, flops, nbytes),
    }
    _raise_if_row_mismatch(ret, m, seq_len)
    return ret


def _indexer_paged_gqa_sweep_batches(batches):
    """Split M into Flash-Next decode (M at most 8) and prefill."""
    decode = [m for m in batches if m <= 8]
    prefill = [m for m in batches if m > 8]
    return decode, prefill


def _skip_m_past_seq(m, seq_len, where):
    if m <= seq_len:
        return False
    aiter.logger.warning("skip %s M=%s L=%s (M must fit in L)", where, m, seq_len)
    return True


def _skip_wide_interleaved(cache_layout, kv_layout, where):
    """A wide cache stays packed. Interleaving it would copy the pages."""
    if cache_layout != "wide" or kv_layout != "interleaved":
        return False
    aiter.logger.warning("skip %s wide+interleaved", where)
    return True


def _skip_pad_narrower_than_context(
    seq_len, page_size, pad_pages, compress_ratio, where
):
    """A pad that cannot hold the context would drop real pages."""
    if pad_pages <= 0:
        return False
    n_blocks = seq_len // compress_ratio
    packed_pages = (n_blocks + page_size - 1) // page_size
    if packed_pages <= pad_pages:
        return False
    aiter.logger.warning(
        "skip %s L=%s page=%s pad_pages=%s (context needs %s pages)",
        where,
        seq_len,
        page_size,
        pad_pages,
        packed_pages,
    )
    return True


def _raise_if_indexer_mismatch(err, m, seq_len, where="FlyDSL indexer"):
    if err != 0:
        raise AssertionError(f"{where} set mismatch at M={m} L={seq_len} (err={err})")


def _raise_if_paged_gqa_above_tolerance(err, m, seq_len):
    if err != 0:
        raise AssertionError(
            f"paged GQA err={err} at M={m} L={seq_len} is above the unit "
            "tolerance rtol=1e-2 atol=1e-2"
        )


def _summarize(name, rows):
    """Log one sweep table."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    aiter.logger.info("%s (markdown):\n%s", name, df.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Flash-Next and Gluon-parity QSA sweeps, FlyDSL indexer and "
            "sparse GQA, and the end-to-end indexer, expand, and sparse GQA"
        ),
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="activation dtype (Flash-Next is BF16)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 8, 512],
        help=(
            "flattened query tokens M. Flash-Next indexer and sparse GQA "
            "report M<=8 as\n"
            "decode and every larger M as prefill."
        ),
    )
    parser.add_argument(
        "-s",
        "--seq",
        type=int,
        nargs="*",
        default=[512, 2048, 8192, 32768],
        help="context length L in tokens (32k default; pass 131072 for 128k).\n"
        "Bar shapes are budget-saturated: decode M in {1,8} at L=32768,\n"
        "prefill M=512 at L=8192. L=512 is a fast smoke row -- only ~25%%\n"
        "of the 2051 selection slots are live there (12.5%% at M=512), so\n"
        "it measures the masked path more than the gather. See valid%%.",
    )
    parser.add_argument(
        "-p",
        "--page-size",
        type=int,
        nargs="*",
        default=[16],
        help="vLLM-style page size (indexer slots and GQA tokens)",
    )
    parser.add_argument(
        "--pad-pages",
        type=int,
        nargs="*",
        default=[0],
        help="Indexer block-table width in pages. 0 keeps the packed table.\n"
        "A positive value zero-fills every indexer column's table out to that\n"
        "many pages, with zeros past the packed pages. A row\n"
        "whose context already needs more pages is skipped. Examples:\n"
        "--pad-pages 4096 with page size 16 is 65536 columns;\n"
        "--page-size 392 --pad-pages 168 is the Flash-Next TP2 table\n"
        "(65856 columns). Pass 0 and a width to sweep both.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        nargs="*",
        default=[1, 4],
        help=(
            "Indexer sequences sharing one row tile. 1 is the single-request\n"
            "sweep. Any larger value adds four rows per sequence (4 is the\n"
            "MTP block, M=16) and does not cross the prefill M list."
        ),
    )
    parser.add_argument(
        "--kv-layout",
        nargs="*",
        default=["packed", "interleaved"],
        choices=["packed", "interleaved"],
        help=(
            "Paged GQA K and V. packed is two caches. interleaved is the\n"
            "K|V view of those pages. A wide cache stays packed: interleaving\n"
            "it would copy the pages and drop the wide span."
        ),
    )
    parser.add_argument(
        "--gqa",
        type=dtypes.str2tuple,
        nargs="*",
        default=list(_FLASH_NEXT_GQA_RANKS),
        help=(
            "Flash-Next sparse GQA as query,kv head pairs for one\n"
            "tensor-parallel rank. 24,2 is TP1, 12,1 is TP2, 6,1 is TP4,\n"
            "and 3,1 is TP8. The indexer stays at 4 heads. Example:\n"
            "--gqa 24,2 12,1"
        ),
    )
    parser.add_argument(
        "--rotate",
        type=int,
        nargs="*",
        default=[0],
        help="run_perftest num_rotate_args (copies of timed tensors).\n"
        "0 = cold cache (default; auto-size copies from L2, matches serving).\n"
        "1 = hot cache, one reused buffer set.\n"
        "N>1 = that many copies.\n"
        "Same value on every named backend in a row. Not combined with HIP graphs.",
    )
    args = parser.parse_args()
    for spec in args.gqa:
        if not isinstance(spec, tuple) or len(spec) != 2:
            parser.error(f"--gqa expects query,kv pairs, got {spec!r}")
        try:
            _flash_next_gqa(*spec)
        except ValueError as exc:
            parser.error(str(exc))
    if any(n_req < 1 for n_req in args.requests):
        parser.error(f"--requests must be positive, got {args.requests}")

    if not torch.cuda.is_available():
        aiter.logger.warning("no CUDA; skipping QSA sweep")
        return
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("QSA unsupported on %s; skipping", get_gfx())
        return

    check_qsa_rejections()
    check_qsa_edges()

    for dtype in args.dtype:
        decode_m, prefill_m = _indexer_paged_gqa_sweep_batches(args.batch)

        decode_shapes = [(m, 1) for m in decode_m]
        seen_shapes = set(decode_shapes)
        for n_req in args.requests:
            if n_req <= 1:
                continue
            shape = (_MTP_ROWS_PER_REQUEST * n_req, n_req)
            if shape not in seen_shapes:
                decode_shapes.append(shape)
                seen_shapes.add(shape)

        sweeps = (
            ("decode", decode_m, decode_shapes),
            ("prefill", prefill_m, [(m, 1) for m in prefill_m]),
        )
        for label, gqa_m, indexer_shapes in sweeps:
            where = f"Flash-Next indexer {label}"
            rows = []
            for (
                (m, requests),
                seq_len,
                page_size,
                rotate,
                pad_pages,
                cache_layout,
            ) in itertools.product(
                indexer_shapes,
                args.seq,
                args.page_size,
                args.rotate,
                args.pad_pages,
                _CACHE_LAYOUTS,
            ):
                if _skip_m_past_seq(m, seq_len, where):
                    continue
                if _skip_pad_narrower_than_context(
                    seq_len,
                    page_size,
                    pad_pages,
                    FLASH_NEXT_INDEXER.compress_ratio,
                    where,
                ):
                    continue
                row = bench_qsa_flash_next_indexer(
                    m,
                    seq_len,
                    page_size,
                    dtype,
                    rotate,
                    pad_pages,
                    cache_layout,
                    requests,
                )
                _raise_if_indexer_mismatch(row["indexer err"], m, seq_len)
                if "indexer_graph err" in row:
                    _raise_if_indexer_mismatch(
                        row["indexer_graph err"],
                        m,
                        seq_len,
                        where="FlyDSL indexer graph",
                    )
                rows.append(row)
            _summarize(f"QSA Flash-Next FlyDSL indexer {label} summary", rows)

            where = f"Flash-Next sparse GQA {label}"
            rows = []
            for (
                m,
                seq_len,
                page_size,
                rotate,
                cache_layout,
                kv_layout,
                (n_heads, kv_heads),
            ) in itertools.product(
                gqa_m,
                args.seq,
                args.page_size,
                args.rotate,
                _CACHE_LAYOUTS,
                args.kv_layout,
                args.gqa,
            ):
                if _skip_wide_interleaved(cache_layout, kv_layout, where):
                    continue
                if _skip_m_past_seq(m, seq_len, where):
                    continue
                row = bench_qsa_flash_next_paged_gqa(
                    m,
                    seq_len,
                    page_size,
                    dtype,
                    n_heads,
                    kv_heads,
                    rotate,
                    cache_layout,
                    kv_layout,
                )
                _raise_if_paged_gqa_above_tolerance(row["paged_gqa err"], m, seq_len)
                rows.append(row)
            _summarize(f"QSA Flash-Next FlyDSL sparse GQA {label} summary", rows)

        gluon_ratio = GLUON_PARITY_INDEXER.compress_ratio
        gluon_sweeps = (
            # title, where, M list, L list, page sizes, indexer heads
            (
                "QSA Gluon-parity FlyDSL indexer H=8 summary",
                "Gluon-parity indexer H=8",
                args.batch,
                [s for s in args.seq if s // gluon_ratio <= 512],
                args.page_size,
                8,
            ),
            # Published indexer point: M=32, H=4, page_size=8, n_blocks=512.
            (
                "QSA Gluon-parity FlyDSL published indexer point",
                "Gluon-parity indexer published",
                [32],
                [2048],
                [8],
                4,
            ),
            (
                "QSA Gluon-parity FlyDSL indexer long-L summary",
                "Gluon-parity indexer long-L",
                args.batch,
                [s for s in args.seq if s // gluon_ratio > 512],
                args.page_size,
                8,
            ),
        )
        for title, where, gluon_m, gluon_seq, page_sizes, index_heads in gluon_sweeps:
            rows = []
            for (
                m,
                seq_len,
                page_size,
                rotate,
                pad_pages,
                cache_layout,
            ) in itertools.product(
                gluon_m,
                gluon_seq,
                page_sizes,
                args.rotate,
                args.pad_pages,
                _CACHE_LAYOUTS,
            ):
                if _skip_m_past_seq(m, seq_len, where):
                    continue
                if _skip_pad_narrower_than_context(
                    seq_len, page_size, pad_pages, gluon_ratio, where
                ):
                    continue
                row = bench_qsa_gluon_parity_indexer(
                    m,
                    seq_len,
                    page_size,
                    dtype,
                    index_heads,
                    rotate,
                    pad_pages,
                    cache_layout,
                )
                _raise_if_indexer_mismatch(row["indexer err"], m, seq_len)
                rows.append(row)
            _summarize(title, rows)

        rows = []
        for (
            m,
            seq_len,
            page_size,
            rotate,
            cache_layout,
            (n_heads, kv_heads),
        ) in itertools.product(
            args.batch,
            args.seq,
            args.page_size,
            args.rotate,
            _CACHE_LAYOUTS,
            args.gqa,
        ):
            if _skip_m_past_seq(m, seq_len, "Flash-Next end-to-end"):
                continue
            row = bench_qsa_flash_next_e2e(
                m,
                seq_len,
                page_size,
                dtype,
                n_heads,
                kv_heads,
                rotate,
                cache_layout,
            )
            rows.append(row)
        _summarize("QSA Flash-Next end-to-end summary", rows)

        rows = []
        for (
            m,
            seq_len,
            page_size,
            cache_layout,
            (n_heads, kv_heads),
        ) in itertools.product(
            [b for b in args.batch if b <= 8],
            args.seq,
            args.page_size,
            _CACHE_LAYOUTS,
            args.gqa,
        ):
            if _skip_m_past_seq(m, seq_len, "Flash-Next decode HIP-graph"):
                continue
            row = bench_qsa_flash_next_e2e_graph(
                m, seq_len, page_size, dtype, n_heads, kv_heads, cache_layout
            )
            rows.append(row)
        _summarize("QSA Flash-Next decode HIP-graph summary", rows)

        rows = []
        for (
            m,
            seq_len,
            page_size,
            index_heads,
            rotate,
            cache_layout,
        ) in itertools.product(
            args.batch,
            args.seq,
            args.page_size,
            (4, 8),
            args.rotate,
            _CACHE_LAYOUTS,
        ):
            if _skip_m_past_seq(m, seq_len, "Gluon-parity end-to-end"):
                continue
            row = bench_qsa_gluon_parity_e2e(
                m,
                seq_len,
                page_size,
                dtype,
                index_heads,
                rotate,
                cache_layout,
            )
            rows.append(row)
        _summarize("QSA Gluon-parity end-to-end summary", rows)


if __name__ == "__main__":
    main()
