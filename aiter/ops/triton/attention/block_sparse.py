# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Host-side SpargeAttn / VFA block-sparse attention preparation.

Turns Q/K into the ragged block-sparse look-up tables and the VFA ``m_init``
running-max estimates consumed by the sage kernel
(:mod:`aiter.ops.triton.attention.fav3_sage`). The underlying Triton kernels
live in ``aiter/ops/triton/_triton_kernels/attention/block_lut.py`` (block-mask
construction / LUT building) and ``.../vfa.py`` (the ``m_init`` estimator);
the per-block mean-pooling proxy lives in ``aiter/ops/triton/pool.py``.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch

from aiter.ops.triton.attention.utils import map_dims
from aiter.ops.triton._triton_kernels.attention.block_lut import (
    block_attn_mask_to_lut_kernel,
    triton_fill_block_map_kernel,
    triton_fill_causal_mask_kernel,
)
from aiter.ops.triton._triton_kernels.attention.vfa import (
    _sage_vfa_m_blockidx_kernel,
)
from aiter.ops.triton.pool import (
    get_pool_sim_triton_simmean,
    compute_pooled_scores,
)


def block_attn_mask_to_lut(
    block_attn_mask: torch.Tensor,
    lut_start: torch.Tensor,
    lut_count: torch.Tensor,
    kv_block_indices: torch.Tensor,
    BLOCK_KB: int = 128,
):
    """
    Launch the LUT-fill kernel. Caller must ensure block_attn_mask is 4D
    (batch, num_heads, num_q_blocks, num_kv_blocks) and kv_block_indices has
    length lut_count.sum().
    """
    batch, num_heads, num_q_blocks, num_kv_blocks = block_attn_mask.shape
    num_programs = batch * num_heads * num_q_blocks

    grid = (num_programs,)
    block_attn_mask_to_lut_kernel[grid](
        block_attn_mask,
        lut_start,
        lut_count,
        kv_block_indices,
        stride_mask_b=block_attn_mask.stride(0),
        stride_mask_h=block_attn_mask.stride(1),
        stride_mask_qb=block_attn_mask.stride(2),
        stride_mask_kb=block_attn_mask.stride(3),
        num_heads=num_heads,
        num_q_blocks=num_q_blocks,
        num_kv_blocks=num_kv_blocks,
        BLOCK_KB=BLOCK_KB,
    )


def block_attn_mask_to_ragged_lut(
    block_attn_mask: torch.Tensor,
    num_heads: Optional[int] = None,
    return_none_if_dense: bool = False,
    BLOCK_KB: int = 128,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Convert a dense block attention mask to a ragged look-up table of KV block
    indices per (batch, head, q_block). Used for block-sparse attention with no
    per-iteration branching in the kernel.

    block_attn_mask: Either (batch, num_q_blocks, num_kv_blocks) boolean for
        same mask for all heads, or (batch, num_heads, num_q_blocks, num_kv_blocks)
        for per-head masks. True = may attend, False = must not attend.
    num_heads: Required when block_attn_mask is 3D (number of Q heads). Ignored when 4D.
    return_none_if_dense: If True and the mask is all True (dense), return None so the
        caller can pass block_lut=None to fav3_sage_wrapper_func and use the dense path.
        Avoids building a very large LUT that can trigger munmap_chunk on MI300X/ROCm.
    Returns:
        kv_block_indices: 1D int32, concatenation of all KV block index lists.
        lut_start: 1D int32, length batch * num_heads * num_q_blocks. Index
            idx = batch_idx * (num_heads * num_q_blocks) + head_idx * num_q_blocks + q_block_idx.
        lut_count: 1D int32, same length as lut_start.
        When return_none_if_dense is True and the mask is all True, returns None instead.
    """
    device = block_attn_mask.device

    # 3D -> 4D: expand and fall through to 4D path
    if block_attn_mask.dim() == 3:
        if num_heads is None:
            raise ValueError("num_heads must be provided when block_attn_mask is 3D")
        batch, num_q_blocks, num_kv_blocks = block_attn_mask.shape
        if return_none_if_dense and block_attn_mask.all():
            return None
        block_attn_mask = block_attn_mask.unsqueeze(1).expand(
            batch, num_heads, num_q_blocks, num_kv_blocks
        )

    # 4D: (batch, num_heads, num_q_blocks, num_kv_blocks) — GPU vectorized path
    batch, num_heads, num_q_blocks, num_kv_blocks = block_attn_mask.shape
    if return_none_if_dense and block_attn_mask.all():
        return None

    counts = block_attn_mask.to(torch.int32).sum(dim=-1)
    lut_count = counts.reshape(-1)
    lut_start = torch.cumsum(lut_count, dim=0) - lut_count

    # NOTE: Overallocating the LUT is a waste of memory, but the
    # alternative lut_count.sum(), will cause graph break with torch compile.
    max_count = batch * num_heads * num_q_blocks * num_kv_blocks
    kv_block_indices = torch.empty(max_count, dtype=torch.int32, device=device)
    block_attn_mask_to_lut(
        block_attn_mask,
        lut_start,
        lut_count,
        kv_block_indices,
        BLOCK_KB=BLOCK_KB,
    )

    return kv_block_indices, lut_start, lut_count


# ============================================================================
# SpargeAttn block-sparse mask construction
#
# Mean-pool Q/K into per-block representatives, score the pooled blocks, and
# select the smallest set of top-scoring K blocks per query block (the
# SpargeAttn "meansim" proxy). The resulting dense block mask is turned into a
# ragged LUT by :func:`build_attention_lut` for the block-sparse sage kernel.
#
# Reference: "SpargeAttn: Accurate Sparse Attention Accelerating Any Model
# Inference" (https://arxiv.org/abs/2502.18137).
# ============================================================================
def fill_block_map_triton(
    final_map: torch.Tensor,
    num_to_select: torch.Tensor,
    sorted_indices: torch.Tensor,
) -> torch.Tensor:
    """Scatter the top-``num_to_select`` ranked K blocks per (B, H, Q) into ``final_map``."""
    final_map = final_map.contiguous()
    num_to_select = num_to_select.contiguous()
    sorted_indices = sorted_indices.contiguous()
    B, H, Q, K = final_map.shape
    grid = (B, H, Q)
    triton_fill_block_map_kernel[grid](final_map, num_to_select, sorted_indices, K)
    return final_map


def fill_causal_mask_triton(mask: torch.Tensor, BqdivBk: float) -> torch.Tensor:
    """Fill a 2-D ``(nq, nk)`` block-level causal mask for a Q/K block-size ratio."""
    assert mask.dim() == 2
    triton_fill_causal_mask_kernel[mask.shape](mask, BqdivBk)
    return mask


def get_block_map_meansim(
    q: torch.Tensor,
    k: torch.Tensor,
    is_causal: bool = False,
    BLKQ: int = 64,
    BLKK: int = 64,
    simthreshd1: float = 0.1,
    cdfthreshd: float = 0.9,
    attention_sink: bool = False,
    attention_scored_only: bool = False,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Build a SpargeAttn block-sparse mask via the mean-similarity proxy.

    Mean-pools Q/K into per-block representatives, forms the block-level score
    ``pooled_q @ pooled_k^T * d**-0.5``, softmaxes over key blocks, and keeps the
    smallest set of top key blocks whose cumulative probability reaches
    ``cdfthreshd`` per query block.  Blocks that fail the intra-block similarity
    test are forced on (their mean is not a faithful summary, so they must be
    evaluated exactly).

    Args:
        attention_scored_only: when ``True``, skip the intra-block similarity
            test and block-selection logic entirely, returning
            ``(None, pooled_score)`` where ``pooled_score`` is the raw
            block-level score ``pooled_q @ pooled_k^T * d**-0.5``.

    Returns:
        A ``(final_map, pooled_score)`` tuple.  ``final_map`` is a
        ``(B, H, num_q_blocks, num_k_blocks)`` bool mask (or ``None`` when
        ``attention_scored_only`` is set), and ``pooled_score`` is the
        block-level score with positions that ``final_map`` masks out set to
        ``-inf``.
    """
    nq = (q.shape[-2] + BLKQ - 1) // BLKQ
    nk = (k.shape[-2] + BLKK - 1) // BLKK
    pooled_q, sim_q = get_pool_sim_triton_simmean(
        q, BLKQ, simthreshd1, attention_scored_only
    )
    pooled_k, sim_k = get_pool_sim_triton_simmean(
        k, BLKK, simthreshd1, attention_scored_only
    )
    pooled_score = pooled_q @ pooled_k.transpose(-1, -2) * q.shape[-1] ** -0.5
    if attention_scored_only:
        return None, pooled_score

    neg_inf = pooled_score.new_full((), float("-inf"))
    sim_k = sim_k.unsqueeze(-2).expand(-1, -1, nq, -1)  # faster than repeat
    sim_q = sim_q.unsqueeze(-1).expand(-1, -1, -1, nk)

    prob = torch.where(sim_k, pooled_score, neg_inf)
    causal_mask = None
    if is_causal:
        causal_mask = fill_causal_mask_triton(
            torch.empty(nq, nk, device=q.device, dtype=torch.bool), BLKQ / BLKK
        )
        prob = torch.where(causal_mask[None, None], prob, neg_inf)
    prob = prob.softmax(-1)

    # Keep the smallest set of top key blocks whose cumulative mass reaches cdfthreshd.
    sorted_score = torch.sort(prob, dim=-1, descending=True)
    cdf = sorted_score.values.cumsum(dim=-1)
    H, K = cdf.shape[1], cdf.shape[-1]
    ge = cdf >= cdfthreshd.view(1, H, 1, 1)
    idx = ge.to(torch.uint8).argmax(dim=-1)
    num_to_select = torch.where(ge.any(dim=-1), idx, idx.new_full((), K))

    final_map = fill_block_map_triton(
        torch.zeros_like(prob, dtype=torch.bool), num_to_select, sorted_score.indices
    )
    final_map = final_map | ~sim_k | ~sim_q
    if is_causal:
        final_map = final_map * causal_mask[None, None]
    if attention_sink:
        final_map[:, :, :, 0] = 1
    return final_map, pooled_score


def _num_text_blocks(text_len: int, block_m: int, block_n: int) -> Tuple[int, int]:
    """Number of (q, k) blocks spanned by ``text_len`` trailing text tokens."""
    return (
        (text_len + block_m - 1) // block_m,
        (text_len + block_n - 1) // block_n,
    )


def _assemble_full_block_mask(
    image_block_mask: torch.Tensor,
    image_len_q: int,
    image_len_k: int,
    text_len: int,
    block_m: int,
    block_n: int,
) -> torch.Tensor:
    """Append dense text rows/columns to an image-only block mask.

    Returns the full ``(B, H, n_iq + n_text_q, n_ik + n_text_k)`` mask in which
    all Q rows attend to the text K columns, all text Q rows attend to
    everything, and any partial image/text boundary block is forced dense (so
    spillover tokens are never dropped).  A no-op when ``text_len == 0``.
    """
    if text_len == 0:
        return image_block_mask

    B, H, n_iq, n_ik = image_block_mask.shape
    n_text_q, n_text_k = _num_text_blocks(text_len, block_m, block_n)

    full = torch.zeros(
        (B, H, n_iq + n_text_q, n_ik + n_text_k),
        dtype=image_block_mask.dtype,
        device=image_block_mask.device,
    )
    full[:, :, :n_iq, :n_ik] = image_block_mask
    full[:, :, :, -n_text_k:] = True  # every Q row attends to text K cols
    full[:, :, -n_text_q:, :] = True  # text Q rows attend to everything
    if image_len_q % block_m != 0:  # partial boundary blocks -> dense
        full[:, :, image_len_q // block_m, :] = True
    if image_len_k % block_n != 0:
        full[:, :, :, image_len_k // block_n] = True
    return full


def block_attn_mask_to_ragged_lut_topn_front(
    block_attn_mask: torch.Tensor,
    pooled_score: torch.Tensor,
    sample_n: int,
    num_heads: Optional[int] = None,
    force_front_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a ragged LUT with the top-``sample_n`` scored K blocks emitted first.

    Like :func:`block_attn_mask_to_ragged_lut`, this turns a block attention
    mask into a ragged ``(kv_block_indices, lut_start, lut_count)`` LUT, but
    within each ``(batch, head, q_block)`` segment it emits blocks in this order:

      1. the ``sample_n`` highest ``pooled_score`` blocks, descending score;
      2. the ``force_front_mask`` blocks (e.g. text), ascending block index;
      3. the remaining attended blocks, ascending block index.

    Building the LUT directly from the mask lets us write each segment out in the
    desired order in a single pass -- no separate reorder of a pre-built LUT. The
    valid entries are compactly packed into the prefix of ``kv_block_indices``
    (indexed by ``lut_start``/``lut_count``); the buffer itself is over-allocated
    to a static size so the whole function is CUDA-graph-safe (no data-dependent
    shapes or device syncs), matching :func:`block_attn_mask_to_ragged_lut`.

    This pairs with the ``freeze_softmax_max_count`` block-sparse path
    (:func:`aiter.ops.triton.attention.fav3_sage.fav3_sage_func`): the
    online-softmax running max is frozen after the first few inner-loop
    iterations, so visiting the highest-scoring (and thus likely highest-max)
    tiles -- plus any always-attended ``force_front`` tiles -- first makes the
    frozen max a tight estimate. See :func:`compute_m_proxy_topn` for the
    analogous pooled-score top-N block selection used by the VFA precomputed-max
    (``m_init``) path.

    Args:
        block_attn_mask: ``(B, num_q_blocks, num_k_blocks)`` (shared across heads)
            or ``(B, H, num_q_blocks, num_k_blocks)`` bool mask. True = attend.
        pooled_score: fp32 ``(B, H, num_q_blocks, num_k_blocks)`` block-level
            attention score (e.g. from :func:`get_block_map_meansim`). Only blocks
            that are attended in ``block_attn_mask`` are ever selected.
        sample_n: number of top-scored tiles to emit first per segment. ``<= 0``
            emits only the ``force_front`` tiles ahead of the rest.
        num_heads: number of Q heads; required when ``block_attn_mask`` is 3D.
        force_front_mask: optional bool mask, same shape/broadcast as
            ``block_attn_mask``, of blocks to place immediately after the sampled
            tiles. Excluded from the top-``sample_n`` selection; only the
            attended ones are emitted.

    Returns:
        ``kv_block_indices`` (1D int32), ``lut_start`` (1D int32) and
        ``lut_count`` (1D int32), indexed by
        ``idx = b * (H * num_q_blocks) + h * num_q_blocks + q_block``.
        ``kv_block_indices`` is over-allocated to the static size ``B*H*Q*K``;
        only the compact prefix selected by ``lut_start``/``lut_count`` is valid
        (the remainder is unused padding).
    """
    if block_attn_mask.dim() == 3:
        if num_heads is None:
            raise ValueError("num_heads must be provided when block_attn_mask is 3D")
        B, Q, K = block_attn_mask.shape
        block_attn_mask = block_attn_mask.unsqueeze(1).expand(B, num_heads, Q, K)
        if force_front_mask is not None and force_front_mask.dim() == 3:
            force_front_mask = force_front_mask.unsqueeze(1).expand(B, num_heads, Q, K)

    B, H, Q, K = block_attn_mask.shape
    assert pooled_score.shape[:3] == (B, H, Q) and pooled_score.shape[-1] == K, (
        f"pooled_score shape {tuple(pooled_score.shape)} does not match mask "
        f"{(B, H, Q, K)}"
    )
    device = block_attn_mask.device

    attended = block_attn_mask.to(torch.bool)
    lut_count = attended.sum(-1).to(torch.int32).reshape(-1)
    lut_start = torch.cumsum(lut_count, 0) - lut_count

    if force_front_mask is None:
        force_front = torch.zeros_like(attended)
    else:
        # Only attended blocks can be emitted at all.
        force_front = force_front_mask.to(torch.bool).expand(B, H, Q, K) & attended

    neg_inf = pooled_score.new_full((), float("-inf"))
    masked_score = torch.where(attended, pooled_score.to(torch.float32), neg_inf)

    # Mark the top-``sample_n`` attended, non-force-front blocks per (B, H, Q) row.
    is_topn = torch.zeros((B, H, Q, K), dtype=torch.bool, device=device)
    n = min(sample_n, K)
    if n > 0:
        sample_score = torch.where(force_front, neg_inf, masked_score)
        topk = sample_score.topk(n, dim=-1)
        # A row with fewer than n candidates pads topk with -inf entries; mark
        # only the finite (genuinely attended, non-force-front) selections.
        is_topn.scatter_(-1, topk.indices, topk.values > neg_inf)

    # Per-row ordering of the K blocks by (priority, tiebreak):
    #   0 = attended & top-n      -> descending score (highest-max first)
    #   1 = attended & force-front -> ascending block index
    #   2 = attended, the rest    -> ascending block index
    #   3 = not attended          -> sorts past the per-row count, so dropped
    col = torch.arange(K, device=device).view(1, 1, 1, K)
    priority = torch.where(
        ~attended,
        3,
        torch.where(is_topn, 0, torch.where(force_front, 1, 2)),
    )
    tiebreak = torch.where(is_topn, -masked_score, col.to(torch.float32))

    # Lexicographic (priority, tiebreak) sort per row via two stable sorts.
    o1 = torch.argsort(tiebreak, dim=-1, stable=True)
    order = torch.gather(
        o1, -1, torch.argsort(torch.gather(priority, -1, o1), dim=-1, stable=True)
    )

    # The first ``count`` entries of each row are exactly the attended blocks in
    # the desired order; pack them row-major into the ragged index list. Done as
    # a scatter into an over-allocated, statically-sized buffer (rather than
    # boolean-mask indexing, which triggers a ``nonzero`` device sync and a
    # data-dependent output shape) so the function stays CUDA-graph-safe.
    R = B * H * Q
    rows = order.reshape(R, K)
    col = torch.arange(K, device=device)
    keep = col[None, :] < lut_count[:, None]
    # Destination of each entry in the packed buffer. Dropped (unattended)
    # entries are routed to a scratch sink slot so they can never clobber a
    # valid neighbouring row's position.
    sink = R * K
    dest = torch.where(
        keep,
        lut_start[:, None].to(torch.long) + col[None, :],
        col.new_full((), sink),
    )
    packed = torch.empty(sink + 1, dtype=torch.int32, device=device)
    packed.scatter_(0, dest.reshape(-1), rows.reshape(-1).to(torch.int32))
    kv_block_indices = packed[:sink]
    return kv_block_indices, lut_start, lut_count


def build_attention_lut(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    simthreshd1: float,
    cdfthreshd: float,
    use_vfa: bool = True,
    n_sample: int = 8,
    is_causal: bool = False,
    static_block_mask: Optional[torch.Tensor] = None,
    text_len: int = 0,
    block_m: int = 128,
    block_n: int = 128,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    """Build a SpargeAttn block-sparse ragged LUT (optionally VFA-front-loaded).

    One entry point that turns Q/K into a ready-to-use ragged LUT plus the
    matching ``freeze_softmax_max_count``. Either way it keeps only the
    SpargeAttn-selected (meansim) blocks; ``use_vfa`` selects the in-segment
    block ordering:

      * ``use_vfa=False`` (plain SpargeAttn): emit the selected blocks in
        ascending order. ``freeze_softmax_max_count = -1`` (the online softmax
        max is never frozen); ``n_sample`` is ignored.
      * ``use_vfa=True`` (SpargeAttn + VFA): front-load the top-``n_sample``
        selected blocks by pooled score, so the VFA running-max freeze sees the
        most important tiles first.

    The standalone dense-VFA mode (keep *all* K blocks + freeze) is not
    supported here; the VFA freeze always rides on a sparge-selected LUT.

    Text handling (``text_len > 0``): every q-block additionally attends to the
    trailing dense text K blocks. With ``use_vfa=True`` those text blocks are
    appended to the front immediately *after* the ``n_sample`` sampled tiles, and
    the returned ``freeze_softmax_max_count`` is ``n_sample + n_text_blocks`` so
    the max is frozen only once both the sampled and text tiles are visited.

    Causality is handled at block granularity (whole K blocks are kept/dropped);
    the block-sparse kernel does not apply an intra-block diagonal mask.

    Returns:
        ``(block_lut, freeze_softmax_max_count)`` where ``block_lut`` is the
        ragged ``(kv_block_indices, lut_start, lut_count)`` tuple to pass as
        ``block_lut`` to
        :func:`aiter.ops.triton.attention.fav3_sage.fav3_sage_wrapper_func` (or
        ``kv_block_indices``/``lut_start``/``lut_count`` to
        :func:`aiter.ops.triton.attention.fav3_sage.fav3_sage_func` with
        ``use_block_sparse=True`` and ``freeze_softmax_max_count=...``).
    """
    image_q = q[:, :, : q.shape[2] - text_len, :] if text_len > 0 else q
    image_k = k[:, :, : k.shape[2] - text_len, :] if text_len > 0 else k
    image_len_q = q.shape[2] - text_len
    image_len_k = k.shape[2] - text_len
    n_text_k = _num_text_blocks(text_len, block_m, block_n)[1] if text_len > 0 else 0

    image_mask, image_score = get_block_map_meansim(
        image_q,
        image_k,
        is_causal=is_causal,
        BLKQ=block_m,
        BLKK=block_n,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
    )

    if static_block_mask is not None:
        image_mask = image_mask | static_block_mask[None, None, ...]

    full_mask = _assemble_full_block_mask(
        image_mask, image_len_q, image_len_k, text_len, block_m, block_n
    )

    if not use_vfa:
        return block_attn_mask_to_ragged_lut(full_mask), -1

    # use_vfa: front-load the top-n sampled image tiles, then the dense text tiles.
    # Scores live only over the image region; text/text-row positions stay -inf
    # so they are never picked as sampled tiles (text is forced front).
    B, H, n_iq, n_ik = image_mask.shape
    n_tq, n_tk = full_mask.shape[-2], full_mask.shape[-1]
    full_score = full_mask.new_full(
        (B, H, n_tq, n_tk), float("-inf"), dtype=torch.float32
    )
    full_score[:, :, :n_iq, :n_ik] = image_score.to(torch.float32)

    force_front = None
    if n_text_k > 0:
        force_front = torch.zeros((B, H, n_tq, n_tk), dtype=torch.bool, device=q.device)
        force_front[:, :, :, -n_text_k:] = True

    block_lut = block_attn_mask_to_ragged_lut_topn_front(
        full_mask, full_score, n_sample, force_front_mask=force_front
    )
    return block_lut, n_sample + n_text_k


# ============================================================================
# VFA (Vector Relieved Flash Attention) m_init estimator
#
# Builds a per-row running-max estimate ``m_init`` that, fed into
# :func:`aiter.ops.triton.attention.fav3_sage.fav3_sage_func` via the ``m_init``
# argument, lets the kernel drop the online-softmax rowmax reduction and the
# per-block acc rescale (the four softmax vector ops), running
# ``p = exp2(qk - m_init)`` with ``m_init`` frozen. See
# ``_sage_vfa_m_blockidx_kernel`` for why a lower-bound estimate is safe.
# ============================================================================
def compute_m_proxy_topn(
    q: torch.Tensor,
    k: torch.Tensor,
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    BLKQ: int,
    BLKK: int,
    layout: str = "bshd",
    n_blocks: int = 8,
) -> torch.Tensor:
    """Guided per-row max over the top-``n_blocks`` pooled-score K blocks.

    Ranks candidate K blocks with a SpargeAttn-style mean-pooled block-attention
    estimate (see :func:`aiter.ops.triton.pool.compute_pooled_scores`) and
    evaluates the per-(q-block)
    top-``n_blocks`` of them exactly.  Only the proposal stage is approximate:
    the selected blocks are evaluated with REAL K rows, so the estimate is a
    lower bound on the true rowmax with far smaller gap than uniform sampling at
    the same ``n_blocks``.  No safety margin.

    The proposal pools the original (pre-quant) float ``q``/``k`` into per-block
    means and forms ``pooled_q @ pooled_k^T``; ``q_int8``/``k_int8`` are used
    only for the exact selected-block evaluation.
    """
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    batch, seqlen_q, nheads_q, head_dim = map_dims(q_int8.shape, bshd_map)
    _, seqlen_k, nheads_k, _ = map_dims(k_int8.shape, bshd_map)
    num_q_blocks = (seqlen_q + BLKQ - 1) // BLKQ
    num_k_blocks = (seqlen_k + BLKK - 1) // BLKK

    n = max(1, min(n_blocks, num_k_blocks))

    if n >= num_k_blocks:
        # Selecting every K block: the pooled-score ranking + top-k is pure
        # overhead (and the result is the exact per-row max).  Use a single
        # shared 1D [num_k_blocks] table broadcast to every program.
        block_idx = torch.arange(num_k_blocks, device=q_int8.device, dtype=torch.int32)
        stride_biz, stride_bih, stride_biqblk = 0, 0, 0
        stride_bis = block_idx.stride(0)
    else:
        score = compute_pooled_scores(q, k, BLKQ=BLKQ, BLKK=BLKK, layout=layout)
        block_idx = score.topk(n, dim=-1).indices.to(torch.int32).contiguous()
        stride_biz, stride_bih, stride_biqblk, stride_bis = block_idx.stride()

    m_init = torch.empty(
        (batch, nheads_q, num_q_blocks, BLKQ),
        dtype=torch.float32,
        device=q_int8.device,
    )

    stride_qz, stride_qm, stride_qh, stride_qd = map_dims(q_int8.stride(), bshd_map)
    stride_kz, stride_kn, stride_kh, stride_kd = map_dims(k_int8.stride(), bshd_map)
    stride_qsz, stride_qsh, stride_qsblk = q_descale.stride()
    stride_ksz, stride_ksh, stride_ksblk = k_descale.stride()
    stride_mz, stride_mh, stride_mblk, stride_mr = m_init.stride()

    padded_d_model_qk = max(16, 1 << (head_dim - 1).bit_length())

    grid = (num_q_blocks, nheads_q, batch)
    _sage_vfa_m_blockidx_kernel[grid](
        q_int8,
        k_int8,
        q_descale,
        k_descale,
        block_idx,
        m_init,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_qsz,
        stride_qsh,
        stride_qsblk,
        stride_ksz,
        stride_ksh,
        stride_ksblk,
        stride_biz,
        stride_bih,
        stride_biqblk,
        stride_bis,
        stride_mz,
        stride_mh,
        stride_mblk,
        stride_mr,
        SEQLEN_Q=seqlen_q,
        SEQLEN_K=seqlen_k,
        HQ=nheads_q,
        HK=nheads_k,
        BLOCK_M=BLKQ,
        BLOCK_N=BLKK,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        ACTUAL_BLOCK_DMODEL_QK=head_dim,
        N_SAMPLES=block_idx.shape[-1],
        num_warps=4,
        num_stages=2,
    )
    return m_init
