# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# High-level wrapper for the Vector Relieved Flash Attention (VFA) variant of
# the SAGE FP8 attention kernel.  See ``fav3_sage_attention_vfa.py`` for the
# Triton kernels and a description of the algorithm.

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton

import aiter
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import map_dims
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_vfa import (
    _sage_vfa_m_blockidx_kernel,
    sage_fwd_vfa,
)
from aiter.ops.triton.attention.fav3_sage import get_sage_fwd_configs
from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant


def compute_m_sampled(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    BLKQ: int,
    BLKK: int,
    layout: str = "bshd",
    n_blocks: int = 8,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-row max over ``n_blocks`` randomly sampled K blocks.

    Dots Q against the real int8 K rows of ``n_blocks`` distinct, randomly
    chosen K blocks and takes the per-row max.  Because it uses coherent K
    rows the estimate is a lower bound on the true rowmax that tightens as
    ``n_blocks`` grows.  Cost scales as ``n_blocks / num_k_blocks`` of a full
    QK pass.

    Returns fp32 ``[batch, num_q_heads, num_q_blocks, BLOCK_M]``.  No safety
    margin.
    """
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    batch, seqlen_q, nheads_q, head_dim = map_dims(q_int8.shape, bshd_map)
    _, seqlen_k, nheads_k, _ = map_dims(k_int8.shape, bshd_map)
    num_q_blocks = (seqlen_q + BLKQ - 1) // BLKQ
    num_k_blocks = (seqlen_k + BLKK - 1) // BLKK

    n = max(1, min(n_blocks, num_k_blocks))
    if n >= num_k_blocks:
        block_idx = torch.arange(num_k_blocks, device=q_int8.device, dtype=torch.int32)
    else:
        # Sample distinct blocks without replacement; sort for K-load locality.
        perm = torch.randperm(num_k_blocks, generator=generator, device=q_int8.device)
        block_idx = perm[:n].sort().values.to(torch.int32)

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

    # The shared 1D [N_SAMPLES] table is broadcast to every (batch, head,
    # q-block) program by zeroing the leading block-index strides.
    grid = (num_q_blocks, nheads_q, batch)
    _sage_vfa_m_blockidx_kernel[grid](
        q_int8, k_int8,
        q_descale, k_descale,
        block_idx,
        m_init,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_qsz, stride_qsh, stride_qsblk,
        stride_ksz, stride_ksh, stride_ksblk,
        0, 0, 0, block_idx.stride(0),
        stride_mz, stride_mh, stride_mblk, stride_mr,
        SEQLEN_Q=seqlen_q,
        SEQLEN_K=seqlen_k,
        HQ=nheads_q,
        HK=nheads_k,
        BLOCK_M=BLKQ,
        BLOCK_N=BLKK,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        ACTUAL_BLOCK_DMODEL_QK=head_dim,
        N_SAMPLES=n,
        num_warps=4,
        num_stages=2,
    )
    return m_init


def _pool_blocks_mean(
    x: torch.Tensor,
    BLK: int,
    layout: str,
) -> torch.Tensor:
    """Mean-pooled block representatives.

    Returns fp32 ``[batch, nheads, num_blocks, head_dim]`` -- the per-block
    token mean of ``x``.
    """
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    batch, seqlen, nheads, head_dim = map_dims(x.shape, bshd_map)
    num_blocks = (seqlen + BLK - 1) // BLK

    # Logical [batch, nheads, seqlen, head_dim] view regardless of layout.
    # Reduce each block directly with a fp32 accumulator (``dtype=torch.float32``)
    # so we never materialize a full-precision copy of the large sequence tensor,
    # and split off any ragged tail instead of padding the whole tensor -- both
    # avoid extra full-size allocations/copies that dominated this proxy.
    xv = x if layout == "bhsd" else x.permute(0, 2, 1, 3)
    n_full = seqlen // BLK
    full = n_full * BLK

    sums = (
        xv[:, :, :full, :]
        .reshape(batch, nheads, n_full, BLK, head_dim)
        .sum(dim=3, dtype=torch.float32)
    )

    rem = seqlen - full
    if rem == 0:
        return sums / BLK

    tail = xv[:, :, full:, :].sum(dim=2, dtype=torch.float32).unsqueeze(2)
    sums = torch.cat([sums, tail], dim=2)
    counts = torch.full((num_blocks,), float(BLK), device=x.device)
    counts[-1] = rem
    return sums / counts[None, None, :, None]


def _compute_pooled_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    BLKQ: int,
    BLKK: int,
    layout: str,
) -> torch.Tensor:
    """SpargeAttn-style block-attention estimate used to rank candidate K blocks.

    Mean-pools Q and K into per-block representatives and forms the block-level
    score ``pooled_q @ pooled_k^T * head_dim**-0.5``.  Operates on the original
    (pre-quant) float Q/K, which keeps the estimate a genuine block-attention
    proxy.

    Returns fp32 ``[batch, nheads_q, num_q_blocks, num_k_blocks]``.
    """
    pooled_q = _pool_blocks_mean(q, BLKQ, layout)
    pooled_k = _pool_blocks_mean(k, BLKK, layout)

    nheads_q, nheads_k = pooled_q.shape[1], pooled_k.shape[1]
    if nheads_q != nheads_k:
        pooled_k = pooled_k.repeat_interleave(nheads_q // nheads_k, dim=1)

    head_dim = pooled_q.shape[-1]
    return torch.matmul(pooled_q, pooled_k.transpose(-1, -2)) * (head_dim ** -0.5)


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
    estimate (see :func:`_compute_pooled_scores`) and evaluates the per-(q-block)
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
        score = _compute_pooled_scores(q, k, BLKQ=BLKQ, BLKK=BLKK, layout=layout)
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
        q_int8, k_int8,
        q_descale, k_descale,
        block_idx,
        m_init,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_qsz, stride_qsh, stride_qsblk,
        stride_ksz, stride_ksh, stride_ksblk,
        stride_biz, stride_bih, stride_biqblk, stride_bis,
        stride_mz, stride_mh, stride_mblk, stride_mr,
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


def fav3_sage_vfa_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    m_init: torch.Tensor,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    layout: str = "bshd",
    config: Optional[dict] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """SageAttention v1 with Vector Relieved Flash Attention -- hot kernel only.

    Dense, non-causal, no-sliding-window, no-block-sparse path only.  Inputs
    follow the same quantization protocol as :func:`fav3_sage_func`; ``m_init``
    must be a precomputed per-row running-max estimate (see
    :func:`compute_m_proxy_topn` for the guided pooled-score estimator and
    :func:`compute_m_sampled` for the uniform-sampling estimator).
    """
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]

    batch, seqlen_q, nheads_q, head_size_qk = map_dims(q.shape, bshd_map)
    _, seqlen_k, nheads_k, _ = map_dims(k.shape, bshd_map)
    _, _, _, head_size_v = map_dims(v.shape, bshd_map)

    assert q.dtype == torch.int8 and k.dtype == torch.int8, "Q and K must be int8"
    assert (
        nheads_q % nheads_k == 0
    ), f"GQA/MQA error: {nheads_q} not divisible by {nheads_k}"

    if config is None:
        config = get_sage_fwd_configs()

    BLKQ, BLKK = config["BLOCK_M"], config["BLOCK_N"]
    num_q_blocks = (seqlen_q + BLKQ - 1) // BLKQ
    num_k_blocks = (seqlen_k + BLKK - 1) // BLKK
    n_extra_tokens = seqlen_k % BLKK

    if softmax_scale is None:
        softmax_scale = head_size_qk ** -0.5

    assert q_descale.shape == (batch, nheads_q, num_q_blocks)
    assert k_descale.shape == (batch, nheads_k, num_k_blocks)
    assert m_init.shape == (batch, nheads_q, num_q_blocks, BLKQ), (
        f"m_init shape {tuple(m_init.shape)} does not match expected "
        f"{(batch, nheads_q, num_q_blocks, BLKQ)}"
    )
    assert m_init.dtype == torch.float32

    out_dtype = torch.bfloat16
    out_shape = (q.shape[0], q.shape[1], q.shape[2], v.shape[-1])
    out = torch.zeros(out_shape, dtype=out_dtype, device=q.device)

    softmax_lse = (
        torch.zeros((batch, nheads_q, seqlen_q), device=q.device, dtype=torch.float32)
        if return_lse
        else None
    )

    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd_map)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd_map)
    stride_vb, stride_vn, stride_vh, stride_vd = map_dims(v.stride(), bshd_map)
    stride_ob, stride_om, stride_oh, stride_od = map_dims(out.stride(), bshd_map)

    stride_lse_z, stride_lse_h, stride_lse_m = (
        softmax_lse.stride() if return_lse else (0, 0, 0)
    )
    stride_qsz, stride_qsh, stride_qsblk = q_descale.stride()
    stride_ksz, stride_ksh, stride_ksblk = k_descale.stride()
    stride_vsz, stride_vsh, _ = v_descale.stride()

    padded_d_model_qk = max(16, 1 << (head_size_qk - 1).bit_length())
    padded_d_model_v = max(16, 1 << (head_size_v - 1).bit_length())

    stride_mz, stride_mh, stride_mblk, stride_mr = m_init.stride()

    def grid(META):
        return (triton.cdiv(seqlen_q, META["BLOCK_M"]), nheads_q, batch)

    sage_fwd_vfa[grid](
        q, k, v,
        q_descale, k_descale, v_descale,
        m_init,
        softmax_lse, out,
        stride_qsz, stride_qsh, stride_qsblk,
        stride_ksz, stride_ksh, stride_ksblk,
        stride_vsz, stride_vsh,
        stride_mz, stride_mh, stride_mblk, stride_mr,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_lse_z, stride_lse_h, stride_lse_m,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=seqlen_q,
        MAX_SEQLENS_K=seqlen_k,
        NUM_K_BLOCKS=num_k_blocks,
        N_EXTRA_TOKENS=n_extra_tokens,
        BLOCK_M=BLKQ,
        BLOCK_N=BLKK,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        PRE_LOAD_V=config.get("PRE_LOAD_V", False),
        USE_EXP2=True,
        RETURN_LSE=return_lse,
        waves_per_eu=config.get("waves_per_eu", 2),
        # num_stages=4 wins on gfx950: the extra K/V prefetch stage hides
        # roughly one full HBM round trip per loop iteration without pushing
        # the per-wave VGPR count past the 256-VGPR / 2-waves-per-EU limit.
        num_stages=config.get("num_stages", 4),
        num_warps=config.get("num_warps", 8),
    )

    if return_lse:
        return out, softmax_lse
    return out, None


def fav3_sage_vfa_wrapper_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    layout: str = "bshd",
    config: Optional[dict] = None,
    n_sample_blocks: int = 16,
    guided_sample: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """High-precision API that handles quantization and ``m_init``.

    ``m_init`` is always estimated from ``n_sample_blocks`` K blocks evaluated
    with real K rows (a lower bound on the true per-row max; no safety margin):
      * ``guided_sample=True`` (default) -> the top-``n_sample_blocks`` blocks
        per q-block ranked by a SpargeAttn mean-pooled block-score.
      * ``guided_sample=False``          -> ``n_sample_blocks`` uniformly random
        K blocks.
    """
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert k.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert v.dtype in [torch.float16, torch.bfloat16, torch.float32]

    if config is None:
        config = get_sage_fwd_configs()

    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    _, _, _, head_dim = map_dims(q.shape, bshd_map)
    softmax_scale = softmax_scale or (head_dim ** -0.5)

    BLKQ, BLKK = config["BLOCK_M"], config["BLOCK_N"]
    fp8_dtype = aiter.dtypes.fp8
    fp8_max = torch.finfo(fp8_dtype).max

    q_int8, q_descale, k_int8, k_descale, v_fp8, v_descale = sage_quant(
        q, k, v,
        fp8_dtype, fp8_max,
        sm_scale=softmax_scale,
        BLKQ=BLKQ,
        BLKK=BLKK,
        layout=layout,
    )

    if guided_sample:
        m_init = compute_m_proxy_topn(
            q, k, q_int8, k_int8, q_descale, k_descale,
            BLKQ=BLKQ, BLKK=BLKK, layout=layout,
            n_blocks=n_sample_blocks,
        )
    else:
        m_init = compute_m_sampled(
            q_int8, k_int8, q_descale, k_descale,
            BLKQ=BLKQ, BLKK=BLKK, layout=layout,
            n_blocks=n_sample_blocks, generator=generator,
        )

    out, lse = fav3_sage_vfa_func(
        q_int8, k_int8, v_fp8,
        q_descale, k_descale, v_descale,
        m_init,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        layout=layout,
        config=config,
    )
    if return_lse:
        return out, lse
    return out
