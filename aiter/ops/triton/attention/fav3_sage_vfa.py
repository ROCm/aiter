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
    _sage_k_sabsmax_kernel,
    _sage_vfa_m_init_kernel,
    sage_fwd_vfa,
)
from aiter.ops.triton.attention.fav3_sage import get_sage_fwd_configs
from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant


# Additive safety bias in log2 units applied to the per-row m_init estimate.
# Chosen so that the typical sabsmax under-shoot on real workloads
# (|m_init - true_max| <= ~0.5 log2 units on the captured payloads we tested)
# is fully absorbed.  +2 log2 units = 4x scaling: the maximum p value drops
# from 1.0 to 0.25, comfortably within fp8 E4M3 dynamic range while still
# leaving headroom for outlier rows whose sabsmax bound is off by ~1.5 log2
# units.  The hot kernel additionally clamps `qk - m_init` to <= 0 before
# exp2 so anything beyond this margin can only bias a few weights up to
# `p == 1`, never inf/NaN.
_M_INIT_SAFETY_LOG2 = 1.0


def compute_k_repr_sabsmax(
    k_int8: torch.Tensor,
    BLKK: int,
    layout: str = "bshd",
) -> torch.Tensor:
    """Compute the signed absmax representative per K block, per feature dim.

    Returns ``[batch, num_kv_heads, num_k_blocks, head_dim]`` int8.  For each
    K block and dim, holds the K element with the largest absolute value
    along the sequence axis, preserving its sign so the m-init dot can
    exploit Q/K sign alignment.
    """
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    batch, seqlen_k, num_kv_heads, head_dim = map_dims(k_int8.shape, bshd_map)
    num_k_blocks = (seqlen_k + BLKK - 1) // BLKK

    k_repr = torch.empty(
        (batch, num_kv_heads, num_k_blocks, head_dim),
        dtype=torch.int8,
        device=k_int8.device,
    )

    stride_kz, stride_kn, stride_kh, stride_kd = map_dims(k_int8.stride(), bshd_map)
    stride_rz, stride_rh, stride_rblk, stride_rd = k_repr.stride()

    grid = (batch, num_kv_heads, num_k_blocks)
    _sage_k_sabsmax_kernel[grid](
        k_int8,
        k_repr,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_rz, stride_rh, stride_rblk, stride_rd,
        SEQLEN_K=seqlen_k,
        BLOCK_N=BLKK,
        D=head_dim,
    )
    return k_repr


def compute_m_init(
    q_int8: torch.Tensor,
    q_descale: torch.Tensor,
    k_repr: torch.Tensor,
    k_descale: torch.Tensor,
    BLKQ: int,
    layout: str = "bshd",
    block_k_repr: int = 64,
    safety_log2: float = _M_INIT_SAFETY_LOG2,
) -> torch.Tensor:
    """Per-row running-max estimate, computed in a dedicated kernel.

    Returns fp32 ``[batch, num_q_heads, num_q_blocks, BLOCK_M]``.  Includes
    an additive ``safety_log2`` margin so that the hot kernel can rely on it
    as a near-upper-bound without rescale.
    """
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    batch, seqlen_q, nheads_q, head_dim = map_dims(q_int8.shape, bshd_map)
    _, nheads_k, num_k_blocks, _ = k_repr.shape
    num_q_blocks = (seqlen_q + BLKQ - 1) // BLKQ

    m_init = torch.empty(
        (batch, nheads_q, num_q_blocks, BLKQ),
        dtype=torch.float32,
        device=q_int8.device,
    )

    stride_qz, stride_qm, stride_qh, stride_qd = map_dims(q_int8.stride(), bshd_map)
    stride_qsz, stride_qsh, stride_qsblk = q_descale.stride()
    stride_krz, stride_krh, stride_krblk, stride_krd = k_repr.stride()
    stride_ksz, stride_ksh, stride_ksblk = k_descale.stride()
    stride_mz, stride_mh, stride_mblk, stride_mr = m_init.stride()

    # Cap BLOCK_K_REPR at next_pow2(num_k_blocks) so very small-K cases
    # (e.g. cross-attention with K < 64*BLKK tokens) do not pay for an
    # oversized tile that is then almost entirely masked out.  16 is the
    # smallest tile we support (MFMA tile size constraints).
    if block_k_repr < 16:
        block_k_repr = 16
    block_k_repr = 1 << (block_k_repr - 1).bit_length()
    if num_k_blocks > 0:
        block_k_repr = min(block_k_repr, max(16, 1 << (num_k_blocks - 1).bit_length()))

    padded_d_model_qk = max(16, 1 << (head_dim - 1).bit_length())

    grid = (num_q_blocks, nheads_q, batch)
    _sage_vfa_m_init_kernel[grid](
        q_int8,
        q_descale,
        k_repr,
        k_descale,
        m_init,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_qsz, stride_qsh, stride_qsblk,
        stride_krz, stride_krh, stride_krblk, stride_krd,
        stride_ksz, stride_ksh, stride_ksblk,
        stride_mz, stride_mh, stride_mblk, stride_mr,
        SEQLEN_Q=seqlen_q,
        NUM_K_BLOCKS=num_k_blocks,
        SAFETY=safety_log2,
        HQ=nheads_q,
        HK=nheads_k,
        BLOCK_M=BLKQ,
        BLOCK_K_REPR=block_k_repr,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        ACTUAL_BLOCK_DMODEL_QK=head_dim,
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
    k_repr: torch.Tensor,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    layout: str = "bshd",
    config: Optional[dict] = None,
    block_k_repr: int = 64,
    safety_log2: float = _M_INIT_SAFETY_LOG2,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """SageAttention v1 with Vector Relieved Flash Attention.

    Dense, non-causal, no-sliding-window, no-block-sparse path only.  Inputs
    follow the same quantization protocol as :func:`fav3_sage_func`; ``k_repr``
    is expected to come from :func:`compute_k_repr_sabsmax`.
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
    assert k_repr.shape == (batch, nheads_k, num_k_blocks, head_size_qk)

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

    # Phase 0: precompute m_init via the standalone helper kernel.
    m_init = compute_m_init(
        q,
        q_descale,
        k_repr,
        k_descale,
        BLKQ=BLKQ,
        layout=layout,
        block_k_repr=block_k_repr,
        safety_log2=safety_log2,
    )
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
) -> torch.Tensor:
    """High-precision API that quantizes Q/K/V internally and runs VFA."""
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

    k_repr = compute_k_repr_sabsmax(k_int8, BLKK=BLKK, layout=layout)

    out, lse = fav3_sage_vfa_func(
        q_int8, k_int8, v_fp8,
        q_descale, k_descale, v_descale,
        k_repr,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        layout=layout,
        config=config,
    )
    if return_lse:
        return out, lse
    return out
