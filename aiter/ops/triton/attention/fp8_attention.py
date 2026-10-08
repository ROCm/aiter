# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public Python wrapper for the FP8 Flash Attention v2 Triton kernels.

Exposes attn_fwd and related utilities at the public aiter.ops.triton.attention
boundary so callers do not need to import internal _triton_kernels symbols.
`flash_attn_block_scale` is the block-scale entry used by `flash_attn_func`
when the caller passes q_block_descale, k_block_descale, and v_scale.
"""

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.fp8_attention_kernel import (
    FIXED_BLOCK_M,
    FIXED_BLOCK_N,
    _bwd_kernel_dkdv,
    _bwd_kernel_dq,
    _bwd_preprocess_use_o,
    attn_fwd,
    compute_fp8_scaling_factors,
    get_padded_headsize,
)
from aiter.ops.triton.utils import types

__all__ = [
    "_bwd_kernel_dkdv",
    "_bwd_kernel_dq",
    "_bwd_preprocess_use_o",
    "attn_fwd",
    "compute_fp8_scaling_factors",
    "flash_attn_block_scale",
    "get_padded_headsize",
]


def _pad_head(tensor, d_pad):
    if tensor.shape[-1] == d_pad:
        return tensor
    padded = torch.zeros(
        *tensor.shape[:-1],
        d_pad,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[..., : tensor.shape[-1]] = tensor
    return padded


def _bshd_strides(tensor):
    """Return (batch, head, seq, dim) strides for a (batch, seq, head, dim) tensor."""
    return tensor.stride(0), tensor.stride(2), tensor.stride(1), tensor.stride(3)


def _compact_lse(lse, seqlen, block_m):
    """Pack the kernel's doubled LSE buffer down to (batch, heads, seqlen)."""
    n_blocks = triton.cdiv(seqlen, block_m)
    blocks = lse[:, :, : n_blocks * 2 * block_m].view(
        lse.shape[0], lse.shape[1], n_blocks, 2, block_m
    )
    compact = blocks[:, :, :, 0, :].reshape(
        lse.shape[0], lse.shape[1], n_blocks * block_m
    )
    return compact[:, :, :seqlen].contiguous()


def _check_block_scale_args(
    q,
    k,
    v,
    q_block_descale,
    k_block_descale,
    v_scale,
    dropout_p,
    window_size,
    bias,
    alibi_slopes,
    return_attn_probs,
    sink,
    p_scale,
):
    if q_block_descale is None or k_block_descale is None or v_scale is None:
        raise ValueError(
            "Block-scale FP8 attention requires q_block_descale, "
            "k_block_descale, and v_scale."
        )
    if dropout_p:
        raise ValueError("Block-scale FP8 attention does not support dropout.")
    if bias is not None:
        raise ValueError("Block-scale FP8 attention does not support bias.")
    if alibi_slopes is not None:
        raise ValueError("Block-scale FP8 attention does not support alibi_slopes.")
    if sink is not None:
        raise ValueError("Block-scale FP8 attention does not support sink.")
    if return_attn_probs:
        raise ValueError(
            "Block-scale FP8 attention does not support return_attn_probs."
        )
    if int(window_size[0]) != -1 or int(window_size[1]) != -1:
        raise ValueError("Block-scale FP8 attention does not support sliding window.")
    if not types._is_fp8(q) or q.dtype != types.e4m3_dtype:
        raise ValueError(
            f"Block-scale FP8 attention requires {types.e4m3_dtype} q/k/v "
            f"(got q={q.dtype})."
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(
            f"q/k/v dtypes must match (got q={q.dtype}, k={k.dtype}, v={v.dtype})."
        )
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "Block-scale FP8 attention expects q/k/v shaped "
            "(batch, seqlen, nheads, headdim)."
        )
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q/k/v batch sizes must match.")
    if k.shape[1] != v.shape[1] or k.shape[2] != v.shape[2]:
        raise ValueError("k and v sequence length and head count must match.")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q/k/v head dims must match.")
    batch, seqlen_q, nheads_q, _ = q.shape
    seqlen_k, nheads_k = k.shape[1], k.shape[2]
    if nheads_q % nheads_k != 0:
        raise ValueError(
            f"num_q_heads ({nheads_q}) must be divisible by num_k_heads ({nheads_k})."
        )
    n_q_blocks = triton.cdiv(seqlen_q, FIXED_BLOCK_M)
    n_k_blocks = triton.cdiv(seqlen_k, FIXED_BLOCK_N)
    if tuple(q_block_descale.shape) != (batch, nheads_q, n_q_blocks):
        raise ValueError(
            "q_block_descale must be shaped "
            f"({batch}, {nheads_q}, {n_q_blocks}), got {tuple(q_block_descale.shape)}."
        )
    if tuple(k_block_descale.shape) != (batch, nheads_k, n_k_blocks):
        raise ValueError(
            "k_block_descale must be shaped "
            f"({batch}, {nheads_k}, {n_k_blocks}), got {tuple(k_block_descale.shape)}."
        )
    if v_scale.numel() != 1:
        raise ValueError(
            f"v_scale must be a single fp32 tensor scale, got shape {tuple(v_scale.shape)}."
        )
    if p_scale <= 0:
        raise ValueError(f"p_scale must be positive, got {p_scale}.")
    return n_q_blocks, n_k_blocks


def _launch_fwd(
    q, k, v, q_block_descale, k_block_descale, v_scale, sm_scale, causal, p_scale
):
    batch, seqlen_q, nheads_q, head_dim = q.shape
    seqlen_k, nheads_k = k.shape[1], k.shape[2]
    d_pad = get_padded_headsize(head_dim)
    q_p, k_p, v_p = _pad_head(q, d_pad), _pad_head(k, d_pad), _pad_head(v, d_pad)
    out = torch.zeros(
        batch,
        seqlen_q,
        nheads_q,
        d_pad,
        dtype=torch.bfloat16,
        device=q.device,
    )
    n_q_blocks = triton.cdiv(seqlen_q, FIXED_BLOCK_M)
    lse = torch.zeros(
        batch,
        nheads_q,
        2 * n_q_blocks * FIXED_BLOCK_M,
        dtype=torch.float32,
        device=q.device,
    )
    q_scale = q_block_descale.contiguous()
    k_scale = k_block_descale.contiguous()
    v_scale = v_scale.to(dtype=torch.float32, device=q.device).reshape(1).contiguous()
    stride_qz, stride_qh, stride_qm, stride_qk = _bshd_strides(q_p)
    stride_kz, stride_kh, stride_kn, stride_kk = _bshd_strides(k_p)
    stride_vz, stride_vh, stride_vk, stride_vn = _bshd_strides(v_p)
    stride_oz, stride_oh, stride_om, stride_on = _bshd_strides(out)
    grid = (n_q_blocks, nheads_q, batch)
    attn_fwd[grid](
        Q=q_p,
        K=k_p,
        V=v_p,
        bias=None,
        p_scale=p_scale,
        q_descale_ptr=q_scale,
        k_descale_ptr=k_scale,
        v_scale_ptr=v_scale,
        USE_FP8=True,
        SM_SCALE=sm_scale,
        LSE=lse,
        Out=out,
        stride_qz=stride_qz,
        stride_qh=stride_qh,
        stride_qm=stride_qm,
        stride_qk=stride_qk,
        stride_kz=stride_kz,
        stride_kh=stride_kh,
        stride_kn=stride_kn,
        stride_kk=stride_kk,
        stride_vz=stride_vz,
        stride_vh=stride_vh,
        stride_vk=stride_vk,
        stride_vn=stride_vn,
        stride_oz=stride_oz,
        stride_oh=stride_oh,
        stride_om=stride_om,
        stride_on=stride_on,
        stride_bz=0,
        stride_bh=0,
        stride_bm=0,
        stride_bn=0,
        stride_az=0,
        stride_ah=0,
        stride_sz=0,
        stride_sh=0,
        stride_sm=0,
        stride_sn=0,
        stride_lse_z=lse.stride(0),
        stride_lse_h=lse.stride(1),
        stride_lse_m=lse.stride(2),
        stride_qdescale_z=q_scale.stride(0),
        stride_qdescale_h=q_scale.stride(1),
        stride_qdescale_m=q_scale.stride(2),
        stride_kdescale_z=k_scale.stride(0),
        stride_kdescale_h=k_scale.shape[-1],
        stride_kdescale_m=k_scale.stride(2),
        padded_kscale_block_num=k_scale.shape[-1],
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        dropout_p=0.0,
        philox_seed=0,
        philox_offset_base=0,
        scores=None,
        scores_scaled_shifted=None,
        exp_scores=None,
        alibi_slopes=None,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=d_pad,
        ACTUAL_BLOCK_DMODEL_V=d_pad,
        MAX_SEQLENS_Q=seqlen_q,
        MAX_SEQLENS_K=seqlen_k,
        VARLEN=False,
        IS_CAUSAL=causal,
        BLOCK_M=FIXED_BLOCK_M,
        BLOCK_DMODEL_QK=d_pad,
        BLOCK_DMODEL_V=d_pad,
        BLOCK_N=FIXED_BLOCK_N,
        USE_BIAS=False,
        ENABLE_DROPOUT=False,
        RETURN_SCORES=False,
        USE_ALIBI=False,
        USE_EXP2=True,
    )
    if head_dim == d_pad:
        out_user = out
    else:
        out_user = out[..., :head_dim].contiguous()
    return out_user, _compact_lse(lse, seqlen_q, FIXED_BLOCK_M)


def flash_attn_block_scale(
    q,
    k,
    v,
    q_block_descale,
    k_block_descale,
    v_scale,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    return_lse=False,
    return_attn_probs=False,
    sink=None,
    p_scale=1.0,
):
    """Block-scale FP8 attention for q/k/v in (batch, seqlen, nheads, headdim) layout.

    q_block_descale and k_block_descale are fp32 descales shaped
    (batch, heads, n_blocks) with one value per 64 tokens. v_scale is one fp32
    value for the whole V tensor. The output is bf16. This entry is forward-only:
    the kernel's dq/dk/dv are bf16 and cannot be attached to fp8 inputs.
    """
    if any(
        tensor is not None and getattr(tensor, "requires_grad", False)
        for tensor in (q, k, v, q_block_descale, k_block_descale, v_scale)
    ):
        raise RuntimeError(
            "Block-scale FP8 attention on flash_attn_func is forward-only. "
            "dq/dk/dv are bf16 and cannot be attached to fp8 q/k/v."
        )
    _check_block_scale_args(
        q,
        k,
        v,
        q_block_descale,
        k_block_descale,
        v_scale,
        dropout_p,
        window_size,
        bias,
        alibi_slopes,
        return_attn_probs,
        sink,
        p_scale,
    )
    p_scale = float(p_scale)
    sm_scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    out, softmax_lse = _launch_fwd(
        q,
        k,
        v,
        q_block_descale,
        k_block_descale,
        v_scale,
        sm_scale,
        bool(causal),
        p_scale,
    )
    if return_lse:
        return out, softmax_lse
    return out
