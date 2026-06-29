# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-torch sparse-attention-with-sink references for the DeepSeek-V4 block.

Ported from ATOM's ``atom/model_ops/sparse_attn_v4.py`` (the ``_sparse_attn_torch``
/ ``_sparse_attn_ragged_torch`` golden paths) so aiter's DSV4 paged decode/prefill
kernels can be validated without an ATOM dependency.
"""

import torch


def _sparse_attn_torch(
    q: torch.Tensor,  # [B, M, H, D]
    kv: torch.Tensor,  # [B, N, D] single (MQA) head
    attn_sink: torch.Tensor,  # [H]
    topk_idxs: torch.Tensor,  # [B, M, K] int32, -1 = skip
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse MQA attention with per-query top-k gather and per-head sink.

    The sink contributes only to the softmax denominator (a zero-value virtual
    KV column). Invalid (-1) topk entries set their logit to -inf. Accumulation
    in fp32; output cast back to q.dtype.
    """
    B, M, H, D = q.shape
    _, N, D_kv = kv.shape
    K = topk_idxs.shape[-1]
    assert D_kv == D and attn_sink.shape == (H,) and topk_idxs.shape == (B, M, K)
    out_dtype = q.dtype
    device = q.device

    valid = topk_idxs != -1
    safe_idxs = topk_idxs.clamp(min=0).long()
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, M, K)
    kv_gathered = kv[batch_idx, safe_idxs]  # [B, M, K, D]

    kv_f32 = kv_gathered.float()
    kv_f32 = torch.where(
        valid.unsqueeze(-1), kv_f32, torch.zeros((), dtype=kv_f32.dtype, device=device)
    )

    q_f32 = q.float()
    scores = torch.einsum("bmhd,bmkd->bmhk", q_f32, kv_f32) * float(softmax_scale)
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    sink = attn_sink.float().view(1, 1, H, 1).expand(B, M, H, 1)
    combined = torch.cat([scores, sink], dim=-1)  # [B, M, H, K+1]
    cmax = combined.amax(dim=-1, keepdim=True)
    cmax = torch.where(
        cmax == float("-inf"), torch.zeros((), dtype=cmax.dtype, device=device), cmax
    )
    weights = (combined - cmax).exp()
    denom = weights.sum(dim=-1, keepdim=True)
    weights = weights / denom.clamp(min=1e-30)
    weights_kv = weights[..., :K]
    out = torch.einsum("bmhk,bmkd->bmhd", weights_kv, kv_f32)
    return out.to(out_dtype)


def _sparse_attn_ragged_torch(
    q: torch.Tensor,  # [T, H, D]
    kv: torch.Tensor,  # [total_kv, D]
    attn_sink: torch.Tensor,  # [H]
    topk_idxs: torch.Tensor,  # [T, K] global indices into kv, -1 = skip
    softmax_scale: float,
) -> torch.Tensor:
    """Flat ragged variant: one batch folded into the token axis."""
    return _sparse_attn_torch(
        q.unsqueeze(0),
        kv.unsqueeze(0),
        attn_sink,
        topk_idxs.unsqueeze(0),
        softmax_scale,
    ).squeeze(0)
