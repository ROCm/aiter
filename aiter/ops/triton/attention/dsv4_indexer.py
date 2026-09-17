# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# DSV4 Indexer op: forward (scoring + topk) + backward.
# Uses hipBLASLt via torch.einsum for the heavy GEMM.

import torch
import torch.nn.functional as F


def _causal_mask(S, P, compress_ratio, device):
    s_idx = torch.arange(S, device=device)[:, None]
    p_idx = torch.arange(P, device=device)[None, :]
    return (p_idx + 1) * compress_ratio - 1 <= s_idx


def indexer_fwd(q, k, w, compress_ratio):
    """Compute indexer scores.

    Args:
        q: [S, H, Hd] bf16
        k: [P, Hd] bf16
        w: [S, H] fp32
        compress_ratio: int

    Returns:
        scores: [S, P] fp32
    """
    S, _H, _Hd = q.shape
    P = k.shape[0]
    with torch.no_grad():
        dot = torch.einsum("shd,pd->shp", q.float(), k.float())
        scores = (F.relu(dot) * w.float().unsqueeze(-1)).sum(dim=1)
        scores = torch.where(
            _causal_mask(S, P, compress_ratio, q.device), scores, float("-inf")
        )
    return scores


def indexer_bwd(q, k, w, d_scores, compress_ratio):
    """Backward pass for indexer scoring.

    Args:
        q: [S, H, Hd] bf16
        k: [P, Hd] bf16
        w: [S, H] fp32
        d_scores: [S, P] fp32
        compress_ratio: int

    Returns:
        dq: [S, H, Hd] fp32
        dk: [P, Hd] fp32
        dw: [S, H] fp32
    """
    S, _H, _Hd = q.shape
    P = k.shape[0]

    q_f = q.float().requires_grad_(True)
    k_f = k.float().requires_grad_(True)
    w_f = w.float().requires_grad_(True)

    dot = torch.einsum("shd,pd->shp", q_f, k_f)
    scores = (F.relu(dot) * w_f.unsqueeze(-1)).sum(dim=1)
    allowed = _causal_mask(S, P, compress_ratio, q.device)
    scores = torch.where(allowed, scores, torch.tensor(0.0, device=q.device))

    (scores * d_scores).sum().backward()
    return q_f.grad, k_f.grad, w_f.grad


def dsv4_indexer(q, k, w, compress_ratio, topk):
    """Full DSV4 indexer: scoring + topk.

    Args:
        q: [S, H, Hd] bf16
        k: [P, Hd] bf16
        w: [S, H] fp32
        compress_ratio: int
        topk: int

    Returns:
        scores: [S, eff_topk] fp32
        indices: [S, eff_topk] int32
    """
    S, _H, _Hd = q.shape
    P = k.shape[0]

    dot = torch.einsum("shd,pd->shp", q.float(), k.float())
    logits = (F.relu(dot) * w.float().unsqueeze(-1)).sum(dim=1)
    logits = torch.where(
        _causal_mask(S, P, compress_ratio, q.device), logits, float("-inf")
    )

    eff_topk = min(topk, P)
    scores, indices = torch.topk(logits, eff_topk, dim=-1)
    indices = indices.to(torch.int32)
    indices = indices.masked_fill(scores == float("-inf"), -1)
    return scores, indices
