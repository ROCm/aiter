# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# from __future__ import annotations

import pytest
import torch

from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse


def _sparse_attn_torch(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Per-batch sparse multi-head attention with sink in the denominator only.

    Shapes:
        q:           [B, M, H, D]
        kv:          [B, N, D]
        attn_sink:   [H]
        topk_idxs:   [B, M, K] int32, -1 means skip
    Returns:
        [B, M, H, D] same dtype as q.
    """
    B, M, H, D = q.shape
    K = topk_idxs.shape[-1]
    device = q.device
    out_dtype = q.dtype

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
    combined = torch.cat([scores, sink], dim=-1)
    cmax = combined.amax(dim=-1, keepdim=True)
    cmax = torch.where(
        cmax == float("-inf"),
        torch.zeros((), dtype=cmax.dtype, device=device),
        cmax,
    )
    weights = (combined - cmax).exp()
    denom = weights.sum(dim=-1, keepdim=True)
    weights = weights / denom.clamp(min=1e-30)
    weights_kv = weights[..., :K]
    out = torch.einsum("bmhk,bmkd->bmhd", weights_kv, kv_f32)
    return out.to(out_dtype)


def pa_decode_sparse_reference(
    q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale
):
    """Pure-torch reference that materialises per-token KV via gather."""
    T = q.size(0)
    indptr = kv_indptr.to(torch.int64)
    spans = (indptr[1:] - indptr[:T]).clamp(min=0)
    k_dim = int(spans.max().item()) if T > 0 else 1
    if k_dim == 0:
        k_dim = 1
    topk_idxs = torch.full((T, k_dim), -1, device=q.device, dtype=torch.int32)
    for t in range(T):
        s = int(indptr[t].item())
        n = int(spans[t].item())
        if n > 0:
            topk_idxs[t, :n] = kv_indices[s : s + n].to(torch.int32)
    return _sparse_attn_torch(
        q.unsqueeze(0),
        unified_kv.unsqueeze(0),
        attn_sink,
        topk_idxs.unsqueeze(0),
        softmax_scale,
    ).squeeze(0)


# ---------------------------------------------------------------------------
# Input builder
# ---------------------------------------------------------------------------


def _make_inputs(
    T: int,
    H: int,
    D: int,
    kv_len_per_token: int,
    total_pages: int,
    dtype=torch.bfloat16,
    seed: int = 0,
    include_sentinels: bool = False,
    variable_len: bool = False,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    q = torch.randn(T, H, D, dtype=dtype, device=device) * 0.5
    unified_kv = torch.randn(total_pages, D, dtype=dtype, device=device) * 0.5
    attn_sink = torch.randn(H, dtype=torch.float32, device=device) * 0.1

    # Per-token kv_len: fixed or random in [1, kv_len_per_token].
    if variable_len:
        kv_lens = torch.randint(
            low=1,
            high=kv_len_per_token + 1,
            size=(T,),
            device=device,
            dtype=torch.int64,
        )
    else:
        kv_lens = torch.full((T,), kv_len_per_token, device=device, dtype=torch.int64)

    indptr = torch.zeros(T + 1, device=device, dtype=torch.int64)
    indptr[1:] = kv_lens.cumsum(0)
    total_indices = int(indptr[-1].item())

    indices = torch.randint(
        low=0,
        high=total_pages,
        size=(total_indices,),
        device=device,
        dtype=torch.int32,
    )
    if include_sentinels and total_indices > 0:
        # Sprinkle a few -1 sentinels.
        n_sentinel = max(1, total_indices // 16)
        sentinel_pos = torch.randperm(total_indices, device=device)[:n_sentinel]
        indices[sentinel_pos] = -1

    indptr = indptr.to(torch.int32)
    softmax_scale = float(D) ** -0.5
    return q, unified_kv, indices, indptr, attn_sink, softmax_scale


@pytest.mark.parametrize("T", [1, 32, 64])
@pytest.mark.parametrize("H", [1, 8, 16, 64, 128])
@pytest.mark.parametrize("D", [512])
@pytest.mark.parametrize("kv_len", [100, 400, 1024, 4096, 8192, 16384])
@pytest.mark.parametrize("var_len", [True, False])
@pytest.mark.parametrize("sentinels", [True, False])
def test_pa_decode_sparse_vs_reference(T, H, D, kv_len, var_len, sentinels):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    pages = 2 * T * kv_len
    q, ukv, indices, indptr, sink, scale = _make_inputs(
        T,
        H,
        D,
        kv_len,
        pages,
        include_sentinels=sentinels,
        variable_len=var_len,
    )

    ref = pa_decode_sparse_reference(q, ukv, indices, indptr, sink, scale)
    out = pa_decode_sparse(q, ukv, indices, indptr, sink, scale)

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)
