# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for the FlyDSL pa_decode_sparse kernel on gfx942.

This mirrors op_tests/triton_tests/attention/test_pa_decode_sparse.py with the
following differences:
- Tests flydsl_pa_decode_sparse (FlyDSL main kernel + Triton reduce).
- has_invalid / sentinels are not tested (not yet implemented in FlyDSL kernel).
- D must be divisible by BLOCK_THREADS=64.

Tolerance: atol=5e-3, rtol=5e-3 (same as the Triton reference test).
"""

import pytest
import torch
import triton

from aiter.ops.flydsl.kernels.pa_decode_sparse import flydsl_pa_decode_sparse


# ---------------------------------------------------------------------------
# Reference implementation (pure torch)
# ---------------------------------------------------------------------------


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


def pa_decode_sparse_reference(q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale):
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
# skip_reduce: reproduce log-sum-exp combine + sink fold in pure torch.
# The FlyDSL kernel uses base-2 domain (USE_EXP2=True) same as the Triton main
# kernel; Triton reduce is called with USE_EXP2=True.
# ---------------------------------------------------------------------------


def _reduce_partials_torch(acc_partial, m_partial, l_partial, attn_sink, kv_indptr, block_k):
    """Pure-torch port of _pa_decode_sparse_reduce for USE_EXP2=True (base-2 domain).

    Shapes:
        acc_partial: [T, KV_SPLITS, H, D] fp32
        m_partial:   [T, KV_SPLITS, H]    fp32
        l_partial:   [T, KV_SPLITS, H]    fp32
    Returns [T, H, D] fp32.
    """
    T, kv_splits, H, D = acc_partial.shape
    device = acc_partial.device
    LOG2E = 1.4426950408889634

    indptr = kv_indptr.to(torch.int64)
    kv_lens = (indptr[1 : T + 1] - indptr[:T]).clamp(min=0)
    seg_ids = torch.arange(kv_splits, device=device)
    sink = attn_sink.float() * LOG2E  # in base-2 domain

    out = torch.empty(T, H, D, dtype=torch.float32, device=device)
    for t in range(T):
        n = int(kv_lens[t].item())
        if n <= 0:
            act_num_segments = 0
        else:
            tiles_per_segment = triton.cdiv(n, kv_splits * block_k)
            act_num_segments = triton.cdiv(n, tiles_per_segment * block_k)
        seg_mask = seg_ids < act_num_segments  # [KV_SPLITS]

        m_p = m_partial[t].clone()    # [KV_SPLITS, H]
        l_p = l_partial[t]            # [KV_SPLITS, H]
        a_p = acc_partial[t]          # [KV_SPLITS, H, D]
        m_p = torch.where(seg_mask[:, None], m_p, torch.full_like(m_p, float("-inf")))

        m_max = m_p.max(dim=0).values  # [H]
        is_dead = m_p == float("-inf")
        alpha = torch.where(is_dead, torch.zeros_like(m_p), torch.exp2(m_p - m_max[None, :]))
        l_comb = torch.where(is_dead, torch.zeros_like(l_p), l_p * alpha).sum(0)  # [H]
        acc_comb = torch.where(
            is_dead[:, :, None], torch.zeros_like(a_p), a_p * alpha[:, :, None]
        ).sum(0)  # [H, D]

        m_final = torch.maximum(m_max, sink)
        alpha_kv = torch.exp2(m_max - m_final)
        alpha_sink = torch.exp2(sink - m_final)
        l_final = l_comb * alpha_kv + alpha_sink
        acc_final = acc_comb * alpha_kv[:, None]
        denom = l_final.clamp(min=1e-30)
        out[t] = torch.where(
            l_final[:, None] > 0.0,
            acc_final / denom[:, None],
            torch.zeros_like(acc_final),
        )
    return out


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
    variable_len: bool = False,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    q = torch.randn(T, H, D, dtype=dtype, device=device) * 0.5
    unified_kv = torch.randn(total_pages, D, dtype=dtype, device=device) * 0.5
    attn_sink = torch.randn(H, dtype=torch.float32, device=device) * 0.1

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

    indptr = indptr.to(torch.int32)
    softmax_scale = float(D) ** -0.5
    return q, unified_kv, indices, indptr, attn_sink, softmax_scale


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# D=512 and D=576 are both divisible by BLOCK_THREADS=64.
# D=576 matches the DSv4 MLA head dimension (kv_lora_rank=512 + rope_rank=64).
@pytest.mark.parametrize("T", [1, 32])
@pytest.mark.parametrize("H", [1, 8, 16, 128])
@pytest.mark.parametrize("D", [512, 576])
@pytest.mark.parametrize("kv_len", [100, 400, 1024])
@pytest.mark.parametrize("var_len", [True, False])
@pytest.mark.parametrize("skip_reduce", [True, False])
def test_flydsl_pa_decode_sparse_vs_reference(T, H, D, kv_len, var_len, skip_reduce):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if D % 64 != 0:
        pytest.skip(f"D={D} not divisible by BLOCK_THREADS=64")

    pages = T * kv_len
    q, ukv, indices, indptr, sink, scale = _make_inputs(
        T, H, D, kv_len, pages, variable_len=var_len
    )

    ref = pa_decode_sparse_reference(q, ukv, indices, indptr, sink, scale)
    result = flydsl_pa_decode_sparse(
        q,
        ukv,
        indices,
        indptr,
        sink,
        scale,
        skip_reduce=skip_reduce,
    )

    if isinstance(result, tuple):
        # skip_reduce with kv_splits > 1: partials returned; combine in torch.
        acc_partial, m_partial, l_partial = result
        block_k = 16 if D >= 256 else 32
        out = _reduce_partials_torch(
            acc_partial, m_partial, l_partial, sink, indptr, block_k
        ).to(q.dtype)
    else:
        out = result

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)
