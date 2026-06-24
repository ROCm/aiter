# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# from __future__ import annotations

import pytest
import torch
import triton

from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse
from aiter.ops.triton.utils._triton import arch_info


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


# ---------------------------------------------------------------------------
# skip_reduce: the wrapper hands back the pre-reduce split-K partials and the
# caller is responsible for the log-sum-exp combine + sink fold. This mirrors
# the _pa_decode_sparse_reduce kernel in pure torch so we can validate the
# partials against the dense reference.
# ---------------------------------------------------------------------------


def _wrapper_main_kernel_params(D: int):
    """Reproduce the (use_exp2, block_k) the wrapper picks for the main kernel.

    The split kernel emits partials in the base-2 domain on non-gfx1250
    (triton path, USE_EXP2=True) and the natural-exp domain on gfx1250 (gluon
    path, USE_EXP2=False). block_k matches the wrapper's per-arch choice.
    """
    use_gluon = arch_info.get_arch() == "gfx1250"
    use_exp2 = not use_gluon
    block_k = 32 if use_gluon else (16 if D >= 256 else 32)
    return use_exp2, block_k


def _reduce_partials_torch(
    acc_partial, m_partial, l_partial, attn_sink, kv_indptr, block_k, use_exp2
):
    """Pure-torch port of _pa_decode_sparse_reduce.

    Shapes:
        acc_partial: [T, KV_SPLITS, H_padded, D] fp32
        m_partial:   [T, KV_SPLITS, H_padded]    fp32
        l_partial:   [T, KV_SPLITS, H_padded]    fp32
    Returns [T, H, D] in attn_sink-implied output dtype (bf16/fp16 caller casts).
    """
    T, kv_splits, _, D = acc_partial.shape
    H = attn_sink.shape[0]
    device = acc_partial.device

    expfn = torch.exp2 if use_exp2 else torch.exp
    LOG2E = 1.4426950408889634
    sink_scale = LOG2E if use_exp2 else 1.0

    indptr = kv_indptr.to(torch.int64)
    kv_lens = (indptr[1 : T + 1] - indptr[:T]).clamp(min=0)
    seg_ids = torch.arange(kv_splits, device=device)
    sink = attn_sink.float() * sink_scale  # [H]

    out = torch.empty(T, H, D, dtype=torch.float32, device=device)
    for t in range(T):
        n = int(kv_lens[t].item())
        # Match the kernel's tiles_per_segment / act_num_segments masking so we
        # ignore the stale (uninitialised) partial-buffer slots that the split
        # kernel early-returned on.
        if n <= 0:
            act_num_segments = 0
        else:
            tiles_per_segment = triton.cdiv(n, kv_splits * block_k)
            act_num_segments = triton.cdiv(n, tiles_per_segment * block_k)
        seg_mask = seg_ids < act_num_segments  # [KV_SPLITS]

        m_p = m_partial[t, :, :H].clone()  # [KV_SPLITS, H]
        l_p = l_partial[t, :, :H]
        a_p = acc_partial[t, :, :H, :]  # [KV_SPLITS, H, D]
        m_p = torch.where(seg_mask[:, None], m_p, torch.full_like(m_p, float("-inf")))

        m_max = m_p.max(dim=0).values  # [H]
        is_dead = m_p == float("-inf")  # [KV_SPLITS, H]
        alpha = torch.where(is_dead, torch.zeros_like(m_p), expfn(m_p - m_max[None, :]))
        l_comb = torch.where(is_dead, torch.zeros_like(l_p), l_p * alpha).sum(0)  # [H]
        acc_comb = torch.where(
            is_dead[:, :, None], torch.zeros_like(a_p), a_p * alpha[:, :, None]
        ).sum(
            0
        )  # [H, D]

        m_final = torch.maximum(m_max, sink)
        alpha_kv = expfn(m_max - m_final)
        alpha_sink = expfn(sink - m_final)
        l_final = l_comb * alpha_kv + alpha_sink
        acc_final = acc_comb * alpha_kv[:, None]
        denom = l_final.clamp(min=1e-30)
        out[t] = torch.where(
            l_final[:, None] > 0.0,
            acc_final / denom[:, None],
            torch.zeros_like(acc_final),
        )
    return out


@pytest.mark.parametrize("T", [1, 32, 512])
@pytest.mark.parametrize("H", [1, 8, 16])
@pytest.mark.parametrize("D", [512])
@pytest.mark.parametrize("kv_len", [100, 400, 1024, 4096])
@pytest.mark.parametrize("var_len", [True, False])
@pytest.mark.parametrize("sentinels", [True, False])
@pytest.mark.parametrize("skip_reduce", [True, False])
def test_pa_decode_sparse_vs_reference(
    T, H, D, kv_len, var_len, sentinels, skip_reduce
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    pages = T * kv_len
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
    result = pa_decode_sparse(
        q,
        ukv,
        indices,
        indptr,
        sink,
        scale,
        has_invalid=sentinels,
        skip_reduce=skip_reduce,
    )

    if isinstance(result, tuple):
        # skip_reduce with the split-K path active (kv_splits > 1): the wrapper
        # returns raw partials, so do the log-sum-exp combine + sink fold here.
        acc_partial, m_partial, l_partial = result
        use_exp2, block_k = _wrapper_main_kernel_params(D)
        out = _reduce_partials_torch(
            acc_partial, m_partial, l_partial, sink, indptr, block_k, use_exp2
        ).to(q.dtype)
    else:
        # kv_splits == 1 (skip_reduce is a no-op) or skip_reduce=False: the
        # wrapper already returns the final output.
        out = result

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


# ---------------------------------------------------------------------------
# FP8 KV cache quantization helpers
# ---------------------------------------------------------------------------

_FP8_GROUP_SIZE = 64
_FP8_DTYPE = torch.float8_e4m3fnuz


def _quantize_kv_fp8(unified_kv, group_size=_FP8_GROUP_SIZE):
    """Quantize bf16/fp16 unified_kv to (fp8, scales) with 1xGROUP_SIZE block scaling.

    Returns (kv_fp8, kv_scales) where kv_fp8 is float8_e4m3fnuz and
    kv_scales is [total_pages, D // group_size] fp32.
    """
    total_pages, D = unified_kv.shape
    assert D % group_size == 0
    num_groups = D // group_size
    kv_f32 = unified_kv.float().view(total_pages, num_groups, group_size)
    amax = kv_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    fp8_max = torch.finfo(_FP8_DTYPE).max
    scales = (amax / fp8_max).squeeze(-1)  # [total_pages, num_groups]
    kv_scaled = kv_f32 / amax * fp8_max
    kv_fp8 = kv_scaled.view(total_pages, D).to(_FP8_DTYPE)
    return kv_fp8, scales.to(torch.float32)


def _dequant_kv_fp8(kv_fp8, kv_scales, group_size=_FP8_GROUP_SIZE):
    """Dequantize for reference comparison."""
    total_pages, D = kv_fp8.shape
    num_groups = D // group_size
    kv_f32 = kv_fp8.float().view(total_pages, num_groups, group_size)
    scales_expanded = kv_scales.unsqueeze(-1).expand(
        total_pages, num_groups, group_size
    )
    return (kv_f32 * scales_expanded).view(total_pages, D)


@pytest.mark.parametrize("T", [1, 32, 128])
@pytest.mark.parametrize("H", [1, 8, 16])
@pytest.mark.parametrize("D", [512])
@pytest.mark.parametrize("kv_len", [100, 400, 1024])
@pytest.mark.parametrize("var_len", [True, False])
def test_pa_decode_sparse_fp8_vs_reference(T, H, D, kv_len, var_len):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    pages = T * kv_len
    q, ukv_bf16, indices, indptr, sink, scale = _make_inputs(
        T,
        H,
        D,
        kv_len,
        pages,
        variable_len=var_len,
    )

    # Quantize KV to fp8 + scales
    kv_fp8, kv_scales = _quantize_kv_fp8(ukv_bf16)

    # Reference: dequant back to bf16, run the torch reference
    ukv_deq = _dequant_kv_fp8(kv_fp8, kv_scales).to(q.dtype)
    ref = pa_decode_sparse_reference(q, ukv_deq, indices, indptr, sink, scale)

    # Triton kernel with fp8 kv + kv_scales
    out = pa_decode_sparse(
        q,
        kv_fp8,
        indices,
        indptr,
        sink,
        scale,
        kv_scales=kv_scales,
        has_invalid=False,
    )

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
