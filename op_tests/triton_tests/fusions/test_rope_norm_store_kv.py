# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the BF16 ``rope_norm_store_kv`` Triton kernel.

The reference is the pure-PyTorch implementation supplied alongside the API spec;
it is reproduced here so the test file is self-contained.
"""

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.fusions.rope_norm_store_kv import rope_norm_store_kv


# ---------- Reference ----------

def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight


def _apply_rope_neox_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat([y1, y2], dim=-1)


def rope_norm_store_kv_reference(
    key_cache, value_cache, qkv, cos_sin,
    num_seqlen_per_req, q_index, kvcache_indices, is_prefill,
    q_norm_weight=None, k_norm_weight=None,
    out_q=None, out_k=None, out_v=None,
    qk_norm_policy=0,
):
    num_rows = qkv.shape[0]
    num_kv_heads = key_cache.shape[2]
    qk_head_dim = key_cache.shape[3]
    v_head_dim = value_cache.shape[3]
    block_size = key_cache.shape[1]
    hidden = qkv.shape[1]
    num_q_heads = (
        hidden - num_kv_heads * qk_head_dim - num_kv_heads * v_head_dim
    ) // qk_head_dim
    num_req = num_seqlen_per_req.shape[0]

    q_dim = num_q_heads * qk_head_dim
    k_dim = num_kv_heads * qk_head_dim

    qkv_f32 = qkv.float()
    q = qkv_f32[:, :q_dim].reshape(num_rows, num_q_heads, qk_head_dim)
    k = qkv_f32[:, q_dim:q_dim + k_dim].reshape(num_rows, num_kv_heads, qk_head_dim)
    v = qkv_f32[:, q_dim + k_dim:].reshape(num_rows, num_kv_heads, v_head_dim)

    positions = torch.zeros(num_rows, dtype=torch.long, device=qkv.device)
    for req_id in range(num_req):
        start = q_index[req_id].item()
        end = q_index[req_id + 1].item()
        seq_len = num_seqlen_per_req[req_id].item()
        for i in range(start, end):
            positions[i] = i + seq_len - end

    cos = cos_sin[positions, :qk_head_dim // 2].unsqueeze(1)
    sin = cos_sin[positions, qk_head_dim // 2:].unsqueeze(1)

    if qk_norm_policy == 2:
        q = _rms_norm_ref(q, q_norm_weight)
    q = _apply_rope_neox_ref(q, cos, sin)
    if qk_norm_policy == 1:
        q = _rms_norm_ref(q, q_norm_weight)

    if qk_norm_policy == 2:
        k = _rms_norm_ref(k, k_norm_weight)
    k = _apply_rope_neox_ref(k, cos, sin)
    if qk_norm_policy == 1:
        k = _rms_norm_ref(k, k_norm_weight)

    q_bf16 = q.bfloat16()
    k_bf16 = k.bfloat16()
    v_bf16 = v.bfloat16()

    if out_q is not None:
        out_q.copy_(q_bf16)
    else:
        out_q = q_bf16.contiguous()

    if out_k is not None:
        out_k.copy_(k_bf16)
    else:
        for req_id in range(num_req):
            start = q_index[req_id].item()
            end = q_index[req_id + 1].item()
            seq_len = num_seqlen_per_req[req_id].item()
            for i in range(start, end):
                tp = i + seq_len - end
                phys = kvcache_indices[req_id, tp // block_size].item()
                key_cache[phys, tp % block_size, :, :] = k_bf16[i]

    if out_v is not None:
        out_v.copy_(v_bf16)
    else:
        for req_id in range(num_req):
            start = q_index[req_id].item()
            end = q_index[req_id + 1].item()
            seq_len = num_seqlen_per_req[req_id].item()
            for i in range(start, end):
                tp = i + seq_len - end
                phys = kvcache_indices[req_id, tp // block_size].item()
                value_cache[phys, tp % block_size, :, :] = v_bf16[i]

    if is_prefill:
        for req_id in range(num_req):
            seq_len = num_seqlen_per_req[req_id].item()
            if seq_len <= 0:
                continue
            last_pos = seq_len - 1
            lbi = last_pos // block_size
            lbr = last_pos % block_size
            phys = kvcache_indices[req_id, lbi].item()
            if lbr + 1 < block_size:
                if out_k is None:
                    key_cache[phys, lbr + 1:, :, :] = 0
                if out_v is None:
                    value_cache[phys, lbr + 1:, :, :] = 0

    return out_q


# ---------- Input generator ----------

def _make_inputs(
    seqlens,
    num_q_heads,
    num_kv_heads,
    qk_head_dim,
    v_head_dim,
    block_size,
    is_prefill,
    seed=0,
    device="cuda",
    extra_pad_blocks=2,
):
    """Build a self-consistent batch.

    ``seqlens`` is a list of *total* sequence lengths per request (including the
    new tokens). For prefill we treat every token in the request as new; for decode
    each request supplies exactly one new token.
    """
    torch.manual_seed(seed)
    num_req = len(seqlens)
    new_tokens_per_req = list(seqlens) if is_prefill else [1] * num_req
    q_index = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(new_tokens_per_req), dim=0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    num_rows = int(q_index[-1].item())
    num_seqlen_per_req = torch.tensor(seqlens, dtype=torch.int32, device=device)

    max_seqlen = max(seqlens)
    max_blocks_per_req = (max_seqlen + block_size - 1) // block_size
    total_blocks_needed = max_blocks_per_req * num_req + extra_pad_blocks
    # Use a fresh shuffled physical block per logical slot to catch any
    # indexing assumptions.
    physical_blocks = torch.randperm(total_blocks_needed, device=device).to(torch.int32)
    kvcache_indices = physical_blocks[: num_req * max_blocks_per_req].reshape(
        num_req, max_blocks_per_req
    )

    hidden = num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim + num_kv_heads * v_head_dim
    qkv = torch.randn(num_rows, hidden, dtype=torch.bfloat16, device=device)

    max_table_len = max_seqlen + 8
    cos_sin = torch.randn(max_table_len, qk_head_dim, dtype=torch.float32, device=device)

    q_norm_weight = torch.randn(qk_head_dim, dtype=torch.float32, device=device)
    k_norm_weight = torch.randn(qk_head_dim, dtype=torch.float32, device=device)

    key_cache = torch.full(
        (total_blocks_needed, block_size, num_kv_heads, qk_head_dim),
        7.0,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.full(
        (total_blocks_needed, block_size, num_kv_heads, v_head_dim),
        -3.0,
        dtype=torch.bfloat16,
        device=device,
    )

    return dict(
        qkv=qkv,
        cos_sin=cos_sin,
        num_seqlen_per_req=num_seqlen_per_req,
        q_index=q_index,
        kvcache_indices=kvcache_indices,
        key_cache=key_cache,
        value_cache=value_cache,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        block_size=block_size,
    )


def _run_pair(inputs, is_prefill, qk_norm_policy, use_out_kv=False):
    ref_kc = inputs["key_cache"].clone()
    ref_vc = inputs["value_cache"].clone()
    out_kc = inputs["key_cache"].clone()
    out_vc = inputs["value_cache"].clone()

    num_rows = inputs["qkv"].shape[0]
    num_q_heads = inputs["num_q_heads"]
    num_kv_heads = inputs["num_kv_heads"]
    qk_head_dim = inputs["qk_head_dim"]
    v_head_dim = inputs["v_head_dim"]

    out_k_ref = out_k_kern = out_v_ref = out_v_kern = None
    if use_out_kv:
        out_k_ref = torch.empty(
            num_rows, num_kv_heads, qk_head_dim,
            dtype=torch.bfloat16, device=inputs["qkv"].device,
        )
        out_k_kern = torch.empty_like(out_k_ref)
        out_v_ref = torch.empty(
            num_rows, num_kv_heads, v_head_dim,
            dtype=torch.bfloat16, device=inputs["qkv"].device,
        )
        out_v_kern = torch.empty_like(out_v_ref)

    qnw = inputs["q_norm_weight"] if qk_norm_policy != 0 else None
    knw = inputs["k_norm_weight"] if qk_norm_policy != 0 else None

    ref_q = rope_norm_store_kv_reference(
        ref_kc, ref_vc, inputs["qkv"], inputs["cos_sin"],
        inputs["num_seqlen_per_req"], inputs["q_index"],
        inputs["kvcache_indices"], is_prefill,
        q_norm_weight=qnw, k_norm_weight=knw,
        out_k=out_k_ref, out_v=out_v_ref,
        qk_norm_policy=qk_norm_policy,
    )
    kern_q = rope_norm_store_kv(
        out_kc, out_vc, inputs["qkv"], inputs["cos_sin"],
        inputs["num_seqlen_per_req"], inputs["q_index"],
        inputs["kvcache_indices"], is_prefill,
        q_norm_weight=qnw, k_norm_weight=knw,
        out_k=out_k_kern, out_v=out_v_kern,
        qk_norm_policy=qk_norm_policy,
    )

    return (
        ref_q, kern_q, ref_kc, out_kc, ref_vc, out_vc,
        out_k_ref, out_k_kern, out_v_ref, out_v_kern,
    )


def _assert_close(a, b, name):
    torch.testing.assert_close(a, b, atol=2e-2, rtol=2e-2,
                                msg=lambda m: f"{name} mismatch: {m}")


# ---------- Tests ----------

PREFILL_CONFIGS = [
    ([4],            1, 1, 64,  64,  16),
    ([1, 3, 5],      4, 2, 64,  64,  16),
    ([17, 5, 1],     8, 2, 128, 128, 16),
    ([32],           8, 8, 128, 128, 16),
    ([7, 11],        4, 1, 64,  32,  8),   # different qk_head_dim and v_head_dim
    ([200],          16, 4, 128, 128, 32),
]


@pytest.mark.parametrize(
    "seqlens, qh, kvh, qk_d, v_d, bs",
    PREFILL_CONFIGS,
    ids=[f"L{','.join(map(str,s))}_qh{qh}_kvh{kvh}_qkd{qk}_vd{vd}_bs{bs}"
         for (s, qh, kvh, qk, vd, bs) in PREFILL_CONFIGS],
)
@pytest.mark.parametrize("policy", [0, 1, 2])
def test_rope_norm_store_kv_prefill_paged_cache(seqlens, qh, kvh, qk_d, v_d, bs, policy):
    inp = _make_inputs(seqlens, qh, kvh, qk_d, v_d, bs, is_prefill=True)
    (rq, kq, rkc, okc, rvc, ovc, *_) = _run_pair(inp, True, policy, use_out_kv=False)
    _assert_close(rq, kq, "out_q")
    _assert_close(rkc, okc, "key_cache")
    _assert_close(rvc, ovc, "value_cache")


@pytest.mark.parametrize(
    "seqlens, qh, kvh, qk_d, v_d, bs",
    PREFILL_CONFIGS,
    ids=[f"L{','.join(map(str,s))}_qh{qh}_kvh{kvh}_qkd{qk}_vd{vd}_bs{bs}"
         for (s, qh, kvh, qk, vd, bs) in PREFILL_CONFIGS],
)
@pytest.mark.parametrize("policy", [0, 1])
def test_rope_norm_store_kv_prefill_out_kv(seqlens, qh, kvh, qk_d, v_d, bs, policy):
    inp = _make_inputs(seqlens, qh, kvh, qk_d, v_d, bs, is_prefill=True)
    (rq, kq, _, _, _, _, rk, ok_, rv, ov) = _run_pair(inp, True, policy, use_out_kv=True)
    _assert_close(rq, kq, "out_q")
    _assert_close(rk, ok_, "out_k")
    _assert_close(rv, ov, "out_v")


DECODE_CONFIGS = [
    ([5],         1, 1, 64,  64,  16),
    ([3, 7, 11],  4, 2, 64,  64,  16),
    ([1, 1, 17, 33], 8, 2, 128, 128, 16),
    ([129],       16, 4, 128, 128, 32),
]


@pytest.mark.parametrize(
    "seqlens, qh, kvh, qk_d, v_d, bs",
    DECODE_CONFIGS,
    ids=[f"L{','.join(map(str,s))}_qh{qh}_kvh{kvh}_qkd{qk}_vd{vd}_bs{bs}"
         for (s, qh, kvh, qk, vd, bs) in DECODE_CONFIGS],
)
@pytest.mark.parametrize("policy", [0, 1, 2])
def test_rope_norm_store_kv_decode_paged_cache(seqlens, qh, kvh, qk_d, v_d, bs, policy):
    inp = _make_inputs(seqlens, qh, kvh, qk_d, v_d, bs, is_prefill=False)
    # Decode: each request appends 1 token, so num_rows == num_req
    assert inp["qkv"].shape[0] == len(seqlens)
    (rq, kq, rkc, okc, rvc, ovc, *_) = _run_pair(inp, False, policy, use_out_kv=False)
    _assert_close(rq, kq, "out_q")
    _assert_close(rkc, okc, "key_cache")
    _assert_close(rvc, ovc, "value_cache")
