# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the FP8 ``rope_norm_store_kv_fp8`` Triton kernel.

Mirrors the pattern from ``aiter_118_test.py``: run the FP8 kernel, dequantize
its outputs back to BF16 using the returned/known scales, then compare against
the unified BF16 reference (which optionally applies the Hadamard rotation for
quant_policy=3).
"""

import math
from typing import Optional

import pytest
import torch

from aiter.ops.triton.fusions.rope_norm_store_kv_fp8 import rope_norm_store_kv_fp8
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype


# ---------- Reference (matches aiter_118_test.py's rope_norm_ref) ----------

def _rms_norm_ref(x, weight, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight


def _apply_rope_neox_ref(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _hadamard_matrix(n, device):
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H


def _apply_hadamard_per_head(x, head_dim):
    H = _hadamard_matrix(head_dim, x.device)
    return torch.matmul(x.to(torch.float32), H.t()) * (1.0 / math.sqrt(head_dim))


def rope_norm_ref(
    kcache, vcache, qkv, cos_sin, num_seqlen_per_req, q_index, kv_indices,
    q_norm_weight, k_norm_weight, qk_norm_policy, apply_hadamard=False,
):
    """BF16 reference matching `rope_norm_ref` from aiter_118_test.py."""
    dtype = qkv.dtype
    num_kv = kcache.shape[2]
    v_dim = vcache.shape[3]
    qk_dim = kcache.shape[3]
    num_q = (qkv.shape[1] - num_kv * qk_dim - num_kv * v_dim) // qk_dim
    num_req = num_seqlen_per_req.shape[0]
    q_lens = (q_index[1:] - q_index[:-1]).tolist()
    num_rows = int(q_index[-1].item())
    blk = kcache.shape[1]

    q = qkv[:, : num_q * qk_dim].to(torch.float32).view(num_rows, num_q, qk_dim)
    k = qkv[:, num_q * qk_dim : (num_q + num_kv) * qk_dim].to(torch.float32).view(
        num_rows, num_kv, qk_dim
    )
    v = qkv[:, (num_q + num_kv) * qk_dim :].view(num_rows, num_kv, v_dim)

    cs = torch.zeros(num_rows, qk_dim, dtype=torch.float32, device=qkv.device)
    off = 0
    for i in range(num_req):
        sl = int(num_seqlen_per_req[i].item())
        ql = int(q_lens[i])
        if ql > 0 and sl > 0:
            cs[off : off + ql] = cos_sin[sl - ql : sl]
        off += ql

    cos = cs[:, : qk_dim // 2].unsqueeze(1)
    sin = cs[:, qk_dim // 2 :].unsqueeze(1)

    if qk_norm_policy == 2:
        q = _rms_norm_ref(q, q_norm_weight)
        k = _rms_norm_ref(k, k_norm_weight)
    q = _apply_rope_neox_ref(q, cos, sin)
    k = _apply_rope_neox_ref(k, cos, sin)
    if qk_norm_policy == 1:
        q = _rms_norm_ref(q, q_norm_weight)
        k = _rms_norm_ref(k, k_norm_weight)

    if apply_hadamard:
        q = _apply_hadamard_per_head(q, qk_dim)
        k = _apply_hadamard_per_head(k, qk_dim)

    tok = 0
    for ri in range(num_req):
        sl = int(num_seqlen_per_req[ri].item())
        ql = int(q_lens[ri])
        if sl <= 0:
            tok += ql
            continue
        for pos in range(sl - ql, sl):
            bi, pb = pos // blk, pos % blk
            cb = int(kv_indices[ri, bi].item())
            kcache[cb, pb] = k[tok].to(dtype)
            vcache[cb, pb] = v[tok].to(dtype)
            if pos == sl - 1 and pb + 1 < blk:
                kcache[cb, pb + 1 :] = 0
                vcache[cb, pb + 1 :] = 0
            tok += 1

    return q.to(dtype)


# ---------- Input generation ----------

def _make_inputs(
    seqlens,
    num_q_heads,
    num_kv_heads,
    qk_head_dim,
    v_head_dim,
    block_size,
    is_prefill,
    quant_policy,
    seed=0,
    device="cuda",
    pad_decode=False,
):
    torch.manual_seed(seed)
    num_req = len(seqlens)
    new_per_req = list(seqlens) if is_prefill else [1] * num_req
    if not is_prefill and pad_decode:
        pad_to = ((num_req + 7) // 8) * 8
    else:
        pad_to = num_req

    real_num_req = num_req
    real_num_rows = sum(new_per_req)

    # Build padded structures
    if pad_to > num_req:
        seqlens_padded = list(seqlens) + [0] * (pad_to - num_req)
        new_per_req_padded = new_per_req + [0] * (pad_to - num_req)
    else:
        seqlens_padded = list(seqlens)
        new_per_req_padded = new_per_req

    q_index_list = [0]
    cum = 0
    for n in new_per_req_padded:
        cum += n
        q_index_list.append(cum)
    num_rows = cum
    q_index = torch.tensor(q_index_list, dtype=torch.int32, device=device)
    num_seqlen_per_req = torch.tensor(seqlens_padded, dtype=torch.int32, device=device)

    max_seqlen = max(seqlens) if seqlens else 1
    max_blocks_per_req = (max_seqlen + block_size - 1) // block_size
    total_blocks_needed = max_blocks_per_req * pad_to + 4
    physical_blocks = torch.randperm(total_blocks_needed, device=device).to(torch.int32)
    kvcache_indices = physical_blocks[: pad_to * max_blocks_per_req].reshape(
        pad_to, max_blocks_per_req
    )

    hidden = num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim + num_kv_heads * v_head_dim
    qkv = torch.randn(num_rows, hidden, dtype=torch.bfloat16, device=device) * 0.5

    max_table_len = max_seqlen + 8
    cos_sin = torch.randn(max_table_len, qk_head_dim, dtype=torch.float32, device=device)

    q_norm_weight = torch.randn(qk_head_dim, dtype=torch.float32, device=device) * 0.3
    k_norm_weight = torch.randn(qk_head_dim, dtype=torch.float32, device=device) * 0.3

    fp8_dtype = get_fp8_e4m3_dtype()
    key_cache = torch.zeros(
        total_blocks_needed, block_size, num_kv_heads, qk_head_dim,
        dtype=fp8_dtype, device=device,
    )
    value_cache = torch.zeros(
        total_blocks_needed, block_size, num_kv_heads, v_head_dim,
        dtype=fp8_dtype, device=device,
    )

    # Scales per policy
    if quant_policy in (0, 3):
        L = qk_head_dim // 4
        R = block_size // L
        k_scale = torch.zeros(total_blocks_needed, R, num_kv_heads, L,
                              dtype=torch.float32, device=device)
        v_scale = torch.rand(num_kv_heads, dtype=torch.float32, device=device) * 0.2 + 0.05
    else:
        k_scale = torch.tensor([0.1], dtype=torch.float32, device=device)
        v_scale = torch.tensor([0.1], dtype=torch.float32, device=device)

    q_scale_val = 2.0
    q_scale_inv = torch.tensor([1.0 / q_scale_val], dtype=torch.float32, device=device)

    if is_prefill:
        max_seqlens = max(new_per_req) if new_per_req else 0
    else:
        max_seqlens = 1

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
        k_scale=k_scale,
        v_scale=v_scale,
        q_scale_val=q_scale_val,
        q_scale_inv=q_scale_inv,
        max_seqlens=max_seqlens,
        real_num_req=real_num_req,
        real_num_rows=real_num_rows,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        block_size=block_size,
        total_blocks_needed=total_blocks_needed,
    )


# ---------- Tests ----------

PREFILL_CONFIGS = [
    # (seqlens, qh, kvh, qk_d, v_d, bs)
    ([16, 32, 8],   8, 1, 128, 128, 64),
    ([200, 50, 80, 30], 64, 8, 128, 128, 64),
    ([5],            8, 1, 128, 128, 64),
]

DECODE_CONFIGS = [
    ([100, 50, 30], 8, 1, 128, 128, 64),
    ([90] * 7,      64, 8, 128, 128, 64),  # 7 reqs → pads to 8 in helper
    ([200],         8, 1, 128, 128, 64),
]


def _dequant_compare(
    res, inp, quant_policy, qk_norm_policy, is_prefill,
    q_fp8, q_scale_out,
    kcache_fp8, vcache_fp8,
):
    """Dequantize FP8 outputs and compare against BF16 reference."""
    num_q_heads = inp["num_q_heads"]
    num_kv_heads = inp["num_kv_heads"]
    qk_head_dim = inp["qk_head_dim"]
    v_head_dim = inp["v_head_dim"]
    block_size = inp["block_size"]
    real_num_req = inp["real_num_req"]
    real_num_rows = inp["real_num_rows"]

    # Reference (BF16) is computed on UNPADDED inputs (padded entries have seqlen=0).
    q_index_r = inp["q_index"][: real_num_req + 1]
    qkv_r = inp["qkv"][:real_num_rows]
    ns_r = inp["num_seqlen_per_req"][:real_num_req]
    ki_r = inp["kvcache_indices"][:real_num_req]
    kcache_ref = torch.zeros_like(kcache_fp8, dtype=torch.bfloat16)
    vcache_ref = torch.zeros_like(vcache_fp8, dtype=torch.bfloat16)

    ref_q = rope_norm_ref(
        kcache_ref, vcache_ref, qkv_r, inp["cos_sin"], ns_r, q_index_r, ki_r,
        inp["q_norm_weight"], inp["k_norm_weight"], qk_norm_policy,
        apply_hadamard=(quant_policy == 3),
    )

    # ----- Dequantize Q and compare -----
    if quant_policy in (0, 1, 3):
        if is_prefill:
            pad128 = q_scale_out.shape[-1]
            seqlens = (q_index_r[1:] - q_index_r[:-1]).to(q_fp8.device)
            mask = (
                torch.arange(pad128, device=q_fp8.device).expand(real_num_req, pad128)
                < seqlens.unsqueeze(1)
            )
            # q_scale_out: [num_req, num_q_heads, pad128] → permute → [num_req, pad128, num_q_heads]
            scale_flat = q_scale_out[:real_num_req].permute(0, 2, 1)[mask]
            q_bf16 = (
                q_fp8[:real_num_rows].to(torch.bfloat16) * scale_flat[:, :, None]
            ).to(torch.bfloat16)
        else:
            q_bf16 = (
                q_fp8[:real_num_rows].to(torch.bfloat16)
                * q_scale_out[:real_num_rows, :, None]
            ).to(torch.bfloat16)
    else:
        assert q_scale_out is None, "static-Q should not return q_scale"
        q_bf16 = (q_fp8[:real_num_rows].to(torch.float32) * inp["q_scale_val"]).to(
            torch.bfloat16
        )

    torch.testing.assert_close(
        ref_q.to(torch.float32), q_bf16.to(torch.float32),
        atol=0.8, rtol=0.0,
        msg=lambda m: f"Q mismatch (policy={quant_policy}, norm={qk_norm_policy}, prefill={is_prefill}): {m}",
    )

    # ----- Dequantize K/V cache and compare -----
    q_lens_r = (q_index_r[1:] - q_index_r[:-1]).tolist()
    L = qk_head_dim // 4
    for ri in range(real_num_req):
        sl = int(ns_r[ri].item())
        ql = int(q_lens_r[ri])
        if sl <= 0:
            continue
        for pos in range(sl - ql, sl):
            bi, pb = pos // block_size, pos % block_size
            cb = int(ki_r[ri, bi].item())
            for h in range(num_kv_heads):
                v_fp8 = vcache_fp8[cb, pb, h, :].to(torch.float32)
                v_ref = vcache_ref[cb, pb, h, :].to(torch.float32)
                if quant_policy in (0, 3):
                    v_dq = v_fp8 * inp["v_scale"][h]
                else:
                    v_dq = v_fp8 * inp["v_scale"][0]
                torch.testing.assert_close(
                    v_ref, v_dq, atol=0.8, rtol=0.0,
                    msg=lambda m: f"V mismatch req={ri} pos={pos} h={h} policy={quant_policy}: {m}",
                )
                k_fp8 = kcache_fp8[cb, pb, h, :].to(torch.float32)
                k_ref = kcache_ref[cb, pb, h, :].to(torch.float32)
                if quant_policy in (0, 3):
                    k_s = res["k_scale"][cb, pb // L, h, pb % L].item()
                else:
                    k_s = inp["k_scale"][0].item()
                k_dq = k_fp8 * k_s
                torch.testing.assert_close(
                    k_ref, k_dq, atol=0.8, rtol=0.0,
                    msg=lambda m: f"K mismatch req={ri} pos={pos} h={h} policy={quant_policy}: {m}",
                )


def _run(inp, is_prefill, qk_norm_policy, quant_policy):
    qnw = inp["q_norm_weight"] if qk_norm_policy != 0 else None
    knw = inp["k_norm_weight"] if qk_norm_policy != 0 else None
    q_scale_inv = inp["q_scale_inv"] if quant_policy == 2 else None

    out_q, q_scale_out, split_k_flag = rope_norm_store_kv_fp8(
        inp["key_cache"], inp["value_cache"], inp["qkv"], inp["cos_sin"],
        inp["num_seqlen_per_req"], inp["q_index"], inp["kvcache_indices"],
        is_prefill,
        inp["k_scale"], inp["v_scale"],
        quant_policy=quant_policy,
        max_seqlens=inp["max_seqlens"],
        q_scale_inv=q_scale_inv,
        q_norm_weight=qnw, k_norm_weight=knw,
        qk_norm_policy=qk_norm_policy,
    )
    return out_q, q_scale_out, split_k_flag


@pytest.mark.parametrize(
    "seqlens, qh, kvh, qk_d, v_d, bs",
    PREFILL_CONFIGS,
    ids=[f"L{','.join(map(str,s))}_qh{qh}_kvh{kvh}_qkd{qk}_vd{vd}_bs{bs}"
         for (s, qh, kvh, qk, vd, bs) in PREFILL_CONFIGS],
)
@pytest.mark.parametrize("qk_norm_policy", [0, 1, 2])
@pytest.mark.parametrize("quant_policy", [0, 1, 2, 3])
def test_fp8_prefill(seqlens, qh, kvh, qk_d, v_d, bs, qk_norm_policy, quant_policy):
    inp = _make_inputs(
        seqlens, qh, kvh, qk_d, v_d, bs,
        is_prefill=True, quant_policy=quant_policy,
    )
    out_q, q_scale_out, split_k_flag = _run(inp, True, qk_norm_policy, quant_policy)

    assert split_k_flag.shape == (inp["num_seqlen_per_req"].shape[0], kvh)
    assert split_k_flag.dtype == torch.int32
    assert torch.all(split_k_flag == 0)

    _dequant_compare(
        {"k_scale": inp["k_scale"]}, inp, quant_policy, qk_norm_policy, True,
        out_q, q_scale_out, inp["key_cache"], inp["value_cache"],
    )


@pytest.mark.parametrize(
    "seqlens, qh, kvh, qk_d, v_d, bs",
    DECODE_CONFIGS,
    ids=[f"L{','.join(map(str,s))}_qh{qh}_kvh{kvh}_qkd{qk}_vd{vd}_bs{bs}"
         for (s, qh, kvh, qk, vd, bs) in DECODE_CONFIGS],
)
@pytest.mark.parametrize("qk_norm_policy", [0, 1, 2])
@pytest.mark.parametrize("quant_policy", [0, 1, 2, 3])
@pytest.mark.parametrize("pad_decode", [False, True])
def test_fp8_decode(seqlens, qh, kvh, qk_d, v_d, bs, qk_norm_policy, quant_policy, pad_decode):
    inp = _make_inputs(
        seqlens, qh, kvh, qk_d, v_d, bs,
        is_prefill=False, quant_policy=quant_policy,
        pad_decode=pad_decode,
    )
    out_q, q_scale_out, split_k_flag = _run(inp, False, qk_norm_policy, quant_policy)

    assert split_k_flag.shape == (inp["num_seqlen_per_req"].shape[0], kvh)
    assert split_k_flag.dtype == torch.int32

    _dequant_compare(
        {"k_scale": inp["k_scale"]}, inp, quant_policy, qk_norm_policy, False,
        out_q, q_scale_out, inp["key_cache"], inp["value_cache"],
    )
