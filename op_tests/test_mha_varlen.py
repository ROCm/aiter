# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from einops import rearrange, repeat
import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, perftest
from aiter.test_mha_common import (
    attention_ref,
    attn_bias_from_alibi_slopes,
    ck_randval_to_dropout_mask,
    convert_flash_attn_S_to_softmax,
    generate_qkv,
    generate_random_padding_mask,
    pad_rearrange_dropout_mask_hts_to_bhss)


def run_torch(
    q,
    k,
    v,
    query_padding_mask,
    key_padding_mask,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
):
    (_, seqlen_q, _, _) = q.shape
    (_, seqlen_k, _, _) = k.shape

    if alibi_slopes is not None:
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, query_padding_mask, key_padding_mask, causal=causal
            )
    else:
        attn_bias = None

    out, _ = attention_ref(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )

    if dout == None:
        return out

    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    return out, dq, dk, dv


def run_ck(
    q,
    k,
    v,
    query_padding_mask,
    key_padding_mask,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    deterministic=False,
    return_lse=False,
    return_attn_probs=False
):
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    out_unpad, sm_lse, S_dmask = aiter.flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_lse=return_lse,
        return_attn_probs=True,
    )

    out = output_pad_fn(out_unpad)
    if dropout_p > 0.0:
        (_, seqlen_q, _, d) = q.shape
        (_, seqlen_k, _, d) = k.shape
        S_dmask = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_dmask = pad_rearrange_dropout_mask_hts_to_bhss(S_dmask, cu_seqlens_q, seqlen_q, seqlen_k)
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
    else:
        dropout_mask = None

    if dout == None:
        return out, sm_lse, dropout_mask


def test_mha_varlen_fwd():
    dtype = torch.bfloat16
    batch_size = 1
    (nheads, nheads_k) = (1, 1)
    (seqlen_q, seqlen_k) = (4, 4)
    d = 64
    dropout_p = 0.5
    causal = False
    window_size = (-1, -1)
    deterministic = False
    return_lse = True
    return_attn_probs = True

    q = torch.randn(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d, device="cuda", dtype=dtype)
    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, "cuda", mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, "cuda", mode="random")
    alibi_slopes = torch.rand(batch_size, nheads, device="cuda", dtype=torch.float32)
    dout = None

    out_ck, _, dropout_mask = run_ck(q, k, v, query_padding_mask, key_padding_mask,
                                     alibi_slopes, dout, dropout_p,
                                     causal, window_size, deterministic, return_lse, return_attn_probs)

    out_ref = run_torch(
        q, k, v, query_padding_mask, key_padding_mask, alibi_slopes, dout,
        dropout_p, dropout_mask, causal, window_size)

    checkAllclose(out_ck, out_ref, atol=0.01)


test_mha_varlen_fwd()
