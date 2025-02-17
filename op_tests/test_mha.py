# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from einops import repeat
import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, perftest
from aiter.test_mha_common import (
    attention_ref,
    attn_bias_from_alibi_slopes,
    ck_randval_to_dropout_mask,
    convert_flash_attn_S_to_softmax)


def run_torch(
    q,
    k,
    v,
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
        attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen_q, seqlen_k, causal=causal)
    else:
        attn_bias = None

    out, _ = attention_ref(
            q,
            k,
            v,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )

    if dout == None:
        return out
    else:
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out, dq, dk, dv


def run_ck(
    q,
    k,
    v,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    deterministic=False,
    return_lse=True,
    return_attn_probs=False
):
    out, _, S_dmask = aiter.flash_attn_func(
            q,
            k,
            v,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
        )

    if dropout_p > 0.0:
        (_, seqlen_q, _, d) = q.shape
        (_, seqlen_k, _, d) = k.shape
        S_dmask = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            None,
            None,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
    else:
        dropout_mask = None

    if dout == None:
        return out, dropout_mask
    else:
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out, dropout_mask, dq, dk, dv


def test_flash_attn_output():
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

    q = torch.randn(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d, device="cuda", dtype=dtype, requires_grad=True)
    alibi_slopes = torch.rand(batch_size, nheads, device="cuda", dtype=torch.float32)
    dout = torch.randn_like(q)

    out, dropout_mask, dq, dk, dv = run_ck(
        q, k, v, alibi_slopes, dout, dropout_p, causal,
        window_size, deterministic, return_lse, return_attn_probs)

    out_ref, dq_ref, dk_ref, dv_ref = run_torch(
        q, k, v, alibi_slopes, dout, dropout_p, dropout_mask, causal, window_size)

    checkAllclose(out, out_ref, atol=0.01)
    checkAllclose(dq, dq_ref, atol=0.01)
    checkAllclose(dk, dk_ref, atol=0.01)
    checkAllclose(dv, dv_ref, atol=0.01)


test_flash_attn_output()
