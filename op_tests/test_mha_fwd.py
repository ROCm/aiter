# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, perftest
from aiter.test_mha_common import attention_ref, attn_bias_from_alibi_slopes


def run_torch(q,
    k,
    v,
    dropout_p=0.0,
    is_causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None
):
    batch_size, seqlen_q, nheads, d = q.shape
    batch_size, seqlen_k, nheads, d = k.shape
    if alibi_slopes is not None:
        attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen_q, seqlen_k, causal=is_causal)
    else:
        attn_bias = None
    dropout_mask = None

    out_ref, attn_ref = attention_ref(
            q,
            k,
            v,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=is_causal,
            window_size=window_size,
        )

    return out_ref


def run_ck(
    q,
    k,
    v,
    dropout_p=0.0,
    is_causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    return_lse=True,
    return_attn_probs=True
):
    out, lse, S_dmask = aiter.flash_attn_func(
            q,
            k,
            v,
            dropout_p,
            causal=is_causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
        )

    return out, lse, S_dmask


def test_mha_fwd(dtype):
    batch_size = 1
    nheads = 1
    nheads_k = nheads
    seqlen_q = 4
    seqlen_k = 4
    d = 64

    q = torch.randn(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d, device="cuda", dtype=dtype)
    alibi_slopes = torch.rand(batch_size, nheads, device="cuda", dtype=torch.float32) * 0.3

    out_ck, _, _ = run_ck(q, k, v, alibi_slopes=alibi_slopes)
    out_torch = run_torch(q, k, v, alibi_slopes=alibi_slopes)
    checkAllclose(out_ck, out_torch, atol=0.01)


test_mha_fwd(torch.bfloat16)
