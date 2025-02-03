# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import aiter
from aiter.test_common import checkAllclose, perftest


# @perftest()
def run_ck(
    q,
    k,
    v,
    dropout_p=0.0,
    is_causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    return_lse=False,
    return_attn_probs=False
):
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape

    q_unpad = rearrange(q, "b s h d -> (b s) h d")
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
    )
    max_seqlen_q = seqlen_q

    k_unpad = rearrange(k, "b s h d -> (b s) h d")
    v_unpad = rearrange(v, "b s h d -> (b s) h d")
    cu_seqlens_k = torch.arange(
        0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
    )
    max_seqlen_k = seqlen_k

    aiter.mha_varlen_fwd(q_unpad,
                         k_unpad,
                         v_unpad,
                         cu_seqlens_q,
                         cu_seqlens_k,
                         max_seqlen_q,
                         max_seqlen_k,
                         dropout_p=dropout_p,
                         softmax_scale=q.shape[-1] ** (-0.5),
                         zero_tensors=False,
                         is_causal=is_causal,
                         window_size_left=window_size[0],
                         window_size_right=window_size[1],
                         return_softmax_lse=return_lse,
                         return_dropout_randval=return_attn_probs,
                         alibi_slopes=alibi_slopes,
    )


def test_mha_varlen_fwd(dtype):
    batch_size = 1
    nheads = 1
    nheads_k = nheads
    seqlen_q = 4
    seqlen_k = 4
    d = 64

    q = torch.randn(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d, device="cuda", dtype=dtype)

    run_ck(q, k, v)


test_mha_varlen_fwd(torch.bfloat16)
