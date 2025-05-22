# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter import dtypes
from aiter.test_mha_common import (
    attention_ref,
    attn_bias_from_alibi_slopes,
    ck_randval_to_dropout_mask,
    convert_flash_attn_S_to_softmax,
    generate_qkv,
    generate_random_padding_mask,
    pad_rearrange_dropout_mask_hts_to_bhss,
)
import pytest


def run_torch(
    q,
    k,
    v,
    query_padding_mask,
    key_padding_mask,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    upcast=True,
    reorder_ops=False,
):
    (b, seqlen_q, _, _) = q.shape
    (_, seqlen_k, _, _) = k.shape

    if bias is not None:
        attn_bias = bias.reshape(b, 1, seqlen_q, seqlen_k)
    elif alibi_slopes is not None:
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
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
        upcast=upcast,
        reorder_ops=reorder_ops,
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
    query_padding_mask,
    key_padding_mask,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
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
    if bias is not None:
        # TODO - implement generate_bias() to unpad
        total_q = q_unpad.shape[0]
        assert total_q == batch_size * max_seqlen_q
        assert q.shape[1] == max_seqlen_q
        assert k.shape[1] == max_seqlen_k
        bias_unpad = bias.reshape(batch_size * max_seqlen_q, max_seqlen_k)
    else:
        bias_unpad = None

    out_unpad, _, S_dmask = aiter.flash_attn_varlen_func(
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
        bias=bias_unpad,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_lse=return_lse,
        return_attn_probs=return_attn_probs,
    )

    out = output_pad_fn(out_unpad)
    if dropout_p > 0.0:
        (_, seqlen_q, _, d) = q.shape
        (_, seqlen_k, _, d) = k.shape
        S_dmask = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_dmask = pad_rearrange_dropout_mask_hts_to_bhss(
            S_dmask, cu_seqlens_q, seqlen_q, seqlen_k
        )
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
        return out, dropout_mask
    else:
        dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(
            out, (q_unpad, k_unpad, v_unpad), dout
        )
        dq = dq_pad_fn(dq_unpad)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        return out, dropout_mask, dq, dk, dv


@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("bias_type", ["no", "alibi"])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("nheads", [9])
@pytest.mark.parametrize(
    "d,d_v",
    [
        (32, 32),
        (40, 40),
        (59, 59),
        (64, 64),
        (96, 96),
        (111, 111),
        (128, 128),
        (160, 160),
        (192, 192),
        (224, 224),
        (256, 256),
    ],
)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 147),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
def test_flash_attn_varlen_func(
    batch_size,
    nheads,
    seqlen_q,
    seqlen_k,
    d,
    d_v,
    dropout_p,
    causal,
    local,
    bias_type,
    deterministic,
    mha_type,
    dtype,
):
    torch.random.manual_seed(0)
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))

    return_lse = True
    return_attn_probs = True

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d_v,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    if bias_type == "bias":
        # TODO - We need to implement unpad bias [batch_size, seqlen_q, seqlen_k] -> [total_q, max_seqlen_k]
        # Let total_q = batch_size * seqlen_q to pass the test for now
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, "cuda", mode="full"
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, "cuda", mode="full"
        )
    else:
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, "cuda", mode="random"
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, "cuda", mode="random"
        )

    attn_bias = None
    alibi_slopes = None
    if bias_type == "bias":
        attn_bias = torch.randn(
            batch_size,
            seqlen_q,
            seqlen_k,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
    elif bias_type == "alibi":
        alibi_slopes = torch.rand(batch_size, nheads, device="cuda", dtype=dtypes.fp32)

    dout = torch.randn(
        batch_size,
        seqlen_q,
        nheads,
        d_v,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    out, dropout_mask, dq, dk, dv = run_ck(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        causal,
        window_size,
        deterministic,
        return_lse,
        return_attn_probs,
    )

    out_ref, dq_ref, dk_ref, dv_ref = run_torch(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        dropout_mask,
        causal,
        window_size,
    )

    out_pt, dq_pt, dk_pt, dv_pt = run_torch(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        dropout_mask,
        causal,
        window_size,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    out_tol = max(4 * (out_pt - out_ref).abs().max().item(), 0.01)
    assert (out - out_ref).abs().max().item() <= out_tol

    # TODO: Support varlen bwd for bias
    if bias_type == "bias":
        pytest.skip("Does not support varlen bwd for bias")

    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")

    dq_tol = max(10 * (dq_pt - dq_ref).abs().max().item(), 0.01)
    dk_tol = max(10 * (dk_pt - dk_ref).abs().max().item(), 0.01)
    dv_tol = max(10 * (dv_pt - dv_ref).abs().max().item(), 0.01)

    assert (dq - dq_ref).abs().max().item() <= dq_tol
    assert (dk - dk_ref).abs().max().item() <= dk_tol
    assert (dv - dv_ref).abs().max().item() <= dv_tol


if __name__ == "__main__":
    batch_size = 4
    nheads = 4
    (seqlen_q, seqlen_k) = (4, 4)
    d = 192
    d_v = 192
    dropout_p = 0.5
    causal = False
    local = False
    bias_type = "no"
    deterministic = True
    mha_type = "mha"
    dtype = dtypes.bf16

    test_flash_attn_varlen_func(
        batch_size,
        nheads,
        seqlen_q,
        seqlen_k,
        d,
        d_v,
        dropout_p,
        causal,
        local,
        bias_type,
        deterministic,
        mha_type,
        dtype,
    )
