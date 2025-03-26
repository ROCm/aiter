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
import pytest


def run_torch(
    q,
    k,
    v,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window,
    upcast=True,
    reorder_ops=False
):
    (_, seqlen_q, _, _) = q.shape
    (_, seqlen_k, _, _) = k.shape

    if bias is not None:
        attn_bias = bias
    elif alibi_slopes is not None:
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
            upcast=upcast,
            reorder_ops=reorder_ops,
        )

    if dout == None:
        return out
    elif bias is not None:
        dq, dk, dv, dbias = torch.autograd.grad(out, (q, k, v, bias), dout)
        # If seqlen_q > seqlen_k with mask, pytorch will output NaN.
        # Align with ck behavior here
        dbias = torch.nan_to_num(dbias, nan=0.0)
        return out, dq, dk, dv, dbias
    else:
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out, dq, dk, dv, None


def run_ck(
    q,
    k,
    v,
    bias=None,
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
            bias=bias,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
        )

    if dropout_p > 0.0:
        (_, seqlen_q, _, d) = q.shape
        (_, seqlen_k, _, d) = k.shape
        (_, seqlen_k, _, d_v) = v.shape
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
    elif bias is not None:
        dq, dk, dv, dbias = torch.autograd.grad(out, (q, k, v, bias), dout)
        return out, dropout_mask, dq, dk, dv, dbias
    else:
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out, dropout_mask, dq, dk, dv, None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("bias_type", ["no", "bias", "alibi"])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("nheads", [6])
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
def test_flash_attn_output(
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
    dtype
):
    torch.random.manual_seed(0)
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))

    return_lse = True
    return_attn_probs = True

    q = torch.randn(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d_v, device="cuda", dtype=dtype, requires_grad=True)

    attn_bias = None
    alibi_slopes = None
    if bias_type == 'bias':
        attn_bias = torch.randn(seqlen_q, seqlen_k, device="cuda", dtype=dtype, requires_grad=True)
    elif bias_type == 'alibi':
        alibi_slopes = torch.rand(batch_size, nheads, device="cuda", dtype=torch.float32)

    dout = torch.randn(batch_size, seqlen_q, nheads, d_v, device="cuda", dtype=dtype, requires_grad=True)

    out, dropout_mask, dq, dk, dv, dbias = run_ck(
        q, k, v, attn_bias, alibi_slopes, dout, dropout_p, causal,
        window_size, deterministic, return_lse, return_attn_probs)

    out_ref, dq_ref, dk_ref, dv_ref, dbias_ref = run_torch(
        q, k, v, attn_bias, alibi_slopes, dout, dropout_p, dropout_mask, causal, window_size)

    out_pt, dq_pt, dk_pt, dv_pt, dbias_pt = run_torch(
        q, k, v, attn_bias, alibi_slopes, dout, dropout_p, dropout_mask, causal, window_size,
        upcast=False, reorder_ops=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    out_tol = max(2 * (out_pt - out_ref).abs().max().item(), 0.01)
    assert (out - out_ref).abs().max().item() <= out_tol

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

    if attn_bias is not None:
        print(f"dBias max diff: {(dbias - dbias_ref).abs().max().item()}")
        print(f"dBias Pytorch max diff: {(dbias_pt - dbias_ref).abs().max().item()}")
        dbias_tol = max(10 * (dbias_pt - dbias_ref).abs().max().item(), 0.01)
        assert (dbias - dbias_ref).abs().max().item() <= dbias_tol


if __name__ == '__main__':
    batch_size = 2
    nheads = 5
    (seqlen_q, seqlen_k) = (4, 4)
    d = 192
    d_v = 192
    dropout_p = 0.5
    causal = False
    local = False
    bias_type = 'bias'
    deterministic = True
    mha_type = 'mha'
    dtype = torch.bfloat16

    test_flash_attn_output(
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
        dtype
    )
