# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import argparse
import itertools
import pandas as pd

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
from aiter.test_common import benchmark, run_perftest


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

    if dout is None:
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
    min_seqlen_q=0,
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
    _, _, nhead, d = q.shape
    _, _, _, d_v = v.shape
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

    (outputs), us_fwd = run_perftest(
        aiter.flash_attn_varlen_func,
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q=min_seqlen_q,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        bias=bias_unpad,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_lse=return_lse,
        return_attn_probs=return_attn_probs,
    )

    if type(outputs) is tuple:
        out = output_pad_fn(outputs[0])
    else:
        out = output_pad_fn(outputs)

    if dropout_p > 0.0 and return_attn_probs:
        (_, seqlen_q, _, d) = q.shape
        (_, seqlen_k, _, d) = k.shape
        S_dmask = outputs[-1]
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

    fwd_flop = 0
    fwd_num_bytes = 0
    bwd_flop = 0
    bwd_num_bytes = 0
    dtype_bytes = torch.finfo(q.dtype).bits // 8
    lse_dtype_bytes = torch.finfo(torch.float).bits // 8
    for i in range(len(cu_seqlens_q) - 1):
        real_seqlen_q = cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item()
        real_seqlen_k = cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item()
        fwd_flop = (
            fwd_flop
            + nhead * 2 * real_seqlen_q * real_seqlen_k * d
            + nhead * 2 * real_seqlen_q * real_seqlen_k * d_v
        )
        fwd_num_bytes = fwd_num_bytes + nhead * dtype_bytes * (
            real_seqlen_q * d
            + real_seqlen_k * d
            + real_seqlen_k * d_v
            + real_seqlen_q * d_v
        )
        bwd_flop = (
            bwd_flop
            + nhead * 3 * 2 * real_seqlen_q * real_seqlen_k * d
            + nhead * 2 * 2 * real_seqlen_q * real_seqlen_k * d_v
        )
        bwd_num_bytes = (
            bwd_num_bytes
            + nhead
            * dtype_bytes
            * (
                real_seqlen_q * d
                + real_seqlen_k * d
                + real_seqlen_k * d_v
                + real_seqlen_q * d_v
            )
            * 2
            + nhead * lse_dtype_bytes * real_seqlen_q
        )

    if dout is None or not return_lse:
        return out, dropout_mask, None, None, None, (us_fwd, fwd_flop, fwd_num_bytes)
    else:
        (dq_unpad, dk_unpad, dv_unpad), us_bwd = run_perftest(
            torch.autograd.grad,
            out,
            (q_unpad, k_unpad, v_unpad),
            dout,
            retain_graph=True,
            num_rotate_args=1,
        )
        dq = dq_pad_fn(dq_unpad)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        return (
            out,
            dropout_mask,
            dq,
            dk,
            dv,
            (us_fwd, fwd_flop, fwd_num_bytes, us_bwd, bwd_flop, bwd_num_bytes),
        )


@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("bias_type", ["no", "alibi"])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("min_seqlen_q", [0])
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
@benchmark()
def test_flash_attn_varlen_func(
    batch_size,
    nheads,
    seqlen_q,
    seqlen_k,
    d,
    d_v,
    min_seqlen_q,
    dropout_p,
    causal,
    local,
    bias_type,
    deterministic,
    mha_type,
    dtype,
):
    return_lse = True
    torch.random.manual_seed(0)
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))

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

    # return_attn_probs is just for host verification (to produce same dropout mask)
    # no need to use in actual case
    if dropout_p > 0:
        return_attn_probs = True
    else:
        return_attn_probs = False

    (
        out,
        dropout_mask,
        dq,
        dk,
        dv,
        (us_fwd, fwd_flop, fwd_num_bytes, us_bwd, bwd_flop, bwd_num_bytes),
    ) = run_ck(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        min_seqlen_q,
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
    # assert (out - out_ref).abs().max().item() <= out_tol

    # TODO: Support varlen bwd for bias
    if bias_type == "bias":
        pytest.skip("Does not support varlen bwd for bias")

    if dq is not None:
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
    ret = {}
    ret["fwd_us"] = us_fwd
    ret["fwd_tflops"] = (fwd_flop) / 1.0e6 / us_fwd
    ret["fwd_gb_per_sec"] = (fwd_num_bytes) / 1.0e3 / us_fwd
    ret["bwd_us"] = us_bwd
    ret["bwd_tflops"] = (bwd_flop) / 1.0e6 / us_fwd
    ret["bwd_gb_per_sec"] = (bwd_num_bytes) / 1.0e3 / us_fwd
    return ret


l_dtype = ["bf16", "fp16"]
l_dim = [32, 40, 64, 96, 111, 128, 160, 192]
l_mha_type = ["mha", "mqa", "gqa"]
l_causal = [False, True]
l_local = [False, True]
l_deterministic = [False, True]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        nargs="?",
        default=4,
        help="""Batch size.
    e.g.: -b 16""",
    )
    parser.add_argument(
        "-nh",
        "--nheads",
        type=int,
        nargs="?",
        default=9,
        help="""Number of attention heads.
    e.g. -nh 4""",
    )
    parser.add_argument(
        "-s",
        "--seqlen_q_k",
        type=dtypes.str2tuple,
        nargs="?",
        default=(4, 8),
        help="""Sequence length of query&key.
    e.g. -s 4,8""",
    )
    parser.add_argument(
        "-d",
        type=int,
        nargs="?",
        default=None,
        help="""Dimension of query&key.
    e.g. -d 128""",
    )
    parser.add_argument(
        "-dv",
        type=int,
        nargs="?",
        default=None,
        help="""Dimension of value.
    e.g. -dv 128""",
    )
    parser.add_argument(
        "-dp",
        "--dropout_p",
        type=float,
        nargs="?",
        default=0.0,
        help="""Dropout probability."
    e.g. -dp 0.0""",
    )
    parser.add_argument(
        "-msq",
        "--min_seqlen_q",
        type=int,
        nargs="?",
        default=0,
        help="""Minimum sequence length of query.
    e.g. -msq 1""",
    )
    parser.add_argument(
        "-c",
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="""Causal attention, default is None.
    -c or --causal    # enable causal attention
    --no-causal       # disable causal attention""",
    )
    parser.add_argument(
        "-l",
        "--local",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="""Local attention. default is None.
        e.g. -l or --local    # enable local attention
        --no-local        # disable local attention""",
    )
    parser.add_argument(
        "-bt",
        "--bias_type",
        type=str,
        default="no",
        help="Type of bias.",
    )
    parser.add_argument(
        "-det",
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="""Deterministic attention, default is None.
    -det or --deterministic    # enable deterministic attention
    --no-deterministic         # disable deterministic attention""",
    )
    parser.add_argument(
        "-mha",
        "--mha_type",
        type=str,
        default=None,
        help="""Type of multi-head attention.
    e.g. -mha mha/mqa/gqa""",
    )
    parser.add_argument(
        "-dt",
        "--dtype",
        type=str,
        default=None,
        help="""Data type.
    e.g.: -dt bf16""",
    )

    args = parser.parse_args()
    if args.dtype is not None:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    else:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    if args.d is not None:
        l_dim = [args.d]
    if args.mha_type is not None:
        l_mha_type = [args.mha_type]
    if args.causal is not None:
        l_causal = [args.causal]
    if args.local is not None:
        l_local = [args.local]
    if args.deterministic is not None:
        l_deterministic = [args.deterministic]
    (seqlen_q, seqlen_k) = args.seqlen_q_k

    collected = []
    for dtype, dim, mha_type, causal, local, deterministic in itertools.product(
        l_dtype, l_dim, l_mha_type, l_causal, l_local, l_deterministic
    ):
        ret = test_flash_attn_varlen_func(
            args.batch_size,
            args.nheads,
            seqlen_q,
            seqlen_k,
            dim,
            dim,
            args.min_seqlen_q,
            args.dropout_p,
            causal,
            local,
            args.bias_type,
            deterministic,
            mha_type,
            dtype,
        )
        collected.append(ret)

    df = pd.DataFrame(collected)
    aiter.logger.info(f"mha_varlen summary:\n{df}")
