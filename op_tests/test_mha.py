# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, run_perftest
from aiter.test_mha_common import (
    attention_ref,
    attn_bias_from_alibi_slopes,
    ck_randval_to_dropout_mask,
    convert_flash_attn_S_to_softmax,
)


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
    reorder_ops=False,
):
    (_, seqlen_q, _, _) = q.shape
    (_, seqlen_k, _, _) = k.shape

    if bias is not None:
        attn_bias = bias
    elif alibi_slopes is not None:
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, causal=causal
        )
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

    if dout is None:
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
    return_attn_probs=False,
):
    (out, _, S_dmask), us_fwd = run_perftest(
        aiter.flash_attn_func,
        q,
        k,
        v,
        dropout_p,
        None,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
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

    if dout is None:
        return out, dropout_mask, us_fwd
    elif bias is not None:
        (dq, dk, dv, dbias), us_bwd = run_perftest(
            torch.autograd.grad,
            out,
            (q, k, v, bias),
            dout,
            retain_graph=True,
            num_rotate_args=1,
        )
        return out, dropout_mask, dq, dk, dv, dbias, (us_fwd, us_bwd)
    else:
        (dq, dk, dv), us_bwd = run_perftest(
            torch.autograd.grad,
            out,
            (q, k, v),
            dout,
            retain_graph=True,
            num_rotate_args=1,
        )
        return out, dropout_mask, dq, dk, dv, None, (us_fwd, us_bwd)


@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16])
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
@benchmark()
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
    dtype,
):
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()
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

    attn_bias = None
    alibi_slopes = None
    if bias_type == "bias":
        attn_bias = torch.randn(
            seqlen_q, seqlen_k, device="cuda", dtype=dtype, requires_grad=True
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

    out, dropout_mask, dq, dk, dv, dbias, (us_fwd, us_bwd) = run_ck(
        q,
        k,
        v,
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

    out_ref, dq_ref, dk_ref, dv_ref, dbias_ref = run_torch(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        dropout_mask,
        causal,
        window_size,
    )

    out_pt, dq_pt, dk_pt, dv_pt, dbias_pt = run_torch(
        q,
        k,
        v,
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

    fwd_flop = nheads * (seqlen_q * seqlen_k * d * 2 + seqlen_q * seqlen_k * d_v * 2)
    dtype_bytes = torch.finfo(dtype).bits // 8
    fwd_num_bytes = (
        nheads
        * dtype_bytes
        * (seqlen_q * d + seqlen_k * d + seqlen_k * d_v + seqlen_q * d_v)
    )
    bwd_flop = nheads * (
        seqlen_q * seqlen_k * d * 2 * 3 + seqlen_q * seqlen_k * d_v * 2 * 2
    )
    bwd_num_bytes = (
        2 * fwd_num_bytes + nheads * (torch.finfo(torch.float).bits // 8) * seqlen_q
    )
    ret = {}
    ret["fwd_us"] = us_fwd
    ret["fwd_tflops"] = (fwd_flop) / 1.0e6 / us_fwd
    ret["fwd_gb_per_sec"] = (fwd_num_bytes) / 1.0e3 / us_fwd
    ret["bwd_us"] = us_bwd
    ret["bwd_tflops"] = (bwd_flop) / 1.0e6 / us_fwd
    ret["bwd_gb_per_sec"] = (bwd_num_bytes) / 1.0e3 / us_fwd
    return ret


l_dtype = ["bf16", "fp16"]
l_dim = [32, 40, 64, 96, 111, 128, 160]
l_mha_type = ["mha", "mqa", "gqa"]
l_causal = [False, True]
l_local = [False, True]
l_seq_q = [108, 256, 512]
l_seq_k = [256, 512, 1024]
l_deterministic = [False, True]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=2,
    help="""Batch size. Default is 2.
    e.g.: -b 16""",
)
parser.add_argument(
    "-n",
    "--nheads",
    type=int,
    default=6,
    help="""Number of heads. Default is 6.
    e.g.: -n 8""",
)
parser.add_argument(
    "-q",
    "--seqlen_q",
    type=int,
    default=None,
    help="""Sequence length for query. Default is 512.
    e.g.: -q 1024""",
)
parser.add_argument(
    "-k",
    "--seqlen_k",
    type=int,
    default=None,
    help="""Sequence length for key. Default is 512.
    e.g.: -k 1024""",
)
parser.add_argument(
    "-qk",
    "--d_qk",
    type=int,
    default=None,
    help="""Dimension of query and key. Default is 128.
    e.g.: -qk 256""",
)
parser.add_argument(
    "-v",
    "--d_v",
    type=int,
    default=None,
    help="""Dimension of value. Default is 128.
    e.g.: -v 256""",
)
parser.add_argument(
    "-p",
    "--dropout_p",
    type=float,
    default=0.0,
    help="""Dropout probability. Default is 0.0.
    e.g.: -p 0.1""",
)
parser.add_argument(
    "-c",
    "--causal",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="""Causal attention. Default is None.
    -c or --causal    # enable causal attention
    --no-causal       # disable causal attention""",
)
parser.add_argument(
    "-l",
    "--local",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="""Local attention. Default is None.
        e.g. -l or --local    # enable local attention
        --no-local        # disable local attention""",
)
parser.add_argument(
    "-bt",
    "--bias_type",
    type=str,
    default="no",
    help="""Bias type. Default is 'no'.
    e.g.: -bt no""",
)
parser.add_argument(
    "-det",
    "--deterministic",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="""Deterministic attention. Default is None.
    -det or --deterministic    # enable deterministic attention
    --no-deterministic         # disable deterministic attention""",
)
parser.add_argument(
    "-m",
    "--mha_type",
    type=str,
    default=None,
    help="""Type of multi-head attention.
    e.g.: -m mha""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
if __name__ == "__main__":
    args = parser.parse_args()
    if args.dtype is not None:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    else:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    if args.d_qk is not None:
        l_dim = [args.d_qk]
    if args.mha_type is not None:
        l_mha_type = [args.mha_type]
    if args.causal is not None:
        l_causal = [args.causal]
    if args.local is not None:
        l_local = [args.local]
    if args.seqlen_q is not None:
        l_seq_q = [args.seqlen_q]
    if args.seqlen_k is not None:
        l_seq_k = [args.seqlen_k]
    if args.deterministic is not None:
        l_deterministic = [args.deterministic]
    collected = []
    for (
        dtype,
        dim,
        mha_type,
        causal,
        local,
        seq_q,
        seq_k,
        deterministic,
    ) in itertools.product(
        l_dtype, l_dim, l_mha_type, l_causal, l_local, l_seq_q, l_seq_k, l_deterministic
    ):
        ret = test_flash_attn_output(
            args.batch_size,
            args.nheads,
            seq_q,
            seq_k,
            dim,
            dim,
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
    aiter.logger.info(f"mha summary:\n{df}")
