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
    generate_qkv,
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
    query_padding_mask=None,
    key_padding_mask=None,
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

    out, _, softmax_lse = attention_ref(
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
        return out, softmax_lse
    elif bias is not None:
        dq, dk, dv, dbias = torch.autograd.grad(out, (q, k, v, bias), dout)
        # If seqlen_q > seqlen_k with mask, pytorch will output NaN.
        # Align with ck behavior here
        dbias = torch.nan_to_num(dbias, nan=0.0)
        return out, softmax_lse, dq, dk, dv, dbias
    else:
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out, softmax_lse, dq, dk, dv, None


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
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
):
    (out, softmax_lse, S_dmask), us_fwd = run_perftest(
        aiter.flash_attn_func,
        q,
        k,
        v,
        dropout_p,
        None,  # softmax_scale
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse=return_lse,
        return_attn_probs=return_attn_probs,
        how_v3_bf16_cvt=2,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        num_rotate_args=1,
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
        return out, softmax_lse, dropout_mask, us_fwd
    elif bias is not None:
        (dq, dk, dv, dbias), us_bwd = run_perftest(
            torch.autograd.grad,
            out,
            (q, k, v, bias),
            dout,
            retain_graph=True,
            num_rotate_args=1,
        )
        return out, softmax_lse, dropout_mask, dq, dk, dv, dbias, (us_fwd, us_bwd)
    else:
        (dq, dk, dv), us_bwd = run_perftest(
            torch.autograd.grad,
            out,
            (q, k, v),
            dout,
            retain_graph=True,
            num_rotate_args=1,
        )
        return out, softmax_lse, dropout_mask, dq, dk, dv, None, (us_fwd, us_bwd)


@pytest.mark.parametrize("input_layout", ["BSHD", "BHSD", "SBHD", "KVPACKED"])
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
        (192, 128),
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
    dtype,
    input_layout,
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

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        q,
        k,
        v,
        _,
        _,
        _,
    ) = generate_qkv(
        q,
        k,
        v,
        None,
        None,
        kvpacked=(input_layout == "KVPACKED"),
        qkvpacked=(input_layout == "QKVPACKED"),
        input_layout=input_layout,
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

    out, softmax_lse, dropout_mask, dq, dk, dv, dbias, (us_fwd, us_bwd) = run_ck(
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

    out_ref, softmax_lse_ref, dq_ref, dk_ref, dv_ref, dbias_ref = run_torch(
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

    out_pt, softmax_lse_pt, dq_pt, dk_pt, dv_pt, dbias_pt = run_torch(
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

    print(f"softmax_lse max diff: {(softmax_lse - softmax_lse_ref).abs().max().item()}")
    print(
        f"softmax_lse Pytorch max diff: {(softmax_lse_pt - softmax_lse_ref).abs().max().item()}"
    )
    softmax_lse_tol = max(
        2 * (softmax_lse_pt - softmax_lse_ref).abs().max().item(), 0.01
    )
    # assert (softmax_lse - softmax_lse_ref).abs().max().item() <= softmax_lse_tol

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

    fwd_flop = (
        batch_size
        * nheads
        * (seqlen_q * seqlen_k * d * 2 + seqlen_q * seqlen_k * d_v * 2)
    )
    dtype_bytes = torch.finfo(dtype).bits // 8
    fwd_num_bytes = (
        batch_size
        * nheads
        * dtype_bytes
        * (seqlen_q * d + seqlen_k * d + seqlen_k * d_v + seqlen_q * d_v)
    )
    bwd_flop = (
        batch_size
        * nheads
        * (seqlen_q * seqlen_k * d * 2 * 3 + seqlen_q * seqlen_k * d_v * 2 * 2)
    )
    bwd_num_bytes = (
        2 * fwd_num_bytes
        + batch_size * nheads * (torch.finfo(torch.float).bits // 8) * seqlen_q
    )
    ret = {}
    ret["fwd_us"] = us_fwd
    ret["fwd_tflops"] = (fwd_flop) / 1.0e6 / us_fwd
    ret["fwd_gb_per_sec"] = (fwd_num_bytes) / 1.0e3 / us_fwd
    ret["bwd_us"] = us_bwd
    ret["bwd_tflops"] = (bwd_flop) / 1.0e6 / us_bwd
    ret["bwd_gb_per_sec"] = (bwd_num_bytes) / 1.0e3 / us_bwd
    return ret


@benchmark()
def flash_attn_output_benchmark(
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
    input_layout,
):
    return test_flash_attn_output(
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
        input_layout,
    )


@pytest.mark.parametrize(
    "padding_scenario",
    ["mixed", "q_only", "k_only", "no_padding", "q_len_1", "k_len_1"],
)
@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("bias_type", ["no"])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dropout_p", [0.0])  # Keep dropout 0 for padding test clarity
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("nheads", [6])
@pytest.mark.parametrize(
    "d,d_v",
    [
        (32, 32),
        (40, 40),
        (59, 59),
        (64, 64),
        # (96, 96), # Skip (96, 96) cases due to a known issue in CK.
        (111, 111),
        (128, 128),
        (160, 160),
        (192, 192),
        (224, 224),
        (256, 256),
        (192, 128),
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
def test_flash_attn_seq_padding(
    padding_scenario,
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

    if bias_type == "bias":
        pytest.skip("Padding test does not include elementwise bias.")

    # Test forward pass only
    return_lse = True
    return_attn_probs = True

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=False
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=False,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d_v,
        device="cuda",
        dtype=dtype,
        requires_grad=False,
    )

    # 1. Generate padding masks and cu_seqlens based on padding_type
    # The convention for padding masks in attention_ref is True = valid data, False = padded
    q_seqlens = [seqlen_q] * batch_size
    k_seqlens = [seqlen_k] * batch_size

    if padding_scenario == "q_only":
        for i in range(batch_size // 2):
            q_seqlens[i] = seqlen_q // 2
    elif padding_scenario == "k_only":
        for i in range(batch_size // 2):
            k_seqlens[i] = seqlen_k // 2
    elif padding_scenario == "mixed":  # was "q_and_k"
        for i in range(batch_size // 2):
            q_seqlens[i] = seqlen_q // 2
            k_seqlens[i] = seqlen_k // 2
    elif padding_scenario == "no_padding":
        pass  # lengths remain full
    elif padding_scenario == "q_len_1":
        q_seqlens = [1] * batch_size
    elif padding_scenario == "k_len_1":
        k_seqlens = [1] * batch_size

    query_padding_mask = (
        torch.arange(seqlen_q, device="cuda")[None, :]
        < torch.tensor(q_seqlens, device="cuda")[:, None]
    )
    key_padding_mask = (
        torch.arange(seqlen_k, device="cuda")[None, :]
        < torch.tensor(k_seqlens, device="cuda")[:, None]
    )

    q_seqlens_tensor = torch.tensor(q_seqlens, dtype=torch.int32, device="cuda")
    k_seqlens_tensor = torch.tensor(k_seqlens, dtype=torch.int32, device="cuda")

    cu_seqlens_q = torch.nn.functional.pad(
        q_seqlens_tensor.cumsum(0, dtype=torch.int32), (1, 0)
    )
    cu_seqlens_kv = torch.nn.functional.pad(
        k_seqlens_tensor.cumsum(0, dtype=torch.int32), (1, 0)
    )

    alibi_slopes = None
    if bias_type == "alibi":
        alibi_slopes = torch.rand(batch_size, nheads, device="cuda", dtype=dtypes.fp32)

    # 2. Run CK with cu_seqlens (forward pass only)
    out, _, _, _ = run_ck(
        q,
        k,
        v,
        None,
        alibi_slopes,
        None,
        dropout_p,
        causal,
        window_size,
        deterministic,
        return_lse,
        return_attn_probs,
        cu_seqlens_q,
        cu_seqlens_kv,
    )

    # 3. Run Torch with padding_mask (forward pass only)
    out_ref, _ = run_torch(
        q,
        k,
        v,
        None,
        alibi_slopes,
        None,
        dropout_p,
        None,
        causal,
        window_size,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
    )

    out_pt, _ = run_torch(
        q,
        k,
        v,
        None,
        alibi_slopes,
        None,
        dropout_p,
        None,
        causal,
        window_size,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        upcast=False,
    )

    # Mask the output for correct comparison
    output_mask = torch.zeros_like(out, dtype=torch.bool)
    for i in range(batch_size):
        output_mask[i, q_seqlens[i] :, :, :] = True

    out_masked = out.masked_fill(output_mask, 0.0)
    out_ref_masked = out_ref.masked_fill(output_mask, 0.0)
    out_pt_masked = out_pt.masked_fill(output_mask, 0.0)

    print(
        f"\nPadding Test ({padding_scenario}) | Output max diff: {(out_masked - out_ref_masked).abs().max().detach().item()}"
    )

    # Add visualization for debugging
    print("--- Debugging Output Mismatch ---")
    # Print a small slice of the first sequence, first head
    print("Aiter output slice:\n", out_masked[0, :5, 0, :5])
    print("Torch ref output slice:\n", out_ref_masked[0, :5, 0, :5])
    print("Difference slice:\n", (out_masked - out_ref_masked).abs()[0, :5, 0, :5])
    print("---------------------------------")

    # --- Begin Error Location Analysis ---
    diff_tensor = (out_masked - out_ref_masked).abs()
    max_diff_val = diff_tensor.max().item()

    print(f"\nMax difference value is: {max_diff_val}")

    # Find and print coordinates of max difference
    max_diff_indices = torch.unravel_index(torch.argmax(diff_tensor), diff_tensor.shape)
    b, s_q, h, d_idx = max_diff_indices
    print(
        f"Coordinates of max difference (batch, seq_q, head, dim): {tuple(x.item() for x in max_diff_indices)}"
    )
    # Check the padding status at this specific query position
    is_q_padded = not query_padding_mask[b, s_q].item()
    print(
        f"Is the query token at position {s_q} in batch {b} a padded token? {'Yes' if is_q_padded else 'No'}, actual length: {q_seqlens[b]}"
    )

    # Also check the original values at the point of maximum difference
    print(f"Value at aiter_out at max_diff_coords: {out_masked[max_diff_indices]}")
    print(f"Value at torch_ref at max_diff_coords: {out_ref_masked[max_diff_indices]}")
    # --- End Error Location Analysis ---

    print(f"Output max diff: {(out_masked - out_ref_masked).abs().max().item()}")
    print(
        f"Output Pytorch max diff: {(out_pt_masked - out_ref_masked).abs().max().item()}"
    )
    out_tol = max(2 * (out_pt_masked - out_ref_masked).abs().max().item(), 0.01)
    diff = (out_masked - out_ref_masked).abs().max().item()
    assert diff <= out_tol


l_causal = [False, True]
l_local = [False, True]
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
    default=512,
    help="""Sequence length for query. Default is 512.
    e.g.: -q 1024""",
)
parser.add_argument(
    "-k",
    "--seqlen_k",
    type=int,
    default=512,
    help="""Sequence length for key. Default is 512.
    e.g.: -k 1024""",
)
parser.add_argument(
    "-d_qk_v",
    type=dtypes.str2tuple,
    nargs="+",
    default=[
        (32, 32),
        (40, 40),
        (64, 64),
        (111, 111),
        (128, 128),
        (160, 160),
        (192, 128),
    ],
    help="""Dimension of query and key. Default is None.
    e.g.: -qk_v 256,256""",
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
    nargs="+",
    choices=["mha", "mqa", "gqa"],
    default=["mha", "mqa", "gqa"],
    help="""Type of multi-head attention.
    e.g.: -m mha""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    nargs="+",
    choices=["bf16", "fp16"],
    default=["bf16", "fp16"],
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-i",
    "--input_layout",
    type=str,
    choices=["BSHD", "BHSD", "SBHD", "QKVPACKED", "KVPACKED"],
    default="BSHD",
    help="""input_layout.
    e.g.: -i BSHD""",
)
if __name__ == "__main__":
    args = parser.parse_args()

    if args.causal is not None:
        l_causal = [args.causal]

    if args.local is not None:
        l_local = [args.local]

    if args.deterministic is not None:
        l_deterministic = [args.deterministic]

    collected = []
    for (
        dtype,
        (dim_qk, dim_v),
        mha_type,
        causal,
        local,
        deterministic,
    ) in itertools.product(
        args.dtype, args.d_qk_v, args.mha_type, l_causal, l_local, l_deterministic
    ):
        ret = flash_attn_output_benchmark(
            args.batch_size,
            args.nheads,
            args.seqlen_q,
            args.seqlen_k,
            dim_qk,
            dim_v,
            args.dropout_p,
            causal,
            local,
            args.bias_type,
            deterministic,
            mha_type,
            dtypes.d_dtypes[dtype],
            args.input_layout,
        )
        collected.append(ret)
        test_flash_attn_seq_padding(
            "mixed",
            args.batch_size,
            args.nheads,
            args.seqlen_q,
            args.seqlen_k,
            dim_qk,
            dim_v,
            args.dropout_p,
            causal,
            local,
            args.bias_type if args.bias_type != "bias" else "no",
            deterministic,
            mha_type,
            dtypes.d_dtypes[dtype],
        )

    df = pd.DataFrame(collected)
    aiter.logger.info(f"mha summary:\n{df}")
