# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha import fmha_fwd_bf16_opus_fwd
from aiter.test_common import benchmark, run_perftest
from aiter.test_mha_common import (
    attention_ref,
    attn_bias_from_alibi_slopes,
    ck_randval_to_dropout_mask,
    convert_flash_attn_S_to_softmax,
    generate_qkv,
    opus_check_lse,
    opus_ref_lse,
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
    _, seqlen_q, _, _ = q.shape
    _, seqlen_k, _, _ = k.shape

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
    num_splits=0,
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
        num_splits=num_splits,
        num_rotate_args=1,
    )

    if dropout_p > 0.0:
        _, seqlen_q, _, d = q.shape
        _, seqlen_k, _, d = k.shape
        _, seqlen_k, _, _d_v = v.shape
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
@pytest.mark.parametrize("gqa_ratio", [1, 8])
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
    gqa_ratio,
    dtype,
    input_layout,
    num_splits=0,
):
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()
    assert nheads % gqa_ratio == 0
    nheads_k = nheads // gqa_ratio
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
        num_splits=num_splits,
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
    max(2 * (softmax_lse_pt - softmax_lse_ref).abs().max().item(), 0.01)
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
    fwd_flop = fwd_flop / 2 if causal else fwd_flop
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
    bwd_flop = bwd_flop / 2 if causal else bwd_flop
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
    gqa_ratio,
    dtype,
    input_layout,
    num_splits=0,
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
        gqa_ratio,
        dtype,
        input_layout,
        num_splits=num_splits,
    )


@pytest.mark.parametrize(
    "padding_scenario",
    ["mixed", "q_only", "k_only", "no_padding", "q_len_1", "k_len_1"],
)
@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16])
@pytest.mark.parametrize("gqa_ratio", [1, 8])
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
    gqa_ratio,
    dtype,
):

    torch.random.manual_seed(0)
    torch.cuda.empty_cache()
    assert nheads % gqa_ratio == 0
    nheads_k = nheads // gqa_ratio
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
    b, s_q, _h, _d_idx = max_diff_indices
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
    default=16,
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
    type=dtypes.str2bool,
    nargs="*",
    default=[False, True],
    help="""Causal attention. Default is [False, True].
    e.g. -c true # enable causal attention
         -c false # disable causal attention""",
)
parser.add_argument(
    "-l",
    "--local",
    type=dtypes.str2bool,
    nargs="*",
    default=[False, True],
    help="""Local attention. Default is [False, True].
        e.g. -l true # enable local attention
             -l false # disable local attention""",
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
    type=dtypes.str2bool,
    nargs="*",
    default=[False, True],
    help="""Deterministic attention. Default is [False, True].
    e.g. -det true # enable deterministic attention
         -det false # disable deterministic attention""",
)
parser.add_argument(
    "-gr",
    "--gqa_ratio",
    type=int,
    nargs="+",
    choices=[1, 8],
    default=[1, 8],
    help="""gqa ratio.
    e.g.: -gr 8""",
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
parser.add_argument(
    "-ns",
    "--num_splits",
    type=int,
    default=0,
    help="native split-K num_splits (0=auto/heuristic, 1=disable split-K, >=2 forces native if capable)",
)
if __name__ == "__main__":
    args = parser.parse_args()

    collected = []
    for (
        dtype,
        (dim_qk, dim_v),
        gqa_ratio,
        causal,
        local,
        deterministic,
    ) in itertools.product(
        args.dtype,
        args.d_qk_v,
        args.gqa_ratio,
        args.causal,
        args.local,
        args.deterministic,
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
            gqa_ratio,
            dtypes.d_dtypes[dtype],
            args.input_layout,
            args.num_splits,
        )
        collected.append(ret)
        # test_flash_attn_seq_padding(
        #     "mixed",
        #     args.batch_size,
        #     args.nheads,
        #     args.seqlen_q,
        #     args.seqlen_k,
        #     dim_qk,
        #     dim_v,
        #     args.dropout_p,
        #     causal,
        #     local,
        #     args.bias_type if args.bias_type != "bias" else "no",
        #     deterministic,
        #     gqa_ratio,
        #     dtypes.d_dtypes[dtype],
        # )

    df = pd.DataFrame(collected)
    aiter.logger.info(f"mha summary:\n{df}")


# ---------------------------------------------------------------------------
# Sink backward tests (mha_bwd with sink / d_sink)
#
# Reference formula (derived from kernel block_fmha_bwd_dot_do_o.hpp):
#   D[b, h, q]      = sum_j(dout[b, q, h, j] * out[b, q, h, j]) * p_undrop
#   P_sink[b, h, q] = exp(sink[b, h] - lse_fwd[b, h, q])
#   d_sink[h]       = sum_{b, q} (-P_sink[b, h, q] * D[b, h, q])
# ---------------------------------------------------------------------------


def _sink_make_qkvo(
    batch, seqlen_q, seqlen_k, nhead, nhead_k, hdim, hdim_v, dtype, device
):
    """Return (q, k, v, dout) in BSHD layout, requires_grad=True."""
    q = torch.randn(
        batch, seqlen_q, nhead, hdim, device=device, dtype=dtype
    ).requires_grad_(True)
    k = torch.randn(
        batch, seqlen_k, nhead_k, hdim, device=device, dtype=dtype
    ).requires_grad_(True)
    v = torch.randn(
        batch, seqlen_k, nhead_k, hdim_v, device=device, dtype=dtype
    ).requires_grad_(True)
    dout = torch.randn(batch, seqlen_q, nhead, hdim_v, device=device, dtype=dtype)
    return q, k, v, dout


def _sink_run_fwd(q, k, v, softmax_scale, causal):
    """Run mha_fwd and return (out, lse)."""
    out, lse, _, _ = aiter.mha_fwd(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=-1,
        window_size_right=0 if causal else -1,
        sink_size=0,
        return_softmax_lse=True,
        return_dropout_randval=False,
    )
    return out, lse


def _sink_reference_d_sink(dout, out, lse, sink, p_undrop=1.0):
    """
    Pure-PyTorch reference for d_sink.

    dout : [B, Sq, H, Dv]
    out  : [B, Sq, H, Dv]
    lse  : [B, H, Sq]       (forward LSE without sink)
    sink : [B, H]
    returns d_sink : [H]
    """
    D_bsh = (dout.float() * out.float()).sum(dim=-1) * p_undrop  # [B, Sq, H]
    D_bhs = D_bsh.permute(0, 2, 1)  # [B, H, Sq]
    sink_bhs = sink.unsqueeze(-1)  # [B, H, 1]
    p_sink = torch.exp(sink_bhs.float() - lse.float())  # [B, H, Sq]
    d_sink = (-p_sink * D_bhs).sum(dim=(0, 2))  # [H]
    return d_sink.float()


_SINK_DTYPES = [dtypes.fp16, dtypes.bf16]
_SINK_CAUSALS = [False, True]
_SINK_CONFIGS = [
    # (batch, seqlen_q, seqlen_k, nhead, nhead_k, hdim)
    (2, 128, 128, 4, 4, 64),
    (1, 64, 64, 6, 2, 128),
]


@pytest.mark.parametrize("causal", _SINK_CAUSALS)
@pytest.mark.parametrize("dtype", _SINK_DTYPES)
@pytest.mark.parametrize("batch,seqlen_q,seqlen_k,nhead,nhead_k,hdim", _SINK_CONFIGS)
def test_mha_bwd_sink_dsink(
    batch, seqlen_q, seqlen_k, nhead, nhead_k, hdim, dtype, causal
):
    """Verify that mha_bwd correctly accumulates d_sink."""
    device = torch.device("cuda")
    hdim_v = hdim
    softmax_scale = hdim**-0.5

    q, k, v, dout = _sink_make_qkvo(
        batch, seqlen_q, seqlen_k, nhead, nhead_k, hdim, hdim_v, dtype, device
    )
    out, lse = _sink_run_fwd(q.detach(), k.detach(), v.detach(), softmax_scale, causal)

    sink = torch.empty(batch, nhead, device=device, dtype=torch.float32).uniform_(
        30.0, 60.0
    )
    d_sink = torch.zeros(nhead, device=device, dtype=torch.float32)

    _dq, _dk, _dv, _softmax_d = aiter.mha_bwd(
        dout,
        q.detach(),
        k.detach(),
        v.detach(),
        out,
        lse,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=-1,
        window_size_right=0 if causal else -1,
        deterministic=False,
        sink=sink,
        d_sink=d_sink,
    )

    assert d_sink.abs().max() > 0, "d_sink was not updated by mha_bwd"

    d_sink_ref = _sink_reference_d_sink(dout, out, lse, sink)
    torch.testing.assert_close(
        d_sink,
        d_sink_ref,
        rtol=0.02,
        atol=0.5,
        msg=f"d_sink mismatch for dtype={dtype}, causal={causal}, B={batch}, Sq={seqlen_q}, H={nhead}",
    )


@pytest.mark.parametrize("causal", _SINK_CAUSALS)
@pytest.mark.parametrize("dtype", _SINK_DTYPES)
@pytest.mark.parametrize("batch,seqlen_q,seqlen_k,nhead,nhead_k,hdim", _SINK_CONFIGS)
def test_mha_bwd_with_sink_dq_dk_dv(
    batch, seqlen_q, seqlen_k, nhead, nhead_k, hdim, dtype, causal
):
    """Verify that passing sink/d_sink does not corrupt the dQ, dK, dV outputs."""
    device = torch.device("cuda")
    hdim_v = hdim
    softmax_scale = hdim**-0.5

    q, k, v, dout = _sink_make_qkvo(
        batch, seqlen_q, seqlen_k, nhead, nhead_k, hdim, hdim_v, dtype, device
    )
    out, lse = _sink_run_fwd(q.detach(), k.detach(), v.detach(), softmax_scale, causal)

    common_bwd_args = {
        "dropout_p": 0.0,
        "softmax_scale": softmax_scale,
        "is_causal": causal,
        "window_size_left": -1,
        "window_size_right": 0 if causal else -1,
        "deterministic": False,
    }

    dq_base, dk_base, dv_base, _ = aiter.mha_bwd(
        dout, q.detach(), k.detach(), v.detach(), out, lse, **common_bwd_args
    )

    sink_small = torch.full((batch, nhead), -1000.0, device=device, dtype=torch.float32)
    d_sink = torch.zeros(nhead, device=device, dtype=torch.float32)

    dq_sink, dk_sink, dv_sink, _ = aiter.mha_bwd(
        dout,
        q.detach(),
        k.detach(),
        v.detach(),
        out,
        lse,
        **common_bwd_args,
        sink=sink_small,
        d_sink=d_sink,
    )

    rtol, atol = (0.01, 0.01) if dtype == dtypes.fp16 else (0.02, 0.02)
    torch.testing.assert_close(
        dq_sink, dq_base, rtol=rtol, atol=atol, msg="dQ mismatch with small sink"
    )
    torch.testing.assert_close(
        dk_sink, dk_base, rtol=rtol, atol=atol, msg="dK mismatch with small sink"
    )
    torch.testing.assert_close(
        dv_sink, dv_base, rtol=rtol, atol=atol, msg="dV mismatch with small sink"
    )


@pytest.mark.parametrize("dtype", _SINK_DTYPES)
def test_mha_bwd_sink_null_gives_same_as_no_sink(dtype):
    """Passing sink=None must give identical output to omitting sink entirely."""
    device = torch.device("cuda")
    batch, seqlen, nhead, hdim = 2, 64, 4, 64
    softmax_scale = hdim**-0.5

    q, k, v, dout = _sink_make_qkvo(
        batch, seqlen, seqlen, nhead, nhead, hdim, hdim, dtype, device
    )
    out, lse = _sink_run_fwd(q.detach(), k.detach(), v.detach(), softmax_scale, False)

    common = {
        "dropout_p": 0.0,
        "softmax_scale": softmax_scale,
        "is_causal": False,
        "window_size_left": -1,
        "window_size_right": -1,
        "deterministic": False,
    }

    dq1, dk1, dv1, d1 = aiter.mha_bwd(
        dout, q.detach(), k.detach(), v.detach(), out, lse, **common
    )
    dq2, dk2, dv2, d2 = aiter.mha_bwd(
        dout,
        q.detach(),
        k.detach(),
        v.detach(),
        out,
        lse,
        **common,
        sink=None,
        d_sink=None,
    )

    torch.testing.assert_close(dq1, dq2, msg="dQ differs with sink=None vs omitted")
    torch.testing.assert_close(dk1, dk2, msg="dK differs with sink=None vs omitted")
    torch.testing.assert_close(dv1, dv2, msg="dV differs with sink=None vs omitted")
    torch.testing.assert_close(
        d1, d2, msg="softmax_d differs with sink=None vs omitted"
    )


# OPUS gfx950 dense (batch) cases, shared by the D=128 and D_QK=192/D_V=128 tests.
# Causal is bottom-right aligned, so seqlen_q > seqlen_kv gives O=0 / LSE=-inf rows.
#   (batch, seqlen_q, seqlen_kv, nheads, nheads_k)
_OPUS_BATCH_CASES = [
    (2, 64, 64, 8, 2),  # 1 KV tile, GQA
    (2, 128, 128, 16, 1),  # 2 KV tiles, MQA
    (2, 100, 100, 8, 2),  # partial last KV tile, GQA
    (2, 129, 129, 32, 8),  # pipelined odd tile count, GQA
    (1, 256, 256, 16, 16),  # exactly one Q block, MHA
    (2, 512, 512, 8, 1),  # multiple Q blocks, MQA
    (2, 1023, 1023, 16, 4),  # partial odd, GQA
    (1, 4096, 4096, 8, 2),  # large, GQA
    (4, 256, 256, 8, 8),  # larger batch, MHA
    (2, 128, 512, 8, 2),  # cross sq < sk, GQA
    (2, 512, 128, 8, 2),  # cross sq > sk, GQA
    (1, 300, 700, 4, 4),  # cross, partial both sides, MHA
    (2, 256, 65, 16, 4),  # cross, sk under one KV tile, GQA
    (2, 1, 1024, 8, 8),  # single query row, MHA
    (2, 1024, 640, 64, 8),  # 4*64*2 = 512 -> causal head/tail merge
    (2, 300, 0, 8, 2),  # no keys at all -> zero KV tiles, every row fully masked
]

_OPUS_BATCH_IDS = [
    f"b{b}_sq{sq}_sk{sk}_h{h}_hkv{hk}" for (b, sq, sk, h, hk) in _OPUS_BATCH_CASES
]


def _opus_sink(nheads, enabled):
    """One learned fp32 logit per query head, or None.

    Swept over the heads so the "sink dominates every score" branch and the
    max-reference path are both exercised within a single case.
    """
    if not enabled:
        return None
    return torch.linspace(-4, 12, nheads, device="cuda", dtype=torch.float32)


def _run_opus_batch_case(
    batch_size,
    seqlen_q,
    seqlen_kv,
    nheads,
    nheads_k,
    d_qk,
    d_v,
    causal,
    with_sink=False,
):
    """Shared body: flash_attn_func with LSE, assert it routed to OPUS, check results.

    with_sink adds one learned logit per query head. A sink joins the softmax
    denominator only, so O is unchanged for rows that see keys, fully-masked rows
    still produce O=0, and their LSE collapses to the sink logit itself.
    """
    torch.manual_seed(0)
    q = torch.randn(
        batch_size, seqlen_q, nheads, d_qk, device="cuda", dtype=dtypes.bf16
    )
    k = torch.randn(
        batch_size, seqlen_kv, nheads_k, d_qk, device="cuda", dtype=dtypes.bf16
    )
    v = torch.randn(
        batch_size, seqlen_kv, nheads_k, d_v, device="cuda", dtype=dtypes.bf16
    )
    sink = _opus_sink(nheads, with_sink)

    with torch.no_grad():
        out, lse = aiter.flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=None,  # -> default 1/sqrt(d_qk), matches attention_ref
            causal=causal,
            window_size=(-1, -1),
            return_lse=True,
            return_attn_probs=False,
            sink_ptr=sink,
        )
        # The dispatch chain tries several backends before OPUS; without this the
        # checks below could be validating a different kernel.
        out_opus = fmha_fwd_bf16_opus_fwd(
            q, k, v, softmax_scale=d_qk**-0.5, causal=causal, sink=sink
        )
    assert torch.equal(
        out, out_opus
    ), f"flash_attn_func did not route to the opus d{d_qk} kernel"

    tag = f"opus-d{d_qk}" + ("-sink" if with_sink else "")
    out_ref, _, _ = attention_ref(q, k, v, causal=causal, sink=sink)
    out_pt, _, _ = attention_ref(
        q, k, v, causal=causal, sink=sink, upcast=False, reorder_ops=True
    )
    out_tol = max(2 * (out_pt - out_ref).abs().max().item(), 0.01)
    out_diff = (out - out_ref).abs().max().item()
    print(f"[{tag}] out max diff: {out_diff} tol={out_tol}")
    assert out_diff <= out_tol

    # A sink does not create keys, so the no-key (-inf) rows come from the
    # sink-free LSE. With a sink the denominator gains one column, which is
    # exactly logaddexp(no-sink LSE, sink); for a fully-masked row that collapses
    # to the sink logit, matching the kernel's "attend only the sink" result.
    no_key = opus_ref_lse(q, k, causal)
    lse_ref = torch.logaddexp(no_key, sink.view(1, nheads, 1)) if with_sink else no_key
    assert tuple(lse.shape) == (batch_size, nheads, seqlen_q), f"lse {tuple(lse.shape)}"
    opus_check_lse(tag, lse, lse_ref)

    dead = torch.isneginf(no_key)  # rows that see no keys -> O must be exactly 0
    if dead.any():
        dead_o = dead.permute(0, 2, 1).unsqueeze(-1).expand_as(out)
        assert (out[dead_o] == 0).all(), f"{tag}: fully-masked rows must produce O=0"
        print(f"[{tag}] fully-masked rows: {int(dead.sum())}")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "batch_size,seqlen_q,seqlen_kv,nheads,nheads_k",
    _OPUS_BATCH_CASES,
    ids=_OPUS_BATCH_IDS,
)
def test_flash_attn_func_opus(
    batch_size, seqlen_q, seqlen_kv, nheads, nheads_k, causal, monkeypatch
):
    """OPUS gfx950 dense D=128 forward through flash_attn_func, with LSE.

    Env-gated (AITER_ENABLE_FMHA_OPUS=1); monkeypatched scoped to this test so the
    other cases keep exercising the default v3/CK dispatch.
    """
    if get_gfx() != "gfx950":
        pytest.skip("opus D=128 kernel requires gfx950")
    monkeypatch.setenv("AITER_ENABLE_FMHA_OPUS", "1")
    _run_opus_batch_case(
        batch_size, seqlen_q, seqlen_kv, nheads, nheads_k, 128, 128, causal
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "batch_size,seqlen_q,seqlen_kv,nheads,nheads_k",
    _OPUS_BATCH_CASES,
    ids=_OPUS_BATCH_IDS,
)
def test_flash_attn_func_opus_d192_v128(
    batch_size, seqlen_q, seqlen_kv, nheads, nheads_k, causal
):
    """OPUS gfx950 dense D_QK=192/D_V=128 forward through flash_attn_func, with LSE.

    Enabled by default (no env), so no monkeypatch.
    """
    if get_gfx() != "gfx950":
        pytest.skip("opus D=192 kernel requires gfx950")
    _run_opus_batch_case(
        batch_size, seqlen_q, seqlen_kv, nheads, nheads_k, 192, 128, causal
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "batch_size,seqlen_q,seqlen_kv,nheads,nheads_k",
    _OPUS_BATCH_CASES,
    ids=_OPUS_BATCH_IDS,
)
def test_flash_attn_func_opus_d64(
    batch_size, seqlen_q, seqlen_kv, nheads, nheads_k, causal, monkeypatch
):
    """OPUS gfx950 dense D=64 forward through flash_attn_func, with LSE.

    The symmetric kernel is traits-parameterised on D; this is the same coverage
    as the D=128 test one head dim down. Env-gated like the D=128 case.
    """
    if get_gfx() != "gfx950":
        pytest.skip("opus D=64 kernel requires gfx950")
    monkeypatch.setenv("AITER_ENABLE_FMHA_OPUS", "1")
    _run_opus_batch_case(
        batch_size, seqlen_q, seqlen_kv, nheads, nheads_k, 64, 64, causal
    )


# Sink correctness: a representative subset, including a no-key case so the
# "fully masked -> attend only the sink" branch is exercised.
_OPUS_SINK_CASES = [
    (2, 64, 64, 8, 2),  # GQA
    (2, 128, 128, 16, 1),  # MQA
    (1, 256, 256, 8, 8),  # MHA
    (2, 300, 0, 8, 2),  # no keys -> every row attends only the sink
]
_OPUS_SINK_IDS = [
    f"b{b}_sq{sq}_sk{sk}_h{h}_hkv{hk}" for (b, sq, sk, h, hk) in _OPUS_SINK_CASES
]


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "batch_size,seqlen_q,seqlen_kv,nheads,nheads_k",
    _OPUS_SINK_CASES,
    ids=_OPUS_SINK_IDS,
)
def test_flash_attn_func_opus_sink(
    batch_size, seqlen_q, seqlen_kv, nheads, nheads_k, causal, d, monkeypatch
):
    """OPUS symmetric kernel with a per-head attention sink (D in {64, 128}).

    Checks O against attention_ref and the fp32 LSE against logaddexp(no-sink LSE,
    sink) -- the LSE is the sharp check since the sink lives in the denominator.
    """
    if get_gfx() != "gfx950":
        pytest.skip("opus symmetric kernel requires gfx950")
    monkeypatch.setenv("AITER_ENABLE_FMHA_OPUS", "1")
    _run_opus_batch_case(
        batch_size, seqlen_q, seqlen_kv, nheads, nheads_k, d, d, causal, with_sink=True
    )


# Single-sequence shapes for the varlen router (batch is always 1 here).
#   (seqlen_q, seqlen_kv, nheads, nheads_k)
_OPUS_VARLEN_CASES = [
    (64, 64, 8, 2),  # 1 KV tile, GQA
    (128, 512, 16, 1),  # cross sq < sk, MQA
    (512, 128, 8, 8),  # cross sq > sk, MHA
    (4096, 1024, 8, 2),  # long-ish prefill, GQA
]
_OPUS_VARLEN_IDS = [
    f"sq{sq}_sk{sk}_h{h}_hkv{hk}" for (sq, sk, h, hk) in _OPUS_VARLEN_CASES
]


def _run_opus_single_seq_varlen(
    seqlen_q, seqlen_kv, nheads, nheads_k, d, causal, with_sink
):
    """A one-sequence varlen batch must route to the dense OPUS kernel and produce
    output bit-identical to sending that sequence straight to the dense entry."""
    torch.manual_seed(0)
    q = torch.randn(seqlen_q, nheads, d, device="cuda", dtype=dtypes.bf16)
    k = torch.randn(seqlen_kv, nheads_k, d, device="cuda", dtype=dtypes.bf16)
    v = torch.randn(seqlen_kv, nheads_k, d, device="cuda", dtype=dtypes.bf16)
    sink = _opus_sink(nheads, with_sink)
    cu_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, seqlen_kv], dtype=torch.int32, device="cuda")

    with torch.no_grad():
        out, lse = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            seqlen_q,
            seqlen_kv,
            causal=causal,
            return_lse=True,
            sink_ptr=sink,
        )
        # The same single sequence sent straight to the dense OPUS op.
        out_dense = fmha_fwd_bf16_opus_fwd(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            softmax_scale=d**-0.5,
            causal=causal,
            sink=sink,
        )
    tag = f"opus-varlen-d{d}" + ("-sink" if with_sink else "")
    assert tuple(out.shape) == (
        seqlen_q,
        nheads,
        d,
    ), f"{tag}: out shape {tuple(out.shape)}"
    assert tuple(lse.shape) == (
        nheads,
        seqlen_q,
    ), f"{tag}: lse shape {tuple(lse.shape)}"
    # Routing + view correctness in one check: varlen must equal the dense path.
    assert torch.equal(
        out, out_dense.squeeze(0)
    ), f"{tag}: single-seq varlen did not route to the dense OPUS kernel"


@pytest.mark.parametrize("with_sink", [False, True])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_kv,nheads,nheads_k",
    _OPUS_VARLEN_CASES,
    ids=_OPUS_VARLEN_IDS,
)
def test_flash_attn_func_opus_varlen_single_seq(
    seqlen_q, seqlen_kv, nheads, nheads_k, causal, d, with_sink, monkeypatch
):
    """flash_attn_varlen_func on a one-sequence batch routes to dense OPUS."""
    if get_gfx() != "gfx950":
        pytest.skip("opus symmetric kernel requires gfx950")
    monkeypatch.setenv("AITER_ENABLE_FMHA_OPUS", "1")
    _run_opus_single_seq_varlen(
        seqlen_q, seqlen_kv, nheads, nheads_k, d, causal, with_sink
    )


# (valid_q, valid_k, buf_q, buf_k): the K/V buffer is LONGER than the valid range,
# mirroring vLLM's fixed-size chunked-context workspace. Q is always exact (the router
# requires it), so buf_q == valid_q.
_OPUS_OVERSIZED_CASES = [
    (128, 300, 128, 512),  # K/V oversized (the vLLM chunked-context shape)
    (256, 512, 256, 1024),  # larger valid range, K/V oversized
]
_OPUS_OVERSIZED_IDS = [
    f"vq{vq}_vk{vk}_bq{bq}_bk{bk}" for (vq, vk, bq, bk) in _OPUS_OVERSIZED_CASES
]


def _run_opus_oversized_buffer(
    valid_q, valid_k, buf_q, buf_k, nheads, nheads_k, d, causal, with_sink
):
    """The single-seq router must attend only the valid prefix of an oversized
    buffer. The tail past the valid range is poisoned so that attending any stale
    row would move the result far outside tolerance."""
    torch.manual_seed(0)
    q = torch.randn(buf_q, nheads, d, device="cuda", dtype=dtypes.bf16)
    k = torch.randn(buf_k, nheads_k, d, device="cuda", dtype=dtypes.bf16)
    v = torch.randn(buf_k, nheads_k, d, device="cuda", dtype=dtypes.bf16)
    k[valid_k:] = 50.0  # poison the stale tail
    v[valid_k:] = -50.0
    sink = _opus_sink(nheads, with_sink)
    cu_q = torch.tensor([0, valid_q], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, valid_k], dtype=torch.int32, device="cuda")

    with torch.no_grad():
        out, _ = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            valid_q,
            valid_k,
            causal=causal,
            return_lse=True,
            sink_ptr=sink,
        )
        # Dense OPUS on the CLEAN valid prefix only. Equality proves both that the
        # call routed to OPUS and that the poisoned tail never entered the softmax.
        out_ref = fmha_fwd_bf16_opus_fwd(
            q[:valid_q].unsqueeze(0),
            k[:valid_k].unsqueeze(0),
            v[:valid_k].unsqueeze(0),
            softmax_scale=d**-0.5,
            causal=causal,
            sink=sink,
        )
    tag = f"opus-oversized-d{d}"
    assert tuple(out.shape) == (
        valid_q,
        nheads,
        d,
    ), f"{tag}: out shape {tuple(out.shape)}"
    assert torch.equal(
        out, out_ref.squeeze(0)
    ), f"{tag}: poisoned tail leaked into the output (or did not route to OPUS)"


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "valid_q,valid_k,buf_q,buf_k",
    _OPUS_OVERSIZED_CASES,
    ids=_OPUS_OVERSIZED_IDS,
)
def test_flash_attn_func_opus_oversized_buffer(
    valid_q, valid_k, buf_q, buf_k, causal, monkeypatch
):
    """An oversized single-seq packed buffer attends only its valid prefix."""
    if get_gfx() != "gfx950":
        pytest.skip("opus symmetric kernel requires gfx950")
    monkeypatch.setenv("AITER_ENABLE_FMHA_OPUS", "1")
    _run_opus_oversized_buffer(
        valid_q,
        valid_k,
        buf_q,
        buf_k,
        nheads=8,
        nheads_k=2,
        d=64,
        causal=causal,
        with_sink=True,
    )
