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
import argparse


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
    cu_seqlens_q_padded=None,
    cu_seqlens_k_padded=None,
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
    batch_size = q.shape[0]

    if bias is not None:
        # TODO - implement generate_bias() to unpad
        total_q = q_unpad.shape[0]
        assert total_q == batch_size * max_seqlen_q
        assert q.shape[1] == max_seqlen_q
        assert k.shape[1] == max_seqlen_k
        bias_unpad = bias.reshape(batch_size * max_seqlen_q, max_seqlen_k)
    else:
        bias_unpad = None

    outputs = aiter.flash_attn_varlen_func(
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
        cu_seqlens_q_padded=cu_seqlens_q_padded,
        cu_seqlens_k_padded=cu_seqlens_k_padded,
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

    if dout is None or not return_lse:
        return out, dropout_mask, None, None, None
    else:
        dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(
            out, (q_unpad, k_unpad, v_unpad), dout
        )
        dq = dq_pad_fn(dq_unpad)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        return out, dropout_mask, dq, dk, dv


def run_ck_seq_padding(
    q,
    k,
    v,
    q_actual_lens,
    k_actual_lens,
    q_padded_lens,
    k_padded_lens,
    deterministic=False,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
):
    """Run CK varlen forward with physically padded inputs."""

    device = q.device
    dtype = q.dtype
    batch_size = q.size(0)
    nheads = q.size(2)
    d = q.size(3)
    d_v = v.size(3)

    assert len(q_actual_lens) == batch_size
    assert len(k_actual_lens) == batch_size
    assert len(q_padded_lens) == batch_size
    assert len(k_padded_lens) == batch_size

    q_actual = torch.tensor(q_actual_lens, dtype=torch.int32, device=device)
    k_actual = torch.tensor(k_actual_lens, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.nn.functional.pad(
        q_actual.cumsum(0, dtype=torch.int32), (1, 0)
    )
    cu_seqlens_k = torch.nn.functional.pad(
        k_actual.cumsum(0, dtype=torch.int32), (1, 0)
    )

    q_padded = torch.tensor(q_padded_lens, dtype=torch.int32, device=device)
    k_padded = torch.tensor(k_padded_lens, dtype=torch.int32, device=device)
    cu_seqlens_q_padded = torch.nn.functional.pad(
        q_padded.cumsum(0, dtype=torch.int32), (1, 0)
    )
    cu_seqlens_k_padded = torch.nn.functional.pad(
        k_padded.cumsum(0, dtype=torch.int32), (1, 0)
    )

    def _flatten(tensor, padded_lens):
        pieces = []
        for i in range(batch_size):
            pieces.append(tensor[i, : padded_lens[i]])
        return torch.cat(pieces, dim=0)

    q_flat = _flatten(q, q_padded_lens)
    k_flat = _flatten(k, k_padded_lens)
    v_flat = _flatten(v, k_padded_lens)

    outputs = aiter.flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        cu_seqlens_q,
        cu_seqlens_k,
        max(q_actual_lens),
        max(k_actual_lens),
        dropout_p=0.0,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_lse=False,
        return_attn_probs=False,
        cu_seqlens_q_padded=cu_seqlens_q_padded,
        cu_seqlens_k_padded=cu_seqlens_k_padded,
    )

    out_flat = outputs[0] if isinstance(outputs, tuple) else outputs

    out_batches = []
    for i in range(batch_size):
        start = int(cu_seqlens_q_padded[i].item())
        end = int(cu_seqlens_q_padded[i + 1].item())
        keep = q_actual_lens[i]
        out_batch = torch.zeros(q.size(1), nheads, d_v, dtype=dtype, device=device)
        out_batch[:keep] = out_flat[start : start + keep]
        out_batches.append(out_batch)

    return torch.stack(out_batches, dim=0)


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

    out, dropout_mask, dq, dk, dv = run_ck(
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

    out_diff = (out - out_ref).abs().max().item()
    ref_diff = (out_pt - out_ref).abs().max().item()
    print(f"Output max diff: {out_diff}")
    print(f"Output Pytorch max diff: {ref_diff}")
    out_tol = max(4 * ref_diff, 0.01)
    assert out_diff <= out_tol, f"forward diff {out_diff} exceeds tolerance {out_tol}"

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


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize(
    "padding_scenario",
    ["mixed", "q_only", "k_only", "no_padding", "q_len_1", "k_len_1"],
)
@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16])
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
@pytest.mark.parametrize("local", [False, True])
def test_varlen_flash_attn_seq_padding(
    batch_size,
    mha_type,
    deterministic,
    padding_scenario,
    dtype,
    d,
    d_v,
    seqlen_q,
    seqlen_k,
    local,
):
    """End-to-end check that CK group-mode varlen path respects padded tokens."""
    torch.random.manual_seed(0)

    nheads = 9
    device = "cuda"

    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    if nheads % nheads_k != 0:
        pytest.skip("nheads must be divisible by nheads_k")

    # Dynamically generate padding configurations
    q_padded_lens = torch.randint(seqlen_q // 2, seqlen_q + 1, (batch_size,)).tolist()
    q_actual_lens = [
        torch.randint(max(1, l // 2), l + 1, (1,)).item() for l in q_padded_lens
    ]
    k_padded_lens = torch.randint(seqlen_k // 2, seqlen_k + 1, (batch_size,)).tolist()
    k_actual_lens = [
        torch.randint(max(1, l // 2), l + 1, (1,)).item() for l in k_padded_lens
    ]

    if padding_scenario == "q_only":
        k_actual_lens = k_padded_lens
    elif padding_scenario == "k_only":
        q_actual_lens = q_padded_lens
    elif padding_scenario == "no_padding":
        q_actual_lens = q_padded_lens
        k_actual_lens = k_padded_lens
    elif padding_scenario == "q_len_1":
        q_actual_lens = [1] * batch_size
    elif padding_scenario == "k_len_1":
        k_actual_lens = [1] * batch_size

    q_s = max(q_padded_lens)
    k_s = max(k_padded_lens)
    window_size = (-1, -1) if not local else torch.randint(0, k_s, (2,))

    q = torch.zeros(batch_size, q_s, nheads, d, device=device, dtype=dtype)
    k = torch.zeros(batch_size, k_s, nheads_k, d, device=device, dtype=dtype)
    v = torch.zeros(batch_size, k_s, nheads_k, d_v, device=device, dtype=dtype)

    for i in range(batch_size):
        q[i, : q_actual_lens[i]] = torch.randn(
            q_actual_lens[i], nheads, d, device=device, dtype=dtype
        )
        k[i, : k_actual_lens[i]] = torch.randn(
            k_actual_lens[i], nheads_k, d, device=device, dtype=dtype
        )
        v[i, : k_actual_lens[i]] = torch.randn(
            k_actual_lens[i], nheads_k, d_v, device=device, dtype=dtype
        )

    query_padding_mask = torch.arange(q_s, device=device).unsqueeze(0).expand(
        batch_size, -1
    ) < torch.tensor(q_actual_lens, device=device).unsqueeze(1)
    key_padding_mask = torch.arange(k_s, device=device).unsqueeze(0).expand(
        batch_size, -1
    ) < torch.tensor(k_actual_lens, device=device).unsqueeze(1)

    out_ck = run_ck_seq_padding(
        q,
        k,
        v,
        q_actual_lens,
        k_actual_lens,
        q_padded_lens,
        k_padded_lens,
        deterministic,
        causal=True,
        window_size=window_size,
    )

    out_ref = run_torch(
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
        causal=True,
        window_size=window_size,
    )

    out_pt = run_torch(
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
        causal=True,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )

    query_mask = (
        (
            torch.arange(q.shape[1], device=device).unsqueeze(0)
            < torch.tensor(q_actual_lens, device=device).unsqueeze(1)
        )
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

    out_ck_masked = out_ck.masked_fill(~query_mask, 0.0)
    out_ref_masked = out_ref.masked_fill(~query_mask, 0.0)
    out_pt_masked = out_pt.masked_fill(~query_mask, 0.0)

    out_diff = (out_ck_masked - out_ref_masked).abs().max().item()
    ref_diff = (out_pt_masked - out_ref_masked).abs().max().item()

    out_tol = max(4 * ref_diff, 0.01)

    print(
        f"\nGroup Mode Test (bs={batch_size}, {mha_type}, {padding_scenario}, {dtype}, local={local}) | Max diff: {out_diff} | Ref diff: {ref_diff} | Tol: {out_tol}"
    )
    assert out_diff <= out_tol


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
        default=4,
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
        default=128,
        help="""Dimension of query&key.
    e.g. -d 128""",
    )
    parser.add_argument(
        "-dv",
        type=int,
        nargs="?",
        default=128,
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
        default=True,
        help="""Causal attention, default is True.
    -c or --causal    # enable causal attention
    --no-causal       # disable causal attention""",
    )
    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        default=False,
        help="""Local attention. default is False.
    e.g. -l or --local    # enable local attention""",
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
        default=True,
        help="""Deterministic attention, default is True.
    -det or --deterministic    # enable deterministic attention
    --no-deterministic         # disable deterministic attention""",
    )
    parser.add_argument(
        "-mha",
        "--mha_type",
        type=str,
        default="mha",
        help="""Type of multi-head attention.
    e.g. -mha mha/mqa/gqa""",
    )
    parser.add_argument(
        "-dt",
        "--dtype",
        type=str,
        default="bf16",
        help="""Data type.
    e.g.: -dt bf16""",
    )

    args = parser.parse_args()
    dtype = dtypes.d_dtypes[args.dtype]
    (seqlen_q, seqlen_k) = args.seqlen_q_k

    test_flash_attn_varlen_func(
        args.batch_size,
        args.nheads,
        seqlen_q,
        seqlen_k,
        args.d,
        args.dv,
        args.min_seqlen_q,
        args.dropout_p,
        args.causal,
        args.local,
        args.bias_type,
        args.deterministic,
        args.mha_type,
        dtype,
    )

    test_varlen_flash_attn_seq_padding(
        args.batch_size,
        args.mha_type,
        args.deterministic,
        "mixed",
        dtype,
        args.d,
        args.dv,
        seqlen_q,
        seqlen_k,
        args.local,
    )
