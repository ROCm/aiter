# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter import dtypes
from aiter.test_common import run_perftest
from aiter import per_tensor_quant
from aiter.test_mha_common import (
    generate_qkv,
    generate_random_padding_mask,
)
import pytest
import pandas as pd
import argparse

benchmark = {}


def run_ck(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    min_seqlen_q,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    if q.dtype == dtypes.fp8 and k.dtype == dtypes.fp8 and v.dtype == dtypes.fp8:
        return run_perftest(
            aiter.flash_attn_varlen_fp8_pertensor_func,
            q,
            k,
            v,
            q_descale,
            k_descale,
            v_descale,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            min_seqlen_q,
            causal=causal,
            window_size=window_size,
        )
    else:
        return run_perftest(
            aiter.flash_attn_varlen_func,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            dropout_p=0.0,
            causal=causal,
            window_size=window_size,
            bias=None,
            alibi_slopes=None,
            deterministic=True,
            return_lse=False,
            return_attn_probs=False,
        )


# @pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("min_seqlen_q", [0])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("nheads, nheads_k", [(8, 1), (40, 8), (32, 8), (5, 1)])
@pytest.mark.parametrize(
    "d,d_v",
    [
        (128, 128),
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
        # (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
        (4096, 4096),
    ],
)
def test_flash_attn_varlen_output(
    batch_size,
    nheads,
    nheads_k,
    seqlen_q,
    seqlen_k,
    d,
    d_v,
    min_seqlen_q,
    causal,
    local,
):
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    dtype = torch.bfloat16
    quant_dtype = dtypes.fp8

    q_pad = torch.rand(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype)
    k_pad = torch.rand(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device="cuda",
        dtype=dtype,
    )
    v_pad = torch.rand(
        batch_size,
        seqlen_k,
        nheads_k,
        d_v,
        device="cuda",
        dtype=dtype,
    )

    query_padding_mask = generate_random_padding_mask(
        seqlen_q, batch_size, "cuda", mode="random"
    )
    key_padding_mask = generate_random_padding_mask(
        seqlen_k, batch_size, "cuda", mode="random"
    )

    (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q_pad,
        k_pad,
        v_pad,
        _,
        _,
        _,
    ) = generate_qkv(
        q_pad, k_pad, v_pad, query_padding_mask, key_padding_mask, kvpacked=False
    )

    q.requires_grad_(False)
    k.requires_grad_(False)
    v.requires_grad_(False)

    q_quant, q_descale = per_tensor_quant(q, quant_dtype=quant_dtype)
    k_quant, k_descale = per_tensor_quant(k, quant_dtype=quant_dtype)
    v_quant, v_descale = per_tensor_quant(v, quant_dtype=quant_dtype)

    out, us_quant_fwd = run_ck(
        q_quant,
        k_quant,
        v_quant,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q,
        causal,
        window_size,
        q_descale,
        k_descale,
        v_descale,
    )

    out_ref, us_fwd = run_ck(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q,
        causal,
        window_size,
    )

    max_diff = (out - out_ref).abs().max().item()
    print(f"Output max diff: {max_diff}")
    assert max_diff < 0.055

    fwd_flop = 0
    dtype_bytes = torch.finfo(dtype).bits // 8
    quant_dtype_bytes = torch.finfo(quant_dtype).bits // 8
    fwd_num_bytes = 0
    quant_fwd_num_bytes = 0
    for i in range(len(cu_seqlens_q) - 1):
        real_seqlen_q = cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item()
        real_seqlen_k = cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item()
        fwd_flop = (
            fwd_flop
            + nheads * 2 * real_seqlen_q * real_seqlen_k * d
            + nheads * 2 * real_seqlen_q * real_seqlen_k * d_v
        )
        fwd_num_bytes = fwd_num_bytes + nheads * dtype_bytes * (
            real_seqlen_q * d
            + real_seqlen_k * d
            + real_seqlen_k * d_v
            + real_seqlen_q * d_v
        )
        quant_fwd_num_bytes = fwd_num_bytes + nheads * quant_dtype_bytes * (
            real_seqlen_q * d
            + real_seqlen_k * d
            + real_seqlen_k * d_v
            + real_seqlen_q * d_v
        )

    benchmark["quant_fwd_us"] = us_quant_fwd
    benchmark["quant_fwd_tflops"] = (fwd_flop) / 1.0e6 / us_quant_fwd
    benchmark["quant_fwd_gb_per_sec"] = (quant_fwd_num_bytes) / 1.0e3 / us_quant_fwd
    benchmark["fwd_us"] = us_fwd
    benchmark["fwd_tflops"] = (fwd_flop) / 1.0e6 / us_fwd
    benchmark["fwd_gb_per_sec"] = (fwd_num_bytes) / 1.0e3 / us_fwd


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
    default=5,
    help="""Number of heads. Default is 5.
    e.g.: -n 8""",
)
parser.add_argument(
    "-nk",
    "--nheads_k",
    type=int,
    default=-1,
    help="""Number of heads. -1 means equal to n (nheads).
    e.g.: -nk 1""",
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
    default=-1,
    help="""Sequence length for key. -1 means equal to q (seqlen_q).
    e.g.: -k 1024""",
)
parser.add_argument(
    "-d",
    "--d_qk",
    type=int,
    default=128,
    help="""Dimension of query and key. Default is 128.
    e.g.: -d 128""",
)
parser.add_argument(
    "-dv",
    "--d_v",
    type=int,
    default=-1,
    help="""Dimension of query and key. -1 means equal to d (d_qk).
    e.g.: -dv 128""",
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
    action="store_true",
    help="""Causal attention. Default is False.
    -c or --causal    # enable causal attention""",
)
parser.add_argument(
    "-l",
    "--local",
    action="store_true",
    help="""Local attention. Default is False.
    -l or --local    # enable local attention""",
)

if __name__ == "__main__":
    args = parser.parse_args()

    nheads_k = args.nheads_k if args.nheads_k > 0 else args.nheads
    seqlen_k = args.seqlen_k if args.seqlen_k > 0 else args.seqlen_q
    d_v = args.d_v if args.d_v > 0 else args.d_qk

    collected = []
    test_flash_attn_varlen_output(
        args.batch_size,
        args.nheads,
        nheads_k,
        args.seqlen_q,
        seqlen_k,
        args.d_qk,
        d_v,
        args.min_seqlen_q,
        args.causal,
        args.local,
    )
    collected.append(benchmark)

    df = pd.DataFrame(collected)
    aiter.logger.info(f"mha summary:\n{df}")
