# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter import dtypes
from aiter.test_common import run_perftest
from aiter.test_mha_common import attention_ref
import pytest
import pandas as pd
import argparse
import math

benchmark = {}

BLOCK_SIZE = 32


def align_to_tile(original, tile_size):
    return (original + tile_size - 1) // tile_size * tile_size


def create_mxfp8_scale_buffer(batch, head_num, seq_len, head_dim, block_size, sub_tile,
                               device="cuda", fill_value=1.0):
    """Create MXFP8 block scale buffer (float8_e8m0fnu).

    fill_value=1.0 means E8M0 exponent 2^0 = no scaling (raw byte 0x7F).
    """
    total_seq = batch * seq_len
    aligned_seq = align_to_tile(total_seq, sub_tile)
    num_scale_elements = aligned_seq * head_dim * head_num // block_size
    scale_buf = torch.full((num_scale_elements,), fill_value, dtype=torch.float8_e8m0fnu, device=device)
    return scale_buf


def run_mxfp8_kernel(
    q_fp8,
    k_fp8,
    v_fp8,
    q_scale,
    k_scale,
    v_scale,
):
    return run_perftest(
        aiter.flash_attn_mxfp8_func,
        q_fp8,
        k_fp8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
    )


def run_ref(q_bf16, k_bf16, v_bf16, causal=False):
    """CPU reference using attention_ref from test_mha_common."""
    out, _, _ = attention_ref(
        q_bf16,
        k_bf16,
        v_bf16,
        causal=causal,
        upcast=True,
    )
    return out


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("nheads, nheads_k", [(8, 1), (32, 8)])
@pytest.mark.parametrize(
    "d",
    [64, 128],
)
@pytest.mark.parametrize(
    "seqlen_q, seqlen_k",
    [
        (256, 256),
        (256, 512),
        (512, 512),
        (1024, 1024),
    ],
)
def test_mha_mxfp8_output(
    batch_size,
    nheads,
    nheads_k,
    seqlen_q,
    seqlen_k,
    d,
):
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()
    d_v = d
    sub_q = 256
    sub_k = 128

    # Generate data in BHSD layout (batch, head, seq, dim)
    q_bf16 = torch.randn(batch_size, nheads, seqlen_q, d, device="cuda", dtype=torch.bfloat16)
    k_bf16 = torch.randn(batch_size, nheads_k, seqlen_k, d, device="cuda", dtype=torch.bfloat16)
    v_bf16 = torch.randn(batch_size, nheads_k, seqlen_k, d_v, device="cuda", dtype=torch.bfloat16)

    q_fp8 = q_bf16.to(dtypes.fp8)
    k_fp8 = k_bf16.to(dtypes.fp8)
    v_fp8 = v_bf16.to(dtypes.fp8)

    # Reference uses BSHD-shaped view of BHSD data for attention_ref
    q_ref = q_fp8.to(torch.bfloat16).transpose(1, 2)
    k_ref = k_fp8.to(torch.bfloat16).transpose(1, 2)
    v_ref = v_fp8.to(torch.bfloat16).transpose(1, 2)

    # Kernel input: BSHD-shaped view with BHSD strides
    q_in = q_fp8.transpose(1, 2)
    k_in = k_fp8.transpose(1, 2)
    v_in = v_fp8.transpose(1, 2)

    q_scale = create_mxfp8_scale_buffer(batch_size, nheads, seqlen_q, d, BLOCK_SIZE, sub_q)
    k_scale = create_mxfp8_scale_buffer(batch_size, nheads_k, seqlen_k, d, BLOCK_SIZE, sub_k)
    v_scale = create_mxfp8_scale_buffer(batch_size, nheads_k, seqlen_k, d_v, BLOCK_SIZE, sub_k)

    out, us_mxfp8_fwd = run_mxfp8_kernel(
        q_in, k_in, v_in,
        q_scale, k_scale, v_scale,
    )

    out_ref = run_ref(q_ref, k_ref, v_ref)

    # Workaround: bf16 element-wise ops hang on gfx1250 (ROCm bf16 codegen bug).
    # Always compare in fp32.
    out_fp32 = out.float()
    out_ref_fp32 = out_ref.float()
    abs_diff = (out_fp32 - out_ref_fp32).abs()
    max_diff = abs_diff.max().item()

    max_item = max(out_fp32.abs().max().item(), out_ref_fp32.abs().max().item(), 1e-7)
    square_diff = (abs_diff / out_ref_fp32.abs().clamp(min=1e-7)).pow(2)
    nrms = square_diff.sum().sqrt() / (math.sqrt(out_ref_fp32.numel()) * max_item)

    print(f"Output nrms: {nrms}")
    print(f"Output max diff: {max_diff}")
    print(f"MXFP8 fwd time: {us_mxfp8_fwd:.1f} us")
    assert max_diff < 0.05, f"max_diff={max_diff} exceeds threshold 0.05"

    fwd_flop = (
        batch_size
        * nheads
        * (seqlen_q * seqlen_k * d * 2 + seqlen_q * seqlen_k * d_v * 2)
    )
    quant_dtype_bytes = 1
    quant_fwd_num_bytes = (
        batch_size
        * nheads
        * quant_dtype_bytes
        * (seqlen_q * d + seqlen_k * d + seqlen_k * d_v + seqlen_q * d_v)
    )

    benchmark["mxfp8_fwd_us"] = us_mxfp8_fwd
    benchmark["mxfp8_fwd_tflops"] = fwd_flop / 1.0e6 / us_mxfp8_fwd
    benchmark["mxfp8_fwd_gb_per_sec"] = quant_fwd_num_bytes / 1.0e3 / us_mxfp8_fwd


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test MXFP8 MHA forward kernel on gfx1250/MI450",
)
parser.add_argument("-b", "--batch_size", type=int, default=1)
parser.add_argument("-n", "--nheads", type=int, default=8)
parser.add_argument("-nk", "--nheads_k", type=int, default=-1)
parser.add_argument("-q", "--seqlen_q", type=int, default=256)
parser.add_argument("-k", "--seqlen_k", type=int, default=-1)
parser.add_argument("-d", "--d_qk", type=int, default=128)

def run_mxfp8_cli(batch_size, nheads, nheads_k, seqlen_q, seqlen_k, d):
    """CLI entry: call kernel directly without torch.profiler (unstable on gfx1250)."""
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()
    d_v = d
    sub_q = 256
    sub_k = 128

    q_bf16 = torch.randn(batch_size, nheads, seqlen_q, d, device="cuda", dtype=torch.bfloat16)
    k_bf16 = torch.randn(batch_size, nheads_k, seqlen_k, d, device="cuda", dtype=torch.bfloat16)
    v_bf16 = torch.randn(batch_size, nheads_k, seqlen_k, d_v, device="cuda", dtype=torch.bfloat16)

    q_fp8 = q_bf16.to(dtypes.fp8)
    k_fp8 = k_bf16.to(dtypes.fp8)
    v_fp8 = v_bf16.to(dtypes.fp8)

    q_ref = q_fp8.to(torch.bfloat16).transpose(1, 2)
    k_ref = k_fp8.to(torch.bfloat16).transpose(1, 2)
    v_ref = v_fp8.to(torch.bfloat16).transpose(1, 2)

    q_in = q_fp8.transpose(1, 2)
    k_in = k_fp8.transpose(1, 2)
    v_in = v_fp8.transpose(1, 2)

    q_scale = create_mxfp8_scale_buffer(batch_size, nheads, seqlen_q, d, BLOCK_SIZE, sub_q)
    k_scale = create_mxfp8_scale_buffer(batch_size, nheads_k, seqlen_k, d, BLOCK_SIZE, sub_k)
    v_scale = create_mxfp8_scale_buffer(batch_size, nheads_k, seqlen_k, d_v, BLOCK_SIZE, sub_k)

    out = aiter.flash_attn_mxfp8_func(q_in, k_in, v_in, q_scale, k_scale, v_scale)
    torch.cuda.synchronize()

    out_ref, _, _ = attention_ref(q_ref, k_ref, v_ref, causal=False, upcast=True)

    abs_diff = (out.float() - out_ref.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    status = "PASS" if max_diff < 0.05 else "FAIL"
    print(f"b={batch_size} sq={seqlen_q:>4} d={d:>3} nh={nheads:>2} nhk={nheads_k:>2} "
          f"| max={max_diff:.4f} mean={mean_diff:.6f} [{status}]")
    return max_diff


if __name__ == "__main__":
    args = parser.parse_args()

    nheads_k = args.nheads_k if args.nheads_k > 0 else args.nheads
    seqlen_k = args.seqlen_k if args.seqlen_k > 0 else args.seqlen_q

    run_mxfp8_cli(
        args.batch_size,
        args.nheads,
        nheads_k,
        args.seqlen_q,
        seqlen_k,
        args.d_qk,
    )
