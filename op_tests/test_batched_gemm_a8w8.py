# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, perftest
import argparse
import numpy as np


@perftest(num_iters=5)
def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=dtypes.bf16, device="cuda")
    for b in range(B):
        b_x = F.linear(x[b, :, :].to(dtypes.fp32), weight[b, :, :].to(dtypes.fp32))
        b_scale = torch.matmul(x_scale[b, :, :], w_scale[b, :, :])
        b_out = torch.mul(b_x, b_scale)
        if bias is not None:
            b_out = b_out.to(bias[b, :, :]) + bias[b, :, :]
        out[b, :, :] = b_out
    return out.to(dtype)


@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    return aiter.batched_gemm_a8w8_CK(x, weight, x_scale, w_scale, bias)


@perftest()
def run_gemm_asm(A, B, ScaleA, ScaleB, dtype=dtypes.bf16):
    return aiter.batched_gemm_a8w8_ASM(A, B, ScaleA, ScaleB, dtype=dtype)


def mxfp8_ref_matmul_cpu(A_fp8, B_fp8, ScaleA_e8m0, ScaleB_e8m0, batch_size, M, N, K,
                          scale_block_size=32):
    """CPU reference for MXFP8 block-scaled batched GEMM.

    A_fp8: [B, M, K] fp8 (row major, NOT preshuffled)
    B_fp8: [B, N, K] fp8 (row major, NOT preshuffled)
    ScaleA_e8m0: [B, M, K/32] uint8 e8m0 (NOT shuffled)
    ScaleB_e8m0: [B, N, K/32] uint8 e8m0 (NOT shuffled)
    """
    # Convert fp8 to float
    A_f32 = A_fp8.to(torch.float32)  # [B, M, K]
    B_f32 = B_fp8.to(torch.float32)  # [B, N, K]

    scale_k = K // scale_block_size
    out = torch.zeros(batch_size, M, N, dtype=torch.float32, device="cpu")

    # e8m0 scale: value = 2^(raw - 127)
    ScaleA_f32 = (2.0 ** (ScaleA_e8m0.to(torch.float32) - 127.0))  # [B, M, scale_k]
    ScaleB_f32 = (2.0 ** (ScaleB_e8m0.to(torch.float32) - 127.0))  # [B, N, scale_k]

    for blk in range(scale_k):
        k_start = blk * scale_block_size
        k_end = k_start + scale_block_size
        # A_block: [B, M, block_size], B_block: [B, N, block_size]
        A_block = A_f32[:, :, k_start:k_end] * ScaleA_f32[:, :, blk:blk+1]
        B_block = B_f32[:, :, k_start:k_end] * ScaleB_f32[:, :, blk:blk+1]
        # matmul: [B, M, block_size] x [B, block_size, N] -> [B, M, N]
        out += torch.bmm(A_block, B_block.transpose(1, 2))

    return out.to(torch.bfloat16)


def test_gemm(dtype, b, m, n, k):
    dim = (b, m, n, k)
    x = torch.randint(-20, 20, (b, m, k), dtype=dtypes.i8).cuda()
    weight = torch.randint(-20, 20, (b, n, k), dtype=dtypes.i8).cuda()
    x_scale = torch.rand([b, m, 1], dtype=dtypes.fp32).cuda() + 1e-6
    w_scale = torch.rand([b, 1, n], dtype=dtypes.fp32).cuda() + 1e-6

    a, avg_a = run_torch(x, weight, x_scale, w_scale, None, dtype)
    b, avg_b = run_gemm_ck(x, weight, x_scale, w_scale, None, dtype)
    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg="a,b: " + msg, rtol=1e-2, atol=0.01)


def test_gemm_asm(dtype, b, m, n, k, scale_block_size=32):
    """Test MXFP8 batched GEMM ASM kernel against CPU reference."""
    dim = (b, m, n, k)

    # Generate fp8_e4m3 input data
    A_fp8 = torch.randn(b, m, k, dtype=torch.float32).clamp(-448, 448).to(torch.float8_e4m3fnuz).cuda()
    B_fp8 = torch.randn(b, n, k, dtype=torch.float32).clamp(-448, 448).to(torch.float8_e4m3fnuz).cuda()

    # Generate e8m0 block-wise scales (uint8, values represent exponent)
    # Use moderate range around 127 (=2^0=1.0 scale factor)
    ScaleA_e8m0 = torch.randint(120, 135, (b, m, k // scale_block_size), dtype=torch.uint8).cuda()
    ScaleB_e8m0 = torch.randint(120, 135, (b, n, k // scale_block_size), dtype=torch.uint8).cuda()

    # CPU reference (on original non-shuffled data)
    ref = mxfp8_ref_matmul_cpu(
        A_fp8.cpu(), B_fp8.cpu(),
        ScaleA_e8m0.cpu(), ScaleB_e8m0.cpu(),
        b, m, n, k, scale_block_size,
    )

    # TODO: Apply preshuffle to A, B, ScaleA, ScaleB before calling ASM kernel
    # A preshuffle: (m, k) -> (m/2, k/128, 2, 128)
    # B preshuffle: (n, k) -> (n/16, k/16, 16, 16)
    # Scale preshuffle: (m_or_n, k/32) -> (m_or_n/32, k/4, 32, 4)
    # For now, pass raw data — will need preshuffle utility functions

    asm_out, avg_asm = run_gemm_asm(A_fp8, B_fp8, ScaleA_e8m0, ScaleB_e8m0, dtype)
    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, asm avg: {avg_asm:<8.2f} us"
    checkAllclose(ref.cuda(), asm_out, msg="ref,asm: " + msg, rtol=1e-1, atol=1.0)


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"]],
    nargs="*",
    default="bf16,",
    metavar="{bf16}",
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-b",
    "--batch",
    type=int,
    choices=[16],
    nargs="*",
    default=[16],
    help="""Batch size.
    e.g.: -b 16""",
)
parser.add_argument(
    "-s",
    "--mnk",
    type=dtypes.str2tuple,
    nargs="*",
    default=[
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (192, 8192, 1024),
        (256, 8192, 1024),
        (320, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (2048, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
    ],
    help="""Shape of mnk.
    e.g.:   -s 1024,8192,1024
            --mnk 1024,8192,1024""",
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    choices=["ck", "asm", "both"],
    default="ck",
    help="""Test mode: ck (CK kernel), asm (MXFP8 ASM kernel), both.
    e.g.: -m asm""",
)

args = parser.parse_args()


for dtype in args.dtype:
    for b in args.batch:
        for m, n, k in args.mnk:
            if args.mode in ("ck", "both"):
                test_gemm(dtype, b, m, n, k)
            if args.mode in ("asm", "both"):
                test_gemm_asm(dtype, b, m, n, k)
