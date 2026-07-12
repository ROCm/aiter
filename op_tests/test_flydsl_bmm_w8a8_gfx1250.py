# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 test / bench for the FlyDSL W8A8 blockwise batched GEMM.

Both operands are fp8_e4m3fn with per-block E8M0 (uint8) scales::

    C[.., n] = sum_k (A_fp8[.., k] * a_scale[.., k//gk])
                    * (B_fp8[b, n, k] * b_scale[b, n//gn, k//gk])

Two output layouts (both logically ``[b, m, n]``), matching the kernel:
    mbn = transposed view (physically ``[m, b, *]``);  bmn = contiguous ``[b, m, *]``.

``B`` / ``b_scale`` are always ``[B, N, K]`` / ``[B, N//gn, K//gk]``.

Style mirrors ``op_tests/test_batched_gemm_bf16.py``: ``@benchmark`` + ``run_torch``
reference + ``run_perftest`` / ``checkAllclose``, argparse-driven ``main()`` that
prints a TFLOPS / bandwidth table.
"""

import argparse
import itertools

import pandas as pd
import torch

from aiter import dtypes, logger
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.bmm_w8a8_gfx1250 import run_bmm_w8a8_gfx1250
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import (
    batched_gemm_bf16 as batched_gemm_bf16_triton,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

# This kernel is gfx1250-only (WMMA scale instructions).
SUPPORTED_GFX = ["gfx1250"]

GROUP_K = 128
GROUP_N = 128
FP8_E4M3 = int(fp4_utils.MxDtypeInt.FP8_E4M3)


# ---------------------------------------------------------------------------
# fp8 + E8M0 block-scale quant helpers
# ---------------------------------------------------------------------------
def quant_a_blockwise(a_bmk):
    """Per-block (over K, size GROUP_K) MXFP8 quant of A ``[B, M, K]``.

    Returns ``(q_fp8 [B,M,K], scale_e8m0 [B,M,K//gk], deq_f32 [B,M,K])``.
    """
    Bn, M, K = a_bmk.shape
    blk = a_bmk.reshape(Bn, M, K // GROUP_K, GROUP_K)
    amax = blk.abs().amax(dim=-1)
    scale_e8m0 = fp4_utils.f32_to_mx_e8m0_scale(amax.contiguous(), dtype=FP8_E4M3)
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0)
    q = (blk / scale_f32.unsqueeze(-1)).to(dtypes.fp8)
    deq = (q.to(dtypes.fp32) * scale_f32.unsqueeze(-1)).reshape(Bn, M, K)
    return q.reshape(Bn, M, K), scale_e8m0, deq


def quant_b_blockwise(b_bnk):
    """Per-block (over N group_n x K group_k) MXFP8 quant of B ``[B, N, K]``.

    One scale per ``(n_group, k_block)``. Returns
    ``(q_fp8 [B,N,K], scale_e8m0 [B,N//gn,K//gk], deq_f32 [B,N,K])``.
    """
    Bn, N, K = b_bnk.shape
    blk = b_bnk.reshape(Bn, N // GROUP_N, GROUP_N, K // GROUP_K, GROUP_K)
    amax = blk.abs().amax(dim=(2, 4))  # [B, N//gn, K//gk]
    scale_e8m0 = fp4_utils.f32_to_mx_e8m0_scale(amax.contiguous(), dtype=FP8_E4M3)
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0)[:, :, None, :, None]
    q = (blk / scale_f32).to(dtypes.fp8)
    deq = (q.to(dtypes.fp32) * scale_f32).reshape(Bn, N, K)
    return q.reshape(Bn, N, K), scale_e8m0, deq


def dequant_a(a_q_bmk, a_scale_e8m0, dtype):
    """fp8 A ``[B,M,K]`` + E8M0 scale ``[B,M,K//gk]`` -> ``dtype`` ``[B,M,K]``."""
    Bn, M, K = a_q_bmk.shape
    scale_f32 = fp4_utils.e8m0_to_f32(a_scale_e8m0).unsqueeze(-1)  # [B,M,K//gk,1]
    blk = a_q_bmk.reshape(Bn, M, K // GROUP_K, GROUP_K).to(dtypes.fp32) * scale_f32
    return blk.reshape(Bn, M, K).to(dtype)


def dequant_b(b_q_bnk, b_scale_e8m0, dtype):
    """fp8 B ``[B,N,K]`` + E8M0 scale ``[B,N//gn,K//gk]`` -> ``dtype`` ``[B,N,K]``."""
    Bn, N, K = b_q_bnk.shape
    scale_f32 = fp4_utils.e8m0_to_f32(b_scale_e8m0)[:, :, None, :, None]
    blk = (
        b_q_bnk.reshape(Bn, N // GROUP_N, GROUP_N, K // GROUP_K, GROUP_K).to(dtypes.fp32)
        * scale_f32
    )
    return blk.reshape(Bn, N, K).to(dtype)


def run_torch(a_deq_bmk, b_deq_bnk, dtype):
    """fp32 reference batched GEMM ``C[b] = A_deq[b] @ B_deq[b]^T`` -> [B, M, N]."""
    out = torch.matmul(a_deq_bmk, b_deq_bnk.transpose(1, 2))
    return out.to(dtype)


# ---------------------------------------------------------------------------
# Benchmark / correctness
# ---------------------------------------------------------------------------
@benchmark()
def test_gemm(b, m, n, k, dtype, layout):
    # Generate everything logically as [B, M, K] / [B, N, K].
    a = torch.randn(b, m, k, dtype=dtypes.fp32) * 0.5
    w = torch.randn(b, n, k, dtype=dtypes.fp32) * 0.5

    a_q, a_scale_bmk, a_deq = quant_a_blockwise(a)
    b_q, b_scale, b_deq = quant_b_blockwise(w)

    ref = run_torch(a_deq, b_deq, dtype)  # [B, M, N]

    b_q_shuf = shuffle_weight(b_q, layout=(16, 16))

    # Lay A / a_scale / C out per the requested physical layout.
    #   mbn: transposed views of contiguous [m, b, *] tensors.
    #   bmn: plain contiguous [b, m, *].
    if layout == "mbn":
        A = a_q.permute(1, 0, 2).contiguous()  # [M, B, K]
        a_scale = a_scale_bmk.permute(1, 0, 2).contiguous()  # [M, B, K//gk]
        C = torch.empty(m, b, n, dtype=dtype)
    else:
        A = a_q.contiguous()  # [B, M, K]
        a_scale = a_scale_bmk.contiguous()  # [B, M, K//gk]
        C = torch.empty(b, m, n, dtype=dtype)

    tile_m, tile_n, tile_k = 64, 256, 128

    def run_flydsl():
        run_bmm_w8a8_gfx1250(
            C,
            A,
            b_q_shuf.contiguous(),
            a_scale,
            b_scale.contiguous(),
            tile_m,
            tile_n,
            tile_k,
            layout=layout,
            group_k=GROUP_K,
            group_n=GROUP_N,
        )
        # flydsl writes C in the requested layout; normalize to [B, M, N].
        return C.permute(1, 0, 2) if layout == "mbn" else C

    # triton bf16 baseline: dequant fp8+E8M0 -> bf16 is timed too, so the row is
    # end-to-end comparable with flydsl's fused fp8 GEMM (both "include dequant").
    x_tri = a_q.contiguous()  # fp8 [B, M, K]
    w_tri = b_q.contiguous()  # fp8 [B, N, K]
    y_tri = torch.empty(b, m, n, dtype=dtype)

    def run_triton():
        a_bf16 = dequant_a(x_tri, a_scale_bmk, dtype)  # [B, M, K]
        w_bf16 = dequant_b(w_tri, b_scale, dtype)  # [B, N, K]
        return batched_gemm_bf16_triton(a_bf16, w_bf16, YQ=y_tri)  # [B, M, N]

    gemm_funcs = {"flydsl": run_flydsl, "triton": run_triton}

    # batched GEMM b x ([m,k] @ [n,k]^T -> [m,n]):
    #   FLOPs = 2 * b * m * n * k;  bytes = (A + B + C) elements * dtype size.
    flops = 2 * b * m * n * k
    nbytes = (b * m * k + b * n * k) * 1 + (b * m * n) * C.element_size()

    ret = {"gfx": get_gfx()}
    if get_gfx() not in SUPPORTED_GFX:
        logger.info("skip bmm_w8a8: requires %s, got %s", SUPPORTED_GFX, get_gfx())
        return ret

    for name, gemm_func in gemm_funcs.items():
        out, us = run_perftest(gemm_func)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=3e-2,
            atol=3e-2,
            msg=f"{name}: bmm_w8a8 gfx1250 ({layout})",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16,fp16}",
        help="""Output data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[16, 8, 4, 2],
        help="""Batch size.
        e.g.: -b 16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (1, 1024, 4096),
            (2, 1024, 4096),
            (4, 1024, 4096),
            (16, 1024, 4096),
            (64, 1024, 4096),
            (128, 1024, 4096),
            (192, 1024, 4096),
            (256, 1024, 4096),
            (512, 1024, 4096),
            (1024, 1024, 4096),
            (2048, 1024, 4096),
            (4096,1024, 4096),
            (8192,1024, 4096),
            (16384, 1024, 4096)
        ],
        help="""Shape of mnk.
        e.g.:   -s 128,1024,4096
                --mnk 128,1024,4096""",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        choices=["bmn", "mbn"],
        nargs="*",
        default=["bmn", "mbn"],
        help="""A / C layout (both logically [b, m, *]):
        mbn = transposed view (physically [m, b, *]),
        bmn = plain contiguous [b, m, *].
        e.g.: -l mbn""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for layout, b, (m, n, k) in itertools.product(args.layout, args.batch, args.mnk):
            ret = test_gemm(b, m, n, k, dtype, layout)
            df.append(ret)
        df = pd.DataFrame(df)
        logger.info("bmm_w8a8_gfx1250 summary (markdown):\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
