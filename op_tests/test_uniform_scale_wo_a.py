# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# fp8 block-scale UNIFORM (Route B fp8, 4-wave full-tile, direct store) batched
# GEMM for the DeepSeek-V4 output-LoRA (wo_a). Exercises both layout surfaces:
#   * bmajor: O/wo_a/Y = [G,M,K]/[G,N,K]/[G,M,N] (batch-major, contiguous).
#   * mmajor: O/Y = [M,G,*] (transposed view of the batch-major tensors) -- the
#     zero-copy DSV4 path (o=[num_tokens, n_groups, K] feeds with no transpose).
# Per-token A scale x_scale (GROUP_M=1), 128x128 block B scale w_scale. Sweeps
# Y dtype {fp32,bf16}. Compares uniform_scale kids (700=128x128, 701=256x128)
# against a dequant einsum reference (and a8w8_scale_mmajor where applicable).
#
# v1 kernel constraints: N % 128 == 0, K % 128 == 0, M % B_M == 0 (no store OOB
# masking). DSV4 N=1024/K=4096 satisfy N/K; M is swept as multiples of 256.

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.opus.gemm_op_a16w16 import (
    _opus_gemm_uniform_scale_raw,
    _opus_gemm_uniform_scale_mmajor_raw,
)

try:
    from aiter.ops.opus.gemm_op_a16w16 import _opus_gemm_a8w8_scale_mmajor_raw
except ImportError:  # pragma: no cover
    _opus_gemm_a8w8_scale_mmajor_raw = None

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]  # fp8 block-scale uniform is gfx950-only
GROUP = 128  # GROUP_N == GROUP_K == 128; GROUP_M == 1 (per-token)
_DT = {"fp32": dtypes.fp32, "bf16": dtypes.bf16}


def _quant_per_token(x_bf16):
    """x_bf16 [G,M,K] -> fp8 O + x_scale [G,M,K/128] (per-token, per-128-K)."""
    G, M, K = x_bf16.shape
    xb = x_bf16.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    scale = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
    q = (xb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return q.view(G, M, K), scale.squeeze(-1).to(dtypes.fp32)


def _quant_block(w_bf16):
    """w_bf16 [G,N,K] -> fp8 W + w_scale [G,N/128,K/128] (128x128 blocks)."""
    G, N, K = w_bf16.shape
    wb = w_bf16.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    scale = wb.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-8) / 448.0
    q = (wb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return q.view(G, N, K), scale.view(G, N // GROUP, K // GROUP).to(dtypes.fp32)


def run_torch(O_fp8, W_fp8, x_scale, w_scale):
    """Reference: dequant fp8 -> fp32 einsum -> [G,M,N]. Not timed."""
    G, M, K = O_fp8.shape
    N = W_fp8.shape[1]
    O = O_fp8.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    O = (O * x_scale.unsqueeze(-1)).view(G, M, K)
    W = W_fp8.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    W = (W * w_scale.view(G, N // GROUP, 1, K // GROUP, 1)).view(G, N, K)
    return torch.einsum("gmk,gnk->gmn", O, W).to(dtypes.fp32)


@benchmark()
def test_uniform_scale(m, n, k, g, layout, out_dtype):
    ydt = _DT[out_dtype]
    # Canonical batch-major tensors [G,M,K] / [G,N,K].
    O_bf16 = (torch.rand((g, m, k), dtype=dtypes.fp32) / 10).to(dtypes.bf16)
    W_bf16 = (torch.rand((g, n, k), dtype=dtypes.fp32) / 10).to(dtypes.bf16)
    O_fp8, x_scale = _quant_per_token(O_bf16)  # [g,m,k], [g,m,k/128]
    W_fp8, w_scale = _quant_block(W_bf16)  # [g,n,k], [g,n/128,k/128]

    ref_bm = run_torch(O_fp8, W_fp8, x_scale, w_scale)  # [g,m,n]

    if layout == "bmajor":
        O_in, xs_in = O_fp8, x_scale  # [g,m,k], [g,m,k/128] contiguous
        ref = ref_bm
        y_shape = (g, m, n)
        run = _opus_gemm_uniform_scale_raw
    else:  # mmajor: transposed views (dim0=M, dim1=batch=G)
        O_in = O_fp8.transpose(0, 1)  # [m,g,k] view
        xs_in = x_scale.transpose(0, 1)  # [m,g,k/128] view
        ref = ref_bm.transpose(0, 1)  # [m,g,n]
        y_shape = (m, g, n)
        run = _opus_gemm_uniform_scale_mmajor_raw

    def _call(kid):
        Y = torch.empty(y_shape, dtype=ydt)
        run(O_in, W_fp8, Y, xs_in, w_scale, kid)
        return Y

    candidates = {
        "uniform700_128x128": lambda: _call(700),
        "uniform701_256x128": lambda: _call(701),
    }
    # a8w8_scale (8-wave quad) baseline: mmajor + fp32 only (its C++ entry
    # hardcodes fp32 Y and the mmajor layout).
    if (
        layout == "mmajor"
        and out_dtype == "fp32"
        and _opus_gemm_a8w8_scale_mmajor_raw is not None
    ):

        def _a8w8():
            Y = torch.empty(y_shape, dtype=dtypes.fp32)
            _opus_gemm_a8w8_scale_mmajor_raw(O_in, W_fp8, Y, xs_in, w_scale)
            return Y

        candidates["a8w8_scale"] = _a8w8

    flops = 2 * m * g * n * k
    nbytes = (m * g * k + g * n * k + m * g * n) * O_fp8.element_size()

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref,
            out.to(dtypes.fp32),
            rtol=5e-2,
            atol=5e-2,
            msg=f"{name}: uniform_scale {layout}/{out_dtype}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("uniform_scale unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[8],
        help="G = n_local_groups (DSV4 wo_a batch)",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (256, 1024, 4096),
            (512, 1024, 4096),
            (1024, 1024, 4096),
            (2048, 1024, 4096),
            (4096, 1024, 4096),
        ],
        help="(M=num_tokens, N=o_lora_rank, K); M must be a multiple of 256",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        nargs="*",
        default=["mmajor", "bmajor"],
        help="tensor layout: mmajor (DSV4 zero-copy) or bmajor (batch-major)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        nargs="*",
        default=["fp32", "bf16"],
        help="Y output dtype: fp32 or bf16",
    )
    args = parser.parse_args()

    df = []
    for layout, out_dtype, g, (m, n, k) in itertools.product(
        args.layout, args.dtype, args.batch, args.mnk
    ):
        df.append(test_uniform_scale(m, n, k, g, layout, out_dtype))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "uniform_scale wo_a summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
