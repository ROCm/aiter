# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression + perf sweep for the opus fp8 e8m0 mxscale flatmm split-K BMM.

Covers the mmajor DeepSeek-V4 wo_a path: O/Y are [M, G, *] (transposed views of
batch-major [G, M, *]); wo_a + w_scale stay batch-major. Activation scale is
per-token e8m0 (GROUP_M=1), weight scale is 128x128-block e8m0. Candidates are
the curated flatmm kernel IDs; the reference is a dequantized fp32 einsum.

Usage:
    python3 op_tests/test_opus_a8w8_bmm.py
    python3 op_tests/test_opus_a8w8_bmm.py -s 512,1024,4096 -g 2 -d bf16
"""

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.opus.bmm_op import _opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_raw

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]  # fp8 e8m0 mxscale flatmm is gfx950-only
GROUP = 128  # GROUP_N == GROUP_K == 128; GROUP_M == 1 (per-token)
_DT = {"fp32": dtypes.fp32, "bf16": dtypes.bf16}

# Curated flatmm kernel IDs (splitK == 1, direct store). Each requires
# m % B_M == 0, n % B_N == 0, k % 128 == 0 for its tile.
FLATMM_KIDS = {
    "m64n64k128": (650, 64, 64),
    "m64n64k128_scale_prefetch": (653, 64, 64),
}


def _to_e8m0_scale(scale):
    # Round scale up to a power of two so quantized fp8 values stay in range.
    e = torch.ceil(torch.log2(scale.to(dtypes.fp32))).to(torch.int32) + 127
    e = torch.clamp(e, 0, 255).to(torch.uint8)
    scale_pow2 = torch.exp2(e.to(dtypes.fp32) - 127.0)
    return e, scale_pow2


def _quant_per_token_e8m0(x_bf16):
    """[G,M,K] bf16 -> fp8 + e8m0 x_scale [G,M,K/128] + fp32 scale."""
    G, M, K = x_bf16.shape
    xb = x_bf16.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    raw = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
    e8m0, scale = _to_e8m0_scale(raw)
    q = (xb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return q.view(G, M, K), e8m0.squeeze(-1), scale.squeeze(-1)


def _quant_block_e8m0(w_bf16):
    """[G,N,K] bf16 -> fp8 + e8m0 w_scale [G,N/128,K/128] + fp32 scale."""
    G, N, K = w_bf16.shape
    wb = w_bf16.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    raw = wb.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-8) / 448.0
    e8m0, scale = _to_e8m0_scale(raw)
    q = (wb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return (
        q.view(G, N, K),
        e8m0.view(G, N // GROUP, K // GROUP),
        scale.view(G, N // GROUP, K // GROUP),
    )


def run_torch(O_fp8, W_fp8, x_scale, w_scale):
    """Reference: dequant fp8 -> fp32 einsum -> [G,M,N]. Not timed."""
    G, M, K = O_fp8.shape
    N = W_fp8.shape[1]
    act = O_fp8.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    act = (act * x_scale.unsqueeze(-1)).view(G, M, K)
    W = W_fp8.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    W = (W * w_scale.view(G, N // GROUP, 1, K // GROUP, 1)).view(G, N, K)
    return torch.einsum("gmk,gnk->gmn", act, W).to(dtypes.fp32)


@benchmark()
def test_mxscale_bmm(g, m, n, k, dtype):
    ydt = _DT[dtype]
    # Canonical batch-major tensors, then feed the kernel transposed (mmajor)
    # views exactly like the DSV4 wo_a call does (zero-copy, no contiguous copy).
    O_bf16 = (torch.rand((g, m, k), dtype=dtypes.fp32) / 10).to(dtypes.bf16)
    W_bf16 = (torch.rand((g, n, k), dtype=dtypes.fp32) / 10).to(dtypes.bf16)
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(O_bf16)
    W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(W_bf16)

    O_in = O_mx.transpose(0, 1)  # [m,g,k] view
    xs_in = xs_mx.transpose(0, 1)  # [m,g,k/128] view
    ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32).transpose(0, 1)  # [m,g,n]
    y_shape = (m, g, n)

    def _call(kid):
        Y = torch.empty(y_shape, dtype=ydt)
        _opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_raw(
            O_in, W_mx, Y, xs_in, ws_mx, 1, kid
        )
        return Y

    candidates = {}
    for name, (kid, bm, bn) in FLATMM_KIDS.items():
        # Skip a tile whose block shape does not divide this (m, n).
        if m % bm == 0 and n % bn == 0 and k % GROUP == 0:
            candidates[name] = (lambda kid=kid: _call(kid), ref)

    flops = 2.0 * g * m * n * k
    # fp8 A + fp8 W + e8m0 scales (uint8) + output.
    nbytes = (
        g * m * k
        + g * n * k
        + g * m * (k // GROUP)
        + g * (n // GROUP) * (k // GROUP)
        + m * g * n * torch.empty((), dtype=ydt).element_size()
    )

    ret = {"gfx": get_gfx()}
    for name, (fn, fn_ref) in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            fn_ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=0.1,
            atol=0.5,
            msg=f"mxscale_bmm {name} g={g} m={m} n={n} k={k}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "opus mxscale flatmm BMM unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="opus fp8 e8m0 mxscale flatmm split-K BMM test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        nargs="*",
        default=["bf16"],
        choices=["bf16", "fp32"],
        help="output dtype(s) to sweep (default: bf16)",
    )
    parser.add_argument(
        "-g",
        "--groups",
        type=int,
        nargs="*",
        default=[2],
        help="batch group counts to sweep (DSV4 wo_a G; default: 2)",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(512, 1024, 4096), (256, 1024, 4096), (128, 1024, 4096)],
        help="(m,n,k) shapes to sweep",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for g, (m, n, k) in itertools.product(args.groups, args.mnk):
            df.append(test_mxscale_bmm(g, m, n, k, dtype))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "opus mxscale flatmm BMM summary (dtype=%s):\n%s",
            dtype,
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
