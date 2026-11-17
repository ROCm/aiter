# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""mxfp8_to_per_row_fp8: an MXFP8 weight served as a per-token FP8 hipBLASLt GEMM.

The weight's E8M0 block scales are folded into one power-of-two scale per
output row. The requantized weight is checked against the MXFP8 dequant, then
run through per_token_quant_hip and hipb_mm on the preshuffled weight, as vLLM
does on gfx942, and checked against a torch reference.
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose

torch.set_default_device("cuda")

BLOCK = 32

# (N, K) of the dense linears of a DeepSeek-V4.1-Flash TP4 rank
NK = [(8192, 1280), (5120, 2048), (1792, 5120), (1152, 5120), (5120, 576), (4096, 1280)]
M = [1, 256, 16384]


def make_mxfp8_weight(n, k, exp_spread):
    """e4m3fn [N, K] and its E8M0 codes, one per row and K block, [N, K/32] uint8.

    Values span the e4m3fn range like MX-quantized blocks, whose largest value
    sits near 448.
    """
    weight = (torch.randn(n, k) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    codes = torch.randint(127 - exp_spread, 128, (n, k // BLOCK), dtype=dtypes.u8)
    return weight, codes


def dequant_mxfp8(weight, codes):
    return weight.float() * torch.exp2(codes.float() - 127).repeat_interleave(
        BLOCK, dim=1
    )


@benchmark()
def test_mxfp8_to_per_row_fp8(m, n, k, exp_spread):
    weight, codes = make_mxfp8_weight(n, k, exp_spread)
    wq, w_scale = aiter.mxfp8_to_per_row_fp8(weight, codes)
    # Blocks shift by whole exponents, so only values pushed into the
    # subnormals may round.
    requant_err = checkAllclose(
        dequant_mxfp8(weight, codes),
        wq.float() * w_scale,
        rtol=0,
        atol=0,
        tol_err_ratio=0.01,
        msg="mxfp8_to_per_row_fp8",
    )

    x = torch.randn(m, k, dtype=dtypes.bf16)
    xq, x_scale = aiter.per_token_quant_hip(x, quant_dtype=dtypes.fp8)
    out = aiter.hipb_mm(
        xq,
        shuffle_weight(wq, layout=(16, 16)).t(),
        solution_index=-1,
        bias=None,
        out_dtype=dtypes.bf16,
        scaleA=x_scale,
        scaleB=w_scale.t().contiguous(),
        scaleOut=None,
        bpreshuffle=True,
    )
    ref = (xq.float() * x_scale) @ (wq.float() * w_scale).t()
    gemm_err = checkAllclose(
        ref, out.to(dtypes.fp32), rtol=1e-2, atol=1e-2, msg="hipb_mm"
    )
    return {"requant err": requant_err, "gemm err": gemm_err}


def main():
    if get_gfx() != "gfx942":
        aiter.logger.warning("hipb_mm per-token fp8 test targets gfx942; skipping")
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="mxfp8_to_per_row_fp8 served as a per-token FP8 hipb_mm GEMM",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(m, n, k) for (n, k), m in itertools.product(NK, M)],
        help="M,N,K shapes, e.g. -s 16384,5120,576",
    )
    parser.add_argument(
        "-e",
        "--exp_spread",
        type=int,
        nargs="*",
        default=[3],
        help="range of the random E8M0 block exponents within a row",
    )
    args = parser.parse_args()

    aiter.hipb_create_extension()
    df = pd.DataFrame(
        [
            test_mxfp8_to_per_row_fp8(m, n, k, spread)
            for spread, (m, n, k) in itertools.product(args.exp_spread, args.mnk)
        ]
    )
    aiter.logger.info(
        "mxfp8_to_per_row_fp8 summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
