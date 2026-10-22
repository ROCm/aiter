# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP8 weights served as per-token x per-row fp8 GEMMs.

The weight's 32x32 E8M0 block scales are folded into one power-of-two scale
per output row at load (mxfp8_to_per_row_fp8). Each call quantizes the bf16
activation per token and runs the tuned gemm_a8w8_bpreshuffle. The bf16
candidate is the dequantize-at-load path: F.linear on the bf16 weight.
"""

import argparse
import itertools

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.gemm_op_a8w8 import get_GEMM_config_with_quant_type
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
BLOCK = 32

# (N, K) of the dense linears of a DeepSeek-V4.1-Flash TP4 rank
NK = [(8192, 1280), (5120, 2048), (1792, 5120), (1152, 5120), (5120, 576), (4096, 1280)]
M = [256, 2048, 16384]


def make_mxfp8_weight(n, k, exp_spread):
    """e4m3fn [N, K] and its E8M0 scales expanded per row, [N, K/32] uint8.

    Values span the e4m3fn range like MX-quantized blocks, whose largest value
    sits near 448.
    """
    weight = (torch.randn(n, k) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    blocks = torch.randint(
        120 - exp_spread, 121, (n // BLOCK, k // BLOCK), dtype=dtypes.u8
    )
    return weight, blocks.repeat_interleave(BLOCK, dim=0)


def dequant_mxfp8(weight, scale):
    return weight.float() * torch.exp2(scale.float() - 127).repeat_interleave(
        BLOCK, dim=1
    )


def run_torch(xq, x_scale, wq, w_scale, dtype=dtypes.bf16):
    return ((xq.float() * x_scale) @ (wq.float() * w_scale).t()).to(dtype)


@benchmark()
def test_mxfp8_ptpc(m, n, k, exp_spread):
    weight, scale = make_mxfp8_weight(n, k, exp_spread)
    w_exact = dequant_mxfp8(weight, scale)
    wq, w_scale = aiter.mxfp8_to_per_row_fp8(weight, scale)
    # only values shifted into the subnormals may round
    requant_err = checkAllclose(
        w_exact,
        wq.float() * w_scale,
        rtol=0,
        atol=0,
        tol_err_ratio=0.01,
        msg="mxfp8_to_per_row_fp8",
    )
    wq_shuffled = shuffle_weight(wq, layout=(16, 16))
    w_bf16 = w_exact.to(dtypes.bf16)

    x = torch.randn(m, k, dtype=dtypes.bf16)
    xq, x_scale = aiter.per_token_quant_hip(x, quant_dtype=dtypes.fp8)

    def ptpc():
        xq, x_scale = aiter.per_token_quant_hip(x, quant_dtype=dtypes.fp8)
        return aiter.gemm_a8w8_bpreshuffle(xq, wq_shuffled, x_scale, w_scale)

    candidates = {
        "bf16": (lambda: F.linear(x, w_bf16), (x.float() @ w_exact.t())),
        "ptpc": (ptpc, run_torch(xq, x_scale, wq, w_scale)),
    }
    config = get_GEMM_config_with_quant_type(m, n, k, dtypes.fp8)

    flops = 2 * m * n * k
    ret = {
        "gfx": get_gfx(),
        "ptpc kernel": config["kernelName"] if config else "untuned",
        "requant err": requant_err,
    }
    for name, (fn, ref) in candidates.items():
        out, us = run_perftest(fn)
        nbytes = (m * k + n * k) * (2 if name == "bf16" else 1) + m * n * 2
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref.to(dtypes.fp32), out.to(dtypes.fp32), rtol=1e-2, atol=1e-2, msg=name
        )
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("mxfp8 ptpc unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MXFP8 weights as per-token x per-row fp8 GEMMs",
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
        help="range of the random E8M0 block exponents within a row block",
    )
    args = parser.parse_args()

    df = pd.DataFrame(
        [
            test_mxfp8_ptpc(m, n, k, spread)
            for spread, (m, n, k) in itertools.product(args.exp_spread, args.mnk)
        ]
    )
    aiter.logger.info("mxfp8 ptpc summary (markdown):\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
