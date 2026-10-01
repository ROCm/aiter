# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 MXFP8 (1x32 e8m0, m32k4 / n32k4 scales) bpreshuffle GEMM.

Runs every tuned (M, N, K) of this arch by default, or the shapes given by -mnk,
through gemm_a8w8_mxfp8_bpreshuffle and checks it against an fp32 reference.
"""

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.gemm_op_a8w8 import _get_mxfp8_bpreshuffle_config
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.shuffle import shuffle_blockscale_to_mxfp8_scale, shuffle_weight
from aiter.test_common import benchmark, checkAllclose, perftest


def run_torch(x, x_scale, weight, w_scale):
    """fp32 reference from row-major 1x32 A-scales and 128x128 B-scales."""
    m, k = x.shape
    n = weight.shape[0]
    x_scale = torch.exp2(x_scale.view(torch.uint8).float() - 127)
    x = (x.float().view(m, k // 32, 32) * x_scale[..., None]).view(m, k)
    w_scale = torch.exp2(w_scale.view(torch.uint8).float() - 127)
    w_scale = w_scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return x @ (weight.float() * w_scale).T


@perftest()
def run_gemm(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_mxfp8_bpreshuffle(x, weight, x_scale, w_scale, dtype)


@benchmark()
def test_gemm(m, n, k, dtype=dtypes.bf16):
    x = torch.randn(m, k, dtype=dtypes.bf16)
    weight = (torch.randn(n, k) * 0.02).to(dtypes.fp8)
    # 128x128 block scales around 2^-3, as a checkpoint carries them
    w_scale = torch.randint(122, 127, (n // 128, k // 128), dtype=torch.uint8)
    w_scale = w_scale.view(dtypes.fp8_e8m0)
    # the fused quant writes the m32k4 scale directly; the row-major twin feeds
    # the reference
    xq, x_scale = per_group_quant_hip(
        x,
        quant_dtype=dtypes.fp8,
        group_size=32,
        scale_type=dtypes.fp8_e8m0,
        scale_layout_m32k4=True,
    )
    _, x_scale_row = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
    )
    ref = run_torch(xq, x_scale_row, weight, w_scale)
    out, us = run_gemm(
        xq,
        shuffle_weight(weight, layout=(16, 16)),
        x_scale,
        shuffle_blockscale_to_mxfp8_scale(w_scale, n),
        dtype,
    )
    err = checkAllclose(ref, out.float(), msg=f"M={m} N={n} K={k}: ")
    libtype, kernel_name, splitk = _get_mxfp8_bpreshuffle_config(m, n, k)
    return {
        "libtype": libtype,
        "splitK": splitk,
        "us": round(us, 2),
        "TFLOPS": round(2 * m * n * k / us / 1e6, 1),
        "TB/s": round((m * k + n * k + m * n * 2) / us / 1e6, 2),
        "err": err,
        "kernelName": kernel_name,
    }


def tuned_shapes():
    table = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_MXFP8_BPRESHUFFLE_FILE)
    table = table[(table["gfx"] == get_gfx()) & (table["cu_num"] == get_cu_num())]
    return list(table[["M", "N", "K"]].itertuples(index=False, name=None))


if __name__ == "__main__":
    torch.set_default_device("cuda")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        help="""M,N,K shapes; default: every tuned shape of this arch.
        e.g.: -mnk 512,7168,16384 1000,65536,1536""",
    )
    args = parser.parse_args()

    if get_gfx() != "gfx1250":
        print(f"skip: gfx1250 only, got {get_gfx()}")
    else:
        df = pd.DataFrame([test_gemm(*mnk) for mnk in args.mnk or tuned_shapes()])
        aiter.logger.info(f"gemm_a8w8_mxfp8_bpreshuffle summary:\n{df.to_markdown()}")
