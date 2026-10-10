# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP8 (1x32 e8m0 scales) bpreshuffle GEMM.

Runs every tuned (M, N, K) of this arch by default (an untuned gfx950 falls
back to a default shape list), or the shapes given by -mnk, through
gemm_a8w8_mxfp8_bpreshuffle and checks it against an fp32 reference.

The 1x32 scales go in the layout each arch reads:
  * gfx950:  x_scale [M, K/32] and w_scale [N, K/32], row-major, unshuffled.
  * gfx1250: x_scale m32k4 / w_scale n32k4.
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

# Shapes for a gfx950 without tuned rows: DeepSeek-V3 dense projections, ragged M included.
DEFAULT_MNK_GFX950 = [
    (m, n, k)
    for n, k in ((2112, 7168), (7168, 2048), (24576, 1536), (7168, 16384))
    for m in (1, 16, 37, 64, 300, 1024, 4096)
]


def run_torch(x, x_scale, weight, w_scale, w_block=(128, 128)):
    """fp32 reference from row-major 1x32 A-scales and w_block B-scales."""
    m, k = x.shape
    n = weight.shape[0]
    x_scale = torch.exp2(x_scale.view(torch.uint8).float() - 127)
    x = (x.float().view(m, k // 32, 32) * x_scale[..., None]).view(m, k)
    w_scale = torch.exp2(w_scale.view(torch.uint8).float() - 127)
    w_scale = w_scale.repeat_interleave(w_block[0], 0)[:n]
    w_scale = w_scale.repeat_interleave(w_block[1], 1)
    return x @ (weight.float() * w_scale).T


@perftest()
def run_gemm(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_mxfp8_bpreshuffle(x, weight, x_scale, w_scale, dtype)


@benchmark()
def test_gemm(m, n, k, dtype=dtypes.bf16):
    x = torch.randn(m, k, dtype=dtypes.bf16)
    weight = (torch.randn(n, k) * 0.02).to(dtypes.fp8)
    if get_gfx() == "gfx950":
        # 1x32 weight scales around 2^-3; both scales go in row-major as is
        w_scale = torch.randint(122, 127, (n, k // 32), dtype=torch.uint8)
        w_scale = w_scale.view(dtypes.fp8_e8m0)
        xq, x_scale = per_group_quant_hip(
            x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
        )
        ref = run_torch(xq, x_scale, weight, w_scale, (1, 32))
        gemm_w_scale = w_scale
    else:
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
        gemm_w_scale = shuffle_blockscale_to_mxfp8_scale(w_scale, n)
    out, us = run_gemm(
        xq,
        shuffle_weight(weight, layout=(16, 16)),
        x_scale,
        gemm_w_scale,
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
    shapes = list(table[["M", "N", "K"]].itertuples(index=False, name=None))
    if not shapes and get_gfx() == "gfx950":
        return DEFAULT_MNK_GFX950  # not tuned on this machine yet
    return shapes


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

    if get_gfx() not in ("gfx950", "gfx1250"):
        print(f"skip: gfx950 / gfx1250 only, got {get_gfx()}")
    else:
        df = pd.DataFrame([test_gemm(*mnk) for mnk in args.mnk or tuned_shapes()])
        aiter.logger.info(f"gemm_a8w8_mxfp8_bpreshuffle summary:\n{df.to_markdown()}")
