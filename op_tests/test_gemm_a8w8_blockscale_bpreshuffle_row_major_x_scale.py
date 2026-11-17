# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gemm_a8w8_blockscale_bpreshuffle with a row-major 1x128 e8m0 x_scale.

For every (M, N, K) the GEMM runs twice on the same operands: once on the
column-major x_scale (the default, ``x_scale_transposed=True``) and once on the
same scale bytes laid out row-major (``x_scale_transposed=False``). Only the
scale layout differs, so the outputs must match byte for byte; both are also
checked against an fp32 reference. The row-major time and whether
``gemm_a8w8_blockscale_bpreshuffle_prefers_row_major_x_scale`` picks it are
reported per shape.
"""

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.gemm_op_a8w8 import (
    gemm_a8w8_blockscale_bpreshuffle,
    gemm_a8w8_blockscale_bpreshuffle_prefers_row_major_x_scale,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, perftest

# Kimi-K3 TP8 projections that read a fused norm's E8M0 activation scale.
DEFAULT_MNK = [
    (m, n, 7168) for n in (2176, 1536, 6400, 8448) for m in (1, 58, 256, 545, 1024)
]


def run_torch(x, x_scale, weight, w_scale):
    m, k = x.shape
    n = weight.shape[0]
    xs = torch.exp2(x_scale.view(torch.uint8).float() - 127)
    x = (x.float().view(m, k // 128, 128) * xs[..., None]).view(m, k)
    ws = torch.exp2(w_scale.view(torch.uint8).float() - 127)
    ws = ws.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return x @ (weight.float() * ws).T


@perftest()
def run_gemm(x, weight, x_scale, w_scale, x_scale_transposed):
    return gemm_a8w8_blockscale_bpreshuffle(
        x, weight, x_scale, w_scale, dtypes.bf16, x_scale_transposed=x_scale_transposed
    )


@benchmark()
def test_gemm(m, n, k):
    x = (torch.randn(m, k) * 2).to(dtypes.fp8)
    weight = (torch.randn(n, k) * 0.5).to(dtypes.fp8)
    codes = torch.randint(118, 124, (m, k // 128), dtype=torch.uint8)
    w_scale = torch.randint(118, 122, (n // 128, k // 128), dtype=torch.uint8)
    w_scale = w_scale.view(dtypes.fp8_e8m0)
    row = codes.contiguous().view(dtypes.fp8_e8m0)
    # column-major bytes behind an (M, K // 128) tensor, as transpose_scale emits
    col = codes.t().contiguous().view(m, k // 128).view(dtypes.fp8_e8m0)
    wsh = shuffle_weight(weight, layout=(16, 16))
    ref = run_torch(x, row, weight, w_scale)
    out_col, us_col = run_gemm(x, wsh, col, w_scale, True)
    out_row, us_row = run_gemm(x, wsh, row, w_scale, False)
    same = torch.equal(out_col.view(torch.int16), out_row.view(torch.int16))
    err = checkAllclose(ref, out_row.float(), msg=f"M={m} N={n} K={k}: ")
    assert same, f"M={m} N={n} K={k}: row-major and column-major outputs differ"
    return {
        "prefers_row_major": gemm_a8w8_blockscale_bpreshuffle_prefers_row_major_x_scale(
            m, n, k
        ),
        "us_col": round(us_col, 2),
        "us_row": round(us_row, 2),
        "bytes_equal": same,
        "err": err,
    }


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
        help="""M,N,K shapes; default: Kimi-K3 TP8 projections.
        e.g.: -mnk 545,2176,7168 1024,6400,7168""",
    )
    args = parser.parse_args()
    df = pd.DataFrame([test_gemm(*mnk) for mnk in args.mnk or DEFAULT_MNK])
    aiter.logger.info(
        f"gemm_a8w8_blockscale_bpreshuffle row-major x_scale on {get_gfx()}:\n"
        f"{df.to_markdown()}"
    )
