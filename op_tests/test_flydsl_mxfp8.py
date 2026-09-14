# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for tuned FlyDSL MXFP8 GEMM (gfx950).

A[M,K], B[N,K], E8M0 scales, caller-owned out[M,N]. The block128 preshuffle
case reproduces DSv4's gemm_a8w8_blockscale_bpreshuffle call, including the
byte-transposed activation scales from per_group_quant_hip.

    python op_tests/test_flydsl_mxfp8.py
    python op_tests/test_flydsl_mxfp8.py -s 32,768,7168 -l preshuffle --scale-block 128
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.gemm_op_mxfp8 import gemm_mxfp8, get_mxfp8_config
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx950"]
# Subset of model_configs/dsv4_a8w8_blockscale_untuned_gemm.csv:
# M = tokens, [N,K] = the model's local dense linear weight shape.
DEFAULT_MNK = [
    (m, n, k)
    for m, (n, k) in itertools.product(
        (1, 32, 256, 4096), ((768, 7168), (2048, 7168), (7168, 384))
    )
]


def run_torch(a, w, scale_a, scale_b, scale_block, dtype):
    # Reference only; neither timed nor included as a table candidate.
    a_scale = torch.exp2(scale_a.view(torch.uint8).float() - 127)
    b_scale = torch.exp2(scale_b.view(torch.uint8).float() - 127)
    if scale_block == 128:
        b_scale = b_scale.repeat_interleave(128, dim=0)[: w.shape[0]]
    a = a.float() * a_scale.repeat_interleave(scale_block, dim=1)
    w = w.float() * b_scale.repeat_interleave(scale_block, dim=1)
    return (a @ w.t()).to(dtype)


@benchmark()
def test_mxfp8(m, n, k, dtype, layout, scale_block):
    torch.manual_seed(0)
    a = torch.empty(m, k, device="cuda").uniform_(-1, 1).to(dtypes.fp8)
    w = torch.empty(n, k, device="cuda").uniform_(-1, 1).to(dtypes.fp8)
    sa = torch.randint(
        124, 129, (m, k // scale_block), device="cuda", dtype=torch.uint8
    )
    sb = torch.randint(
        124,
        129,
        (n if scale_block == 32 else (n + 127) // 128, k // scale_block),
        device="cuda",
        dtype=torch.uint8,
    )
    sa, sb = sa.view(dtypes.fp8_e8m0), sb.view(dtypes.fp8_e8m0)
    ref = run_torch(a, w, sa, sb, scale_block, dtype)
    bpreshuffle = layout == "preshuffle"
    weight = shuffle_weight(w) if bpreshuffle else w
    out = torch.randn(m, n, device="cuda", dtype=dtype)
    if scale_block == 128 and bpreshuffle:
        # Model ABI: column-major bytes in a contiguous [M,K/128] scale tensor,
        # NOT just a transposed logical tensor. The kernel consumes these bytes.
        packed_sa = sa.t().contiguous().reshape(m, -1)
        candidates = {
            "flydsl": lambda: gemm_a8w8_blockscale_bpreshuffle(
                a, weight, packed_sa, sb, dtype=dtype, out=out
            )
        }
    else:
        candidates = {
            "flydsl": lambda: gemm_mxfp8(
                a,
                weight,
                sa,
                sb,
                dtype=dtype,
                out=out,
                scale_block=scale_block,
                bpreshuffle=bpreshuffle,
            )
        }

    flops = 2 * m * n * k
    nbytes = sum(t.numel() * t.element_size() for t in (a, weight, sa, sb, out))
    config = get_mxfp8_config(
        m,
        n,
        k,
        dtype,
        False,
        scale_block,
        bpreshuffle,
        scale_block == 128 and bpreshuffle,
    )
    ret = {
        "gfx": get_gfx(),
        "tile": f"{config['block_m']}x{config['block_n']}x{config['block_k']}",
        "stages": config["stages"],
        "split_k": config["split_k"],
        "k_waves": config["k_waves"],
    }
    for name, fn in candidates.items():
        y, us = run_perftest(fn)
        assert y.data_ptr() == out.data_ptr()
        err = checkAllclose(
            ref.to(dtypes.fp32),
            y.to(dtypes.fp32),
            rtol=0.03,
            atol=0.1,
            msg=f"{name}: MXFP8 GEMM",
            tol_err_ratio=0,
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("FlyDSL MXFP8 unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.bf16],
        nargs="*",
        default="bf16,",
    )
    parser.add_argument(
        "-s", "--mnk", type=dtypes.str2tuple, nargs="*", default=DEFAULT_MNK
    )
    parser.add_argument(
        "-l",
        "--layout",
        choices=["plain", "preshuffle"],
        nargs="*",
        default=["preshuffle"],
    )
    parser.add_argument(
        "--scale-block", type=int, choices=[32, 128], nargs="*", default=[128]
    )
    args = parser.parse_args()
    for dtype in args.dtype:
        rows = [
            test_mxfp8(m, n, k, dtype, layout, block)
            for (m, n, k), layout, block in itertools.product(
                args.mnk, args.layout, args.scale_block
            )
        ]
        aiter.logger.info(
            "FlyDSL MXFP8 summary (markdown):\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
