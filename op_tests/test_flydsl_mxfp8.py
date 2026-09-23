# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standard MXFP8 correctness + perf: E4M3 data, E8M0 scale per 32 K elements.

A[M,K], B[N,K], scales [M,K/32] and [N,K/32], caller-owned output.
B data may be shuffle_weight(B, (16,16)); scales remain unshuffled.
The default shapes are the MiniMax-M3 MXFP8 tuned table for this device.

    python op_tests/test_flydsl_mxfp8.py
    python op_tests/test_flydsl_mxfp8.py -s 32,2304,6144 -l plain preshuffle
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_mxfp8
from aiter.ops.gemm_op_mxfp8 import get_mxfp8_config
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx950"]


def default_mnk():
    # Same table as runtime/AOT, containing only MiniMax-M3 shapes.
    df = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_GEMM_MXFP8_FILE)
    df = df[df.gfx.eq(get_gfx()) & df.cu_num.eq(get_cu_num())]
    return list(
        df[["M", "N", "K"]].drop_duplicates().itertuples(index=False, name=None)
    )


def run_torch(a, w, scale_a, scale_b, dtype):
    # Reference only, never timed.
    sa = torch.exp2(scale_a.view(torch.uint8).float() - 127)
    sb = torch.exp2(scale_b.view(torch.uint8).float() - 127)
    a = a.float() * sa.repeat_interleave(32, dim=1)
    w = w.float() * sb.repeat_interleave(32, dim=1)
    return (a @ w.t()).to(dtype)


@benchmark()
def test_mxfp8(m, n, k, dtype, layout):
    torch.manual_seed(0)
    a = torch.empty(m, k, device="cuda").uniform_(-1, 1).to(dtypes.fp8)
    w = torch.empty(n, k, device="cuda").uniform_(-1, 1).to(dtypes.fp8)
    sa = torch.randint(124, 129, (m, k // 32), device="cuda", dtype=torch.uint8)
    sb = torch.randint(124, 129, (n, k // 32), device="cuda", dtype=torch.uint8)
    sa, sb = sa.view(dtypes.fp8_e8m0), sb.view(dtypes.fp8_e8m0)
    ref = run_torch(a, w, sa, sb, dtype)
    bp = layout == "preshuffle"
    weight = shuffle_weight(w) if bp else w
    out = torch.full((m, n), float("nan"), device="cuda", dtype=dtype)

    def run(a, w, sa, sb, out):
        return gemm_a8w8_mxfp8(a, w, sa, sb, dtype=dtype, out=out, bpreshuffle=bp)

    candidates = {"flydsl": run}
    config = get_mxfp8_config(m, n, k, dtype, False, bp)
    flops = 2 * m * n * k
    nbytes = sum(t.numel() * t.element_size() for t in (a, weight, sa, sb, out))
    ret = {
        "gfx": get_gfx(),
        "tile": f"{config['block_m']}x{config['block_n']}x{config['block_k']}",
        "split_k": config["split_k"],
        "k_waves": config["k_waves"],
        "direct_b": config["direct_b"],
    }
    for name, fn in candidates.items():
        # Explicit arguments actually rotate tensor buffers, unlike closures.
        y, us = run_perftest(fn, a, weight, sa, sb, out, num_rotate_args=3)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            y.to(dtypes.fp32),
            rtol=0.03,
            atol=0.1,
            tol_err_ratio=0,
            msg=f"{name}: standard MXFP8",
        )
        assert err == 0
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
        choices=[dtypes.bf16, dtypes.fp32],
        nargs="*",
        default="bf16,",
    )
    parser.add_argument(
        "-s", "--mnk", type=dtypes.str2tuple, nargs="*", default=default_mnk()
    )
    parser.add_argument(
        "-l",
        "--layout",
        choices=["plain", "preshuffle"],
        nargs="*",
        default=["preshuffle"],
    )
    args = parser.parse_args()
    for dtype in args.dtype:
        rows = [
            test_mxfp8(m, n, k, dtype, layout)
            for (m, n, k), layout in itertools.product(args.mnk, args.layout)
        ]
        aiter.logger.info(
            "standard MXFP8 summary (markdown):\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
