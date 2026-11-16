# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP8 (1x32 e8m0 scales) bpreshuffle GEMM.

Runs every tuned (M, N, K) of this arch by default (an untuned gfx950 falls
back to a default shape list), or the shapes given by -mnk, through
gemm_a8w8_mxfp8_bpreshuffle and checks it against an fp32 reference.

Both gfx950 and gfx1250 read row-major x_scale [M, K/32]. gfx950 reads
row-major w_scale [N, K/32]; gfx1250 reads 32x4 shuffled w_scale.
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
from aiter.ops.shuffle import shuffle_mxfp8fp4_scale, shuffle_weight
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
    w_scale = torch.randint(122, 127, (n, k // 32), dtype=torch.uint8)
    w_scale = w_scale.view(dtypes.fp8_e8m0)
    xq, x_scale = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
    )
    ref = run_torch(xq, x_scale, weight, w_scale, (1, 32))
    out, us = run_gemm(
        xq,
        shuffle_weight(weight, layout=(16, 16)),
        x_scale,
        shuffle_mxfp8fp4_scale(w_scale) if get_gfx() == "gfx1250" else w_scale,
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


def check_large_a():
    """gfx950's FlyDSL kernel past 4 GiB of A (and of C) stays exact.

    Each workgroup reads A from its own first row through a 64-bit base;
    unfixed, 91% of the output was wrong at a 4 GiB A. Inputs in {-1, 0, 1}
    with unit scales make every fp32 sum exact, so the bf16 output must equal
    the rounded reference bit for bit; M leaves a partial last tile. The split-K fp32 workspace is still read
    whole and must stay below 4 GiB.
    """
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
        run_gemm_a8w8_mxfp8_gfx950,
    )

    limit, n, k = 1 << 32, 128, 512
    free, _ = torch.cuda.mem_get_info()
    if free < 3 * limit:
        aiter.logger.info(
            "skipping the past-4 GiB A check: %.1f GiB free", free / 2**30
        )
        return
    m = limit // k + 100
    step = 1 << 20

    def ints(rows):
        return torch.randint(-1, 2, (rows, k), dtype=torch.int8).to(dtypes.bf16)

    torch.manual_seed(0)
    xq = torch.empty((m, k), dtype=dtypes.fp8)
    for i in range(0, m, step):
        xq[i : i + step] = ints(min(step, m - i)).to(dtypes.fp8)
    w = ints(n).to(dtypes.fp8)
    x_scale = torch.full((m, k // 32), 127, dtype=torch.uint8).view(dtypes.fp8_e8m0)
    w_scale = torch.full((n, k // 32), 127, dtype=torch.uint8).view(dtypes.fp8_e8m0)
    wsh = shuffle_weight(w, layout=(16, 16))
    for name in (None, "flydsl_mxpsh_16x32x256_F8_F8_B16_w2_x0_sk1_pf_nt"):
        out = torch.empty((m, n), dtype=dtypes.bf16)
        run_gemm_a8w8_mxfp8_gfx950(xq, wsh, x_scale, w_scale, out, kernel_name=name)
        for i in range(0, m, step):
            ref = (xq[i : i + step].float() @ w.float().t()).to(dtypes.bf16)
            assert torch.equal(out[i : i + step], ref), f"{name}: rows {i}+ differ"
        del out
    del xq, x_scale
    torch.cuda.empty_cache()

    m, n, k = 262244, 2048, 512
    try:
        run_gemm_a8w8_mxfp8_gfx950(
            torch.zeros((m, k), dtype=dtypes.fp8),
            shuffle_weight(torch.zeros((n, k), dtype=dtypes.fp8), layout=(16, 16)),
            torch.full((m, k // 32), 127, dtype=torch.uint8).view(dtypes.fp8_e8m0),
            torch.full((n, k // 32), 127, dtype=torch.uint8).view(dtypes.fp8_e8m0),
            torch.empty((m, n), dtype=dtypes.bf16),
            kernel_name="flydsl_mxpsh_128x256x128_F8_F8_B16_w0_x4_sk2",
        )
    except ValueError as e:
        assert "workspace" in str(e), e
    else:
        raise AssertionError("a 4 GiB split-K workspace was launched")
    aiter.logger.info("check_large_a: A and C past 4 GiB exact; workspace limit kept")


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
        if get_gfx() == "gfx950":
            check_large_a()
