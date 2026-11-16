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
import itertools

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
from aiter.test_common import benchmark, checkAllclose, perftest, run_perftest

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


@benchmark()
def test_8wave(m, n, k, w_scale_block, block_m, b_non_temporal):
    """Compare the public model dispatch and an explicitly selected 8-wave tile."""
    from aiter.ops.flydsl.gemm_a8w8_mxfp8_8wave import flydsl_mxfp8_8wave_gemm

    group_n = int(w_scale_block.split("x")[0])
    x = torch.randn(m, k, dtype=dtypes.bf16)
    weight = (torch.randn(n, k) * 0.5).to(dtypes.fp8)
    w_scale = torch.randint(122, 130, (n // group_n, k // 32), dtype=torch.uint8)
    w_scale = w_scale.view(dtypes.fp8_e8m0)
    xq, x_scale = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
    )
    ref = run_torch(xq, x_scale, weight, w_scale, (group_n, 32))
    packed = shuffle_weight(weight, layout=(16, 16))
    out = torch.empty(m, n, dtype=dtypes.bf16)
    candidates = {
        "dispatch": lambda: aiter.gemm_a8w8_blockscale_bpreshuffle(
            xq, packed, x_scale, w_scale, out=out
        ),
        "8wave": lambda: flydsl_mxfp8_8wave_gemm(
            xq,
            packed,
            x_scale,
            w_scale,
            out,
            block_m=block_m,
            b_non_temporal=b_non_temporal,
            waves_per_eu=2,
            xcd_swizzle=4,
            w_scale_block=w_scale_block,
        ),
    }
    flops = 2 * m * n * k
    nbytes = m * k + n * k + m * n * 2 + x_scale.numel() + w_scale.numel()
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        y, us = run_perftest(fn)
        err = checkAllclose(
            ref.float(),
            y.float(),
            rtol=0.01,
            atol=0.01,
            tol_err_ratio=0.01,
            msg=f"{name}: MXFP8 8-wave",
        )
        relative = (y.float() - ref).abs().mean() / ref.abs().mean().clamp_min(1e-9)
        assert err < 0.01 and relative <= 0.003
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def tuned_shapes():
    table = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_MXFP8_BPRESHUFFLE_FILE)
    table = table[(table["gfx"] == get_gfx()) & (table["cu_num"] == get_cu_num())]
    shapes = list(table[["M", "N", "K"]].itertuples(index=False, name=None))
    if not shapes and get_gfx() == "gfx950":
        return DEFAULT_MNK_GFX950  # not tuned on this machine yet
    return shapes


def main():
    torch.set_default_device("cuda")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-s",
        "--mnk",
        "-mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        help="""M,N,K shapes; default: every tuned shape of this arch.
        e.g.: -mnk 512,7168,16384 1000,65536,1536""",
    )
    parser.add_argument(
        "--block-m",
        nargs="*",
        type=int,
        choices=(128, 192, 256),
        default=[],
        help="8-wave tile M candidates (gfx950 only)",
    )
    parser.add_argument(
        "--w-scale-block", nargs="*", choices=("1x32", "32x32"), default=["1x32"]
    )
    parser.add_argument(
        "--b-non-temporal", nargs="*", type=int, choices=(0, 1), default=[0]
    )
    args = parser.parse_args()

    if get_gfx() not in ("gfx950", "gfx1250"):
        print(f"skip: gfx950 / gfx1250 only, got {get_gfx()}")
    elif args.block_m:
        if get_gfx() != "gfx950":
            aiter.logger.info("8-wave tiles require gfx950; skipping")
            return
        rows = [
            test_8wave(*mnk, block, bm, bool(bnt))
            for mnk, block, bm, bnt in itertools.product(
                args.mnk or DEFAULT_MNK_GFX950,
                args.w_scale_block,
                args.block_m,
                args.b_non_temporal,
            )
        ]
        aiter.logger.info(
            "8-wave MXFP8 summary:\n%s", pd.DataFrame(rows).to_markdown(index=False)
        )
    else:
        df = pd.DataFrame([test_gemm(*mnk) for mnk in args.mnk or tuned_shapes()])
        aiter.logger.info(f"gemm_a8w8_mxfp8_bpreshuffle summary:\n{df.to_markdown()}")


if __name__ == "__main__":
    main()
