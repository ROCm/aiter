# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import functools
import os
import sys

# Add parent directory to path to ensure we use local aiter module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat as eirp

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.gemm_op_a8w8 import (
    gemm_a8w8_blockscale_ck,
    gemm_a8w8_blockscale_cktile,
    gemm_a8w8_mxfp8_128_bpreshuffle_flydsl,
    get_CKGEMM_config,
)
from aiter.ops.shuffle import shuffle_mxfp8fp4_a, shuffle_weight
from aiter.test_common import benchmark, checkAllclose, perftest
from aiter.utility import fp4_utils

block_shape = (128, 128)
TEST_NUM_ITERS = 100

# Not under aiter/configs/: those feed the production dispatch, which hands the
# kernel a row-major A and so must never select an "_apre" kernel.
APRE_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs",
    "flydsl_a8w8_apre_tuned_gemm.csv",
)


@functools.lru_cache(maxsize=1)
def _apre_configs():
    if not os.path.exists(APRE_CONFIG_FILE):
        return {}
    df = pd.read_csv(APRE_CONFIG_FILE)
    df = df[df["gfx"] == get_gfx()]
    return {(r.M, r.N, r.K): r.kernelName for r in df.itertuples()}


def apre_kernel_name(m, n, k, tuned_name):
    """Best a_preshuffle kernel for this shape, else the tuned config with the
    flag flipped -- valid, but usually not the optimal apre config."""
    return _apre_configs().get((m, n, k), tuned_name + "_apre")


@perftest(num_iters=TEST_NUM_ITERS)
def run_torch(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    if x_scale.dtype == dtypes.fp8_e8m0:
        x_scale = fp4_utils.e8m0_to_f32(x_scale)
    if w_scale.dtype == dtypes.fp8_e8m0:
        w_scale = fp4_utils.e8m0_to_f32(w_scale)
    x = x.to(x_scale.dtype).view(
        m, k // block_shape[1], block_shape[1]
    ) * x_scale.unsqueeze(-1)
    x = x.view(m, k)

    w_scale = rearrange(
        w_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k),
        "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
    )
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32))
    return out.to(dtype)


@perftest(num_iters=TEST_NUM_ITERS)
def run_gemm(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_blockscale(x, weight, x_scale, w_scale, dtype)


@perftest(num_iters=TEST_NUM_ITERS)
def run_gemm_bpreshuffle(x, weightshuffle, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_blockscale_bpreshuffle(
        x, weightshuffle, x_scale, w_scale, dtype
    )


@perftest(num_iters=TEST_NUM_ITERS)
def run_gemm_flydsl(x, weightshuffle, x_scale, w_scale, dtype, kernel_name):
    # Out is allocated here, as gemm_a8w8_blockscale_bpreshuffle does internally:
    # perftest rotates deep copies of its arguments to defeat L2 reuse, so an out
    # buffer passed in would inflate that working set (64MB at n=65536).
    out = torch.empty(x.shape[0], weightshuffle.shape[0], dtype=dtype, device=x.device)
    return gemm_a8w8_mxfp8_128_bpreshuffle_flydsl(
        x,
        weightshuffle,
        x_scale,
        w_scale,
        out,
        {"kernelName": kernel_name, "a_preshuffled_input": True},
    )


@perftest(num_iters=TEST_NUM_ITERS)
def run_triton(x, weightshuffle, x_scale, w_scale, dtype=dtypes.bf16, backend=None):
    # Direct call into the triton preshuffle kernel, mirroring the dispatch in
    # gemm_a8w8_blockscale_bpreshuffle: reshape the (n, k) preshuffled weight to
    # (n // 16, k * 16) and pass the transposed x_scale.
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale_preshuffle,
    )

    n, k = weightshuffle.shape
    return gemm_a8w8_blockscale_preshuffle(
        x,
        weightshuffle.reshape(n // 16, k * 16),
        x_scale,
        w_scale,
        dtype=dtype,
        backend=backend,
    )


# Input initialization modes. "random" is the historical init (uniform / 10, random
# per-block scales) and stresses accuracy; "const" fills A/B with a single exactly
# representable fp8 value and neutral (1.0) scales, so the numbers are deterministic
# and free of denormal/outlier effects -- the init used for stable perf comparison.
CONST_VAL = 0.5


def make_inputs(init, m, n, k, scale_m, scale_n, scale_k):
    if init == "const":
        x = torch.full((m, k), CONST_VAL, dtype=dtypes.fp32, device="cuda").to(
            dtypes.fp8
        )
        weight = torch.full((n, k), CONST_VAL, dtype=dtypes.fp32, device="cuda").to(
            dtypes.fp8
        )
        x_scale = torch.ones([scale_m, scale_k], dtype=dtypes.fp32, device="cuda")
        w_scale = torch.ones([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")
    else:
        x = (torch.rand((m, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
        weight = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(
            dtypes.fp8
        )
        x_scale = torch.rand([scale_m, scale_k], dtype=dtypes.fp32, device="cuda")
        w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")
    return x, weight, x_scale, w_scale


@benchmark()
def test_gemm(
    dtype, m, n, k, ck_preshuffle=True, use_flydsl=False, init="random", apre=False
):
    ret = {}
    block_shape_n, block_shape_k = block_shape
    scale_m = m
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x, weight, x_scale, w_scale = make_inputs(init, m, n, k, scale_m, scale_n, scale_k)
    use_flydsl_fp8_scale = use_flydsl and ck_preshuffle
    if use_flydsl_fp8_scale:
        FP8_E4M3_MAX = 448.0
        x_scale = fp4_utils.f32_to_mx_e8m0_scale(
            x_scale * FP8_E4M3_MAX, dtype=fp4_utils.MxDtypeInt.FP8_E4M3
        )
        w_scale = fp4_utils.f32_to_mx_e8m0_scale(
            w_scale * FP8_E4M3_MAX, dtype=fp4_utils.MxDtypeInt.FP8_E4M3
        )

    a, _ = run_torch(x, weight, x_scale, w_scale, dtype)

    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    gemm_x_scale = x_scale_t if ck_preshuffle else x_scale
    gemm_weight = shuffle_weight(weight, layout=(16, 16)) if ck_preshuffle else weight
    run_func = run_gemm_bpreshuffle if ck_preshuffle else run_gemm
    b, avg_b = run_func(x, gemm_weight, gemm_x_scale, w_scale, dtype)

    err_ck = checkAllclose(a, b, msg="ck", catastrophic_check=True)
    if ck_preshuffle:
        x_scale_strided = x_scale.transpose(0, 1).contiguous().transpose(0, 1)
        b_strided = aiter.gemm_a8w8_blockscale_bpreshuffle(
            x, gemm_weight, x_scale_strided, w_scale, dtype
        )
        checkAllclose(
            a,
            b_strided,
            msg="ck strided x_scale",
            catastrophic_check=True,
        )
    ret["ck us"] = avg_b
    ret["ck TFLOPS"] = m * n * k * 2 / avg_b / 1e6
    ret["ck TB/s"] = (x.nbytes + weight.nbytes) / avg_b / 1e6
    ret["ck err"] = err_ck

    # A-preshuffle candidate, measured against the "ck" columns: on gfx1250 with
    # e8m0 scales that dispatch lands on the same FlyDSL kernel, so "ck" is the
    # apre=0 baseline. a_preshuffle changes which config is optimal, so the
    # candidate comes from APRE_CONFIG_FILE rather than the apre=0 winner with
    # the flag flipped. Needs a tuned FlyDSL row and an even M.
    if use_flydsl_fp8_scale and apre:
        cfg = get_CKGEMM_config(
            m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
        )
        knl = str(cfg.get("kernelName", "")) if cfg is not None else ""
        if knl and cfg.get("libtype") == "flydsl" and m % 2 == 0:
            e, avg_e = run_gemm_flydsl(
                shuffle_mxfp8fp4_a(x),
                gemm_weight,
                gemm_x_scale,
                w_scale,
                dtype,
                apre_kernel_name(m, n, k, knl),
            )
            ret["apre us"] = avg_e
            ret["apre TFLOPS"] = m * n * k * 2 / avg_e / 1e6
            ret["apre TB/s"] = (x.nbytes + weight.nbytes) / avg_e / 1e6
            ret["apre err"] = checkAllclose(a, e, msg="apre", catastrophic_check=True)
        else:
            aiter.logger.warning(
                "skipping apre for M=%s N=%s K=%s (tuned flydsl row=%s, M even=%s)",
                m,
                n,
                k,
                bool(knl) and cfg.get("libtype") == "flydsl",
                m % 2 == 0,
            )

    if not use_flydsl_fp8_scale:
        tag = "asm"
        weight_asm = shuffle_weight(weight, layout=(16, 16))
        c, avg_c = run_asm(x, weight_asm, x_scale_t, w_scale, dtype)

        err_asm = checkAllclose(a, c, msg=f"{tag}", catastrophic_check=True)
        ret[f"{tag} us"] = avg_c
        ret[f"{tag} TFLOPS"] = m * n * k * 2 / avg_c / 1e6
        ret[f"{tag} TB/s"] = (x.nbytes + weight.nbytes) / avg_c / 1e6
        ret[f"{tag} err"] = err_asm
        ret["asm/ck"] = avg_c / avg_b

        # Triton path requires a preshuffled weight. When not preshuffled we simply omit
        # these columns; pd.DataFrame NaN-fills them for those rows in the summary.
        if ck_preshuffle:
            d, avg_d = run_triton(x, gemm_weight, x_scale_t, w_scale, dtype)
            err_triton = checkAllclose(a, d, msg="triton", catastrophic_check=True)
            ret["triton us"] = avg_d
            ret["triton TFLOPS"] = m * n * k * 2 / avg_d / 1e6
            ret["triton TB/s"] = (x.nbytes + weight.nbytes) / avg_d / 1e6
            ret["triton err"] = err_triton
            ret["triton/ck"] = avg_d / avg_b

    return ret


@perftest(num_iters=TEST_NUM_ITERS)
def run_torch2(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]

    x_scale_ = eirp(x_scale, "m k -> m (k repeat)", repeat=block_shape_k)
    x_scale_ = x_scale_[:m, :k]

    w_scale_ = eirp(w_scale, "n k -> (n repeat) k", repeat=block_shape_n)
    w_scale_ = eirp(w_scale_, "n k -> n (k repeat)", repeat=block_shape_k)
    w_scale_ = w_scale_[:n, :k]

    x_ = x.to(x_scale.dtype) * x_scale_
    weight_ = weight.to(w_scale.dtype) * w_scale_

    out = F.linear(x_.to(dtypes.fp32), weight_.to(dtypes.fp32))
    return out.to(dtype)


@perftest(num_iters=TEST_NUM_ITERS)
def run_asm(x, weight, x_scale, w_scale, dtype=dtypes.bf16, kernel_name=None):
    m, _k = x.shape
    n, _ = weight.shape
    out = torch.empty((m, n), dtype=dtype, device=x.device)
    return aiter.gemm_a8w8_blockscale_bpreshuffle_asm(x, weight, out, x_scale, w_scale)


def test_splitk_correctness(m=4, n=2112, k=7168, dtype=dtypes.bf16, splitK=1):
    """Verify that splitK > 0 produces the same output as splitK=0 (within fp tolerance).

    split-K accumulates partial tiles via atomic_add, which changes the floating-point
    reduction order.  We therefore use a relaxed tolerance that matches the cumulative
    rounding error introduced by K-splitting.
    """
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    x = (torch.rand((m, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")

    # CK path (no preshuffle): compare splitK=0 vs splitK>0
    Y_base = torch.empty((m, n), dtype=dtype, device="cuda")
    Y_split = torch.empty((m, n), dtype=dtype, device="cuda")
    gemm_a8w8_blockscale_ck(x, weight, x_scale, w_scale, Y_base, splitK=0)
    gemm_a8w8_blockscale_ck(x, weight, x_scale, w_scale, Y_split, splitK=splitK)
    ck_err = checkAllclose(
        Y_base,
        Y_split,
        msg=f"ck splitK={splitK} vs splitK=0",
        rtol=1e-2,
        atol=1e-2,
        catastrophic_check=True,
    )

    # CKTile path (no preshuffle): compare splitK=0 vs splitK>0
    Y_base_tile = torch.empty((m, n), dtype=dtype, device="cuda")
    Y_split_tile = torch.empty((m, n), dtype=dtype, device="cuda")
    gemm_a8w8_blockscale_cktile(
        x, weight, x_scale, w_scale, Y_base_tile, False, splitK=0
    )
    gemm_a8w8_blockscale_cktile(
        x, weight, x_scale, w_scale, Y_split_tile, False, splitK=splitK
    )
    cktile_err = checkAllclose(
        Y_base_tile,
        Y_split_tile,
        msg=f"cktile splitK={splitK} vs splitK=0",
        rtol=1e-2,
        atol=1e-2,
        catastrophic_check=True,
    )

    print(
        f"test_splitk_correctness(m={m}, n={n}, k={k}, splitK={splitK}): "
        f"ck_err={ck_err:.4g}, cktile_err={cktile_err:.4g}"
    )


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"]],
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16}",
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        96,
        128,
        160,
        192,
        224,
        256,
        288,
        320,
        352,
        384,
        416,
        448,
        480,
        512,
        1024,
        2048,
        4096,
        6144,
        8192,
        10240,
    ],
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-nk",
    type=dtypes.str2tuple,
    nargs="*",
    default=[
        (24576, 1536),
        # (32768, 512),
        # (7168, 16384),
        # (36864, 7168),
    ],
    help="""N&K of mnk.
    e.g.: -nk 24576,1536""",
)
parser.add_argument(
    "--ck_preshuffle",
    type=dtypes.str2bool,
    nargs="*",
    default=[True, False],
    help="""weight ck_preshuffle or not.
    e.g.: --ck_preshuffle True
        or --ck_preshuffle False
    """,
)
parser.add_argument(
    "--flydsl",
    action="store_true",
    help="use flydsl fp8 e8m0 scale path (requires --ck_preshuffle True)",
)
parser.add_argument(
    "--apre",
    type=dtypes.str2bool,
    nargs="*",
    default=[False],
    help="""also measure the FlyDSL A-preshuffle candidate (requires --flydsl
    --ck_preshuffle True and an even M). Sweeps like --ck_preshuffle.
    e.g.: --apre True
        or --apre True False""",
)
parser.add_argument(
    "--init",
    type=str,
    nargs="*",
    choices=["const", "random"],
    default=["const", "random"],
    help="""input init mode(s); default runs both.
    const  = x/weight filled with 0.5 and neutral (1.0) scales (deterministic)
    random = uniform(0,1)/10 data with random per-block scales
    e.g.: --init const""",
)
parser.add_argument(
    "--csv",
    type=str,
    default=None,
    help="""CSV file containing M, N, K columns (one shape per row).
    e.g.: --csv shapes.csv""",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="""Directory to save results CSV.
    e.g.: -o results/""",
)
parser.add_argument(
    "--suffix",
    type=str,
    default="results",
    help="""Suffix for output CSV filename.
    e.g.: --suffix branch""",
)

args = parser.parse_args()

l_preshuffle = (
    args.ck_preshuffle if isinstance(args.ck_preshuffle, list) else [args.ck_preshuffle]
)
l_apre = args.apre if isinstance(args.apre, list) else [args.apre]

df = []
if args.csv is not None:
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    shapes_df = pd.read_csv(args.csv)
    print(f"Loaded {len(shapes_df)} shapes from {args.csv}", flush=True)
    for dtype in args.dtype:
        for init in args.init:
            for preshuffle in l_preshuffle:
                for apre in l_apre:
                    for _, row in shapes_df.iterrows():
                        ret = test_gemm(
                            dtype,
                            int(row["M"]),
                            int(row["N"]),
                            int(row["K"]),
                            ck_preshuffle=preshuffle,
                            use_flydsl=args.flydsl,
                            init=init,
                            apre=apre,
                        )
                        df.append(ret)
else:
    for dtype in args.dtype:
        for init in args.init:
            for m in args.m:
                for n, k in args.nk:
                    for ck_p in l_preshuffle:
                        for apre in l_apre:
                            ret = test_gemm(
                                dtype,
                                m,
                                n,
                                k,
                                ck_preshuffle=ck_p,
                                use_flydsl=args.flydsl,
                                init=init,
                                apre=apre,
                            )
                            df.append(ret)

df = pd.DataFrame(df)

# Configure pandas to show all columns without truncation
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)

print("\n" + "=" * 150)
print("COMPLETE PERFORMANCE SUMMARY (All Columns)")
print("=" * 150)
print(df.to_string(index=False))
print("=" * 150)

df_md = df.to_markdown(index=False)
aiter.logger.info("gemm_a8w8_blockscale summary (markdown):\n%s", df_md)

# Correctness check: verify split-K produces matching results
print("\nRunning split-K correctness checks ...")
for splitK in [1, 2]:
    test_splitk_correctness(m=4, n=512, k=16384, splitK=splitK)

# Save results from benchmarks
if args.output:
    os.makedirs(args.output, exist_ok=True)
    if args.csv:
        csv_filename = os.path.basename(args.csv).replace(".csv", f"_{args.suffix}.csv")
    else:
        csv_filename = f"gemm_a8w8_blockscale_{args.suffix}.csv"
    out_path = os.path.join(args.output, csv_filename)
    df.to_csv(out_path, index=False)
    print(f"Saved results to: {out_path}")
