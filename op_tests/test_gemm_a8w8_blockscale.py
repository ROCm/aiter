# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
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
from aiter import benchmark_data_init as bench_init
from aiter import dtypes
from aiter.benchmark_reporting import print_json_table
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_ck, gemm_a8w8_blockscale_cktile
from aiter.ops.shuffle import (
    shuffle_mxfp8fp4_a,
    shuffle_mxfp8fp4_scale,
    shuffle_weight,
)
from aiter.test_common import benchmark, checkAllclose, perftest
from aiter.utility import fp4_utils

block_shape = (128, 128)
SCALE_BLOCK_SHAPES = {"1x32": (1, 32), "32x32": (32, 32), "128x128": block_shape}
# Native E8M0 preshuffle dispatch is supported on these architectures.
NATIVE_SCALE_GFX = ("gfx950", "gfx1250")
TEST_NUM_ITERS = 100


@perftest(num_iters=TEST_NUM_ITERS)
def run_torch(x, weight, x_scale, w_scale, dtype=dtypes.bf16, scale_shape=block_shape):
    block_shape_n, block_shape_k = scale_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    if x_scale.dtype == dtypes.fp8_e8m0:
        x_scale = fp4_utils.e8m0_to_f32(x_scale)
    if w_scale.dtype == dtypes.fp8_e8m0:
        w_scale = fp4_utils.e8m0_to_f32(w_scale)
    x = x.to(x_scale.dtype).view(
        m, k // block_shape_k, block_shape_k
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
def run_gemm_abpreshuffle(
    x_shuffled, weightshuffle, x_scale, w_scale, dtype=dtypes.bf16
):
    return aiter.gemm_a8w8_blockscale_abpreshuffle(
        x_shuffled, weightshuffle, x_scale, w_scale, dtype
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


@benchmark()
def test_gemm(
    dtype,
    m,
    n,
    k,
    b_preshuffle=True,
    use_flydsl=False,
    data_init="uniform",
    scale_init="auto",
    seed=0,
    apre=False,
    w_scale_block="128x128",
):
    ret = {}
    if w_scale_block not in SCALE_BLOCK_SHAPES:
        raise ValueError(f"unsupported w_scale_block {w_scale_block!r}")
    scale_shape = SCALE_BLOCK_SHAPES[w_scale_block]
    use_flydsl_fp8_scale = (use_flydsl and b_preshuffle) or w_scale_block != "128x128"
    if w_scale_block != "128x128" and apre:
        raise ValueError("Group32 scales do not support the A-preshuffle candidate")

    block_shape_n, block_shape_k = scale_shape
    scale_m = m
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    generator = bench_init.make_generator(seed)
    if data_init == "constant":
        x = torch.full((m, k), 0.5, dtype=dtypes.fp32, device="cuda").to(dtypes.fp8)
        weight = torch.full((n, k), 0.5, dtype=dtypes.fp32, device="cuda").to(
            dtypes.fp8
        )
    else:
        x = bench_init.fill_fp8((m, k), data_init, generator, dtype=dtypes.fp8)
        weight = bench_init.fill_fp8((n, k), data_init, generator, dtype=dtypes.fp8)

    if scale_init == "constant":
        x_scale_raw = torch.full(
            (scale_m, scale_k), 0x7F, dtype=torch.uint8, device="cuda"
        )
        w_scale_raw = torch.full(
            (scale_n, scale_k), 0x7F, dtype=torch.uint8, device="cuda"
        )
    else:
        x_scale_raw = bench_init.fill_scale_e8m0(
            (scale_m, scale_k), scale_init, generator
        )
        w_scale_raw = bench_init.fill_scale_e8m0(
            (scale_n, scale_k), scale_init, generator
        )
    if use_flydsl_fp8_scale:
        x_scale = x_scale_raw.view(dtypes.fp8_e8m0)
        w_scale = w_scale_raw.view(dtypes.fp8_e8m0)
    else:
        x_scale = fp4_utils.e8m0_to_f32(x_scale_raw)
        w_scale = fp4_utils.e8m0_to_f32(w_scale_raw)

    a, _ = run_torch(x, weight, x_scale, w_scale, dtype, scale_shape)

    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    gemm_weight = shuffle_weight(weight, layout=(16, 16)) if b_preshuffle else weight
    gemm_x_scale = x_scale if block_shape_k == 32 or not b_preshuffle else x_scale_t
    run_func = run_gemm_bpreshuffle if b_preshuffle else run_gemm
    gemm_w_scale = (
        shuffle_mxfp8fp4_scale(w_scale)
        if b_preshuffle and w_scale_block == "1x32" and get_gfx() == "gfx1250"
        else w_scale
    )
    b, avg_b = run_func(x, gemm_weight, gemm_x_scale, gemm_w_scale, dtype)

    if a.numel() < 1 << 31:
        err = checkAllclose(a, b, msg="gemm", catastrophic_check=True)
    else:
        # Keep each HIP boolean mask below the int32 element-count limit.
        err = 0.0
        for a_part, b_part in zip(
            a.flatten().split(1 << 30), b.flatten().split(1 << 30)
        ):
            part_err = checkAllclose(
                a_part, b_part, msg="gemm", catastrophic_check=True
            )
            err += part_err * a_part.numel()
        err /= a.numel()
    if b_preshuffle and not use_flydsl_fp8_scale:
        x_scale_strided = x_scale.transpose(0, 1).contiguous().transpose(0, 1)
        b_strided = aiter.gemm_a8w8_blockscale_bpreshuffle(
            x, gemm_weight, x_scale_strided, w_scale, dtype
        )
        checkAllclose(
            a,
            b_strided,
            msg="strided x_scale",
            catastrophic_check=True,
        )
    ret["gfx"] = get_gfx()
    ret["w_scale_block"] = w_scale_block
    ret["us"] = avg_b
    ret["TFLOPS"] = m * n * k * 2 / avg_b / 1e6
    # Effective bandwidth in TB/s: one read of A/B/scales and one write of D.
    num_bytes = sum(t.nbytes for t in (x, gemm_weight, gemm_x_scale, gemm_w_scale, b))
    ret["BW"] = num_bytes / avg_b / 1e6
    ret["err"] = err

    if apre and use_flydsl_fp8_scale:
        # A-preshuffle packs adjacent A row pairs, so an odd M needs A -- and only
        # A -- padded to M+1 rows; x_scale and the result keep the true M.
        if m % 2:
            x_apre = torch.zeros((m + 1, k), dtype=x.dtype, device=x.device)
            x_apre[:m] = x
        else:
            x_apre = x
        e, avg_e = run_gemm_abpreshuffle(
            shuffle_mxfp8fp4_a(x_apre), gemm_weight, gemm_x_scale, w_scale, dtype
        )
        ret["apre us"] = avg_e
        ret["apre TFLOPS"] = m * n * k * 2 / avg_e / 1e6
        num_bytes = sum(
            t.nbytes for t in (x_apre, gemm_weight, gemm_x_scale, w_scale, e)
        )
        ret["apre BW"] = num_bytes / avg_e / 1e6
        ret["apre err"] = checkAllclose(a, e, msg="apre", catastrophic_check=True)
        ret["apre/base"] = avg_e / avg_b

    if not use_flydsl_fp8_scale:
        # The ASM blockscale implementation has no gfx1250 kernel.
        if get_gfx() != "gfx1250":
            tag = "asm"
            weight_asm = shuffle_weight(weight, layout=(16, 16))
            c, avg_c = run_asm(x, weight_asm, x_scale_t, w_scale, dtype)

            err_asm = checkAllclose(a, c, msg=f"{tag}", catastrophic_check=True)
            ret[f"{tag} us"] = avg_c
            ret[f"{tag} TFLOPS"] = m * n * k * 2 / avg_c / 1e6
            num_bytes = sum(t.nbytes for t in (x, weight_asm, x_scale_t, w_scale, c))
            ret[f"{tag} BW"] = num_bytes / avg_c / 1e6
            ret[f"{tag} err"] = err_asm
            ret["asm/base"] = avg_c / avg_b

        # Triton path requires a preshuffled weight. When not preshuffled we simply omit
        # these columns; pd.DataFrame NaN-fills them for those rows in the summary.
        if b_preshuffle:
            d, avg_d = run_triton(x, gemm_weight, x_scale_t, w_scale, dtype)
            err_triton = checkAllclose(a, d, msg="triton", catastrophic_check=True)
            ret["triton us"] = avg_d
            ret["triton TFLOPS"] = m * n * k * 2 / avg_d / 1e6
            num_bytes = sum(t.nbytes for t in (x, gemm_weight, x_scale_t, w_scale, d))
            ret["triton BW"] = num_bytes / avg_d / 1e6
            ret["triton err"] = err_triton
            ret["triton/base"] = avg_d / avg_b

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
    assert ck_err < 0.05 and cktile_err < 0.05, (
        f"split-K mismatch (m={m}, n={n}, k={k}, splitK={splitK}): "
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
    "--data-init",
    dest="data_init",
    nargs="+",
    choices=bench_init.DATA_DISTS,
    default=None,
    help="DATA initialization distribution(s), paired position-wise with "
    "--scale-init (length-1 broadcasts). Default: constant uniform",
)
parser.add_argument(
    "--scale-init",
    dest="scale_init",
    nargs="+",
    choices=bench_init.E8M0_SCALE_DISTS,
    default=None,
    help="E8M0 SCALE initialization distribution(s), paired position-wise "
    "with --data-init (length-1 broadcasts). Default: constant auto",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="RNG seed for input, weight, and scales (default: 0)",
)
parser.add_argument(
    "--b_preshuffle",
    "--ck_preshuffle",
    dest="b_preshuffle",
    type=dtypes.str2bool,
    nargs="*",
    default=None,
    help="""weight B preshuffle or not (default: CSV layout, else True and False).
    e.g.: --b_preshuffle True
        or --b_preshuffle False
    """,
)
parser.add_argument(
    "--flydsl",
    action="store_true",
    help="use native E8M0 scales (selected automatically for native CSV rows)",
)
parser.add_argument(
    "--w-scale-block",
    choices=tuple(SCALE_BLOCK_SHAPES),
    default=None,
    help="weight scale block (default: CSV w_scale_block column, else 128x128)",
)
parser.add_argument(
    "--apre",
    type=dtypes.str2bool,
    nargs="*",
    default=[False],
    help="""also measure the FlyDSL A-preshuffle candidate (requires --flydsl
    --b_preshuffle True). Odd M is padded to M+1 rows for A only.
    Sweeps like --b_preshuffle.
    e.g.: --apre True
        or --apre True False""",
)
parser.add_argument(
    "--csv",
    type=str,
    default=None,
    help="""CSV file containing M, N, K columns (one shape per row).
    An optional w_scale_block column selects 1x32, 32x32 or 128x128 per row.
    Without --csv or explicit -m/-nk, sweep all configured CSVs for this GPU.
    e.g.: --csv shapes.csv""",
)
parser.add_argument(
    "--table",
    action="store_true",
    help="""Also print the summary as a human-readable table.
    The default output is the single-line JSON record.""",
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

data_init_list = args.data_init or ["constant", "uniform"]
scale_init_list = args.scale_init or ["constant", "auto"]
if len(data_init_list) == 1:
    data_init_list *= len(scale_init_list)
if len(scale_init_list) == 1:
    scale_init_list *= len(data_init_list)
if len(data_init_list) != len(scale_init_list):
    parser.error(
        "--data-init and --scale-init must have equal length "
        "(or length 1 to broadcast)"
    )
init_pairs = list(zip(data_init_list, scale_init_list))

l_preshuffle = args.b_preshuffle or ([True] if args.flydsl else [True, False])
l_apre = args.apre
# Preserve the -m/-nk defaults; only explicit shape arguments select shape mode.
default_csv = args.csv is None and not any(
    arg.startswith(("-m", "-nk")) for arg in sys.argv[1:]
)

df = []
if args.csv is not None or default_csv:
    sources = [
        ("", False, False, "128x128"),
        ("_BPRESHUFFLE", True, False, "128x128"),
        ("_GROUP32", False, True, "32x32"),
        ("_MXSCALE_BPRESHUFFLE", True, True, "128x128"),
    ]
    if args.csv is not None:
        sources = [(args.csv, None, args.flydsl, "128x128")]
    tables = []
    for suffix, preshuffle, native, block in sources:
        path = args.csv or getattr(
            AITER_CONFIGS, f"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE{suffix}_FILE"
        )
        name = os.path.basename(path)
        native = native or "mxscale" in name or "group32" in name
        if "group32" in name:
            preshuffle, block = False, "32x32"
        table = pd.read_csv(path)
        if "gfx" in table:
            table = table[table["gfx"].astype(str).str.strip() == get_gfx()]
        if "cu_num" in table:
            table = table[table["cu_num"] == get_cu_num()]
        table = table.filter(["M", "N", "K", "w_scale_block"]).copy()
        table["w_scale_block"] = table.get("w_scale_block", block)
        table["w_scale_block"] = (
            table["w_scale_block"]
            .replace(r"^\s*$", pd.NA, regex=True)
            .fillna(block)
            .astype(str)
            .str.strip()
        )
        table["b_preshuffle"] = preshuffle
        table["use_flydsl"] = native
        tables.append(table)
        print(f"Loaded {len(table)} shapes from {path}", flush=True)
    shapes_df = pd.concat(tables, ignore_index=True).drop_duplicates()
    if args.w_scale_block:
        if default_csv:
            shapes_df = shapes_df[shapes_df["w_scale_block"] == args.w_scale_block]
        else:
            shapes_df["w_scale_block"] = args.w_scale_block
    shapes_df["use_flydsl"] |= shapes_df["w_scale_block"].isin(("1x32", "32x32"))
    if args.csv is not None and args.b_preshuffle is not None:
        shapes_df = pd.concat(
            [shapes_df.assign(b_preshuffle=p) for p in args.b_preshuffle]
        )
    shapes_df["use_flydsl"] &= shapes_df["b_preshuffle"].ne(False) | (
        shapes_df["w_scale_block"] != "128x128"
    )
    if default_csv and args.flydsl:
        shapes_df = shapes_df[shapes_df["use_flydsl"]]
    if default_csv and args.b_preshuffle is not None:
        shapes_df = shapes_df[shapes_df["b_preshuffle"].isin(args.b_preshuffle)]
    if shapes_df.empty:
        print("No matching CSV rows, nothing to run")
        sys.exit(0)
    scale_blocks = shapes_df["w_scale_block"].tolist()
else:
    scale_blocks = [args.w_scale_block or "128x128"]
    if scale_blocks[0] != "128x128":
        if get_gfx() not in NATIVE_SCALE_GFX:
            print(f"Skipping the group32 cases: {get_gfx()} has no native dispatch")
            sys.exit(0)
        if args.b_preshuffle is None:
            l_preshuffle = [True]

for w_scale_block in set(scale_blocks):
    if w_scale_block not in SCALE_BLOCK_SHAPES:
        parser.error(f"unsupported w_scale_block {w_scale_block!r}")
    if w_scale_block != "128x128" and any(l_apre):
        parser.error("Group32 scales do not support --apre True")

if args.csv is not None or default_csv:
    for dtype in args.dtype:
        for _, row in shapes_df.iterrows():
            preshuffles = (
                [bool(row["b_preshuffle"])]
                if pd.notna(row["b_preshuffle"])
                else ([True] if row["use_flydsl"] else l_preshuffle)
            )
            for preshuffle in preshuffles:
                for data_init, scale_init in init_pairs:
                    for apre in l_apre:
                        ret = test_gemm(
                            dtype,
                            int(row["M"]),
                            int(row["N"]),
                            int(row["K"]),
                            b_preshuffle=preshuffle,
                            use_flydsl=bool(row["use_flydsl"]),
                            data_init=data_init,
                            scale_init=scale_init,
                            seed=args.seed,
                            apre=apre,
                            w_scale_block=row["w_scale_block"],
                        )
                        df.append(ret)
else:
    for dtype in args.dtype:
        for m in args.m:
            for n, k in args.nk:
                for b_p in l_preshuffle:
                    for apre in l_apre:
                        for data_init, scale_init in init_pairs:
                            ret = test_gemm(
                                dtype,
                                m,
                                n,
                                k,
                                b_preshuffle=b_p,
                                use_flydsl=args.flydsl,
                                data_init=data_init,
                                scale_init=scale_init,
                                seed=args.seed,
                                apre=apre,
                                w_scale_block=args.w_scale_block or "128x128",
                            )
                            df.append(ret)

df = pd.DataFrame([row for row in df if row is not None])
print_json_table("gemm_a8w8_blockscale summary", df)
if args.table and not df.empty:
    print("\n" + "=" * 150)
    print("GEMM PERFORMANCE SUMMARY (BW in TB/s)")
    print("=" * 150)
    print(df.to_string(index=False))
    print("=" * 150)

# The legacy HIP split-K kernels do not support gfx1250.
if get_gfx() in ("gfx942", "gfx950"):
    # Correctness check: verify split-K produces matching results
    print("\nRunning split-K correctness checks ...")
    for splitK in [1, 2]:
        test_splitk_correctness(m=4, n=512, k=16384, splitK=splitK)
    # Two K loops per split for the default CK-Tile tile, the shortest split it runs.
    for m, n, k, splitK in [(8, 256, 512, 1), (8, 256, 1024, 2), (32, 512, 2048, 3)]:
        test_splitk_correctness(m=m, n=n, k=k, splitK=splitK)


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
