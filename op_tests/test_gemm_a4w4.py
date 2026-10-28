# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import os

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.gemm_op_a4w4 import gemm_a4w4
from aiter.ops.shuffle import shuffle_mxfp8fp4_scale, shuffle_weight
from aiter.test_common import (
    assertAllclose,
    benchmark,
    checkAllclose,
    perftest,
    run_perftest,
)
from aiter.utility import fp4_utils

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
SCALE_GROUP_SIZE = 32
SUPPORTED_GFX = ["gfx950", "gfx1250"]
# gfx1250 M the tuned table lacks, with whether the kernel each must reach is the compute
# one (every tuned compute row fills whole M tiles). 1 is below the smallest row; 17 and 257
# leave a partial M tile of a general kernel, 2049 one of the compute kernel. 129 and 255 are
# below the 256-row compute tile of N=36864 and reach it through the padded M, as one partial
# tile. Together they reach the masked-M stores and partial-tile cluster reduction of both
# kernel families.
OFF_GRID_SHAPES = {
    (1, 7168, 16384): False,
    (17, 7168, 16384): False,
    (257, 7168, 16384): False,
    (2049, 7168, 16384): True,
    (129, 36864, 7168): True,
    (255, 36864, 7168): True,
}
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 30)


@perftest(num_iters=5)
def run_torch(x, w, x_scales, w_scales, dtype):
    m, _k = x.shape
    n, _k = w.shape
    # First convert the x and w inputs to f32.
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, out):
    return aiter.gemm_a4w4_blockscale(x, weight, x_scale, w_scale, out)


@perftest()
def run_triton(x, w, x_scales, w_scales, out, dtype=dtypes.bf16):
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    gemm_afp4wfp4(x, w, x_scales, w_scales, dtype, out)
    return out


@perftest()
def run_gemm_asm(
    x,
    weightshuffle,
    x_scale,
    w_scale,
    out,
    kernelName="",
    bias=None,
    dtype=dtypes.bf16,
    bpreshuffle=True,
    log2_k_split=None,
):
    # if log2_k_split is not None and log2_k_split > 0:
    #     out_reset = torch.zeros(
    #         (out.shape[0] + 31) // 32 * 32, out.shape[1], dtype=dtype
    #     )
    #     out = out_reset

    aiter.gemm_a4w4_asm(
        x,
        weightshuffle,
        x_scale,
        w_scale,
        out,
        kernelName,
        bias,
        bpreshuffle=bpreshuffle,
        log2_k_split=log2_k_split,
    )
    return out


def _test_gemm_gfx1250(dtype, M, N, K):
    """gemm_a4w4 as ATOM calls it on gfx1250: the weight and the activation scale in the
    32x4 layout, the activation quantized straight into it. ``tuned`` is the public op, so
    it runs the backend the tuned CSV (or the FlyDSL fallback) picks for this shape."""
    # Importing aiter.ops.flydsl needs flydsl itself, so only a FlyDSL run may import it.
    from aiter.ops.gemm_op_a4w4 import (
        _FLYDSL_AVAILABLE,
        _a4w4_mxscale_config,
        gemm_mxfp4_asm,
    )

    quant = aiter.get_hip_quant(aiter.QuantType.per_1x32)
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    # The row-major quant is the reference operand; every layout below holds these bytes.
    xq_raw, xs_raw = quant(x, quant_dtype=dtypes.fp4x2, shuffle=False)
    wq_raw, ws_raw = quant(w, quant_dtype=dtypes.fp4x2, shuffle=False)
    ref, _ = run_torch(
        xq_raw, wq_raw, xs_raw.view(torch.uint8), ws_raw.view(torch.uint8), dtype
    )

    wq = shuffle_weight(wq_raw, layout=(16, 16))
    ws = shuffle_mxfp8fp4_scale(ws_raw)
    xq, xs = quant(x, quant_dtype=dtypes.fp4x2, scale_layout_m32k4=True)
    candidates = {
        "tuned": lambda: gemm_a4w4(xq, wq, xs, ws, dtype=dtype),
        "asm": lambda: gemm_mxfp4_asm(xq, wq, xs, ws, dtype=dtype, a_preshuffle=False),
    }
    config = _a4w4_mxscale_config(M, N, K) or ("asm", "")
    # An asm pick, or the other kernel family, would hide the coverage an off-grid M is for.
    # Without FlyDSL there is no family to check: both candidates still verify ASM.
    want_compute = OFF_GRID_SHAPES.get((M, N, K))
    if want_compute is not None and _FLYDSL_AVAILABLE:
        from aiter.ops.flydsl.a4w4_mxscale_bpreshuffle_gemm_gfx1250 import (
            parse_a4w4_kernel_name,
        )

        kernel = parse_a4w4_kernel_name(config[1]) if config[0] == "flydsl" else None
        assert kernel and kernel["compute_bound"] == want_compute, (
            f"M={M} N={N} K={K}: expected a FlyDSL "
            f"{'compute' if want_compute else 'general'} kernel, got {config[1] or 'asm'}"
        )
    ret = {
        "gfx": get_gfx(),
        "tuned kernel": config[1] if config[0] == "flydsl" else "asm",
    }
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = assertAllclose(
            ref, out, msg=f"{name}: a4w4 M={M} N={N} K={K}", catastrophic_check=True
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = M * N * K * 2 / us / 1e6
        ret[f"{name} TB/s"] = (
            (xq.nbytes + wq.nbytes + xs.nbytes + ws.nbytes + out.nbytes) / us / 1e6
        )
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_gemm(dtype, M, N, K):
    if get_gfx() == "gfx1250":
        return _test_gemm_gfx1250(dtype, M, N, K)
    ret = {}
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    wshuffle = shuffle_weight(w, layout=(16, 16))
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)
    a, _avg_a = run_torch(x, w, x_scales, w_scales, dtype)
    # out1 = torch.empty(M, N, dtype=dtype)
    # b, avg_b = run_triton(x, w.T, x_scales, w_scales, out1, dtype)
    # b, avg_b = a, 0
    # err_b = checkAllclose(a, b, msg="triton        ")

    c, us = run_perftest(
        aiter.gemm_a4w4,
        x,
        wshuffle,
        x_scales_shuffle,
        w_scales_shuffle,
        bpreshuffle=True,
    )
    err = checkAllclose(a, c, msg="unified api", catastrophic_check=True)
    ret["us"] = us
    ret["TFLOPS"] = M * N * K * 2 / us / 1e6
    ret["TB/s"] = (x.nbytes + w.nbytes) / us / 1e6
    ret["err"] = err

    # kernelName = "" # "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E"
    # log2_k_split = 1
    # out2 = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    # d, us = run_gemm_asm(
    #     x,
    #     wshuffle,
    #     x_scales_shuffle,
    #     w_scales_shuffle,
    #     out2,
    #     kernelName,
    #     bias_f32,
    #     bpreshuffle=True,
    #     log2_k_split=log2_k_split,
    # )
    # err = checkAllclose(a, d[:M], msg=f"asm {kernelName} log2_k_split_{log2_k_split}")
    # tag = "asm_dbg"
    # ret[f"us {tag}"] = us
    # ret[f"TFLOPS {tag}"] = M * N * K * 2 / us / 1e6
    # ret[f"TB/s {tag}"] = (x.nbytes + w.nbytes) / us / 1e6
    # ret[f"err {tag}"] = err

    # out3 = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    # e, us = run_gemm_ck(x, wshuffle, x_scales_shuffle, w_scales_shuffle, out3)
    # err = checkAllclose(a, e[:M], msg="ck            ")
    # tag = "ck"
    # ret[f"us {tag}"] = us
    # ret[f"TFLOPS {tag}"] = M * N * K * 2 / us / 1e6
    # ret[f"TB/s {tag}"] = (x.nbytes + w.nbytes) / us / 1e6
    # ret[f"err {tag}"] = err

    return ret


def tuned_shapes():
    """Every (M, N, K) the a4w4_mxscale_bpreshuffle tuned table holds for this machine."""
    try:
        table = pd.read_csv(
            AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_MXSCALE_BPRESHUFFLE_FILE
        )
    except OSError:
        return []  # no tuned shapes on this install
    table = table[(table["gfx"] == get_gfx()) & (table["cu_num"] == get_cu_num())]
    return sorted(set(table[["M", "N", "K"]].itertuples(index=False, name=None)))


def main():
    # Whole-op arch gate: gate here, not inside the @benchmark fn (that would emit an
    # args-only row).
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("a4w4 GEMM unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        choices=[dtypes.d_dtypes["bf16"]],
        metavar="{bf16}",
        default=[dtypes.d_dtypes["bf16"]],
        help="""Data type.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-mnk",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            # pure_compute
            (256, 2048, 8192),
            (2048, 8192, 8192),
            (16384, 16384, 16384),
            (32768, 106496, 16384),
            (32768, 16384, 53248),
            (32768, 18432, 16384),
            (32768, 16384, 16384),
            (128, 106496, 16384),
            (128, 16384, 53248),
            (128, 18432, 16384),
            (128, 16384, 16384),
            (64, 106496, 16384),
            (64, 16384, 53248),
            (64, 18432, 16384),
            (64, 16384, 16384),
            (64, 106496, 16384),
            (32, 106496, 16384),
            (32, 16384, 53248),
            (32, 18432, 16384),
            (32, 16384, 16384),
            # qkv_proj
            (1, 1280, 8192),
            (64, 1280, 8192),
            (127, 1280, 8192),
            (129, 1280, 8192),
            (65, 1280, 8192),
            (32, 1280, 8192),
            (64, 1280, 8192),
            (128, 1280, 8192),
            (192, 1280, 8192),
            (256, 1280, 8192),
            (320, 1280, 8192),
            (512, 1280, 8192),
            (1024, 1280, 8192),
            (2048, 1280, 8192),
            (4096, 1280, 8192),
            (8192, 1280, 8192),
            # attn_out
            (1, 8192, 1024),
            (32, 8192, 1024),
            (64, 8192, 1024),
            (128, 8192, 1024),
            (192, 8192, 1024),
            (256, 8192, 1024),
            (320, 8192, 1024),
            (512, 8192, 1024),
            (1024, 8192, 1024),
            (2048, 8192, 1024),
            (4096, 8192, 1024),
            (8192, 8192, 1024),
            (16384, 8192, 1024),
            # tune
            (1552, 8192, 8192),
            (1664, 8192, 8192),
            (1792, 8192, 8192),
            (1920, 8192, 8192),
            (3072, 8192, 8192),
            (1552, 10240, 8192),
            (1664, 10240, 8192),
            (1792, 10240, 8192),
            (1920, 10240, 8192),
            (3072, 10240, 8192),
            (1552, 57344, 8192),
            (1664, 57344, 8192),
            (1792, 57344, 8192),
            (1920, 57344, 8192),
            (3072, 57344, 8192),
            (1552, 8192, 28672),
            (1664, 8192, 28672),
            (1792, 8192, 28672),
            (1920, 8192, 28672),
            (3072, 8192, 28672),
        ],
        help="""Shape of mnk.
    e.g. -mnk 1280,8192,1024""",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="""CSV file containing M, N, K columns (one shape per row); overrides -mnk.
    On gfx1250, with neither -mnk nor --csv, every shape of the tuned CSV runs, plus the
    off-grid M it lacks.
    e.g.: --csv aiter/configs/model_configs/dsr1_a4w4_mxscale_bpreshuffle_tuned_gemm.csv""",
    )
    args = parser.parse_args()

    gfx1250 = get_gfx() == "gfx1250"
    if args.csv is not None:
        if not os.path.exists(args.csv):
            raise FileNotFoundError(f"CSV file not found: {args.csv}")
        shapes_df = pd.read_csv(args.csv)
        print(f"Loaded {len(shapes_df)} shapes from {args.csv}", flush=True)
        shapes = [
            (int(m), int(n), int(k))
            for m, n, k in shapes_df[["M", "N", "K"]].itertuples(index=False)
        ]
    elif gfx1250 and args.shape is parser.get_default("shape"):
        # The built-in -mnk list is gfx950's; here the tuned table's shapes run.
        shapes = tuned_shapes()
        if not shapes:
            aiter.logger.warning(
                "no tuned a4w4 shapes for gfx1250 cu_num=%s; pass -mnk or --csv",
                get_cu_num(),
            )
            return
        shapes.extend(OFF_GRID_SHAPES)
    else:
        shapes = args.shape
    df = []
    for dtype in args.dtype:
        for m, n, k in shapes:
            ret = test_gemm(dtype, m, n, k)
            df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("gemm_a4w4 summary (markdown):\n%s", df_md)


if __name__ == "__main__":
    main()
