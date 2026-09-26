# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for FlyDSL MX-scale preshuffle GEMM (gfx950).

MiniMax-M3 TP4 dense MXFP8 (``gemm_mxscale_preshuffle``, shuffled B):

    a    : [M, K] packed MXFP8, M = tokens
    w    : shuffled [N, K] MXFP8
    out  : [M, N] preallocated bf16
    n,k  : local linear dims from the MiniMax-M3 dense table
           decode uses M=1, N=1536/2560, K=6144

Run:
    python op_tests/test_flydsl_mxscale_preshuffle.py
    python op_tests/test_flydsl_mxscale_preshuffle.py -s 1,2560,6144 4,1536,6144
"""

import argparse
import itertools

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
    flydsl_mxscale_preshuffle_gemm,
    gemm_mxscale_preshuffle,
)
from aiter.ops.quant import per_1x32_f4_quant, per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
SEED = 0

# MiniMax-M3 TP4 dense MXFP8: m=tokens, (n,k)=local linear.
_TUNED_MNK = [
    (1, 1536, 6144),
    (1, 2560, 6144),
    (4, 1536, 6144),
]

# (m, n, k, a_dtype, b_dtype, tile_m, tile_n, tile_k, split_k)
_EXPLICIT = [
    (1, 2560, 6144, "fp8", "fp8", 32, 16, 256, 1),
    (32, 8192, 8192, "fp8", "fp8", 32, 128, 256, 1),
    (64, 8192, 8192, "fp8", "fp8", 64, 128, 128, 1),
    (32, 8192, 8192, "fp4", "fp4", 32, 128, 256, 1),
    (64, 8192, 8192, "fp4", "fp4", 64, 128, 128, 1),
    (8, 2048, 7168, "fp8", "fp8", 32, 128, 256, 2),
    (8, 2048, 7168, "fp8", "fp8", 32, 128, 256, 4),
    (1, 2048, 7168, "fp8", "fp8", 32, 128, 128, 2),
    (16, 4096, 8192, "fp8", "fp8", 32, 128, 256, 8),
]


def _default_mnk():
    seen = []
    got = set()
    for m, n, k, *_rest in itertools.chain(
        ((m, n, k) for m, n, k in _TUNED_MNK),
        _EXPLICIT,
    ):
        key = (m, n, k)
        if key not in got:
            got.add(key)
            seen.append(key)
    return seen


def _quant_pair(a_float, b_float, a_dtype, b_dtype):
    if a_dtype == "fp8":
        a_quant, a_scale_raw = per_1x32_f8_scale_f8_quant(
            a_float, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    else:
        a_quant, a_scale_raw = per_1x32_f4_quant(a_float, quant_dtype=dtypes.fp4x2)
    if b_dtype == "fp8":
        b_quant, b_scale_raw = per_1x32_f8_scale_f8_quant(
            b_float, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    else:
        b_quant, b_scale_raw = per_1x32_f4_quant(b_float, quant_dtype=dtypes.fp4x2)
    return a_quant, a_scale_raw, b_quant, b_scale_raw


def _dequant(codes, scale_raw, dtype, rows):
    scale_f32 = fp4_utils.e8m0_to_f32(scale_raw[:rows].repeat_interleave(32, dim=1))
    if dtype == "fp8":
        return codes.float() * scale_f32
    return fp4_utils.mxfp4_to_f32(codes) * scale_f32


def run_torch(a_deq, b_deq, dtype):
    return F.linear(a_deq.to(dtypes.fp32), b_deq.to(dtypes.fp32)).to(dtype)


def _prepare(m, n, k, a_dtype, b_dtype, dtype):
    torch.manual_seed(SEED)
    m_aligned = (m + 31) // 32 * 32
    n_aligned = (n + 31) // 32 * 32
    a_float = torch.zeros(m_aligned, k)
    b_float = torch.zeros(n_aligned, k)
    a_float[:m] = torch.randn(m, k)
    b_float[:n] = torch.randn(n, k)
    a_quant, a_scale_raw, b_quant, b_scale_raw = _quant_pair(
        a_float, b_float, a_dtype, b_dtype
    )
    a_codes, b_codes = a_quant[:m], b_quant[:n]
    b_shuffled = shuffle_weight(b_codes, layout=(16, 16))
    a_scale = shuffle_scale_a16w4(a_scale_raw, 1, False)
    b_scale = shuffle_scale_a16w4(b_scale_raw, 1, False)
    a_deq = _dequant(a_codes, a_scale_raw, a_dtype, m)
    b_deq = _dequant(b_codes, b_scale_raw, b_dtype, n)
    ref = run_torch(a_deq, b_deq, dtype)
    out = torch.randn((m, n), dtype=dtype)
    return a_codes, b_shuffled, a_scale, b_scale, out, ref


def _traffic_bytes(a, b, a_scale, b_scale, out):
    return (
        a.numel() * a.element_size()
        + b.numel() * b.element_size()
        + a_scale.numel() * a_scale.element_size()
        + b_scale.numel() * b_scale.element_size()
        + out.numel() * out.element_size()
    )


def _record(ret, candidates, ref, flops, nbytes):
    for name, fn in candidates.items():
        y, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            y.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: mxscale preshuffle",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err


@benchmark()
def test_mxscale_tuned(m, n, k, dtype):
    a, b, a_scale, b_scale, out, ref = _prepare(m, n, k, "fp8", "fp8", dtype)

    def _run():
        return gemm_mxscale_preshuffle(
            a,
            b,
            a_scale,
            b_scale,
            out,
            a_dtype="fp8",
            b_dtype="fp8",
            require_tuned=True,
        )

    flops = 2 * m * n * k
    nbytes = _traffic_bytes(a, b, a_scale, b_scale, out)
    ret = {"gfx": get_gfx()}
    _record(ret, {"flydsl": _run}, ref, flops, nbytes)
    return ret


@benchmark()
def test_mxscale_explicit(
    m, n, k, dtype, a_dtype, b_dtype, tile_m, tile_n, tile_k, split_k
):
    a, b, a_scale, b_scale, out, ref = _prepare(m, n, k, a_dtype, b_dtype, dtype)

    def _run():
        return flydsl_mxscale_preshuffle_gemm(
            a,
            b,
            a_scale,
            b_scale,
            out,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            split_k=split_k,
        )

    flops = 2 * m * n * k
    nbytes = _traffic_bytes(a, b, a_scale, b_scale, out)
    ret = {
        "gfx": get_gfx(),
        "tile": f"{tile_m}x{tile_n}x{tile_k}",
        "split_k": split_k,
    }
    _record(ret, {"flydsl": _run}, ref, flops, nbytes)
    return ret


def summarize(title, rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    try:
        table = df.to_markdown(index=False)
    except ImportError:
        table = df.to_string(index=False)
    aiter.logger.info("%s summary (markdown):\n%s", title, table)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl mxscale preshuffle unsupported on %s; skipping", get_gfx()
        )
        return

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
        default="bf16,",
        metavar="{bf16}",
        help="""Data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=_default_mnk(),
        help="""Shape of mnk.
        e.g.:   -s 1,2560,6144 4,1536,6144""",
    )
    args = parser.parse_args()
    wanted = set(args.mnk)

    for dtype in args.dtype:
        tuned_rows = [
            test_mxscale_tuned(m, n, k, dtype)
            for (m, n, k) in _TUNED_MNK
            if (m, n, k) in wanted
        ]
        summarize(f"flydsl_mxscale_tuned {dtype}", tuned_rows)

        explicit_rows = [
            test_mxscale_explicit(
                m, n, k, dtype, a_dtype, b_dtype, tile_m, tile_n, tile_k, split_k
            )
            for m, n, k, a_dtype, b_dtype, tile_m, tile_n, tile_k, split_k in _EXPLICIT
            if (m, n, k) in wanted
        ]
        summarize(f"flydsl_mxscale_explicit {dtype}", explicit_rows)


if __name__ == "__main__":
    main()
