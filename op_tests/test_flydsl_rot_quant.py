# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes, gemm_a4w4
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_rot_quant
from aiter.ops.quant import per_1x32_f4_quant, per_1x32_f4_quant_hip
from aiter.ops.shuffle import shuffle_scale, shuffle_weight
from aiter.test_common import assertAllclose, benchmark, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
# gemm_a4w4 does not match the dequantized product below N=256 (rel. err 0.68
# at N=128, whatever quantized the activation), and for K not a multiple of 256
# it reads past K, so its result depends on the padding. The gemm check stays
# within N >= 256 and K % 256 == 0.
GEMM_MIN_N = 256


def run_torch(x, rs, shuffle):
    """Rotate with the kernel's FWHT stage order in fp32, round to bf16, then
    the shared per-1x32 MXFP4 torch reference."""
    m, k = x.shape
    v = x.float().view(m, k // rs, rs)
    h = 1
    while h < rs:
        v = v.view(m, k // rs, rs // (2 * h), 2, h)
        a, b = v[..., 0, :], v[..., 1, :]
        v = torch.stack([a + b, a - b], dim=-2)
        h *= 2
    xr = (v.reshape(m, k) * (1.0 / math.sqrt(rs))).to(dtypes.bf16)
    xq, xs = per_1x32_f4_quant(xr, quant_dtype=dtypes.fp4x2, shuffle=shuffle)
    return xq.view(torch.uint8), xs.view(torch.uint8)


def scale_mask(shape, rows, cols, shuffle):
    """True on the scales of `rows` x first `cols` columns, in either layout."""
    mask = torch.zeros(shape, dtype=torch.uint8)
    mask[rows, :cols] = 1
    return (shuffle_scale(mask).view(shape) if shuffle else mask) != 0


def dequant(xq, xs):
    vals = fp4_utils.mxfp4_to_f32(xq)
    return vals * fp4_utils.e8m0_to_f32(xs.repeat_interleave(32, dim=1))


@benchmark()
def test_rot_quant(m, k, rs, shuffle):
    x = torch.randn((m, k), dtype=dtypes.bf16)
    # Outlier channels in every other row drive groups to the top of the e2m1
    # range, where the e8m0 rounding and the saturation decide the result.
    x[::2, ::37] *= 64
    # Rows 1, 3 and 5 when present: a zero row (a padded token), a row far
    # below the HIP amax floor, and a NaN whose groups must get e8m0 NaN (255).
    zero_row, tiny_row, nan_row = 1, 3, 5
    x[zero_row : zero_row + 1] = 0
    x[tiny_row : tiny_row + 1] *= 1e-30
    x[nan_row : nan_row + 1, 7] = float("nan")
    rows = torch.arange(m)
    ref_q, ref_s = run_torch(x, rs, shuffle)

    # e2m1 has no NaN, so the bytes packed for a NaN group are arbitrary; only
    # its scale is compared. The padding columns of the CK layout are compared
    # too, the padding rows are uninitialized.
    candidates = {
        "flydsl": (
            lambda: flydsl_rot_quant(x, rs, shuffle_scales=shuffle),
            rows != nan_row,
            rows,
        )
    }
    if rs == 32:
        # #6734 fuses the same rotation. It floors amax (scale 0x5C for zero
        # groups), leaves the padding columns unwritten, and its NaN group
        # bytes are arbitrary, so those rows and columns are left out.
        hip_rows = (rows != zero_row) & (rows != tiny_row) & (rows != nan_row)
        candidates["hip"] = (
            lambda: per_1x32_f4_quant_hip(x, shuffle=shuffle, hadamard_rotate=True),
            hip_rows,
            rows[hip_rows],
        )

    flops = m * k * int(math.log2(rs))  # butterfly adds
    nbytes = m * k * x.element_size() + m * k // 2 + m * k // 32
    ret = {"gfx": get_gfx()}
    for name, (fn, q_rows, s_rows) in candidates.items():
        (xq, xs), us = run_perftest(fn)
        # Packed bytes: any difference is a bug, so compare exactly.
        err = assertAllclose(
            ref_q.to(dtypes.fp32),
            xq.view(torch.uint8).to(dtypes.fp32),
            rtol=0,
            atol=0,
            tol_err_ratio=0,
            mask=q_rows.unsqueeze(1),
            msg=f"{name}: fp4",
        )
        cols = k // 32 if name == "hip" else ref_s.shape[1]
        assertAllclose(
            ref_s.to(dtypes.fp32),
            xs.view(torch.uint8).to(dtypes.fp32),
            rtol=0,
            atol=0,
            tol_err_ratio=0,
            mask=scale_mask(ref_s.shape, s_rows, cols, shuffle),
            msg=f"{name}: e8m0",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_rot_quant_gemm_a4w4(m, n, k, rs):
    """The model call: rotate and quantize the activation into the CK scale
    layout, then gemm_a4w4. The reference multiplies the dequantized bytes, so
    a scale-layout mismatch shows up as a wrong product."""
    x = torch.randn((m, k), dtype=dtypes.bf16)
    w = torch.randn((n, k), dtype=dtypes.bf16)
    wq, ws = per_1x32_f4_quant_hip(w, shuffle=False)
    wq_shuffled = shuffle_weight(wq, layout=(16, 16))
    ws_shuffled = shuffle_scale(ws)
    ref_q, ref_s = run_torch(x, rs, False)
    ref = dequant(ref_q, ref_s) @ dequant(wq, ws.view(torch.uint8)).T

    def run():
        xq, xs = flydsl_rot_quant(x, rs, shuffle_scales=True)
        return gemm_a4w4(xq, wq_shuffled, xs, ws_shuffled, dtype=dtypes.bf16)

    out, us = run_perftest(run)
    err = assertAllclose(
        ref.to(dtypes.fp32), out.to(dtypes.fp32), msg="flydsl + gemm_a4w4"
    )
    nbytes = m * k * x.element_size() + n * k // 2 + n * k // 32 + m * n * 2
    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": 2 * m * n * k / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "err": err,
    }


def test_rot_quant_rejects_bad_input():
    """Each of these would otherwise be read as contiguous bf16 and quantized
    to plausible-looking garbage."""
    x = torch.randn((64, 4096), dtype=dtypes.bf16)
    cases = {
        "fp16": lambda: flydsl_rot_quant(x.half()),
        "fp32": lambda: flydsl_rot_quant(x.float()),
        "non-contiguous": lambda: flydsl_rot_quant(x[:, ::2]),
        "3D": lambda: flydsl_rot_quant(x.view(2, 32, 4096)),
        "K % RS": lambda: flydsl_rot_quant(x[:, :96].contiguous()),
        "RS=16": lambda: flydsl_rot_quant(x, 16),
    }
    for name, fn in cases.items():
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError(f"{name}: expected ValueError")


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl_rot_quant unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="*",
        default=[1, 32, 300, 1024, 4096, 16384],
        help="""M (tokens).
    e.g.: -m 32""",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        nargs="*",
        default=[2880, 4096, 7168],
        help="""K (hidden size); skipped where RS does not divide it.
    e.g.: -k 4096""",
    )
    parser.add_argument(
        "-r",
        "--rs",
        type=int,
        nargs="*",
        default=[32, 64, 128],
        help="""Rotation size.
    e.g.: -r 64""",
    )
    parser.add_argument(
        "--shuffle",
        type=dtypes.str2bool,
        nargs="*",
        default=[False, True],
        help="""Write scales in the CK layout of shuffle_scale.
    e.g.: --shuffle true""",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        nargs="*",
        default=[256, 4096],
        help=f"""N of the gemm_a4w4 check, at least {GEMM_MIN_N}.
    e.g.: -n 4096""",
    )
    args = parser.parse_args()

    df = [
        test_rot_quant(m, k, rs, shuffle)
        for rs, shuffle, k, m in itertools.product(
            args.rs, args.shuffle, args.k, args.m
        )
        if k % rs == 0
    ]
    aiter.logger.info(
        "rot_quant summary (markdown):\n%s", pd.DataFrame(df).to_markdown(index=False)
    )

    df = [
        test_rot_quant_gemm_a4w4(m, n, k, rs)
        for rs, k, n, m in itertools.product(args.rs, args.k, args.n, args.m)
        if n >= GEMM_MIN_N and k % 256 == 0
    ]
    aiter.logger.info(
        "rot_quant + gemm_a4w4 summary (markdown):\n%s",
        pd.DataFrame(df).to_markdown(index=False),
    )

    test_rot_quant_rejects_bad_input()


if __name__ == "__main__":
    main()
