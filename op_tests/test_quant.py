# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes, get_hip_quant, get_torch_quant, get_triton_quant
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import dynamic_per_group_scaled_quant
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.utility.fp4_utils import f32_to_mx_e8m0_scale, f32_to_mxfp4
from aiter.utility.mx_types import MxDtypeInt

torch.set_default_device("cuda")


@benchmark()
def test_quant(m, n, q_type, q_dtype, h_dtype):
    dim = (m, n)

    input = torch.randn(dim, dtype=h_dtype)
    ref, ref_scale = get_torch_quant(q_type)(input, quant_dtype=q_dtype)

    q_funcs = {
        "triton": get_triton_quant,
        "hip": get_hip_quant,
    }
    ret = {}
    for name, q_func in q_funcs.items():
        q_func = q_func(q_type)
        (out, scale), us1 = run_perftest(q_func, input, quant_dtype=q_dtype)
        err1 = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-3,
            atol=1e-3,
            msg=f"{name}: dynamic quant",
        )
        checkAllclose(
            ref_scale.to(dtypes.fp32),
            scale.to(dtypes.fp32),
            rtol=1e-3,
            atol=1e-3,
            msg=f"{name}: dynamic quant scale",
        )
        ret[f"{name} dq"] = us1
        ret[f"{name} dq err"] = err1
        if q_type == aiter.QuantType.per_Tensor:
            (out, scale), us2 = run_perftest(
                q_func, input, ref_scale, quant_dtype=q_dtype
            )
            err2 = checkAllclose(
                ref.to(dtypes.fp32),
                out.to(dtypes.fp32),
                rtol=1e-3,
                atol=1e-3,
                msg=f"{name}: static  quant",
            )
            ret[f"{name} sq"] = us2
            ret[f"{name} sq err"] = err2

    return ret


@benchmark()
def test_quant_1x128_e8m0(m, n, q_dtype, h_dtype, shuffle, strided):
    """Byte-exact check of dynamic_per_group_scaled_quant, group 128, e8m0 scales."""
    G = 128
    fp4 = q_dtype == dtypes.fp4x2
    src = torch.randn((m, n + (G if strided else 0)), dtype=h_dtype)
    x = src[:, :n]
    out = torch.empty((m, n // 2 if fp4 else n), dtype=q_dtype)
    scale = torch.empty((m, n // G), dtype=dtypes.fp8_e8m0)
    _, us = run_perftest(
        dynamic_per_group_scaled_quant, out, x, scale, G, shuffle_scale=shuffle
    )

    xf = x.float().view(m, n // G, G)
    mx = MxDtypeInt.FP4_E2M1 if fp4 else MxDtypeInt.FP8_E4M3
    ref_scale = f32_to_mx_e8m0_scale(xf.abs().amax(-1), dtype=mx).view(torch.uint8)
    xs = xf / (ref_scale.to(torch.int32) - 127).float().exp2()[..., None]
    if fp4:
        ref_out = f32_to_mxfp4(xs.view(m, n)).view(torch.uint8)
    else:
        fp8_max = torch.finfo(dtypes.fp8).max
        ref_out = xs.clamp(-fp8_max, fp8_max).to(dtypes.fp8).view(m, n)
        ref_out = ref_out.view(torch.uint8)
    got_scale = scale.view(torch.uint8)
    if shuffle:
        got_scale = got_scale.view(n // G, m).t()
    scale_err = int((got_scale != ref_scale).sum())
    out_err = int((out.view(torch.uint8) != ref_out).sum())
    assert scale_err == 0 and out_err == 0, f"{scale_err=} {out_err=}"
    return {"us": us}


def test_mxfp8_nonfinite(group_size, shuffle, h_dtype, pattern):
    """A NaN/Inf makes its MXFP8 group invalid (scale 0xff, data 0xff); others unchanged."""
    x = torch.ones((32, group_size * 8), dtype=h_dtype)
    bad = torch.zeros((32, 8), dtype=torch.bool)
    for row in range(32):
        col = (row * 3) % 8
        bad[row, col] = True
        group = x[row, col * group_size : (col + 1) * group_size]
        if pattern == "all_nan":
            group.fill_(float("nan"))
        elif pattern == "snan":
            group.view(torch.int16)[-1] = 0x7F81 if h_dtype == dtypes.bf16 else 0x7C01
        elif pattern == "nan_inf":
            group[:3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
        else:
            group[-1] = {"nan": float("nan"), "posinf": float("inf")}.get(
                pattern, -float("inf")
            )
    out = torch.empty_like(x, dtype=dtypes.fp8)
    scale = torch.empty((32, 8), dtype=torch.uint8)
    dynamic_per_group_scaled_quant(out, x, scale, group_size, shuffle)

    expected = torch.full(x.shape, 0x78, dtype=torch.uint8)  # 1 / 2^-8 = FP8 256
    expected.view(32, 8, group_size)[bad] = 0xFF
    expected_scale = torch.where(bad, 255, 119).to(torch.uint8)
    if shuffle and group_size == 32:
        shuffled = torch.empty_like(expected_scale).flatten()
        for row, col in itertools.product(range(32), range(8)):
            idx = (row % 16) * 4 + row // 16 + (col % 4) * 64 + (col // 4) * 2
            shuffled[idx] = expected_scale[row, col]
        expected_scale = shuffled
    elif shuffle:
        expected_scale = expected_scale.t().contiguous()
    assert torch.equal(out.view(torch.uint8), expected), pattern
    assert torch.equal(scale.flatten(), expected_scale.flatten()), pattern


def test_mxfp8_large_finite(group_size, shuffle, amax_bits):
    """Finite BF16 values up to the largest stay finite and follow the scale rule."""
    amax = torch.tensor([amax_bits], dtype=torch.int16).view(dtypes.bf16).float()
    group = (torch.linspace(-1, 1, group_size) * amax).to(dtypes.bf16)
    x = group.repeat(32, 8)
    out = torch.empty_like(x, dtype=dtypes.fp8)
    scale = torch.empty((32, 8), dtype=torch.uint8)
    dynamic_per_group_scaled_quant(out, x, scale, group_size, shuffle)

    exponent = torch.ceil(torch.log2(amax.double() / torch.finfo(dtypes.fp8).max))
    expected = (x.double() / torch.exp2(exponent).item()).to(dtypes.fp8)
    assert torch.isfinite(out.float()).all().item()
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8)), amax_bits
    assert torch.all(scale == int(exponent.item()) + 127).item(), amax_bits


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    nargs="*",
    default=[dtypes.d_dtypes["fp16"], dtypes.d_dtypes["bf16"]],
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-n",
    "--n",
    type=int,
    nargs="*",
    default=None,
    help="""N of mnk (default 4096 8192; *_e8m0: see E8M0_SHAPES).
    e.g.: -n 1024""",
)
parser.add_argument(
    "-m",
    "--m",
    type=int,
    nargs="*",
    default=None,
    help="""M of mnk (default 1 .. 163840; *_e8m0: see E8M0_SHAPES).
    e.g.: -m 32""",
)
d_quant = {
    "fp8_tensor": (aiter.QuantType.per_Tensor, dtypes.fp8),
    "fp8_token": (aiter.QuantType.per_Token, dtypes.fp8),
    "fp8_1x128": (aiter.QuantType.per_1x128, dtypes.fp8),
    "i8_token": (aiter.QuantType.per_Token, dtypes.i8),
    # 'fp4x2-1x32': (aiter.QuantType.per_1x32, dtypes.fp4x2),
}
d_quant_e8m0 = {
    "fp8_1x128_e8m0": dtypes.fp8,
    "fp4_1x128_e8m0": dtypes.fp4x2,
}
d_quant_special = ["fp8_mx_nonfinite"]
# Odd M, the paired-store and TDM tile paths (N a multiple of 4096 >= 12288).
E8M0_SHAPES = [
    (1, 7168),
    (100, 1536),
    (1535, 7168),
    (1536, 1152),
    (1536, 7168),
    (1536, 16384),
    (1528, 16384),
    (2048, 12288),
    (16384, 7168),
]
parser.add_argument(
    "-q",
    "--quant",
    type=str,
    choices=list(d_quant) + list(d_quant_e8m0) + d_quant_special,
    nargs="*",
    default=list(d_quant) + list(d_quant_e8m0) + d_quant_special,
    help="""Quantization type.
    e.g.: -q fp8_tensor""",
)

args = parser.parse_args()
list_quant = [d_quant[key] for key in args.quant if key in d_quant]
list_n = args.n or [4096, 8192]
list_m = args.m or [1, 2, 16, 32, 64, 128, 192, 256, 512, 1024, 16384, 163840]

for (
    (q_type, q_dtype),
    h_dtype,
) in itertools.product(list_quant, args.dtype):
    df = []
    for n in list_n:
        for m in list_m:
            ret = test_quant(m, n, q_type, q_dtype, h_dtype)
            df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("quant summary (markdown):\n%s", df_md)

e8m0_shapes = (
    list(itertools.product(args.m or [1536], args.n or [7168]))
    if args.m or args.n
    else E8M0_SHAPES
)
for key in [k for k in args.quant if k in d_quant_e8m0]:
    df = []
    for h_dtype, (m, n), shuffle, strided in itertools.product(
        args.dtype, e8m0_shapes, [True, False], [False, True]
    ):
        df.append(
            test_quant_1x128_e8m0(m, n, d_quant_e8m0[key], h_dtype, shuffle, strided)
        )
    df = pd.DataFrame(df)
    aiter.logger.info("%s summary (markdown):\n%s", key, df.to_markdown(index=False))

if "fp8_mx_nonfinite" in args.quant and get_gfx() in ("gfx950", "gfx1250"):
    for gs, shuffle in itertools.product([32, 64, 128], [True, False]):
        for h_dtype, pattern in itertools.product(
            [dtypes.bf16, dtypes.fp16],
            ["nan", "posinf", "neginf", "nan_inf", "all_nan", "snan"],
        ):
            test_mxfp8_nonfinite(gs, shuffle, h_dtype, pattern)
        for amax_bits in [0x7F60, 0x7F61, 0x7F69, 0x7F7F]:
            test_mxfp8_large_finite(gs, shuffle, amax_bits)
    aiter.logger.info("fp8_mx_nonfinite passed")
