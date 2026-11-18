# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

OPS = {
    "add": (aiter.add, torch.add),
    "sub": (aiter.sub, torch.sub),
    "mul": (aiter.mul, torch.mul),
    "div": (aiter.div, torch.div),
}

_BITS = {
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
}


def make_input(shape, stride, dtype):
    # distinct values with both signs, plus +-0, +-inf and nan at the front. Magnitudes stay at or
    # above 1e-3 so that no fp32 result is denormal: aiter flushes fp32 denormals, torch does not.
    t = torch.empty_strided(shape, stride, dtype=dtype, device="cuda")
    src = torch.randn(shape, dtype=torch.float32, device="cuda") * 4
    src = torch.where(src < 0, src - 1e-3, src + 1e-3)
    flat = src.view(-1)
    specials = [0.0, -0.0, float("inf"), -float("inf"), float("nan")]
    n = min(len(specials), flat.numel())
    flat[:n] = torch.tensor(specials[:n], device="cuda")
    t.copy_(src)
    return t


def assert_bitwise(out, ref, msg):
    # bitwise equal to torch, sign of zero included; any nan matches any nan
    assert (
        out.shape == ref.shape and out.dtype == ref.dtype
    ), f"{msg}: got {out.dtype}{tuple(out.shape)}, torch {ref.dtype}{tuple(ref.shape)}"
    bits = _BITS[out.dtype]
    o = out.contiguous().view(bits)
    r = ref.contiguous().view(bits)
    same = (o == r) | (out.contiguous().isnan() & ref.contiguous().isnan())
    bad = int((~same).sum().item())
    assert bad == 0, f"{msg}: {bad} of {out.numel()} elements differ from torch"


def compare(out, ref, msg):
    nan_both = out.isnan() & ref.isnan()
    err = checkAllclose(
        ref.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=0,
        atol=0,
        msg=msg,
        mask=~nan_both,
    )
    assert_bitwise(out, ref, msg)
    return err


def run_torch(op, input, other):
    return OPS[op][1](input, other)


@benchmark()
def test_binary(shape, stride, other_shape, other_stride, op, dtype):
    a = make_input(shape, stride, dtype)
    b = make_input(other_shape, other_stride, dtype)
    ref = run_torch(op, a, b)
    fn = OPS[op][0]
    out = fn(a, b)
    err = compare(out, ref, f"{op} {dtype} {shape} {other_shape}: ")
    _, us = run_perftest(fn, a, b, num_iters=20, num_rotate_args=1)
    nbytes = a.nbytes + b.nbytes + out.nbytes
    return {
        "gfx": get_gfx_runtime(),
        "aiter us": us,
        "aiter TB/s": nbytes / us / 1e6,
        "aiter err": err,
    }


def summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-i",
        "--input_shapes",
        nargs="*",
        type=dtypes.str2tuple,
        default=[
            (512,),
            (1280, 232, 256),
            (256, 256),
            (256, 8192),
            (256,),
            (1280, 32, 256),
            (384, 256),
            (384,),
            (65536,),
            (65536, 256),
            (1, 8, 256),
            (512, 256),
            (1280, 532, 256),
            (6144, 100, 96),
            (1, 100, 96),
            (6144, 16, 96),
            (6144, 1, 96),
            (6144, 289, 96),
            (289, 1),
            (6144, 16, 192),
            (192,),
            (6144, 8, 1),
            (1,),
        ],
        help="""Input shapes.
    e.g.: -i 1280,232,256""",
    )
    parser.add_argument(
        "-s",
        "--input_strides",
        nargs="*",
        type=dtypes.str2tuple,
        default=[
            (1,),
            (59392, 256, 1),
            (256, 1),
            (8192, 1),
            (1,),
            (8192, 256, 1),
            (256, 1),
            (1,),
            (1,),
            (256, 1),
            (2048, 256, 1),
            (256, 1),
            (136192, 256, 1),
            (9600, 96, 1),
            (9600, 96, 1),
            (16 * 96, 96, 1),
            (96, 96, 1),
            (289 * 96, 96, 1),
            (1, 1),
            (16 * 192, 192, 1),
            (1,),
            (8, 1, 1),
            (1,),
        ],
        help="""Input strides.
    e.g.: -s 59392,256,1""",
    )
    parser.add_argument(
        "-o",
        "--other_shapes",
        nargs="*",
        type=dtypes.str2tuple,
        default=[
            (512,),
            (1280, 232, 256),
            (256, 256),
            (256, 8192),
            (256,),
            (1280, 32, 256),
            (384, 256),
            (384,),
            (65536,),
            (65536, 256),
            (1, 8, 256),
            (512, 256),
            (1280, 532, 256),
            (1, 100, 96),
            (6144, 100, 96),
            (6144, 1, 96),
            (6144, 16, 96),
            (289, 1),
            (6144, 289, 96),
            (192,),
            (6144, 16, 192),
            (1,),
            (6144, 8, 1),
        ],
        help="""Other shapes.
    e.g.: -o 1280,232,256""",
    )
    parser.add_argument(
        "-os",
        "--other_strides",
        nargs="*",
        type=dtypes.str2tuple,
        default=[
            (1,),
            (59392, 256, 1),
            (256, 1),
            (8192, 1),
            (1,),
            (8192, 256, 1),
            (256, 1),
            (1,),
            (1,),
            (256, 1),
            (2048, 256, 1),
            (256, 1),
            (136192, 256, 1),
            (9600, 96, 1),
            (9600, 96, 1),
            (96, 96, 1),
            (16 * 96, 96, 1),
            (1, 1),
            (289 * 96, 96, 1),
            (1,),
            (16 * 192, 192, 1),
            (1,),
            (8, 1, 1),
        ],
        help="""Other strides.
    e.g.: -os 59392,256,1""",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        nargs="*",
        type=dtypes.str2Dtype,
        default=[dtypes.bf16],
        help="""Data types. Each one JIT-builds four more modules, so CI runs bf16 only.
    e.g.: -d bf16 fp16 fp32""",
    )
    parser.add_argument(
        "--op",
        nargs="*",
        choices=list(OPS),
        default=list(OPS),
        help="""Binary operators.
    e.g.: --op add mul""",
    )
    args = parser.parse_args()

    pairs = list(
        zip(
            args.input_shapes,
            args.input_strides,
            args.other_shapes,
            args.other_strides,
        )
    )
    rows = [
        test_binary(*pair, op, dtype)
        for dtype, op, pair in itertools.product(args.dtype, args.op, pairs)
    ]
    summarize("binary", rows)


if __name__ == "__main__":
    main()
