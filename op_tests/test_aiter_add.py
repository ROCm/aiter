# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import ctypes
import itertools
import math
import time
from functools import partial

import pandas as pd
import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.jit.utils.torch_guard import is_torch_equal_or_newer
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


INPLACE_OPS = {
    "add": aiter.add_,
    "sub": aiter.sub_,
    "mul": aiter.mul_,
    "div": aiter.div_,
}

# gfx archs on which aiter.add / sub / mul run the flat 16-byte-vector kernel
FLAT_GFX = ["gfx950", "gfx942"]

# output sizes around the flat kernel's limits: whole 16-byte vectors, and the 1 MiB switch to streaming
SAME_SHAPE_CASES = {
    "16B": 16,
    "1MiB-16": (1 << 20) - 16,
    "1MiB": 1 << 20,
    "1MiB+16": (1 << 20) + 16,
    "2MiB": 2 << 20,
    "1MiB+2": (1 << 20) + 2,
    "misaligned": 1 << 20,
    "partial_block": None,
}


def same_shape_operands(case, dtype):
    esize = torch.finfo(dtype).bits // 8
    if case == "partial_block":
        # not a whole number of 256-vector blocks for any of the three dtypes
        shape = (5, 7176)
        return make_input(shape, (7176, 1), dtype), make_input(shape, (7176, 1), dtype)
    n = -(-SAME_SHAPE_CASES[case] // esize)
    b = make_input((1, n), (n, 1), dtype)
    if case == "misaligned":
        a = make_input((1, n + 1), (n + 1, 1), dtype)[:, 1:]
    else:
        a = make_input((1, n), (n, 1), dtype)
    return a, b


def flat_kernels_launched(fn, fresh=None, label="", pattern="operator_flat_kernel"):
    # fn's result and the flat kernels it launched. The profiler now and then records no device
    # kernel at all; then the call fresh() returns, the same op on new operands made outside the
    # profile, is profiled again after a short pause, up to 3 times, with a warning each time. A
    # profile with device kernels but not the expected one is taken as it is.
    call = fn
    for attempt in range(4):
        if attempt:
            aiter.logger.warning(
                "profile recorded no device kernel, re-profiling %s (attempt %d/4)",
                label,
                attempt + 1,
            )
            call = fresh()
            time.sleep(0.075)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            res = call()
            torch.cuda.synchronize()
        if attempt == 0:
            out = res
        events = prof.key_averages()
        names = [e.key for e in events if pattern in e.key]
        if fresh is None or any(e.device_type == DeviceType.CUDA for e in events):
            break
    return out, names


def assert_flat_variant(names, kernel, streaming, msg):
    # exactly one flat kernel ran, with the expected streaming flag; the profiler may report the
    # name demangled (", false>") or mangled ("Lb0E")
    flag = "true" if streaming else "false"
    bool_arg = (f", {flag}>", "Lb1E" if streaming else "Lb0E")
    assert len(names) == 1 and any(
        t in names[0] for t in bool_arg
    ), f"{msg}expected {kernel}<..., {flag}>, got {names}"


@benchmark()
def test_same_shape(case, op, dtype, inplace):
    a, b = same_shape_operands(case, dtype)
    nbytes = a.numel() * a.element_size()
    gfx = get_gfx_runtime()
    eligible = (
        gfx in FLAT_GFX and op != "div" and nbytes % 16 == 0 and a.data_ptr() % 16 == 0
    )
    ref = run_torch(op, a.clone(), b)
    fn = INPLACE_OPS[op] if inplace else OPS[op][0]
    # every case launches a kernel; new operands of the same case keep its layout and alignment
    label = f"{op}{'_' if inplace else ''} {dtype} {case}"
    out, names = flat_kernels_launched(
        lambda: fn(a, b), lambda: partial(fn, *same_shape_operands(case, dtype)), label
    )
    if inplace:
        assert out is a, f"{op}_ must return its input"
    msg = f"{label}: "
    err = compare(out, ref, msg)

    if eligible:
        assert_flat_variant(names, "operator_flat_kernel", nbytes >= (1 << 20), msg)
    else:
        assert not names, f"{msg}unexpected flat kernel {names}"

    _, us = run_perftest(fn, a, b, num_iters=20, num_rotate_args=1)
    return {
        "gfx": gfx,
        "flat": bool(names),
        "aiter us": us,
        "aiter TB/s": 3 * nbytes / us / 1e6,
        "aiter err": err,
    }


HOST_SCALAR_SHAPES = [(256, 7168), (3, 5), (1, 1), (0,)]
# output bytes from which operator_flat_scalar_kernel streams, out of place and in place, per arch;
# other flat archs never stream. Must match binary_flat_scalar_streams() in binary_operator.cuh.
SCALAR_STREAM_FROM = {"gfx942": {False: 5 << 19, True: 5 << 18}}
# bf16 1-D outputs on both sides of those thresholds: (shape, inplace) for mul by 0.5
SCALAR_STREAM_CASES = [
    ((((5 << 18) - 16) // 2,), True),
    (((5 << 18) // 2,), True),
    ((((5 << 19) - 16) // 2,), False),
    (((5 << 19) // 2,), False),
]
HOST_SCALARS = [
    0.5,
    3,
    -2.5,
    1,
    0.0,
    -0.0,
    float("inf"),
    -float("inf"),
    float("nan"),
    1e-30,
    1e-39,
    1e-40,
    -1e-40,
    1e39,
    3.4028235e38,
    True,
]

FLT_MIN = torch.finfo(torch.float32).tiny
# torch divides by a CPU scalar as a multiply by its reciprocal, formed in double from 2.10 on
RECIPROCAL_IN_DOUBLE = is_torch_equal_or_newer("2.10.0")


def fp32_scalar(op, s, device):
    # the fp32 scalar the flat kernel applies: for div, the reciprocal torch multiplies by
    if op == "div":
        if RECIPROCAL_IN_DOUBLE:
            s = math.copysign(math.inf, s) if s == 0 else 1.0 / float(s)
        else:
            s32 = ctypes.c_float(s).value
            s = math.copysign(math.inf, s32) if s32 == 0 else 1.0 / s32
    return torch.tensor(ctypes.c_float(s).value, dtype=torch.float32, device=device)


def apply_fp32(op, x32, s32):
    return {"add": x32 + s32, "sub": x32 - s32, "mul": x32 * s32, "div": x32 * s32}[op]


def ftz(v):
    # what -fgpu-flush-denormals-to-zero does to fp32 values: subnormals become signed zero
    return torch.where(v.abs() < FLT_MIN, v * 0, v)


def subnormal_involved(op, x, s):
    s32 = fp32_scalar(op, s, x.device)
    r32 = apply_fp32(op, x.float(), s32)
    return any(bool(((v != 0) & (v.abs() < FLT_MIN)).any()) for v in (s32, r32))


def run_ftz_reference(op, x, s):
    # aiter's flat kernel flushes subnormal inputs, scalars and results to signed zero, as every
    # aiter kernel built with -fgpu-flush-denormals-to-zero does; torch keeps subnormals
    s32 = ftz(fp32_scalar(op, s, x.device))
    return ftz(apply_fp32(op, ftz(x.float()), s32)).to(x.dtype)


def run_torch_scalar(op, inplace, x, s):
    # the torch result, or the exception torch raises
    try:
        if inplace:
            return getattr(x, f"{op}_")(s), None
        return OPS[op][1](x, s), None
    except Exception as e:  # noqa: BLE001
        return None, e


@benchmark()
def test_host_scalar(shape, scalar, op, dtype, inplace):
    stride = tuple(int(torch.Size(shape[i + 1 :]).numel()) for i in range(len(shape)))
    x = make_input(shape, stride, dtype)
    x_bits = x.clone()
    ref, ref_err = run_torch_scalar(op, inplace, x.clone(), scalar)
    fn = INPLACE_OPS[op] if inplace else OPS[op][0]
    label = f"{op}{'_' if inplace else ''} {dtype} {shape} {scalar!r}"
    msg = f"{label}: "

    if ref_err is not None:
        try:
            fn(x, scalar)
        except type(ref_err):
            return {"gfx": get_gfx_runtime(), "flat": False, "torch error": True}
        raise AssertionError(f"{msg}torch raises {ref_err!r}, aiter does not")

    unit = op == "mul" and scalar == 1 and not isinstance(scalar, bool)
    gfx = get_gfx_runtime()
    nbytes = x.numel() * x.element_size()
    expect_flat = (
        gfx in FLAT_GFX
        and not isinstance(scalar, bool)
        and not (inplace and unit)
        and 0 < nbytes
        and nbytes % 16 == 0
    )
    # every case that launches a kernel (flat or torch's) re-profiles; mul_(x, 1) and [0] launch
    # nothing
    launches = nbytes > 0 and not (inplace and unit)
    fresh = (lambda: partial(fn, x_bits.clone(), scalar)) if launches else None
    out, names = flat_kernels_launched(
        lambda: fn(x, scalar), fresh, label, pattern="operator_flat_scalar_kernel"
    )
    # Flat-kernel cases with a subnormal fp32 scalar or result (div by 1e39 or 3.4028235e38,
    # mul by 1e-39 or +-1e-40, add / sub of +-1e-40 at x = +-0, ...) compare against the
    # flush-to-zero reference; every other case compares against torch directly.
    ref_kind = "torch"
    if expect_flat and subnormal_involved(op, x_bits, scalar):
        ref, ref_kind = run_ftz_reference(op, x_bits, scalar), "ftz"
    if unit:
        # x * 1 is x exactly, so x is accepted as well as torch's result, which may differ in the
        # sign of zero (torch on gfx942 gives +0.0 for an fp16 -0.0 * 1)
        try:
            err = compare(out, ref, f"{msg}[{ref_kind} reference] ")
        except AssertionError:
            assert_bitwise(out, x_bits, msg)
            err = 0
    else:
        err = compare(out, ref, f"{msg}[{ref_kind} reference] ")

    if inplace:
        assert out is x, f"{msg}must return its input"
        if unit:
            assert_bitwise(x, x_bits, f"{msg}input changed")
    else:
        assert out is not x, f"{msg}returns its input"
        assert x.numel() == 0 or out.data_ptr() != x.data_ptr(), f"{msg}aliases input"
        assert_bitwise(x, x_bits, f"{msg}input changed")

    assert (
        bool(names) == expect_flat
    ), f"{msg}expected flat scalar kernel: {expect_flat}, launched {names}"
    if names:
        stream_from = SCALAR_STREAM_FROM.get(gfx)
        streams = stream_from is not None and nbytes >= stream_from[inplace]
        assert_flat_variant(names, "operator_flat_scalar_kernel", streams, msg)

    ret = {
        "gfx": gfx,
        "flat": bool(names),
        "torch error": False,
        "reference": ref_kind,
        "aiter err": err,
    }
    if shape == HOST_SCALAR_SHAPES[0] and scalar == 0.5:
        _, us = run_perftest(fn, x, scalar, num_iters=20, num_rotate_args=1)
        ret["aiter us"] = us
        ret["aiter TB/s"] = 2 * nbytes / us / 1e6
    return ret


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

    rows = [
        test_same_shape(case, op, dtype, inplace)
        for dtype, op, case, inplace in itertools.product(
            args.dtype, args.op, SAME_SHAPE_CASES, [False, True]
        )
    ]
    summarize("same_shape", rows)

    rows = [
        test_host_scalar(shape, scalar, op, dtype, inplace)
        for dtype, op, shape, scalar, inplace in itertools.product(
            args.dtype, args.op, HOST_SCALAR_SHAPES, HOST_SCALARS, [False, True]
        )
    ]
    if dtypes.bf16 in args.dtype and "mul" in args.op:
        rows += [
            test_host_scalar(shape, 0.5, "mul", dtypes.bf16, inplace)
            for shape, inplace in SCALAR_STREAM_CASES
        ]
    summarize("host_scalar", rows)


if __name__ == "__main__":
    main()
