# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import os
import subprocess
import sys
import tempfile

import pandas as pd
import torch

import aiter
from aiter import dtypes, get_hip_quant, get_torch_quant, get_triton_quant
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

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


# For cols <= 2048, dynamic_per_token_scaled_quant runs several rows per block on gfx942
# (dynamic_per_token_scaled_quant_multirow_kernel). It has to reproduce the one-row-per-block
# kernel byte for byte; AITER_PER_TOKEN_QUANT_DISABLE_MULTIROW=1 selects that kernel, so a child
# process with it set gives the bytes to compare against. Asserts, unlike checkAllclose.
_ONE_ROW_CHILD = """
import sys, torch, aiter
cases = torch.load(sys.argv[1])
res = []
for x, q_name, num_rows, factor in cases:
    x, q_dtype = x.cuda(), getattr(torch, q_name)
    out = torch.full(x.shape, 0x5A, dtype=torch.uint8, device="cuda").view(q_dtype)
    scale = torch.full((x.shape[0], 1), 0x7FA5A5A5, dtype=torch.int32, device="cuda")
    nr = None if num_rows is None else torch.tensor([num_rows], dtype=torch.int32, device="cuda")
    aiter.dynamic_per_token_scaled_quant(out, x, scale.view(torch.float32), num_rows=nr, num_rows_factor=factor)
    res.append((out.view(torch.uint8).cpu(), scale.cpu()))
torch.save(res, sys.argv[2])
"""


def _per_token_input(m, n, h_dtype, seed):
    """randn rows, with the rows below 16 replaced by zero / inf / NaN / denormal / huge rows."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn((m, n), dtype=dtypes.fp32, generator=g).to(h_dtype)
    xi = x.view(torch.int16)
    snan = 0x7F81 if h_dtype == dtypes.bf16 else 0x7C01
    tiny = 1e-39 if h_dtype == dtypes.bf16 else 1e-6
    huge = 3e38 if h_dtype == dtypes.bf16 else 6e4
    special = [
        lambda r: x[r].zero_(),
        lambda r: x[r].fill_(-0.0),
        lambda r: x[r, n // 3].fill_(float("inf")),
        lambda r: x[r, n // 2].fill_(float("-inf")),
        lambda r: x[r, n - 1].fill_(float("nan")),
        lambda r: x[r].fill_(float("nan")),
        lambda r: xi[r, 7].fill_(snan),  # ends a thread's 8-wide vector
        lambda r: xi[r, 7::8].fill_(snan),
        lambda r: xi[r, : min(n, 8)].fill_(snan - 0x8000),
        lambda r: x[r].mul_(tiny),
        lambda r: x[r].clamp_(-1, 1).mul_(huge),
        lambda r: x[r].fill_(1.0),
    ]
    for r, f in enumerate(special[: min(m, len(special))]):
        f(r)
    return x, min(m, 16)


def test_per_token_quant_multirow(h_dtype, q_dtype):
    # cols 256: two rows a wave; 384: one wave a row; 1000 / 2048: 2 / 4 vectors a lane.
    # M sweeps the rows-per-thread choice.
    cases = []
    for n in [256, 384, 1000, 2048]:
        for m in [3, 300, 30000]:
            cases.append((m, n, None, 1))
    cases += [(30000, 384, 20001, 1), (300, 2048, 100, 2), (300, 256, 0, 1)]
    dmax = (
        torch.iinfo(q_dtype).max if q_dtype == dtypes.i8 else torch.finfo(q_dtype).max
    )
    inv_dmax = torch.tensor(1.0 / dmax, dtype=dtypes.fp32)

    inputs, results = [], []
    for i, (m, n, num_rows, factor) in enumerate(cases):
        x, n_special = _per_token_input(m, n, h_dtype, seed=i)
        out = torch.full((m, n), 0x5A, dtype=torch.uint8).view(q_dtype)
        scale = torch.full((m, 1), 0x7FA5A5A5, dtype=torch.int32)
        nr = None if num_rows is None else torch.tensor([num_rows], dtype=torch.int32)
        aiter.dynamic_per_token_scaled_quant(
            out, x, scale.view(dtypes.fp32), num_rows=nr, num_rows_factor=factor
        )
        valid = m if num_rows is None else max(0, min(m, num_rows * factor))
        ob, sb = out.view(torch.uint8), scale
        tag = f"{h_dtype} -> {q_dtype}, m={m} n={n} num_rows={num_rows}x{factor}"
        assert (ob[valid:] == 0x5A).all() and (sb[valid:] == 0x7FA5A5A5).all(), tag

        # torch reference on the randn rows: scale bit-exact, values within one step
        rows = slice(n_special, valid)
        xf = x[rows].to(dtypes.fp32)
        ref_scale = xf.abs().amax(-1, keepdim=True) * inv_dmax.cuda()
        assert torch.equal(sb[rows].view(dtypes.fp32), ref_scale), tag
        y = (xf * (1.0 / ref_scale)).clamp(-dmax, dmax)
        if q_dtype == dtypes.i8:
            ref = y.to(q_dtype).to(dtypes.fp32)
            assert (out[rows].to(dtypes.fp32) - ref).abs().le(1).all(), tag
        else:
            ref = y.to(q_dtype).to(dtypes.fp32)
            torch.testing.assert_close(
                out[rows].to(dtypes.fp32), ref, rtol=2**-3, atol=2**-9, msg=tag
            )
        inputs.append((x.cpu(), str(q_dtype).split(".")[-1], num_rows, factor))
        results.append((ob.cpu(), sb.cpu()))

    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.pt"), os.path.join(d, "out.pt")
        torch.save(inputs, src)
        env = dict(os.environ, AITER_PER_TOKEN_QUANT_DISABLE_MULTIROW="1")
        subprocess.run(
            [sys.executable, "-c", _ONE_ROW_CHILD, src, dst], env=env, check=True
        )
        one_row = torch.load(dst)
    for (m, n, num_rows, factor), (ob, sb), (ob1, sb1) in zip(cases, results, one_row):
        tag = f"{h_dtype} -> {q_dtype}, m={m} n={n} num_rows={num_rows}x{factor}"
        assert torch.equal(ob, ob1) and torch.equal(
            sb, sb1
        ), f"differs from one-row kernel: {tag}"
    aiter.logger.info(
        f"per-token multi-row quant {h_dtype} -> {q_dtype}: {len(cases)} cases match"
    )


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
    default=[4096, 8192],
    help="""N of mnk.
    e.g.: -n 1024""",
)
parser.add_argument(
    "-m",
    "--m",
    type=int,
    nargs="*",
    default=[1, 2, 16, 32, 64, 128, 192, 256, 512, 1024, 16384, 163840],
    help="""M of mnk.
    e.g.: -m 32""",
)
d_quant = {
    "fp8_tensor": (aiter.QuantType.per_Tensor, dtypes.fp8),
    "fp8_token": (aiter.QuantType.per_Token, dtypes.fp8),
    "fp8_1x128": (aiter.QuantType.per_1x128, dtypes.fp8),
    "i8_token": (aiter.QuantType.per_Token, dtypes.i8),
    # 'fp4x2-1x32': (aiter.QuantType.per_1x32, dtypes.fp4x2),
}
parser.add_argument(
    "-q",
    "--quant",
    type=str,
    choices=list(d_quant.keys()),
    nargs="*",
    default=list(d_quant.keys()),
    help="""Quantization type.
    e.g.: -q fp8_tensor""",
)

args = parser.parse_args()
list_quant = [d_quant[key] for key in args.quant]

for (q_type, q_dtype), h_dtype in itertools.product(list_quant, args.dtype):
    if q_type == aiter.QuantType.per_Token:
        test_per_token_quant_multirow(h_dtype, q_dtype)

for (
    (q_type, q_dtype),
    h_dtype,
) in itertools.product(list_quant, args.dtype):
    df = []
    for n in args.n:
        for m in args.m:
            ret = test_quant(m, n, q_type, q_dtype, h_dtype)
            df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("quant summary (markdown):\n%s", df_md)
