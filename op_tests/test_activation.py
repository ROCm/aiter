import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import run_perftest, checkAllclose, benchmark
from aiter import dtypes
import pandas as pd
import argparse


def torch_silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    out = F.silu(x) * y
    return out


@benchmark()
def test_scaled_silu_and_mul(m, n, dtype, output_dtype=None):
    """
    Test scaled_silu_and_mul with flexible input/output types.
    If output_dtype is None, defaults to fp8 for quantization.
    """
    ret = {}
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    scale = torch.max(input).to(torch.float32)
    out_dtype = output_dtype if output_dtype is not None else dtypes.fp8
    out = torch.empty((m, n // 2), dtype=out_dtype, device="cuda")

    # Reference: compute, scale, convert to output dtype
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    ref = (F.silu(x) * y / scale).to(out_dtype)

    _, us_aiter = run_perftest(
        aiter.scaled_silu_and_mul,
        out,
        input,
        scale,
    )

    # Check if the results are close
    err = checkAllclose(ref.to(torch.float), out.to(torch.float))

    # Record input/output types for clarity
    dtype_map = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        dtypes.fp8: "fp8",
    }
    ret["input_dtype"] = dtype_map.get(dtype, str(dtype))
    ret["output_dtype"] = dtype_map.get(out_dtype, str(out_dtype))
    ret["M"] = m
    ret["N"] = n
    ret["us"] = us_aiter
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


@benchmark()
def test_silu_and_mul(m, n, dtype, output_dtype=None):
    """
    Test silu_and_mul with flexible input/output types.
    If output_dtype is None, output matches input dtype.
    """
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    out_dtype = output_dtype if output_dtype is not None else dtype
    out = torch.empty((m, n // 2), dtype=out_dtype, device="cuda")

    # Reference: compute in input dtype, convert to output dtype if needed
    ref = torch_silu_and_mul(input)
    if output_dtype is not None:
        ref = ref.to(output_dtype)

    _, us_aiter = run_perftest(
        aiter.silu_and_mul,
        out,
        input,
    )

    # Check if the results are close
    err = checkAllclose(ref, out)

    # Record input/output types for clarity
    dtype_map = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
    ret = {}
    ret["input_dtype"] = dtype_map.get(dtype, str(dtype))
    ret["output_dtype"] = dtype_map.get(out_dtype, str(out_dtype))
    ret["M"] = m
    ret["N"] = n
    ret["us"] = us_aiter
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


@benchmark()
def test_scaled_silu_and_mul_mixed_dtype(m, n, input_dtype, output_dtype):
    """Test fp32 input with fp16/bf16 output for scaled activation"""
    input = torch.randn(m, n, dtype=input_dtype, device="cuda")
    scale = torch.max(input).to(torch.float32)
    out = torch.empty((m, n // 2), dtype=output_dtype, device="cuda")

    # Reference: compute in fp32, scale, convert to output dtype
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    ref = (F.silu(x) * y / scale).to(output_dtype)

    _, us_aiter = run_perftest(
        aiter.scaled_silu_and_mul,
        out,
        input,
        scale,
    )

    err = checkAllclose(ref.to(torch.float), out.to(torch.float))
    dtype_map = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        dtypes.fp8: "fp8",
    }
    ret = {}
    ret["input_dtype"] = dtype_map.get(input_dtype, str(input_dtype))
    ret["output_dtype"] = dtype_map.get(output_dtype, str(output_dtype))
    ret["M"] = m
    ret["N"] = n
    ret["us"] = us_aiter
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


l_dtype = ["fp16", "bf16", "fp32"]
l_m = [1, 32, 64, 128, 256, 512, 1024, 4096, 8192, 163840]
l_n = [1024, 4096, 6400, 8192]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16, -d fp32""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-n",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="""N of mnk.
    e.g.: -n 1024""",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.m is not None:
    l_m = [args.m]
if args.n is not None:
    l_n = [args.n]

df = []
# Standard same-dtype tests
for dtype in l_dtype:
    for m in l_m:
        for n in l_n:
            ret = test_scaled_silu_and_mul(m, n, dtype)
            df.append(ret)
df = pd.DataFrame(df)
df = df[
    ["M", "N", "input_dtype", "output_dtype", "us", "TB/s", "RD TB/s", "WR TB/s", "err"]
]
df_md = df.to_markdown(index=False)
aiter.logger.info("scaled_silu_and_mul summary (markdown):\n%s", df_md)

df = []
# Standard same-dtype tests
for dtype in l_dtype:
    if dtype == torch.float32:
        continue
    for m in l_m:
        for n in l_n:
            ret = test_silu_and_mul(m, n, dtype)
            df.append(ret)
# Add fp32 input with fp16/bf16 output (bandwidth optimization)
for output_dtype in [torch.float16, torch.bfloat16]:
    for m in l_m:
        for n in l_n:
            ret = test_silu_and_mul(m, n, torch.float32, output_dtype=output_dtype)
            df.append(ret)
df = pd.DataFrame(df)
df = df[
    ["M", "N", "input_dtype", "output_dtype", "us", "TB/s", "RD TB/s", "WR TB/s", "err"]
]

df_md = df.to_markdown(index=False)
aiter.logger.info("silu_and_mul summary (markdown):\n%s", df_md)
