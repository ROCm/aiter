import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import run_perftest, checkAllclose, benchmark
from aiter import dtypes
import pandas as pd
import argparse


def torch_scaled_silu_and_mul(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    out = F.silu(x) * y / scale
    return out.to(dtypes.fp8)


def torch_silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    out = F.silu(x) * y
    return out


@benchmark()
def test_scaled_silu_and_mul(m, n, dtype):
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    scale = torch.max(input).to(torch.float32)
    out = torch.empty((m, n // 2), dtype=dtypes.fp8, device="cuda")

    ref, us = run_perftest(
        torch_scaled_silu_and_mul,
        input,
        scale,
        num_warmup=2,
        num_iters=3,
    )

    _, us_aiter = run_perftest(
        aiter.scaled_silu_and_mul,
        out,
        input,
        scale,
        num_warmup=10,
        num_iters=100,
    )

    # Check if the results are close
    checkAllclose(ref.to(torch.float), out.to(torch.float))
    return {"us_aiter": us_aiter}


@benchmark()
def test_silu_and_mul(m, n, dtype):
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    out = torch.empty((m, n // 2), dtype=dtype, device="cuda")

    ref, us = run_perftest(
        torch_silu_and_mul,
        input,
        num_warmup=2,
        num_iters=3,
    )

    _, us_aiter = run_perftest(
        aiter.silu_and_mul,
        out,
        input,
        num_warmup=10,
        num_iters=100,
    )

    # Check if the results are close
    checkAllclose(ref, out)
    return {"us_aiter": us_aiter}


l_dtype = ["fp16", "bf16"]
l_m = [1, 32, 64, 128, 256, 512, 1024, 4096, 8192]
l_n = [1024, 4096, 8192]

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
    e.g.: -d bf16""",
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
for dtype in l_dtype:
    for m in l_m:
        for n in l_n:
            ret = test_scaled_silu_and_mul(m, n, dtype)
            df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"scaled_silu_and_mul summary:\n{df}")

df = []
for dtype in l_dtype:
    for m in l_m:
        for n in l_n:
            ret = test_silu_and_mul(m, n, dtype)
            df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"silu_and_mul summary:\n{df}")
