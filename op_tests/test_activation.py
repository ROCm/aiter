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
    ret = {}
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    scale = torch.max(input).to(torch.float32)
    out = torch.empty((m, n // 2), dtype=dtypes.fp8, device="cuda")

    ref = torch_scaled_silu_and_mul(input, scale)

    _, us_aiter = run_perftest(
        aiter.scaled_silu_and_mul,
        out,
        input,
        scale,
    )

    # Check if the results are close
    err = checkAllclose(ref.to(torch.float), out.to(torch.float))
    ret["us"] = us_aiter
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


@benchmark()
def test_silu_and_mul(m, n, dtype):
    ret = {}
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    out = torch.empty((m, n // 2), dtype=dtype, device="cuda")

    ref = torch_silu_and_mul(input)

    _, us_aiter = run_perftest(
        aiter.silu_and_mul,
        out,
        input,
    )

    # Check if the results are close
    err = checkAllclose(ref, out)
    ret["us"] = us_aiter
    ret["TB/s"] = (input.nbytes + out.nbytes) / us_aiter / 1e6
    ret["RD TB/s"] = (input.nbytes) / us_aiter / 1e6
    ret["WR TB/s"] = (out.nbytes) / us_aiter / 1e6
    ret["err"] = err
    return ret


l_dtype = ["fp16", "bf16"]
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
