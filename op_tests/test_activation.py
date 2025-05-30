import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import run_perftest, checkAllclose, benchmark
from aiter import dtypes
import pandas as pd


def torch_scaled_silu_and_mul(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    out = F.silu(x) * y / scale
    return out.to(dtypes.fp8)


def torch_silu_and_mul(input: torch.Tensor, act_first: bool) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    out = F.silu(x) * y if act_first else x * F.silu(y)
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
def test_silu_and_mul(m, n, dtype, act_first):
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    out = torch.empty((m, n // 2), dtype=dtype, device="cuda")

    ref, us = run_perftest(
        torch_silu_and_mul,
        input,
        act_first,
        num_warmup=2,
        num_iters=3,
    )

    _, us_aiter = run_perftest(
        aiter.silu_and_mul,
        out,
        input,
        act_first,
        num_warmup=10,
        num_iters=100,
    )

    # Check if the results are close
    checkAllclose(ref, out)
    return {"us_aiter": us_aiter}


df = []
for dtype in [torch.float16, torch.bfloat16]:
    for m in [1, 32, 64, 128, 256, 512, 1024, 4096, 8192]:
        for n in [1024, 4096, 8192]:
            ret = test_scaled_silu_and_mul(m, n, dtype)
            df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"scaled_silu_and_mul summary:\n{df}")


df = []
for act_first in [True, False]:
    for dtype in [torch.float16, torch.bfloat16]:
        for m in [1, 32, 64, 128, 256, 512, 1024, 4096, 8192]:
            for n in [1024, 4096, 8192]:
                ret = test_silu_and_mul(m, n, dtype, act_first)
                df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"silu_and_mul summary:\n{df}")
