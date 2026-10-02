# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import torch

import aiter
from aiter import dtypes
from aiter.ops.triton.softmax import softmax
from aiter.ops.triton.topk import topk
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
g_gpu = torch.Generator(device="cuda").manual_seed(42)
state_gpu = torch.cuda.get_rng_state()


def run_greedy_sample(input):
    input = input.to(torch.float)
    # Match the native ArgMax tie policy: equal values select the lower index.
    return torch.argmax(input, dim=-1)


def run_aiter_greedy_sample(input):
    sampled_tokens = torch.empty(input.size(0), dtype=torch.int32, device="cuda")
    aiter.greedy_sample(sampled_tokens, input)
    return sampled_tokens


@benchmark()
def test_greedy_sample(M, N, dtype=torch.bfloat16):
    input = torch.randn(M, N, device="cuda", dtype=dtype)
    o_a, us_a = run_perftest(run_greedy_sample, input)
    o_b, us_b = run_perftest(run_aiter_greedy_sample, input)
    err = checkAllclose(o_a.to(torch.int), o_b, atol=0, rtol=0)
    return {"origin_us": us_a, "aiter_us": us_b, "aiter_err": err}


def run_random_sample(input, temperatures, eps, use_aiter_exponential=False):
    # Always own the benchmark working buffer. Tensor.to() aliases FP32 input
    # unless copy=True, and the in-place temperature scaling would otherwise
    # corrupt the input reused by run_perftest.
    logits = input.to(torch.float, copy=True)
    logits = logits.div_(temperatures.unsqueeze(dim=1))
    probs = softmax(logits)
    torch.cuda.set_rng_state(state_gpu)
    if use_aiter_exponential:
        exponential = torch.empty_like(probs)
        aiter.exponential(exponential, lambd=1.0, eps=eps)
    else:
        exponential = torch.empty_like(probs).exponential_(1) + eps
    logits = probs.div_(exponential)
    _, sampled_tokens = topk(logits, 1)
    # sampled_tokens = torch.argmax(logits, dim=-1)

    return sampled_tokens.view(-1)


def run_aiter_random_sample(input, temperatures, eps, inner_exponential=False):
    sampled_tokens = torch.empty(input.size(0), dtype=torch.int32, device="cuda")
    torch.cuda.set_rng_state(state_gpu)
    if inner_exponential:
        aiter.random_sample(sampled_tokens, input, temperatures, lambd=1.0, eps=eps)
    else:
        exponential = torch.empty(input.size(), dtype=torch.float32).exponential_(1)
        aiter.random_sample_outer_exponential(
            sampled_tokens, input, exponential, temperatures, eps=eps
        )
    return sampled_tokens


@benchmark()
def test_random_sample(M, N, dtype=torch.bfloat16, eps=1e-6):
    input = torch.randn(M, N, device="cuda", dtype=dtype)
    temperatures = torch.rand(M, device="cuda", dtype=torch.float)
    temperatures = torch.where(
        temperatures < 0.3, torch.ones_like(temperatures), temperatures
    )
    o_a, us_a = run_perftest(
        run_random_sample, input, temperatures, eps, use_aiter_exponential=False
    )
    o_b, us_b = run_perftest(
        run_aiter_random_sample, input, temperatures, eps, inner_exponential=False
    )
    err = checkAllclose(o_a.to(torch.int), o_b, atol=0, rtol=0)

    o_c, us_c = run_perftest(
        run_random_sample, input, temperatures, eps, use_aiter_exponential=True
    )
    o_d, us_d = run_perftest(
        run_aiter_random_sample, input, temperatures, eps, inner_exponential=True
    )
    err2 = checkAllclose(o_c.to(torch.int), o_d, atol=0, rtol=0)
    return {
        "origin_us": min(us_a, us_c),
        "exp_out_aiter_us": us_b,
        "exp_out_aiter_err": err,
        "exp_in_aiter_us": us_d,
        "exp_in_aiter_err": err2,
    }


def run_mixed_sample(input, temperatures, eps, use_aiter_exponential=False):
    # See run_random_sample: this tensor is modified in-place below.
    logits = input.to(torch.float, copy=True)
    # _, greedy_tokens = topk(logits, 1)
    greedy_tokens = torch.argmax(logits, dim=-1)
    logits.div_(temperatures.unsqueeze(dim=1))
    probs = softmax(logits)
    torch.cuda.set_rng_state(state_gpu)
    if use_aiter_exponential:
        exponential = torch.empty_like(probs)
        aiter.exponential(exponential, lambd=1.0, eps=eps)
    else:
        exponential = torch.empty_like(probs).exponential_(1) + eps
    sample_tokens = probs.div_(exponential)
    # _, sample_tokens = topk(sample_tokens, 1)
    sample_tokens = torch.argmax(sample_tokens, dim=-1)
    return torch.where(temperatures == 0, greedy_tokens, sample_tokens)


def run_aiter_mixed_sample(input, temperatures, eps, inner_exponential=False):
    sampled_tokens = torch.empty(input.size(0), dtype=torch.int32, device="cuda")
    torch.cuda.set_rng_state(state_gpu)
    if inner_exponential:
        aiter.mixed_sample(sampled_tokens, input, temperatures, lambd=1.0, eps=eps)
    else:
        exponential = torch.empty(input.size(), dtype=torch.float32).exponential_(1)
        aiter.mixed_sample_outer_exponential(
            sampled_tokens, input, exponential, temperatures, eps=eps
        )
    return sampled_tokens


@benchmark()
def test_mixed_sample(M, N, dtype=torch.bfloat16, eps=1e-6):
    input = torch.randn(M, N, device="cuda", dtype=dtype)
    temperatures = torch.rand(M, device="cuda", dtype=torch.float)
    temperatures = torch.where(
        temperatures < 0.3, torch.zeros_like(temperatures), temperatures
    )
    o_a, us_a = run_perftest(
        run_mixed_sample, input, temperatures, eps, use_aiter_exponential=False
    )
    o_b, us_b = run_perftest(
        run_aiter_mixed_sample, input, temperatures, eps, inner_exponential=False
    )
    err = checkAllclose(o_a.to(torch.int), o_b, atol=0, rtol=0)

    o_c, us_c = run_perftest(
        run_mixed_sample, input, temperatures, eps, use_aiter_exponential=True
    )
    o_d, us_d = run_perftest(
        run_aiter_mixed_sample, input, temperatures, eps, inner_exponential=True
    )
    err2 = checkAllclose(o_c.to(torch.int), o_d, atol=0, rtol=0)
    return {
        "origin_us": min(us_a, us_c),
        "exp_out_aiter_us": us_b,
        "exp_out_aiter_err": err,
        "exp_in_aiter_us": us_d,
        "exp_in_aiter_err": err2,
    }


def _check_tail_tokens(name, actual, expected, n):
    if not torch.equal(actual, expected):
        raise AssertionError(
            f"{name} N={n}: expected {expected.tolist()}, got {actual.tolist()}"
        )
    if not ((actual >= 0) & (actual < n)).all().item():
        raise AssertionError(f"{name} N={n}: sampled an out-of-range token")


def run_tail_checks(dtype, sizes=(1, 17, 4097, 16385), eps=1e-6):
    """Exercise partial vectors and row boundaries with deterministic winners."""
    for n in sizes:
        logits = torch.full((2, n), -1000.0, dtype=dtype, device="cuda")
        logits[0, -1] = 1000.0
        logits[1, 0] = 2000.0
        expected = torch.tensor([n - 1, 0], dtype=torch.int32, device="cuda")
        temperatures = torch.ones(2, dtype=torch.float32, device="cuda")
        mixed_temperatures = torch.tensor(
            [1.0, 0.0], dtype=torch.float32, device="cuda"
        )
        exponentials = torch.ones((2, n), dtype=torch.float32, device="cuda")
        out = torch.empty(2, dtype=torch.int32, device="cuda")

        aiter.greedy_sample(out, logits)
        _check_tail_tokens("greedy_sample", out, expected, n)

        aiter.random_sample_outer_exponential(
            out, logits, exponentials, temperatures, eps=eps
        )
        _check_tail_tokens("random_sample_outer_exponential", out, expected, n)

        generator = torch.Generator(device="cuda").manual_seed(42)
        aiter.random_sample(
            out, logits, temperatures, generator=generator, eps=eps
        )
        _check_tail_tokens("random_sample", out, expected, n)

        aiter.mixed_sample_outer_exponential(
            out, logits, exponentials, mixed_temperatures, eps=eps
        )
        _check_tail_tokens("mixed_sample_outer_exponential", out, expected, n)

        generator = torch.Generator(device="cuda").manual_seed(42)
        aiter.mixed_sample(
            out, logits, mixed_temperatures, generator=generator, eps=eps
        )
        _check_tail_tokens("mixed_sample", out, expected, n)

        sentinel = -1234.0
        backing = torch.full((2, n + 4), sentinel, dtype=dtype, device="cuda")
        exponential_out = backing[:, :n]
        guard = backing[:, n:].clone()
        generator = torch.Generator(device="cuda").manual_seed(42)
        aiter.exponential(exponential_out, generator=generator, eps=eps)
        if not torch.isfinite(exponential_out).all().item():
            raise AssertionError(f"exponential N={n}: produced a non-finite value")
        if not (exponential_out > 0).all().item():
            raise AssertionError(f"exponential N={n}: produced a non-positive value")
        if not torch.equal(backing[:, n:], guard):
            raise AssertionError(f"exponential N={n}: overwrote the row tail guard")

        print(f"PASS sampling vector tails dtype={dtype} M=2 N={n}")


d_sample = {
    "greedy": test_greedy_sample,
    "random": test_random_sample,
    "mixed": test_mixed_sample,
}

list_dtype = ["bf16"]
l_n = [129280, 151936][-1:]
l_m = [1, 8, 16, 32, 64, 128, 192, 256, 512]
import pandas as pd

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=["bf16", "fp16", "fp32"],
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-n",
    "--n",
    type=int,
    nargs="*",
    default=None,
    help="""N of mnk.
    e.g.: -n 1024""",
)
parser.add_argument(
    "-m",
    "--m",
    type=int,
    nargs="*",
    default=None,
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-s",
    "--sample_type",
    type=str,
    choices=list(d_sample.keys()),
    nargs="*",
    default=list(d_sample.keys()),
    help="""Sample type.
    e.g.: -s greedy random mixed""",
)
parser.add_argument(
    "--tail-only",
    action="store_true",
    help="Run deterministic partial-vector correctness checks and exit.",
)

args = parser.parse_args()
if args.dtype is None:
    list_dtype = [dtypes.d_dtypes[key] for key in list_dtype]
else:
    list_dtype = [dtypes.d_dtypes[args.dtype]]
if args.n is not None:
    l_n = args.n
if args.m is not None:
    l_m = args.m
if len(args.sample_type) > 0:
    l_sample_type = args.sample_type

list_sample_func = [d_sample[key] for key in args.sample_type if key in d_sample]

if args.tail_only:
    tail_sizes = tuple(l_n) if args.n is not None else (1, 17, 4097, 16385)
    for dtype in list_dtype:
        run_tail_checks(dtype, tail_sizes)
else:
    for test_func in list_sample_func:
        df = []
        for dtype in list_dtype:
            for n in l_n:
                for m in l_m:
                    ret = test_func(m, n, dtype)
                    df.append(ret)
        df = pd.DataFrame(df)
        df_md = df.to_markdown(index=False)
        aiter.logger.info("sample summary (markdown):\n%s", df_md)
