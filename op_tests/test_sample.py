# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, run_perftest, benchmark
from aiter.ops.triton.topk import topk
from aiter.ops.triton.softmax import softmax
from aiter import dtypes, greedy_sample, random_sample
import argparse

torch.set_default_device("cuda")
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
g_gpu = torch.Generator(device="cuda").manual_seed(42)
state_gpu = torch.cuda.get_rng_state()


def run_greedy_sample(input):
    input = input.to(torch.float)
    _, sampled_tokens = topk(input, 1)
    # sampled_tokens = torch.argmax(input, dim=-1)
    return sampled_tokens.view(-1)


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
    logits = input.to(torch.float)
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
    logits = input.to(torch.float)
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

list_sample_func = [d_sample[key] for key in args.sample_type if key in d_sample.keys()]

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
