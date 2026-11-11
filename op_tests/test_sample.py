# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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


def check_exponential_row_bounds(l_dtype=(torch.float32, torch.bfloat16)):
    """aiter.exponential must fill its row and write nothing past it.

    Same partial-last-vector shape as check_vocab_edges, on the store side.
    The output descriptor's extent is rounded up to a dword (`oob_o`) because
    the buffer bound is enforced per dword, so at a 2-byte dtype with
    N % VecSize != 0 the last store reached one element past the row and a
    two-row buffer had row 1 clobbered by row 0's noise. Shrinking the extent
    to N instead drops a real element, since at an odd N elements N-1 and N
    share a dword -- hence the masked scalar tail in the kernel.
    """
    sentinel = -7.0
    for dtype in l_dtype:
        for N in (4094, 4095, 4096, 4097, 16385):
            buf = torch.full((2, N), sentinel, dtype=dtype)
            aiter.exponential(buf[0:1], lambd=1.0, eps=1e-10)
            torch.cuda.synchronize()
            unwritten = int((buf[0] == sentinel).sum())
            assert (
                unwritten == 0
            ), f"exponential {dtype} {N=}: {unwritten} elements left unwritten"
            assert bool(
                (buf[1] == sentinel).all()
            ), f"exponential {dtype} {N=}: wrote past the row into the next one"
    aiter.logger.info(
        "exponential row-bounds check passed for %s", [str(d) for d in l_dtype]
    )


def check_vocab_edges(l_dtype=(torch.bfloat16, torch.float32), eps=1e-10):
    """Every sampler must return a token id in [0, N) for any vocab size.

    Two shapes used to escape that, and neither is reachable from the perf
    sweep's N (both of its values are multiples of the vector width and far
    above one block's span):

      * N <= (BlockSize - 1) * VecSize: threads whose first element is past N
        skipped the prologue load, then ran the tail on undefined registers
        with a negative base index.
      * N % VecSize != 0: the row's last vector is partial, and the lanes past
        N fall outside the row. For the outer-exponential kernels the noise
        descriptor ends exactly at N, so the 0 it yields lands in the divisor
        and exp(x - max) / (0 + eps) won the argmax for any logits. The other
        kernels take their out-of-range value from the logits descriptor
        instead, so it only wins when every real logit in the row is negative
        -- hence the all-negative rows below.

    Both block spans are covered: 16384 for greedy_sample (block 1024, 16-wide)
    and 4096 for every other entry point (block 1024, 4-wide), including
    argmax_impl as reached through the mixed kernels at temperature 0.

    fp32 is swept as well as bf16: `oob_i` rounds the logits descriptor up to a
    dword, so at a 2-byte dtype the first lane past an odd N still reads real
    memory and the corruption is data-dependent, while at fp32 it is a hard 0.
    """
    l_vocab = [
        256,
        1000,
        2048,
        4092,
        4095,
        4096,
        4097,
        4099,
        8193,
        16368,
        16383,
        16385,
        50001,
        163840,
        163841,
    ]
    M = 4
    for dtype in l_dtype:
        for N in l_vocab:
            for all_negative in (False, True):
                logits = torch.randn(M, N, device="cuda", dtype=torch.float32) * 4
                if all_negative:
                    logits = -logits.abs() - 1.0
                logits = logits.to(dtype)
                ref_argmax = logits.to(torch.float).argmax(dim=-1).to(torch.int32)
                hot = torch.ones(M, dtype=torch.float32, device="cuda")
                cold = torch.zeros(M, dtype=torch.float32, device="cuda")
                exponential = torch.empty(M, N, dtype=torch.float32).exponential_(1)
                tag = f"{dtype} {N=} {all_negative=}"

                out = torch.empty(M, dtype=torch.int32, device="cuda")
                aiter.greedy_sample(out, logits)
                assert torch.equal(out, ref_argmax), f"greedy_sample {tag} {out=}"

                for name, fn in (
                    (
                        "mixed_sample_outer_exponential",
                        aiter.mixed_sample_outer_exponential,
                    ),
                    (
                        "random_sample_outer_exponential",
                        aiter.random_sample_outer_exponential,
                    ),
                ):
                    out = torch.empty(M, dtype=torch.int32, device="cuda")
                    fn(out, logits, exponential, hot, eps=eps)
                    probs = softmax(logits.to(torch.float))
                    ref = probs.div_(exponential + eps).argmax(dim=-1).to(torch.int32)
                    assert torch.equal(out, ref), f"{name} {tag} {out=}"

                # temperature 0 routes the mixed kernels into argmax_impl, which
                # the greedy entry point only ever reaches 16 elements wide.
                out = torch.empty(M, dtype=torch.int32, device="cuda")
                aiter.mixed_sample_outer_exponential(
                    out, logits, exponential, cold, eps=eps
                )
                assert torch.equal(
                    out, ref_argmax
                ), f"mixed_sample_outer_exponential T=0 {tag} {out=}"

                out = torch.empty(M, dtype=torch.int32, device="cuda")
                aiter.mixed_sample(out, logits, cold, lambd=1.0, eps=eps)
                assert torch.equal(out, ref_argmax), f"mixed_sample T=0 {tag} {out=}"

                # The inner-exponential kernels draw their own noise, so only
                # the index is checkable against a reference.
                for name, fn in (
                    ("mixed_sample", aiter.mixed_sample),
                    ("random_sample", aiter.random_sample),
                ):
                    out = torch.empty(M, dtype=torch.int32, device="cuda")
                    fn(out, logits, hot, lambd=1.0, eps=eps)
                    assert bool(
                        ((out >= 0) & (out < N)).all()
                    ), f"{name} {tag} out of range: {out=}"
    # The most negative finite value of the dtype is something logits can
    # actually take, so it must not tie with argmax_impl's sentinel and lose it
    # on the smaller-key rule. Taken per dtype: -FLT_MAX is outside bf16's
    # range and rounds to -inf, the one case where token 0 is the right answer.
    for dtype in l_dtype:
        neg_max = -torch.finfo(dtype).max
        for N, tok in ((17, 5), (4095, 4094), (16385, 16384)):
            logits = torch.full((1, N), float("-inf"), dtype=torch.float32)
            logits[0, tok] = neg_max
            logits = logits.to(dtype)
            assert torch.isfinite(logits[0, tok]), f"{dtype} cannot hold {neg_max}"
            cold = torch.zeros(1, dtype=torch.float32, device="cuda")
            e = torch.empty(1, N, dtype=torch.float32).exponential_(1)
            tag = f"{dtype} {N=} finite-logit=-FLT_MAX at {tok}"

            out = torch.empty(1, dtype=torch.int32, device="cuda")
            aiter.greedy_sample(out, logits)
            assert int(out[0]) == tok, f"greedy_sample {tag} -> {int(out[0])}"

            out = torch.empty(1, dtype=torch.int32, device="cuda")
            aiter.mixed_sample_outer_exponential(out, logits, e, cold, eps=eps)
            assert int(out[0]) == tok, f"mixed T=0 {tag} -> {int(out[0])}"

    aiter.logger.info(
        "vocab-edge check passed over %s for %s", l_vocab, [str(d) for d in l_dtype]
    )


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

list_sample_func = [d_sample[key] for key in args.sample_type if key in d_sample]

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

# After the sweep: this asserts, and it draws from the same global RNG the
# sweep does, so running it first would both hide the perf tables on a
# regressed tree and change the inputs every table above is measured on.
check_vocab_edges()
check_exponential_row_bounds()
