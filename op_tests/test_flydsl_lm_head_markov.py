# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CI sweep of the FlyDSL lm_head GEMM and fused DSpark Markov draft sampler
(gfx950) against torch references, one markdown table per op.

- ``test_lm_head_gemm``: ``flydsl_lm_head_gemm`` (row-major and shuffled
  weight) and the stock path for this shape (``tgemm.mm``; no tuned gfx950 row,
  so it resolves to its default backend) against an fp32 torch GEMM.
- ``test_markov_chain``: ``flydsl_markov_chain_sample`` against a torch
  implementation of the 5-step chain (bias GEMV, bf16 add, murmur3 Gumbel-max).

Edge cases (ties, NaN / inf rows, strides, graph replay, chunking, bitwise
equality with vLLM's Gumbel draws) are in
``op_tests/flydsl_tests/test_flydsl_{lm_head_gemm,markov_chain_sample}.py``.
"""

import argparse
import functools
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
N, K = 129280, 5120  # vocabulary, hidden
V, RANK, NSPEC = 129280, 256, 5  # Markov vocabulary, rank, draft steps
SLOTS = 512
_M32 = 0xFFFFFFFF


# ------------------------------------------------------------------ lm_head


@functools.cache
def lm_head_weight():
    g = torch.Generator(device="cuda").manual_seed(0)
    return (torch.randn(N, K, generator=g) * 0.02).to(dtypes.bf16)


def run_torch_gemm(x, w):
    return (x.float() @ w.float().t()).to(dtypes.bf16)


@benchmark()
def test_lm_head_gemm(m, layout):
    from aiter.ops.flydsl.lm_head_gemm import flydsl_lm_head_gemm, shuffle_weight
    from aiter.tuned_gemm import tgemm

    w = lm_head_weight()
    x = torch.randn(m, K).to(dtypes.bf16)
    ref = run_torch_gemm(x, w)
    # the model passes a preallocated logits buffer; NaN exposes unwritten rows
    out = torch.full((m, N), float("nan"), dtype=dtypes.bf16)
    if layout == "shuffled":
        ws = shuffle_weight(w)
        flydsl = functools.partial(flydsl_lm_head_gemm, x, ws, out, shuffled=True)
    else:
        flydsl = functools.partial(flydsl_lm_head_gemm, x, w, out)
    candidates = {"flydsl": flydsl, "tgemm": functools.partial(tgemm.mm, x, w)}
    flops = 2 * m * N * K
    nbytes = (m * K + N * K + m * N) * x.element_size()
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        y, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            y.to(dtypes.fp32),
            rtol=1e-2,
            atol=2e-2,
            msg=f"{name}: lm_head m={m} {layout}",
        )
        assert err == 0, f"{name}: {err:.2%} of elements outside rtol/atol"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# ------------------------------------------------------------------ Markov


def _rotl(x, s):
    return ((x << s) | (x >> (32 - s))) & _M32


def _mix(h, key):
    key = _rotl((key * 0xCC9E2D51) & _M32, 15)
    h = _rotl(h ^ ((key * 0x1B873593) & _M32), 13)
    return (h * 5 + 0xE6546B64) & _M32


def gumbel_noise(seed, step, pos, cols):
    """vLLM's fp32 draft Gumbel noise of (seed ^ step << 32, pos + 2^30, cols),
    evaluated in fp64 from the fp32 uniform."""
    s = (seed ^ (step << 32)) & ((1 << 64) - 1)
    h = _mix(_mix(_mix(0, s & _M32), (s >> 32) & _M32), (pos + (1 << 30)) & _M32)
    x = torch.full_like(cols, h) ^ (
        (_rotl((cols * 0xCC9E2D51) & _M32, 15) * 0x1B873593) & _M32
    )
    x = ((_rotl(x, 13) * 5 + 0xE6546B64) & _M32) ^ 16
    x = x ^ (x >> 16)
    x = (x * 0x85EBCA6B) & _M32
    x = x ^ (x >> 13)
    x = (x * 0xC2B2AE35) & _M32
    r = x ^ (x >> 16)
    u = (
        (r >> 16).double() * 2.0**-16 + ((r & 0xFFFF).double() + 0.5) * 2.0**-32
    ).to(torch.float32)
    u = u.double()
    p = torch.full_like(u, 1.0 / 8.0)
    for k in (7, 6, 5, 4, 3, 2, 1):
        p = 1.0 / k + u * p
    l1 = torch.where(u < 0.25, -u * p, torch.log(torch.clamp(1.0 - u, min=2**-24)))
    return (-torch.log(-l1)).to(torch.float32)


def run_torch_chain(base, w1, w2, ids, anchor, idx, pos, temp, seeds, cache, tokens):
    """DSpark's sequential Markov sampling, step by step in torch."""
    n = anchor.shape[0]
    prev = ids[anchor].long()
    b3 = base.view(n, NSPEC, V)
    cols = torch.arange(V, dtype=torch.int64)
    for s in range(NSPEC):
        bias = (w1[prev].float() @ w2.float().t()).to(dtypes.bf16)
        y = (b3[:, s].float() + bias.float()).to(dtypes.bf16)
        for r in range(n):
            slot = int(idx[r, s])
            t = float(temp[slot]) if slot >= 0 else 0.0
            if slot >= 0:
                cache[slot, s] = y[r]
            v = y[r].float()
            if t != 0.0:
                v = v / t + gumbel_noise(int(seeds[slot]), s, int(pos[r, s]) - 1, cols)
            tokens[r, s] = v.argmax()
        prev = tokens[:n, s]


@functools.cache
def markov_weights():
    g = torch.Generator(device="cuda").manual_seed(1234)
    w1 = (torch.randn(V, RANK, generator=g) * 0.3).to(dtypes.bf16)
    w2 = (torch.randn(V, RANK, generator=g) * 0.3).to(dtypes.bf16)
    return w1, w2


@benchmark()
def test_markov_chain(n):
    from aiter.ops.flydsl.markov_chain_sample import flydsl_markov_chain_sample

    g = torch.Generator(device="cuda").manual_seed(n)
    w1, w2 = markov_weights()
    base = (torch.randn(n * NSPEC, V, generator=g) * 2.0).to(dtypes.bf16)
    ids = torch.randint(0, V, (4096,), generator=g, dtype=torch.int32)
    anchor = torch.randint(0, 4096, (n,), generator=g)
    slots = torch.randperm(SLOTS, generator=g)[:n]
    idx = slots.view(n, 1).expand(n, NSPEC).contiguous()
    idx[torch.arange(n) % 7 == 6] = -1  # inert rows: no cache write, greedy
    pos = 131072 + torch.randint(0, 4096, (n, 1), generator=g) + torch.arange(NSPEC)
    temp = torch.tensor([(0.0, 1.0, 0.7, 1.3)[i % 4] for i in range(SLOTS)])
    seeds = torch.randint(-(2**62), 2**62, (SLOTS,), generator=g)
    args = (base, w1, w2, ids, anchor, idx, pos, temp, seeds)
    cache_ref = torch.full((SLOTS, NSPEC, V), 7.0, dtype=dtypes.bf16)
    tok_ref = torch.full((n, NSPEC), -1, dtype=torch.int64)
    run_torch_chain(*args, cache_ref, tok_ref)
    # caller buffers with a sentinel: unwritten or extra-written rows fail
    cache = torch.full((SLOTS, NSPEC, V), 7.0, dtype=dtypes.bf16)
    tokens = torch.full((n, NSPEC), -1, dtype=torch.int64)
    _, us = run_perftest(flydsl_markov_chain_sample, *args, cache, tokens)
    # step 0 has no previous-step dependence: its cache rows are the bf16 sums
    # (a few last-bit bias differences: the fp32 sum order is the MFMA's)
    err = checkAllclose(
        cache_ref[:, 0].to(dtypes.fp32),
        cache[:, 0].to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        tol_err_ratio=1e-3,
        msg=f"flydsl: markov cache n={n}",
    )
    assert err < 1e-3, f"{err:.2%} of the step-0 cache elements differ"
    agree = float((tokens == tok_ref).float().mean())
    # free-running chains part only after a near tie or a last-bit bias change
    assert agree > 0.9 and int(tokens.min()) >= 0 and int(tokens.max()) < V
    rows = n * NSPEC
    nbytes = rows * V * 2 * 2 + V * RANK * 2  # base in, cache out, markov_w2
    return {
        "gfx": get_gfx_runtime(),
        "flydsl us": us,
        "flydsl TFLOPS": 2 * rows * V * RANK / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
        "flydsl tok agree": agree,
    }


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "lm_head / Markov FlyDSL ops unsupported on %s; skipping",
            get_gfx_runtime(),
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL lm_head GEMM and Markov draft sampler",
    )
    parser.add_argument(
        "-m",
        "--rows",
        type=int,
        nargs="*",
        default=[1, 16, 64, 96, 128, 160, 256, 512, 768],
        help="lm_head rows (tokens)",
    )
    parser.add_argument(
        "-l", "--layout", nargs="*", default=["row", "shuffled"], help="weight layout"
    )
    parser.add_argument(
        "-n",
        "--requests",
        type=int,
        nargs="*",
        default=[1, 8, 32, 64, 160],
        help="Markov chain requests (5 draft rows each)",
    )
    args = parser.parse_args()

    df = [
        test_lm_head_gemm(m, lay)
        for lay, m in itertools.product(args.layout, args.rows)
    ]
    df = pd.DataFrame(df)
    aiter.logger.info(
        "lm_head GEMM summary (markdown):\n%s", df.to_markdown(index=False)
    )

    df = pd.DataFrame([test_markov_chain(n) for n in args.requests])
    aiter.logger.info(
        "Markov draft chain summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
