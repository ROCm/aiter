# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""``flydsl_markov_chain_sample`` (gfx950) against a torch reference of
DSpark's sequential Markov draft sampling (5 steps, vocab 129280, rank 256).

``test_markov_chain`` checks every step of the chain with teacher forcing: the
reference recomputes each step from the kernel's previous token, so every
step's draft-logits cache rows and tokens are graded, not only step 0's.

- Cache rows: the reference is ``bf16(base + bf16(w1[prev] @ w2^T))`` with an
  fp32 sum. The kernel sums in the MFMA's order, so an element may differ by
  the last bf16 bit of the bias (then of the sum): at most 1e-4 of a step's
  elements may differ, each by at most that. The cache is filled with a
  sentinel, so rows the kernel should not write (inert rows, unused slots) and
  rows it misses fail.
- Tokens: the argmax of the noised rows, the noise from vLLM's murmur3 draw.
  torch has no gfx950 ``v_log_f32``, so the reference noise is computed in fp64
  (within ~2 fp32 ulps of the kernel's): every token must be exact wherever the
  top two values are more than 1e-4 + 1e-6 |top| apart and at temperature 0
  (no noise: ties included), and that must cover > 90% of the chain's tokens.

``test_markov_edge_case`` covers what the sweep does not: strided base rows
and a wider token buffer, launch chunks (> 160 rows) bit-identical to separate
calls, exact ties and tile boundaries (lowest column wins), NaN / inf rows
(tokens stay in [0, 129280)), bitwise determinism, CUDA-graph replay with new
inputs and no compile after the warm-up, the eligibility helper, and, when vLLM
is importable, bit-identical tokens to vLLM's ``gumbel_sample``.
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
V, RANK, NSPEC = 129280, 256, 5  # vocabulary, Markov rank, draft steps
SLOTS = 256  # request-state slots of the draft-logits cache
TEMPS = (0.0, 1.0, 0.7, 1.3, 2.0)
FILL = 7.0  # cache sentinel
_M32 = 0xFFFFFFFF
_SPEC = {
    "num_speculative_steps": NSPEC,
    "use_fp64_gumbel": False,
    "draft_vocab_size": V,
    "has_confidence_head": False,
    "has_draft_watermarker": False,
    "has_acceptance_estimator": False,
}


# ---------------------------------------------------------------- reference


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


def run_torch_step(base, prev, w1, w2):
    """(cache rows, bias) of one step, bf16 [n, V]: bf16(base + bf16(w1[prev] @ w2^T))."""
    bias = (w1[prev].float() @ w2.float().t()).to(dtypes.bf16)
    return (base.float() + bias.float()).to(dtypes.bf16), bias


def sample_reference(y, temp, seed, step, pos):
    """(tokens [n], gap [n]): argmax of the noised rows ``y``; NaN / +inf never
    win, a row without a winner gives 0; gap = top - second value."""
    n = y.shape[0]
    cols = torch.arange(V, dtype=torch.int64)
    vals = torch.empty(n, V, dtype=torch.float32)
    for r in range(n):
        if float(temp[r]) == 0.0:
            vals[r] = y[r].float()
        else:
            vals[r] = y[r].float() / temp[r] + gumbel_noise(
                int(seed[r]), step, int(pos[r]), cols
            )
    top = vals.topk(2, dim=1).values
    ninf = torch.full_like(vals, float("-inf"))
    vals = torch.where(vals.isnan() | (vals == float("inf")), ninf, vals)
    ids = torch.where(vals.max(dim=1).values == float("-inf"), 0, vals.argmax(dim=1))
    return ids, top[:, 0] - top[:, 1]


def check_chain(d, w, cache, tokens):
    """Teacher-forced check of every step (see the module docstring); returns
    the largest fraction of differing cache elements of a step."""
    n = d["anchor"].shape[0]
    w1, w2 = w
    base = d["base"].view(n, NSPEC, V)
    idx = d["idx"].long()
    prev = d["ids"][d["anchor"]].long()
    touched = torch.zeros(SLOTS, NSPEC, dtype=torch.bool)
    worst = 0.0
    clear_ok = 0
    for s in range(NSPEC):
        valid = idx[:, s] >= 0
        slot = idx[:, s].clamp(min=0)
        y, bias = run_torch_step(base[:, s], prev, w1, w2)
        yk = cache[slot, s]
        if bool(valid.any()):
            err = checkAllclose(
                y[valid].to(dtypes.fp32),
                yk[valid].to(dtypes.fp32),
                rtol=0,
                atol=0,
                tol_err_ratio=1e-4,
                msg=f"markov cache rows n={n} step {s}",
            )
            assert err <= 1e-4, f"step {s}: {err:.4%} of the cache elements differ"
            worst = max(worst, err)
            diff = (yk.float() - y.float()).abs()[valid]
            tol = 2.0**-7 * (y.float().abs() + bias.float().abs())[valid] + 1e-30
            assert bool((diff <= tol).all()), f"step {s}: more than a last-bit error"
        touched[slot[valid], s] = True
        temp = torch.where(valid, d["temp"][slot], torch.zeros_like(d["temp"][slot]))
        # valid rows from the kernel's cache rows; inert rows (greedy) from the reference's
        src = torch.where(valid[:, None], yk, y)
        ids, gap = sample_reference(src, temp, d["seeds"][slot], s, d["pos"][:, s] - 1)
        # temperature 0 takes no noise: the kernel's values are the reference's
        # exactly, so ties are graded too (lowest column wins)
        clear = (temp == 0) | (gap > 1e-4 + 1e-6 * src.float().abs().max(dim=1).values)
        got = tokens[:n, s]
        assert bool(
            (got[clear] == ids[clear]).all()
        ), f"step {s}: {int((got != ids)[clear].sum())} clear tokens differ"
        clear_ok += int(clear.sum())
        prev = got
    assert clear_ok > 0.9 * n * NSPEC, f"only {clear_ok} of {n * NSPEC} tokens graded"
    assert bool((cache[~touched] == FILL).all()), "inert or unused rows were written"
    return worst


# ---------------------------------------------------------------- inputs


@functools.cache
def markov_weights():
    g = torch.Generator(device="cuda").manual_seed(1234)
    w1 = (torch.randn(V, RANK, generator=g) * 0.3).to(dtypes.bf16)
    w2 = (torch.randn(V, RANK, generator=g) * 0.3).to(dtypes.bf16)
    return w1, w2


def make_inputs(n, seed, inert=7, idx_dtype=torch.int64, temps=TEMPS):
    """Request rows as DSpark's speculator passes them; LM-like base logits
    (a Gaussian body and a heavy head)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn(n * NSPEC, V, generator=g) * 2.0
    head = torch.randint(0, V, (n * NSPEC, 48), generator=g)
    base.scatter_(1, head, torch.rand(n * NSPEC, 48, generator=g) * 10 + 4)
    slots = torch.randperm(SLOTS, generator=g)[:n]
    idx = slots.view(n, 1).expand(n, NSPEC).clone()
    if inert:
        idx[torch.arange(n) % inert == inert - 1] = -1  # no cache write, greedy
    pos = 131072 + torch.randint(0, 4096, (n, 1), generator=g)
    return {
        "base": base.to(dtypes.bf16),
        "ids": torch.randint(0, V, (4096,), generator=g, dtype=torch.int32),
        "anchor": torch.randint(0, 4096, (n,), generator=g),
        "idx": idx.to(idx_dtype),
        "pos": pos + torch.arange(NSPEC),
        "temp": torch.tensor([temps[i % len(temps)] for i in range(SLOTS)]),
        "seeds": torch.randint(-(2**62), 2**62, (SLOTS,), generator=g),
    }


def run_kernel(d, w, cache=None, tokens=None):
    from aiter.ops.flydsl.markov_chain_sample import flydsl_markov_chain_sample

    n = d["anchor"].shape[0]
    if cache is None:
        cache = torch.full((SLOTS, NSPEC, V), FILL, dtype=dtypes.bf16)
    if tokens is None:
        tokens = torch.full((n, NSPEC), -1, dtype=torch.int64)
    flydsl_markov_chain_sample(
        d["base"],
        w[0],
        w[1],
        d["ids"],
        d["anchor"],
        d["idx"],
        d["pos"],
        d["temp"],
        d["seeds"],
        cache,
        tokens,
    )
    return cache, tokens


def _bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


# ---------------------------------------------------------------- sweep


@benchmark()
def test_markov_chain(n, idx_dtype):
    from aiter.ops.flydsl.markov_chain_sample import flydsl_markov_chain_sample

    w1, w2 = markov_weights()
    d = make_inputs(n, n, idx_dtype=idx_dtype)
    # caller buffers with a sentinel: unwritten or extra-written rows fail
    cache = torch.full((SLOTS, NSPEC, V), FILL, dtype=dtypes.bf16)
    tokens = torch.full((n, NSPEC), -1, dtype=torch.int64)
    args = [d[k] for k in ("base",)] + [w1, w2]
    args += [d[k] for k in ("ids", "anchor", "idx", "pos", "temp", "seeds")]
    _, us = run_perftest(flydsl_markov_chain_sample, *args, cache, tokens)
    err = check_chain(d, (w1, w2), cache, tokens)
    rows = n * NSPEC
    nbytes = rows * V * 2 * 2 + V * RANK * 2  # base in, cache out, markov_w2
    return {
        "gfx": get_gfx_runtime(),
        "flydsl us": us,
        "flydsl TFLOPS": 2 * rows * V * RANK / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
    }


# ---------------------------------------------------------------- edge cases


def case_strided(w):
    """Base rows with a larger (even) stride, a token buffer wider than the batch."""
    n = 20
    d = make_inputs(n, 8)
    wide = torch.zeros(n * NSPEC, V + 64, dtype=dtypes.bf16)
    wide[:, :V] = d["base"]
    tokens = torch.full((n + 9, NSPEC), -1, dtype=torch.int64)
    ca, ta = run_kernel(d, w)
    cb, tb = run_kernel(dict(d, base=wide[:, :V]), w, tokens=tokens)
    assert torch.equal(ta, tb[:n]) and _bits_equal(ca, cb)
    assert bool((tb[n:] == -1).all()), "rows past the batch were written"


def case_chunks(w):
    """> 160 rows run as launches of 160: bit-identical to separate calls, and
    every step checked."""
    from aiter.ops.flydsl.markov_chain_sample import MAX_ROWS

    n = 200
    d = make_inputs(n, n)
    cache, tokens = run_kernel(d, w)
    check_chain(d, w, cache, tokens)
    c2 = torch.full_like(cache, FILL)
    t2 = torch.empty_like(tokens)
    for r0 in range(0, n, MAX_ROWS):
        m = min(MAX_ROWS, n - r0)
        e = dict(
            d,
            base=d["base"][r0 * NSPEC : (r0 + m) * NSPEC],
            anchor=d["anchor"][r0 : r0 + m],
            idx=d["idx"][r0 : r0 + m],
            pos=d["pos"][r0 : r0 + m],
        )
        run_kernel(e, w, c2, t2[r0 : r0 + m])
    assert torch.equal(tokens, t2) and _bits_equal(cache, c2)


def case_ties(w):
    """Zero bias: the logits are the base rows; at temperature 0 the token is
    the lowest column of the maximum (tile edges, last column, ties)."""
    w0 = (w[0], torch.zeros_like(w[1]))
    n = 16
    base = torch.full((n, NSPEC, V), -1.0, dtype=dtypes.bf16)
    peaks = [
        [0],
        [511],
        [512],
        [129279],
        [129023, 129024],
        [5, 50000, 129279],
        [1023, 1024, 4096],
        [255, 256],
    ]
    for r in range(n):
        for s in range(NSPEC):
            base[r, s, peaks[(r + s) % len(peaks)]] = 3.0
    base[3] = 0.0  # all equal: column 0
    d = make_inputs(n, 2, inert=0, temps=(0.0,))
    d["base"] = base.view(n * NSPEC, V)
    _, tokens = run_kernel(d, w0)
    exp = torch.tensor(
        [
            [0 if r == 3 else min(peaks[(r + s) % len(peaks)]) for s in range(NSPEC)]
            for r in range(n)
        ]
    )
    assert torch.equal(tokens, exp), (tokens != exp).nonzero().tolist()


def _nonfinite(w, temp):
    """Rows without a winner (all NaN, all -inf) give token 0 at every step; a
    NaN or +inf never wins and the chain goes on: every token is in [0, V)."""
    w0 = (w[0], torch.zeros_like(w[1]))
    n = 40
    d = make_inputs(n, 9, inert=0, temps=(temp,))
    base = d["base"].view(n, NSPEC, V)
    nan, ninf, pinf = float("nan"), float("-inf"), float("inf")
    base[0] = nan
    base[1] = ninf
    base[2] = nan
    base[2, :, 70001] = 0.5
    base[2, :, 129279] = 0.25
    base[3] = ninf
    base[3, :, 100000] = -3.0
    base[4] = nan
    base[4, :, ::2] = ninf
    base[5] = -1.0
    base[5, :, 129279] = pinf
    base[5, :, 33] = pinf
    base[6, 2] = nan  # one step only
    _, tokens = run_kernel(d, w0)
    assert int(tokens.min()) >= 0 and int(tokens.max()) < V
    assert bool((tokens[0] == 0).all() & (tokens[1] == 0).all())
    assert bool((tokens[4] == 0).all())
    assert bool(((tokens[2] == 70001) | (tokens[2] == 129279)).all())
    assert bool((tokens[3] == 100000).all())
    assert int(tokens[6, 2]) == 0
    if temp == 0.0:
        assert bool((tokens[2] == 70001).all())
        assert bool((tokens[5] == 0).all())  # +inf never wins: lowest column
    assert torch.equal(tokens, run_kernel(d, w0)[1])


def case_deterministic(w):
    d = make_inputs(40, 5)
    c1, t1 = run_kernel(d, w)
    c2, t2 = run_kernel(d, w)
    assert torch.equal(t1, t2) and _bits_equal(c1, c2)


def case_graph_replay(w):
    """Replay with new inputs into a poisoned cache: every step checked, and no
    compile after the warm-up for any row count."""
    from flydsl.compiler.jit_function import MlirCompiler

    from aiter.ops.flydsl.markov_chain_sample import warmup_markov_chain_sample

    warmup_markov_chain_sample(torch.device("cuda"))
    compiled = []
    original = MlirCompiler.compile

    def track(*args, **kwargs):
        compiled.append(1)
        return original(*args, **kwargs)

    MlirCompiler.compile = track
    try:
        n = 48
        d = make_inputs(n, 9)
        cache = torch.full((SLOTS, NSPEC, V), FILL, dtype=dtypes.bf16)
        tokens = torch.zeros(n, NSPEC, dtype=torch.int64)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            run_kernel(d, w, cache, tokens)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run_kernel(d, w, cache, tokens)
        for seed in (10, 11):
            e = make_inputs(n, seed)
            for k in d:
                d[k].copy_(e[k])
            cache.fill_(FILL)
            g.replay()
            g.replay()
            torch.cuda.synchronize()
            check_chain(e, w, cache, tokens)
        for n in (1, 17, 33, 64, 100, 160):
            run_kernel(make_inputs(n, n), w)
        torch.cuda.synchronize()
    finally:
        MlirCompiler.compile = original
    assert compiled == [], "a row-count variant compiled after the warm-up"


def case_eligibility(w):
    from aiter.ops.flydsl.markov_chain_sample import (
        flydsl_markov_chain_sample,
        markov_chain_sample_supported,
    )

    w1, w2 = w
    cache = torch.zeros(4, NSPEC, V, dtype=dtypes.bf16)
    assert markov_chain_sample_supported(w1, w2, cache, **_SPEC)
    for k, v in (
        ("num_speculative_steps", 3),
        ("use_fp64_gumbel", True),
        ("draft_vocab_size", 32000),
        ("has_confidence_head", True),
        ("has_draft_watermarker", True),
        ("has_acceptance_estimator", True),
    ):
        assert not markov_chain_sample_supported(w1, w2, cache, **dict(_SPEC, **{k: v}))
    assert not markov_chain_sample_supported(w1.float(), w2, cache, **_SPEC)
    assert not markov_chain_sample_supported(w1, w2[:, :128], cache, **_SPEC)
    assert not markov_chain_sample_supported(w1, w2, cache.float(), **_SPEC)
    assert not markov_chain_sample_supported(w1, w2, cache[:, :4], **_SPEC)
    try:  # the speculator features are required: no silent defaults
        markov_chain_sample_supported(w1, w2, cache)
        raise AssertionError("eligibility without the speculator features")
    except TypeError:
        pass
    # per-call contract: a ValueError (not an assert that python -O strips),
    # raised before anything is launched
    d = make_inputs(4, 3)
    full = torch.full((SLOTS, NSPEC, V), FILL, dtype=dtypes.bf16)
    args = [d["base"], w1, w2, d["ids"], d["anchor"], d["idx"], d["pos"]]
    args += [d["temp"], d["seeds"], full, torch.zeros(4, NSPEC, dtype=torch.int64)]
    bad_args = {
        "int32 sample_pos": (6, d["pos"].to(torch.int32)),
        "odd base row stride": (
            0,
            torch.zeros(4 * NSPEC, V + 1, dtype=dtypes.bf16)[:, :V],
        ),
        "markov_w2 on the host": (2, w2.cpu()),
        "int32 draft tokens": (10, torch.zeros(4, NSPEC, dtype=torch.int32)),
    }
    if torch.cuda.device_count() > 1:
        bad_args["markov_w2 on another GPU"] = (2, w2.to("cuda:1"))
        bad_args["seeds on another GPU"] = (8, d["seeds"].to("cuda:1"))
    for what, (i, t) in bad_args.items():
        bad = list(args)
        bad[i] = t
        try:
            flydsl_markov_chain_sample(*bad)
            raise AssertionError(f"{what} accepted")
        except ValueError:
            pass
    torch.cuda.synchronize()
    assert bool((full == FILL).all()), "a rejected call wrote the cache"


def case_vllm_gumbel(w):
    """The kernel's draws are vLLM's: ``gumbel_sample`` on the kernel's cache
    rows picks the same tokens, bit for bit."""
    try:
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    except Exception as e:  # noqa: BLE001
        aiter.logger.warning("vLLM not importable, skipping the bitwise check: %s", e)
        return False
    n = 64
    d = make_inputs(n, 33, inert=9)
    cache, tokens = run_kernel(d, w)
    idx = d["idx"].long()
    scratch = torch.zeros(SLOTS, NSPEC, V, dtype=dtypes.bf16)
    cols = torch.arange(NSPEC, dtype=torch.int32)
    for s in range(NSPEC):
        rows = cache[idx[:, s].clamp(min=0), s].clone()
        ok = idx[:, s] >= 0
        rows[~ok] = 0  # inert rows: no cache row, not compared
        ids = gumbel_sample(
            rows,
            d["idx"][:, s].contiguous(),
            d["temp"],
            d["seeds"],
            d["pos"][:, s] - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=scratch,
            logits_cache_col=cols[s],
        )
        assert torch.equal(ids[ok], tokens[ok, s]), f"step {s}"
    return True


CASES = {
    "strided": case_strided,
    "chunks": case_chunks,
    "ties": case_ties,
    "nonfinite_t0": functools.partial(_nonfinite, temp=0.0),
    "nonfinite_t1": functools.partial(_nonfinite, temp=1.0),
    "deterministic": case_deterministic,
    "graph_replay": case_graph_replay,
    "eligibility": case_eligibility,
    "vllm_gumbel": case_vllm_gumbel,
}


@benchmark()
def test_markov_edge_case(case):
    ran = CASES[case](markov_weights())
    return {"gfx": get_gfx_runtime(), "checked": ran is not False}


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl_markov_chain_sample unsupported on %s; skipping",
            get_gfx_runtime(),
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL fused DSpark Markov draft sampler",
    )
    parser.add_argument(
        "-n",
        "--requests",
        type=int,
        nargs="*",
        default=[1, 5, 16, 17, 32, 64, 100, 160],
        help="requests (5 draft rows each)",
    )
    parser.add_argument(
        "-i",
        "--idx_dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[torch.int64, torch.int32],
        help="dtype of the request-slot map",
    )
    parser.add_argument(
        "-c", "--case", nargs="*", default=list(CASES), choices=list(CASES)
    )
    args = parser.parse_args()

    df = pd.DataFrame(
        [
            test_markov_chain(n, t)
            for t, n in itertools.product(args.idx_dtype, args.requests)
        ]
    )
    aiter.logger.info(
        "Markov draft chain summary (markdown):\n%s", df.to_markdown(index=False)
    )
    df = pd.DataFrame([test_markov_edge_case(c) for c in args.case])
    aiter.logger.info(
        "Markov draft chain edge cases (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
