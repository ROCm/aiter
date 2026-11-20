# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""``flydsl_markov_chain_sample`` against a torch reference of DSpark's
sequential Markov draft sampling (5 steps, vocab 129280, rank 256).

The reference follows the kernel's own chain (teacher forcing): per step the
bf16 cache rows from the previous token (all but a few last-bit elements
exact: the fp32 bias sum has the kernel's order) and the token as the argmax of
the noised cache rows. torch has no gfx950 ``v_log_f32``, so the reference
noise is computed in fp64 (within ~2 fp32 ulps of the kernel's): ids are
compared except where the top two values differ by less than
1e-4 + 1e-6 |top|. With vLLM installed, the draws are compared bit for bit with
vLLM's Triton ``gumbel_sample``.

Covers LM-like logits, temperatures 0 / 1 / 0.7 / 1.3 / 2, inert rows (slot -1:
no cache write, greedy), every row-count variant (1-200 rows, chunks above
160), ties (lowest column at temperature 0) and tile boundaries, NaN / inf
rows, int32 / int64 slot maps and ids, strided base rows, bitwise determinism
and graph replay with new inputs.

Run: ``python3 -m pytest op_tests/flydsl_tests/test_flydsl_markov_chain_sample.py -q``
"""

import sys

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

if not torch.cuda.is_available() or get_gfx_runtime() != "gfx950":
    pytest.skip("flydsl_markov_chain_sample needs gfx950", allow_module_level=True)

from aiter.ops.flydsl import markov_chain_sample as mc
from aiter.ops.flydsl.markov_chain_sample import (
    NSPEC,
    R,
    V,
    flydsl_markov_chain_sample,
    markov_chain_sample_supported,
    warmup_markov_chain_sample,
)

SLOTS = 512
TEMPS = (0.0, 1.0, 0.7, 1.3, 2.0)
_M32 = 0xFFFFFFFF


# ---------------------------------------------------------------- reference


def _rotl(x, s):
    return ((x << s) | (x >> (32 - s))) & _M32


def _mix(h, key):
    key = (key * 0xCC9E2D51) & _M32
    key = _rotl(key, 15)
    key = (key * 0x1B873593) & _M32
    h = h ^ key
    h = _rotl(h, 13)
    return (h * 5 + 0xE6546B64) & _M32


def _tail(h, c):
    """murmur3_hash32 after the position, for vocab ids ``c`` (int64 tensor)."""
    key = (c * 0xCC9E2D51) & _M32
    key = ((key << 15) | (key >> 17)) & _M32
    key = (key * 0x1B873593) & _M32
    x = torch.full_like(c, h) ^ key
    x = ((x << 13) | (x >> 19)) & _M32
    x = (x * 5 + 0xE6546B64) & _M32
    x = x ^ 16
    x = x ^ (x >> 16)
    x = (x * 0x85EBCA6B) & _M32
    x = x ^ (x >> 13)
    x = (x * 0xC2B2AE35) & _M32
    return x ^ (x >> 16)


def uniform_reference(seed, step, pos, cols):
    """vLLM's murmur3_uniform32 of (seed ^ (step << 32), pos + 2^30, cols)."""
    s = (seed ^ (step << 32)) & ((1 << 64) - 1)
    h = _mix(_mix(_mix(0, s & _M32), (s >> 32) & _M32), (pos + (1 << 30)) & _M32)
    r = _tail(h, cols.to(torch.int64))
    hi = (r >> 16).to(torch.float64)
    lo = (r & 0xFFFF).to(torch.float64)
    return (hi * 2.0**-16 + (lo + 0.5) * 2.0**-32).to(torch.float32)


def noise_reference(u):
    """-log(-log1p_neg_stable(u)) in fp64 from the fp32 u."""
    u = u.to(torch.float64)
    p = torch.full_like(u, 1.0 / 8.0)
    for k in (7, 6, 5, 4, 3, 2, 1):
        p = 1.0 / k + u * p
    series = -u * p
    direct = torch.log(torch.clamp(1.0 - u, min=5.960464477539063e-08))
    l1 = torch.where(u < 0.25, series, direct)
    return (-torch.log(-l1)).to(torch.float32)


def bias_reference(prev, w1, w2):
    """bf16 [n, V]: bf16(w1[prev] @ w2^T) with fp32 sums."""
    return (w1[prev].float() @ w2.float().t()).to(torch.bfloat16)


def logits_reference(base, bias):
    """bf16 [n, V]: bf16(f32(base) + f32(bias)), the draft-cache row."""
    return (base.float() + bias.float()).to(torch.bfloat16)


def sample_reference(y, temp, seed, step, pos):
    """(ids [n], gap [n]): argmax of the noised rows ``y``; NaN / +inf never
    win, a row without a winner gives 0; gap = top - second value."""
    n = y.shape[0]
    cols = torch.arange(V, device=y.device)
    vals = torch.empty(n, V, dtype=torch.float32, device=y.device)
    for r in range(n):
        yr = y[r].float()
        if float(temp[r]) == 0.0:
            vals[r] = yr
        else:
            u = uniform_reference(int(seed[r]), step, int(pos[r]), cols)
            vals[r] = yr / temp[r] + noise_reference(u)
    top = vals.topk(2, dim=1)
    bad = vals.isnan() | (vals == float("inf"))
    vals = torch.where(bad, torch.full_like(vals, float("-inf")), vals)
    ids = vals.argmax(dim=1)
    ids = torch.where(vals.max(dim=1).values == float("-inf"), 0, ids)
    return ids, top.values[:, 0] - top.values[:, 1]


def chain_reference(base, w1, w2, ids, anchor, idx, pos, temperature, seeds):
    """The whole chain in torch: (cache rows {(slot, step): row}, tokens)."""
    n = anchor.shape[0]
    prev = ids[anchor].long()
    b3 = base.view(n, NSPEC, V)
    slots = idx.long()
    tokens = torch.empty(n, NSPEC, dtype=torch.int64, device=base.device)
    rows = {}
    for s in range(NSPEC):
        y = logits_reference(b3[:, s], bias_reference(prev, w1, w2))
        valid = slots[:, s] >= 0
        slot = slots[:, s].clamp(min=0)
        for r in valid.nonzero().flatten().tolist():
            rows[(int(slot[r]), s)] = y[r]
        temp = torch.where(
            valid, temperature[slot], torch.zeros_like(temperature[slot])
        )
        tokens[:, s], _ = sample_reference(y, temp, seeds[slot], s, pos[:, s] - 1)
        prev = tokens[:, s]
    return rows, tokens


# ---------------------------------------------------------------- harness


@pytest.fixture(scope="module")
def weights():
    gen = torch.Generator().manual_seed(1234)
    w1 = (torch.randn(V, R, generator=gen) * 0.3).to(torch.bfloat16).cuda()
    w2 = (torch.randn(V, R, generator=gen) * 0.3).to(torch.bfloat16).cuda()
    return w1, w2


def lm_rows(n, gen):
    """bf16 logits of n * 5 rows: a Gaussian body and a heavy head, as an LM's."""
    x = torch.randn(n * NSPEC, V, generator=gen) * 2.0
    head = torch.randint(0, V, (n * NSPEC, 48), generator=gen)
    x.scatter_(1, head, torch.rand(n * NSPEC, 48, generator=gen) * 10 + 4)
    return x.to(torch.bfloat16).cuda()


def make(n, gen, inert=7, idx_dtype=torch.int64, ids_dtype=torch.int32, temps=TEMPS):
    slots = torch.randperm(SLOTS, generator=gen)[:n]
    idx = slots.view(n, 1).expand(n, NSPEC).clone()
    if inert:
        idx[torch.arange(n) % inert == inert - 1] = -1
    pos = 131072 + torch.randint(0, 4096, (n, 1), generator=gen)
    temp = [temps[i % len(temps)] for i in range(SLOTS)]
    d = {
        "base": lm_rows(n, gen),
        "ids": torch.randint(0, V, (4096,), generator=gen).to(ids_dtype),
        "anchor": torch.randint(0, 4096, (n,), generator=gen),
        "idx": idx.to(idx_dtype),
        "pos": pos + torch.arange(NSPEC),
        "temp": torch.tensor(temp, dtype=torch.float32),
        "seeds": torch.randint(-(2**62), 2**62, (SLOTS,), generator=gen),
    }
    return {k: v.cuda() for k, v in d.items()}


def run(d, w, cache=None, tokens=None, fill=0.0):
    n = d["anchor"].shape[0]
    if cache is None:
        cache = torch.full((SLOTS, NSPEC, V), fill, dtype=torch.bfloat16, device="cuda")
    if tokens is None:
        tokens = torch.full((n, NSPEC), -1, dtype=torch.int64, device="cuda")
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


def check(d, w, cache, tokens, fill=0.0):
    n = d["anchor"].shape[0]
    w1, w2 = w
    base = d["base"].view(n, NSPEC, V)
    idx = d["idx"].long()
    prev = d["ids"][d["anchor"]].long()
    touched = torch.zeros(SLOTS, NSPEC, dtype=torch.bool, device="cuda")
    clear_total = clear_ok = 0
    for s in range(NSPEC):
        valid = idx[:, s] >= 0
        bias = bias_reference(prev, w1, w2)
        y = logits_reference(base[:, s], bias)
        slot = idx[:, s].clamp(min=0)
        yk = cache[slot, s]
        diff = (yk.float() - y.float()).abs()
        bad = (diff > 0) & valid[:, None]
        assert float(bad.float().mean()) < 1e-4, (s, int(bad.sum()))
        # at most a last-bit difference of the bias, then of the sum
        tol = 2.0**-7 * (y.float().abs() + bias.float().abs()) + 1e-30
        assert bool((diff <= tol)[valid].all()), s
        touched[slot[valid], s] = True
        temp = torch.where(valid, d["temp"][slot], torch.zeros_like(d["temp"][slot]))
        # valid rows from the kernel's cache values; inert rows (greedy) from
        # the reference's
        src = torch.where(valid[:, None], yk, y)
        ids, gap = sample_reference(src, temp, d["seeds"][slot], s, d["pos"][:, s] - 1)
        top = src.float().abs().max(dim=1).values
        clear = gap > 1e-4 + 1e-6 * top
        got = tokens[:, s]
        assert bool((got[clear] == ids[clear]).all()), (s, int((got != ids).sum()))
        clear_total += clear.numel()
        clear_ok += int(clear.sum())
        prev = got
    assert clear_ok > 0.9 * clear_total
    untouched = ~touched
    assert bool((cache[untouched] == fill).all()), "inert or unused rows were written"


# ---------------------------------------------------------------- tests


@pytest.mark.parametrize("n", [1, 5, 11, 16, 17, 32, 48, 64, 65, 100, 128, 160])
def test_chain_rows(weights, n):
    d = make(n, torch.Generator().manual_seed(n))
    cache, tokens = run(d, weights, fill=7.0)
    torch.cuda.synchronize()
    check(d, weights, cache, tokens, fill=7.0)


@pytest.mark.parametrize("n", [161, 200])
def test_chunked_rows(weights, n):
    d = make(n, torch.Generator().manual_seed(n))
    cache, tokens = run(d, weights)
    check(d, weights, cache, tokens)
    # the same bits as separate calls of the chunks
    c2 = torch.zeros_like(cache)
    t2 = torch.empty_like(tokens)
    for r0 in range(0, n, mc.MAX_ROWS):
        m = min(mc.MAX_ROWS, n - r0)
        e = dict(
            d,
            base=d["base"][r0 * NSPEC : (r0 + m) * NSPEC],
            anchor=d["anchor"][r0 : r0 + m],
            idx=d["idx"][r0 : r0 + m],
            pos=d["pos"][r0 : r0 + m],
        )
        run(e, weights, c2, t2[r0 : r0 + m])
    assert torch.equal(tokens, t2)
    assert torch.equal(cache.view(torch.int16), c2.view(torch.int16))


@pytest.mark.parametrize(
    "idx_dtype,ids_dtype", [(torch.int32, torch.int64), (torch.int64, torch.int32)]
)
def test_index_dtypes(weights, idx_dtype, ids_dtype):
    gen = torch.Generator().manual_seed(21)
    a = make(24, gen, idx_dtype=torch.int64, ids_dtype=torch.int32)
    b = dict(a, idx=a["idx"].to(idx_dtype), ids=a["ids"].to(ids_dtype))
    ca, ta = run(a, weights)
    cb, tb = run(b, weights)
    assert torch.equal(ta, tb)
    assert torch.equal(ca.view(torch.int16), cb.view(torch.int16))


def test_strided_inputs(weights):
    """Base rows with a larger stride, a token buffer wider than the batch."""
    n = 20
    d = make(n, torch.Generator().manual_seed(8))
    wide = torch.zeros(n * NSPEC, V + 64, dtype=torch.bfloat16, device="cuda")
    wide[:, :V] = d["base"]
    e = dict(d, base=wide[:, :V])
    tokens = torch.full((n + 9, NSPEC), -1, dtype=torch.int64, device="cuda")
    ca, ta = run(d, weights)
    cb, tb = run(e, weights, tokens=tokens)
    assert torch.equal(ta, tb[:n])
    assert torch.equal(ca.view(torch.int16), cb.view(torch.int16))
    assert bool((tb[n:] == -1).all())


def test_ties_and_boundaries(weights):
    """Zero bias head: the logits are the base rows; at temperature 0 the token
    is the lowest column of the maximum (tile edges, last column, ties)."""
    w1, _ = weights
    w2 = torch.zeros_like(weights[1])
    n = 16
    base = torch.full((n, NSPEC, V), -1.0, dtype=torch.bfloat16)
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
    d = make(n, torch.Generator().manual_seed(2), inert=0, temps=(0.0,))
    d["base"] = base.view(n * NSPEC, V).cuda()
    _, tokens = run(d, (w1, w2))
    for r in range(n):
        for s in range(NSPEC):
            exp = 0 if r == 3 else min(peaks[(r + s) % len(peaks)])
            assert int(tokens[r, s]) == exp, (r, s, int(tokens[r, s]), exp)


@pytest.mark.parametrize("temp", [0.0, 1.0])
def test_degenerate_rows_stay_in_range(weights, temp):
    """Rows without a winner (all NaN, all -inf) give token 0 at every step; a
    NaN or +inf never wins and the chain goes on: every token is in [0, V)."""
    w1, _ = weights
    w2 = torch.zeros_like(weights[1])
    n = 40
    d = make(n, torch.Generator().manual_seed(9), inert=0, temps=(temp,))
    base = d["base"].view(n, NSPEC, V)
    nan, ninf, pinf = float("nan"), float("-inf"), float("inf")
    base[0] = nan
    base[1] = ninf
    base[2, :, :] = nan
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
    _, tokens = run(d, (w1, w2))
    torch.cuda.synchronize()
    assert int(tokens.min()) >= 0 and int(tokens.max()) < V
    assert bool((tokens[0] == 0).all()) and bool((tokens[1] == 0).all())
    assert bool((tokens[4] == 0).all())
    assert bool(((tokens[2] == 70001) | (tokens[2] == 129279)).all())
    assert bool((tokens[3] == 100000).all())
    assert int(tokens[6, 2]) == 0
    if temp == 0.0:
        assert bool((tokens[2] == 70001).all())
        # +inf never wins: all other values equal, lowest column
        assert bool((tokens[5] == 0).all())
    assert torch.equal(tokens, run(d, (w1, w2))[1])


@pytest.mark.parametrize("n", [3, 40])
def test_whole_chain_against_torch(weights, n):
    """Free-running torch chain (its own previous tokens): most tokens agree,
    the first step's almost all (later steps diverge after a near tie or a
    last-bit bias difference)."""
    d = make(n, torch.Generator().manual_seed(n + 50))
    cache, tokens = run(d, weights)
    rows, t2 = chain_reference(
        d["base"],
        weights[0],
        weights[1],
        d["ids"],
        d["anchor"],
        d["idx"],
        d["pos"],
        d["temp"],
        d["seeds"],
    )
    assert float((tokens[:, 0] == t2[:, 0]).float().mean()) > 0.95
    assert float((tokens == t2).float().mean()) > 0.9
    for r in (d["idx"][:, 0] >= 0).nonzero().flatten().tolist():
        slot = int(d["idx"][r, 0])
        mism = (cache[slot, 0] != rows[(slot, 0)]).float().mean()
        assert float(mism) < 1e-4


def test_deterministic(weights):
    d = make(40, torch.Generator().manual_seed(5))
    c1, t1 = run(d, weights)
    c2, t2 = run(d, weights)
    assert torch.equal(t1, t2) and torch.equal(
        c1.view(torch.int16), c2.view(torch.int16)
    )


def test_graph_replay_new_inputs(weights, monkeypatch):
    from flydsl.compiler.jit_function import MlirCompiler

    warmup_markov_chain_sample(torch.device("cuda"))
    compiled = []
    original = MlirCompiler.compile

    def track(*args, **kwargs):
        compiled.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(MlirCompiler, "compile", track)
    d = make(48, torch.Generator().manual_seed(9))
    cache = torch.zeros(SLOTS, NSPEC, V, dtype=torch.bfloat16, device="cuda")
    tokens = torch.zeros(48, NSPEC, dtype=torch.int64, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run(d, weights, cache, tokens)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run(d, weights, cache, tokens)
    for seed in (10, 11):
        e = make(48, torch.Generator().manual_seed(seed))
        for k in d:
            d[k].copy_(e[k])
        cache.zero_()
        g.replay()
        g.replay()
        torch.cuda.synchronize()
        c2, t2 = run(e, weights)
        assert torch.equal(tokens, t2)
        assert torch.equal(cache.view(torch.int16), c2.view(torch.int16))
        check(e, weights, cache, tokens)
    # every row-count variant was compiled by the warm-up
    for n in (1, 17, 33, 64, 100, 160):
        run(make(n, torch.Generator().manual_seed(n)), weights)
    assert compiled == []


def test_supported(weights):
    w1, w2 = weights
    cache = torch.zeros(4, NSPEC, V, dtype=torch.bfloat16, device="cuda")
    assert markov_chain_sample_supported(w1, w2, cache)
    assert not markov_chain_sample_supported(w1, w2, cache, num_speculative_steps=4)
    assert not markov_chain_sample_supported(w1, w2, cache, use_fp64_gumbel=True)
    assert not markov_chain_sample_supported(w1.float(), w2, cache)
    assert not markov_chain_sample_supported(w1, w2[:, :128], cache)
    assert not markov_chain_sample_supported(w1, w2, cache.float())
    assert not markov_chain_sample_supported(w1, w2, cache[:, :4])


def test_matches_vllm_gumbel(weights):
    """The kernel's draws are vLLM's: ``gumbel_sample`` on the kernel's cache
    rows picks the same tokens, bit for bit."""
    pytest.importorskip("vllm")
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    n = 64
    d = make(n, torch.Generator().manual_seed(33), inert=9)
    cache, tokens = run(d, weights)
    idx = d["idx"].long()
    scratch = torch.zeros(SLOTS, NSPEC, V, dtype=torch.bfloat16, device="cuda")
    cols = torch.arange(NSPEC, dtype=torch.int32, device="cuda")
    for s in range(NSPEC):
        slot = idx[:, s].clamp(min=0)
        rows = cache[slot, s].clone()
        rows[idx[:, s] < 0] = 0  # inert rows: no cache row, not compared
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
        ok = idx[:, s] >= 0
        assert torch.equal(ids[ok], tokens[ok, s]), s


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
