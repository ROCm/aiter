# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL standard rejection sampling (5 draft tokens, vocab 129280).

Reference: the same rule in fp64 (``rejection_ref`` below; Triton's Philox
``tl.rand`` for the acceptance draw and vLLM's murmur3 Gumbel draw for the
resample). The kernel's fp32 sums and hardware exp / log differ in the last
bits, so tokens are compared wherever the reference's smallest decision margin
of the request exceeds 1e-3. Inputs: LM-like target rows (fp32 with the
temperature applied, or raw bf16), draft rows of three noise levels (accepted
counts 0..5), temperatures 0 / 0.7 / 1 / 1.3 mixed over random request states,
placeholder drafts, masked columns, synthetic acceptance. Also: requests do not
depend on the batch, repeat runs are bitwise equal and leave the tickets at
zero, graph replay with new inputs, degenerate rows return in-range ids.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("flydsl")

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import rejection_sample as rs
from aiter.ops.flydsl.rejection_sample import (
    flydsl_rejection_sample_standard,
    rejection_sample_standard_supported,
    warmup_rejection_sample_standard,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="gfx950 and FlyDSL are required",
)

V = 129280
NSPEC = 5
ROWS = NSPEC + 1
U_MIN = 4.6566127342e-10
MARGIN = 1e-3
_M32 = 0xFFFFFFFF
_M64 = (1 << 64) - 1


def philox_uniform(seed: int, pos: int) -> float:
    """Triton's ``tl.rand(seed, pos)`` (Philox 4x32-10), clamped to its smallest
    positive output."""
    k0, k1 = seed & _M32, (seed >> 32) & _M32
    c0, c1, c2, c3 = pos & _M32, (pos >> 32) & _M32, 0, 0
    for _ in range(10):
        a, b = c0, c2
        c0 = (((0xCD9E8D57 * b) >> 32) ^ c1 ^ k0) & _M32
        c2 = (((0xD2511F53 * a) >> 32) ^ c3 ^ k1) & _M32
        c1 = (0xCD9E8D57 * b) & _M32
        c3 = (0xD2511F53 * a) & _M32
        k0 = (k0 + 0x9E3779B9) & _M32
        k1 = (k1 + 0xBB67AE85) & _M32
    x = c0 - (1 << 32) if c0 >= (1 << 31) else c0
    x = -x - 1 if x < 0 else x
    return max(float(np.float32(x) * np.float32(U_MIN)), U_MIN)


def _rotl(x, s):
    return ((x << s) | (x >> (32 - s))) & _M32


def _mix(h, key):
    key = (key * 0xCC9E2D51) & _M32
    key = _rotl(key, 15)
    key = (key * 0x1B873593) & _M32
    return (_rotl(h ^ key, 13) * 5 + 0xE6546B64) & _M32


def _log1p_neg_stable(v):
    p = torch.full_like(v, 1.0 / 8.0)
    for k in (7, 6, 5, 4, 3, 2, 1):
        p = 1.0 / k + v * p
    return torch.where(
        v < 0.25, -v * p, torch.log(torch.clamp(1.0 - v, min=5.960464477539063e-08))
    )


def gumbel_noise(seed: int, pos: int, cols: torch.Tensor) -> torch.Tensor:
    """fp64 ``-log(-log1p_neg_stable(u))`` of vLLM's fp32 murmur3 uniform of
    (seed, pos, vocab id)."""
    h = _mix(_mix(_mix(0, seed & _M32), (seed >> 32) & _M32), pos & _M32)
    c = cols.to(torch.int64)
    key = (c * 0xCC9E2D51) & _M32
    key = ((key << 15) | (key >> 17)) & _M32
    key = (key * 0x1B873593) & _M32
    x = torch.full_like(c, h) ^ key
    x = ((x << 13) | (x >> 19)) & _M32
    x = ((x * 5 + 0xE6546B64) & _M32) ^ 16
    x = x ^ (x >> 16)
    x = (x * 0x85EBCA6B) & _M32
    x = x ^ (x >> 13)
    x = (x * 0xC2B2AE35) & _M32
    r = x ^ (x >> 16)
    u = (
        ((r >> 16).double() * 2.0**-16 + ((r & 0xFFFF).double() + 0.5) * 2.0**-32)
        .float()
        .double()
    )
    return -torch.log(-_log1p_neg_stable(u))


def _argmax_lowest(x):
    return int((x == x.max()).nonzero()[0])


def _top_gap(x):
    t = x.topk(2).values
    return float(t[0] - t[1])


def _target_row(target_logits, row, temp):
    x = target_logits[row]
    if x.dtype == torch.bfloat16 and temp != 0.0:
        x = x.float() / torch.tensor(temp, dtype=torch.float32, device=x.device)
    return x.double()


def rejection_ref(
    target_logits,
    draft_logits,
    draft_sampled,
    pos,
    idx_mapping,
    temperature,
    seed,
    synthetic_rates=None,
):
    """(sampled int64 [B, 6], num_sampled int32 [B], margin fp32 [B])."""
    b = idx_mapping.shape[0]
    dev = target_logits.device
    sampled = torch.full((b, ROWS), -1, dtype=torch.int64)
    num_sampled = torch.zeros(b, dtype=torch.int32)
    margin = torch.full((b,), float("inf"), dtype=torch.float32)
    ds = draft_sampled.cpu().tolist()
    ps = pos.cpu().tolist()
    cols = torch.arange(V, device=dev)
    for r in range(b):
        slot = int(idx_mapping[r])
        temp = float(temperature[slot])
        sd = int(seed[slot]) & _M64
        base = r * ROWS
        acc = 0
        mg = float("inf")
        for j in range(NSPEC):
            tok = ds[base + j + 1]
            u = philox_uniform(sd, ps[base + j])
            t = _target_row(target_logits, base + j, temp)
            ok = tok >= 0
            if synthetic_rates is not None:
                accept = ok and u < float(synthetic_rates[j])
                mg = min(mg, abs(u - float(synthetic_rates[j])))
            elif temp == 0.0:
                accept = ok and tok == _argmax_lowest(t)
            elif ok:
                lp_t = t - torch.logsumexp(t, 0)
                d = draft_logits[slot, j].double() / float(np.float32(temp))
                lp_d = d - torch.logsumexp(d, 0)
                lhs = float(lp_t[tok])
                rhs = float(np.log(np.float32(u))) + float(lp_d[tok])
                accept = lhs > rhs
                mg = min(mg, abs(lhs - rhs))
            else:
                accept = False
            if not accept:
                break
            acc += 1
        t = _target_row(target_logits, base + acc, temp)
        bonus = acc == NSPEC
        tok = -1 if bonus else ds[base + acc + 1]
        if temp == 0.0:
            token = _argmax_lowest(t)
        else:
            if bonus or tok < 0:
                res = t
            else:
                lp_t = t - torch.logsumexp(t, 0)
                d = draft_logits[slot, acc].double() / float(np.float32(temp))
                lp_d = d - torch.logsumexp(d, 0)
                ratio = torch.exp(lp_d - lp_t)
                res = torch.where(
                    ratio < 1.0,
                    lp_t + _log1p_neg_stable(ratio),
                    torch.full_like(t, float("-inf")),
                )
            val = res + gumbel_noise(sd, ps[base + acc], cols)
            token = _argmax_lowest(val)
            if torch.isfinite(val.max()):
                mg = min(mg, _top_gap(val))
        sampled[r, :acc] = torch.tensor(ds[base + 1 : base + 1 + acc])
        sampled[r, acc] = token
        num_sampled[r] = acc + 1
        margin[r] = mg
    return sampled, num_sampled, margin


def make(
    b,
    gen,
    temps=(0.0, 0.7, 1.0, 1.3),
    noise=1.0,
    placeholders=0.0,
    extra_slots=3,
    bf16_target=False,
):
    """``b`` requests at random request states; draft tokens drawn from the
    draft rows. ``bf16_target``: raw bf16 target rows."""
    r = b + extra_slots
    base = torch.randn(b * ROWS, V, generator=gen) * 2
    base.scatter_(
        1,
        torch.randint(0, V, (b * ROWS, 16), generator=gen),
        torch.rand(b * ROWS, 16, generator=gen) * 10 + 6,
    )
    idx = torch.randperm(r, generator=gen)[:b].to(torch.int32)
    temp = torch.tensor(temps)[torch.randint(0, len(temps), (r,), generator=gen)]
    temp = temp.float()
    scale = temp[idx.long()].repeat_interleave(ROWS).unsqueeze(1)
    if bf16_target:
        target = base.bfloat16()
    else:
        target = torch.where(scale > 0, base / scale.clamp_min(1e-6), base)
    draft = torch.randn(r, NSPEC, V, generator=gen).to(torch.bfloat16)
    tokens = torch.full((b * ROWS,), -1, dtype=torch.int64)
    for i in range(b):
        s = int(idx[i])
        for j in range(NSPEC):
            d = (base[i * ROWS + j] + torch.randn(V, generator=gen) * noise).bfloat16()
            draft[s, j] = d
            t = float(temp[s])
            if t == 0:
                tok = int(d.float().argmax())
            else:
                probs = torch.softmax(d.float() / t, 0)
                tok = int(torch.multinomial(probs, 1, generator=gen))
            tokens[i * ROWS + j + 1] = tok
    if placeholders:
        tokens[torch.rand(b * ROWS, generator=gen) < placeholders] = -1
    pos = torch.arange(b * ROWS, dtype=torch.int64)
    pos += int(torch.randint(0, 2**20, (1,), generator=gen))
    seed = torch.randint(-(2**62), 2**62, (r,), generator=gen)
    return tuple(x.cuda() for x in (target, draft, tokens, pos, idx, temp, seed))


def check(args, rates=None):
    sampled, num = flydsl_rejection_sample_standard(*args, rates)
    ref_s, ref_n, margin = rejection_ref(*args, rates)
    torch.cuda.synchronize()
    sampled, num = sampled.cpu(), num.cpu()
    compared = 0
    for i in range(sampled.shape[0]):
        if margin[i] < MARGIN:
            continue
        compared += 1
        assert int(num[i]) == int(ref_n[i]) and torch.equal(sampled[i], ref_s[i]), (
            i,
            sampled[i].tolist(),
            ref_s[i].tolist(),
            float(margin[i]),
        )
    assert compared >= sampled.shape[0] // 2
    return num


@pytest.mark.parametrize("bf16_target", [False, True])
@pytest.mark.parametrize("b", [1, 3, 8])
@pytest.mark.parametrize("noise", [0.3, 1.0, 3.0])
def test_matches_reference(b, noise, bf16_target):
    gen = torch.Generator().manual_seed(b * 31 + int(noise * 10))
    args = make(b, gen, noise=noise, placeholders=0.05, bf16_target=bf16_target)
    num = check(args)
    assert num.min() >= 1 and num.max() <= ROWS


def test_accepted_counts_cover_the_range():
    seen = set()
    for noise, seed in ((0.2, 1), (1.0, 2), (4.0, 3)):
        gen = torch.Generator().manual_seed(seed)
        seen |= set(check(make(12, gen, noise=noise, temps=(0.0, 1.0))).tolist())
    assert {1, 2, ROWS} <= seen, seen


@pytest.mark.parametrize("bf16_target", [False, True])
def test_served_batch_sizes(bf16_target):
    for b in (32, 64):
        gen = torch.Generator().manual_seed(b)
        check(make(b, gen, noise=1.0, bf16_target=bf16_target))


@pytest.mark.parametrize("bf16_target", [False, True])
@pytest.mark.parametrize(
    "rates",
    [[1.0] * 5, [0.0] * 5, [0.9, 0.8, 0.6, 0.4, 0.2], [0.5] * 5],
)
def test_synthetic_acceptance(rates, bf16_target):
    gen = torch.Generator().manual_seed(5)
    num = check(make(8, gen, bf16_target=bf16_target), torch.tensor(rates).cuda())
    if rates[0] == 0.0:
        assert bool((num == 1).all())
    if rates[-1] == 1.0:
        assert bool((num == ROWS).all())


@pytest.mark.parametrize("bf16_target", [False, True])
def test_masked_columns_and_ties(bf16_target):
    gen = torch.Generator().manual_seed(6)
    args = list(make(6, gen, temps=(0.0, 1.0), bf16_target=bf16_target))
    target = args[0]
    target[:, ::2] = float("-inf")  # top-k style mask: odd columns survive
    target[3] = float("-inf")
    target[3, 777] = 1.0
    target[4] = 0.0  # a flat row: greedy picks id 0
    for i in range(6 * ROWS):  # drafts on surviving columns
        if args[2][i] >= 0:
            args[2][i] |= 1
    check(tuple(args))


@pytest.mark.parametrize("bf16_target", [False, True])
def test_degenerate_rows_return_in_range_ids(bf16_target):
    """All-NaN and all -inf rows: every id is in [0, V) or the -1 pad."""
    gen = torch.Generator().manual_seed(8)
    args = list(make(8, gen, temps=(0.0, 1.0), bf16_target=bf16_target))
    target, draft = args[0], args[1]
    target[0:ROWS] = float("nan")
    target[ROWS : 2 * ROWS] = float("-inf")
    target[2 * ROWS + 1] = float("nan")
    draft[args[4][3].long()] = float("nan")
    draft[args[4][4].long()] = float("-inf")
    sampled, num = flydsl_rejection_sample_standard(*args, None)
    torch.cuda.synchronize()
    assert int(sampled.min()) >= -1 and int(sampled.max()) < V
    assert int(num.min()) >= 1 and int(num.max()) <= ROWS
    for i in range(sampled.shape[0]):
        n = int(num[i])
        assert bool((sampled[i, :n] >= 0).all())
        assert bool((sampled[i, n:] == -1).all())


def test_request_independent_of_batch():
    gen = torch.Generator().manual_seed(7)
    args = make(16, gen, noise=0.6)
    full_s, full_n = flydsl_rejection_sample_standard(*args)
    target, draft, tokens, pos, idx, temp, seed = args
    for i in (0, 7, 15):
        sl = slice(i * ROWS, (i + 1) * ROWS)
        one = (
            target[sl].contiguous(),
            draft,
            tokens[sl].contiguous(),
            pos[sl].contiguous(),
            idx[i : i + 1].contiguous(),
            temp,
            seed,
        )
        s, n = flydsl_rejection_sample_standard(*one)
        assert torch.equal(s[0], full_s[i]) and int(n[0]) == int(full_n[i]), i


def test_repeat_bitwise_and_tickets_reset():
    gen = torch.Generator().manual_seed(8)
    args = make(16, gen, noise=0.6)
    a = flydsl_rejection_sample_standard(*args)
    b = flydsl_rejection_sample_standard(*args)
    torch.cuda.synchronize()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    from aiter.ops.flydsl.kernels import rejection_sample_standard as k

    ws = rs._workspace(args[0].device.index)
    assert not bool(ws[k.STATS_BYTES + k.PART_BYTES :].any())


def test_graph_replay_new_inputs():
    gen = torch.Generator().manual_seed(9)
    bufs = make(8, gen, noise=0.6)
    warmup_rejection_sample_standard(bufs[0].device)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = flydsl_rejection_sample_standard(*bufs)
    for noise in (0.3, 2.0):
        new = make(8, gen, noise=noise)
        for buf, x in zip(bufs, new):
            buf.copy_(x)
        out[0].fill_(-7)
        out[1].fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        want = flydsl_rejection_sample_standard(*new)
        assert torch.equal(out[0], want[0]) and torch.equal(out[1], want[1])


def test_unsupported_steps_decline():
    t = torch.zeros((ROWS, V), dtype=torch.bfloat16, device="cuda")
    d = torch.zeros((1, NSPEC, V), dtype=torch.bfloat16, device="cuda")
    assert rejection_sample_standard_supported(t, d, 1, NSPEC)
    assert not rejection_sample_standard_supported(t, d, 1, 3)
    assert not rejection_sample_standard_supported(t, d, 2, NSPEC)
    assert not rejection_sample_standard_supported(t.half(), d, 1, NSPEC)
    assert not rejection_sample_standard_supported(t, d.float(), 1, NSPEC)
    t2 = torch.zeros((ROWS, V + 64), dtype=torch.bfloat16, device="cuda")[:, :V]
    assert not rejection_sample_standard_supported(t2, d, 1, NSPEC)


@pytest.mark.parametrize("bf16_target", [False, True])
def test_matches_vllm_triton_rejection_sample(bf16_target):
    """Same tokens as vLLM's Triton ``rejection_sample`` (standard method,
    replicated draft cache) where the reference margin exceeds 1e-3; for bf16
    rows vLLM first converts to fp32 and applies the temperature. Optional:
    needs vLLM."""
    pytest.importorskip("vllm")
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    class Cache:
        row_mapping = None

        def __init__(self, values):
            self.values = values

    gen = torch.Generator().manual_seed(21 + bf16_target)
    b = 48
    args = make(b, gen, noise=0.8, placeholders=0.03, bf16_target=bf16_target)
    target, draft, tokens, pos, idx, temp, seed = args
    cu = (torch.arange(b + 1, device="cuda") * ROWS).int()
    local_pos = torch.arange(ROWS, device="cuda").repeat(b).int()
    eidx = idx.repeat_interleave(ROWS)
    tgt = target
    if bf16_target:
        tgt = target.float()
        apply_temperature(tgt, eidx, temp)
    v_s, v_n = rejection_sample(
        tgt, Cache(draft), tokens, cu, pos, idx, eidx, local_pos, temp, seed, NSPEC
    )
    f_s, f_n = flydsl_rejection_sample_standard(*args)
    _, _, margin = rejection_ref(*args)
    torch.cuda.synchronize()
    agree = 0
    for i in range(b):
        if margin[i] < MARGIN:
            continue
        n = int(v_n[i])
        assert int(f_n[i]) == n, i
        assert torch.equal(f_s[i, :n].cpu(), v_s[i, :n].cpu()), i
        agree += 1
    assert agree >= b // 2
