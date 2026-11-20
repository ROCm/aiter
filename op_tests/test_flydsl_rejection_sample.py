# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL standard rejection sampling: 5 draft tokens, vocab 129280 (gfx950).

    python3 op_tests/test_flydsl_rejection_sample.py
    python3 op_tests/test_flydsl_rejection_sample.py -b 8 64 -d bf16 -m standard

Reference: the same rule in fp64 (``run_torch``): Triton's Philox ``tl.rand``
for the acceptance draw and vLLM's murmur3 Gumbel draw for the resample,
reimplemented here. The kernel's fp32 sums and hardware exp / log differ in the
last bits, so tokens are compared for every request whose smallest decision
margin in the reference exceeds ``MARGIN`` (1e-4), and at least half of the
requests must be compared. When vLLM is installed, its Triton
``rejection_sample`` (the path the op replaces; for raw bf16 rows preceded by
the fp32 copy and ``apply_temperature``) is a second candidate, checked against
the same reference.

Inputs: LM-like target rows (fp32 with the temperature applied, or raw bf16),
draft rows at three noise levels (accepted counts 0..5), temperatures 0 / 0.7 /
1 / 1.3 mixed over random request states, placeholder drafts, and the synthetic
acceptance mode. ``main()`` also runs the edge cases: masked columns and ties,
NaN / -inf rows, batch independence, bitwise repeatability with the ticket
counters back at zero, CUDA-graph replay with new inputs, the capture guard,
declined shapes and the package's lazy exports.
"""

import argparse
import itertools

import numpy as np
import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx950"]

V = 129280
NSPEC = 5
ROWS = NSPEC + 1
U_MIN = 4.6566127342e-10
MARGIN = 1e-4
RATES = (0.9, 0.8, 0.6, 0.4, 0.2)
_M32 = 0xFFFFFFFF
_M64 = (1 << 64) - 1


# ---- fp64 reference ------------------------------------------------------


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


def run_torch(
    target_logits,
    draft_logits,
    draft_sampled,
    pos,
    idx_mapping,
    temperature,
    seed,
    synthetic_rates=None,
):
    """fp64 reference: (sampled int64 [B, 6], num_sampled int32 [B], smallest
    decision margin of each request fp32 [B])."""
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
                # exact: the kernel compares the same fp32 values
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
            token = _argmax_lowest(t)  # exact, ties to the lowest id
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


# ---- inputs ---------------------------------------------------------------


def make_inputs(
    b,
    gen,
    temps=(0.0, 0.7, 1.0, 1.3),
    noise=1.0,
    placeholders=0.0,
    extra_slots=3,
    bf16_target=False,
):
    """``b`` requests at random request states of a larger draft cache; the
    draft tokens are sampled from the draft rows. ``bf16_target``: raw bf16
    lm_head rows (the kernel applies the temperature), else fp32 rows with the
    temperature applied."""
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


def _padded(sampled, num):
    """Tokens past ``num`` set to -1 (vLLM leaves them unspecified)."""
    keep = torch.arange(ROWS, device=sampled.device) < num.to(sampled.device)[:, None]
    return torch.where(keep, sampled, torch.full_like(sampled, -1))


def _vllm_rejection_sample(bf16_target):
    """vLLM's path for this step (needs vLLM; not an aiter dependency): for raw
    bf16 rows the fp32 copy and ``apply_temperature``, then the Triton
    ``rejection_sample`` with a replicated draft cache. None without vLLM."""
    try:
        from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
            rejection_sample,
        )
    except ImportError:
        return None

    class DraftCache:
        row_mapping = None

        def __init__(self, values):
            self.values = values

    def run(target, draft, tokens, pos, idx, temp, seed, rates=None):
        b = idx.shape[0]
        cu = (torch.arange(b + 1, device=target.device) * ROWS).int()
        local_pos = torch.arange(ROWS, device=target.device).repeat(b).int()
        eidx = idx.repeat_interleave(ROWS)
        if bf16_target:
            target = target.float()
            apply_temperature(target, eidx, temp)
        return rejection_sample(
            target,
            DraftCache(draft),
            tokens,
            cu,
            pos,
            idx,
            eidx,
            local_pos,
            temp,
            seed,
            NSPEC,
            synthetic_conditional_rates=rates,
        )

    return run


# ---- correctness and performance sweep ---------------------------------------


@benchmark()
def test_rejection_sample(reqs, dtype, mode, noise):
    from aiter.ops.flydsl.rejection_sample import flydsl_rejection_sample_standard

    bf16 = dtype == dtypes.bf16
    gen = torch.Generator().manual_seed(reqs * 31 + int(noise * 10) + 7 * bf16)
    args = make_inputs(reqs, gen, noise=noise, placeholders=0.05, bf16_target=bf16)
    rates = (
        torch.tensor(RATES, dtype=torch.float32, device="cuda")
        if mode == "synthetic"
        else None
    )
    ref_s, ref_n, margin = run_torch(*args, rates)
    compared = margin >= MARGIN
    assert int(compared.sum()) >= max(1, reqs // 2), "too few requests compared"

    candidates = {"flydsl": flydsl_rejection_sample_standard}
    vllm = _vllm_rejection_sample(bf16)
    if vllm is not None:  # optional: vLLM is not an aiter dependency
        candidates["vllm"] = vllm

    tbytes = args[0].element_size()
    # One pass over the 5 verified target and draft rows, plus the resampled
    # row and its draft row.
    nbytes = reqs * (ROWS * V * tbytes + ROWS * V * 2)
    # Per logit read: scale, exp and accumulate.
    flops = 3 * reqs * ROWS * V * 2

    ret = {"gfx": get_gfx_runtime(), "compared": int(compared.sum())}
    for name, fn in candidates.items():
        (out_s, out_n), us = run_perftest(fn, *args, rates)
        out_s = _padded(out_s, out_n).cpu()
        err = checkAllclose(
            ref_s.to(dtypes.fp32),
            out_s.to(dtypes.fp32),
            rtol=0,
            atol=0,
            mask=compared[:, None],
            msg=f"{name}: sampled tokens",
        )
        err_n = checkAllclose(
            ref_n.to(dtypes.fp32),
            out_n.cpu().to(dtypes.fp32),
            rtol=0,
            atol=0,
            mask=compared,
            msg=f"{name}: num_sampled",
        )
        assert err == 0 and err_n == 0, f"{name}: tokens differ from the reference"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# ---- edge cases ---------------------------------------------------------------


def _check_tokens(args, rates=None):
    from aiter.ops.flydsl.rejection_sample import flydsl_rejection_sample_standard

    sampled, num = flydsl_rejection_sample_standard(*args, rates)
    ref_s, ref_n, margin = run_torch(*args, rates)
    compared = margin >= MARGIN
    assert int(compared.sum()) >= sampled.shape[0] // 2
    err = checkAllclose(
        ref_s.to(dtypes.fp32),
        _padded(sampled, num).cpu().to(dtypes.fp32),
        rtol=0,
        atol=0,
        mask=compared[:, None],
        msg="sampled tokens",
        printLog=False,
    )
    assert err == 0 and torch.equal(num.cpu()[compared], ref_n[compared])
    return num.cpu()


def check_accepted_counts_and_synthetic():
    seen = set()
    for noise, seed in ((0.2, 1), (1.0, 2), (4.0, 3)):
        gen = torch.Generator().manual_seed(seed)
        args = make_inputs(12, gen, noise=noise, temps=(0.0, 1.0))
        seen |= set(_check_tokens(args).tolist())
    assert {1, 2, ROWS} <= seen, seen
    for bf16 in (False, True):
        gen = torch.Generator().manual_seed(5)
        args = make_inputs(8, gen, bf16_target=bf16)
        for value, want in ((0.0, 1), (1.0, ROWS)):
            rates = torch.full((NSPEC,), value, device="cuda")
            assert bool((_check_tokens(args, rates) == want).all())


def check_masked_columns_and_ties():
    for bf16 in (False, True):
        gen = torch.Generator().manual_seed(6)
        args = list(make_inputs(6, gen, temps=(0.0, 1.0), bf16_target=bf16))
        target = args[0]
        target[:, ::2] = float("-inf")  # top-k style mask: odd columns survive
        target[3] = float("-inf")
        target[3, 777] = 1.0
        target[4] = 0.0  # a flat row: greedy picks id 0
        for i in range(6 * ROWS):  # drafts on surviving columns
            if args[2][i] >= 0:
                args[2][i] |= 1
        _check_tokens(tuple(args))


def check_degenerate_rows():
    """All-NaN and all -inf rows: every id is in [0, V) or the -1 pad."""
    from aiter.ops.flydsl.rejection_sample import flydsl_rejection_sample_standard

    for bf16 in (False, True):
        gen = torch.Generator().manual_seed(8)
        args = list(make_inputs(8, gen, temps=(0.0, 1.0), bf16_target=bf16))
        target, draft = args[0], args[1]
        target[0:ROWS] = float("nan")
        target[ROWS : 2 * ROWS] = float("-inf")
        target[2 * ROWS + 1] = float("nan")
        draft[args[4][3].long()] = float("nan")
        draft[args[4][4].long()] = float("-inf")
        sampled, num = flydsl_rejection_sample_standard(*args)
        sampled, num = sampled.cpu(), num.cpu()
        assert int(sampled.min()) >= -1 and int(sampled.max()) < V
        assert int(num.min()) >= 1 and int(num.max()) <= ROWS
        for i in range(sampled.shape[0]):
            n = int(num[i])
            assert bool((sampled[i, :n] >= 0).all())
            assert bool((sampled[i, n:] == -1).all())


def check_batch_independence_and_repeat():
    from aiter.ops.flydsl import rejection_sample as rs
    from aiter.ops.flydsl.kernels import rejection_sample_standard as k

    gen = torch.Generator().manual_seed(7)
    args = make_inputs(16, gen, noise=0.6)
    full_s, full_n = rs.flydsl_rejection_sample_standard(*args)
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
        s, n = rs.flydsl_rejection_sample_standard(*one)
        assert torch.equal(s[0], full_s[i]) and int(n[0]) == int(full_n[i]), i
    again = rs.flydsl_rejection_sample_standard(*args)
    assert torch.equal(again[0], full_s) and torch.equal(again[1], full_n)
    ws = rs._buffers[target.device.index][0]
    assert not bool(ws[k.STATS_BYTES + k.PART_BYTES :].any()), "tickets not reset"


def check_graph_replay_and_capture_guard():
    from aiter.ops.flydsl import rejection_sample as rs

    gen = torch.Generator().manual_seed(9)
    bufs = make_inputs(8, gen, noise=0.6)
    dev = bufs[0].device
    # Without warmup the workspace would be allocated inside the capture.
    saved = rs._buffers.pop(dev.index, None)
    raised = False
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            rs.flydsl_rejection_sample_standard(*bufs)
    except RuntimeError as e:
        raised = "warmup_rejection_sample_standard" in str(e)
    if saved is not None:
        rs._buffers[dev.index] = saved
    assert raised, "capture without warmup did not raise"

    rs.warmup_rejection_sample_standard(dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = rs.flydsl_rejection_sample_standard(*bufs)
    for noise in (0.3, 2.0):
        new = make_inputs(8, gen, noise=noise)
        for buf, x in zip(bufs, new):
            buf.copy_(x)
        out[0].fill_(-7)
        out[1].fill_(-7)
        g.replay()
        want = rs.flydsl_rejection_sample_standard(*new)
        assert torch.equal(out[0], want[0]) and torch.equal(out[1], want[1])


def check_declined_shapes():
    from aiter.ops.flydsl.rejection_sample import rejection_sample_standard_supported

    t = torch.zeros((ROWS, V), dtype=torch.bfloat16, device="cuda")
    d = torch.zeros((1, NSPEC, V), dtype=torch.bfloat16, device="cuda")
    assert rejection_sample_standard_supported(t, d, 1, NSPEC)
    assert not rejection_sample_standard_supported(t, d, 1, 3)
    assert not rejection_sample_standard_supported(t, d, 2, NSPEC)
    assert not rejection_sample_standard_supported(t.half(), d, 1, NSPEC)
    assert not rejection_sample_standard_supported(t, d.float(), 1, NSPEC)
    t2 = torch.zeros((ROWS, V + 64), dtype=torch.bfloat16, device="cuda")[:, :V]
    assert not rejection_sample_standard_supported(t2, d, 1, NSPEC)


def check_lazy_exports():
    """Every lazy export of ``aiter.ops.flydsl`` resolves (each table entry is a
    (module, attribute) pair), and this op's names are in ``__all__`` (host
    only: no kernel runs)."""
    import aiter.ops.flydsl as fly

    for name, entry in fly._LAZY_IMPORTS.items():
        assert len(entry) == 2, f"{name}: {entry}"
        getattr(fly, name)
    for name in (
        "flydsl_rejection_sample_standard",
        "rejection_sample_standard_supported",
        "warmup_rejection_sample_standard",
    ):
        assert name in fly._LAZY_IMPORTS and name in fly.__all__, name
        assert callable(getattr(fly, name)), name


def main():
    check_lazy_exports()
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl rejection sampler unsupported on %s; skipping", get_gfx_runtime()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL standard rejection sampling (5 drafts, vocab 129280)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.fp32, dtypes.bf16],
        help="target logits dtype: fp32 (temperature applied) or bf16 (raw)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 8, 64],
        help="requests (6 logits rows each)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        nargs="*",
        default=["standard", "synthetic"],
        choices=["standard", "synthetic"],
        help="probability-ratio test or synthetic acceptance rates",
    )
    parser.add_argument(
        "--noise",
        type=float,
        nargs="*",
        default=[0.3, 1.0, 3.0],
        help="draft-row noise (spreads the accepted counts)",
    )
    args = parser.parse_args()

    check_declined_shapes()
    check_accepted_counts_and_synthetic()
    check_masked_columns_and_ties()
    check_degenerate_rows()
    check_batch_independence_and_repeat()
    check_graph_replay_and_capture_guard()

    df = [
        test_rejection_sample(reqs, dtype, mode, noise)
        for dtype, mode, reqs, noise in itertools.product(
            args.dtype, args.mode, args.batch, args.noise
        )
    ]
    df = pd.DataFrame(df)
    aiter.logger.info(
        "flydsl rejection sampler summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
