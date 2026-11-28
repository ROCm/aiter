# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter.spec_verify_pick against the unfused chain it replaces.

Reference, on the same explicit noise:

    index_select(target / bonus rows); searchsorted(cu_draft) -> owner temps;
    mixed_sample_outer_exponential + topk_select(.., 1, tie="low") + where

Greedy rows must match exactly. Sampled rows run the same device function on
the same noise and must match too, bit for bit: no sampled row may differ from
the chain on these seeded inputs (measured 0 on MI355X over every dtype,
vocabulary, concurrency and pitch here).

Perf (-p): the fused op vs the chain an inference engine runs today, with the
noise drawn outside the timed region in both: the Triton chain
(index_select x2 + searchsorted + gather + fused_mixed_sample x2) and the HIP
chain above.
"""

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.triton.fusions.fused_mixed_sample import fused_mixed_sample
from aiter.test_common import benchmark

EPS = 1e-10


def _argmax_low(x):
    try:
        from aiter import topk_select

        return topk_select(x, 1, tie="low")[1].view(-1).to(torch.int32)
    except ImportError:
        # Same rule on float: NaN highest, -0.0 == 0.0, ties to the first column.
        return x.float().argmax(dim=-1).to(torch.int32)


def build(conc, max_draft, V, dtype, greedy_frac=0.3, pitch=0, seed=0, ragged=True):
    """A step of `conc` requests with draft counts in [0, max_draft] (all
    max_draft unless `ragged`).

    Request i owns d_i verification rows followed by its bonus row, as a
    verify step lays them out. Some requests are greedy (temperature clamped
    to EPS), the rest sampled.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    drafts = torch.randint(0, max_draft + 1, (conc,), generator=g, device="cuda")
    drafts[0] = max_draft
    if not ragged:
        drafts.fill_(max_draft)
    cu = drafts.cumsum(0)
    starts = cu - drafts + torch.arange(conc, device="cuda")
    bonus = (starts + drafts).to(torch.int64)
    owner = torch.repeat_interleave(torch.arange(conc, device="cuda"), drafts)
    first_t = torch.cumsum(drafts, 0) - drafts
    target = (
        starts[owner] + torch.arange(owner.numel(), device="cuda") - first_t[owner]
    ).to(torch.int64)
    rows = conc + int(drafts.sum())
    full = torch.randn(rows, V + pitch, device="cuda", generator=g) * 3.0
    logits = full.to(dtype)[:, :V]
    temps = torch.rand(conc, device="cuda", generator=g) + 0.3
    temps = torch.where(
        torch.rand(conc, device="cuda", generator=g) < greedy_frac,
        torch.zeros_like(temps),
        temps,
    ).clamp_min(EPS)
    expo_t = torch.empty(target.numel(), V, device="cuda").exponential_(1, generator=g)
    expo_b = torch.empty(conc, V, device="cuda").exponential_(1, generator=g)
    return logits, expo_t, expo_b, temps, target, bonus, cu.to(torch.int32)


def run_chain(logits, expo_t, expo_b, temps, target, bonus, cu):
    """The unfused HIP chain: the definition of the op."""
    out = []
    owner = torch.searchsorted(
        cu, torch.arange(target.numel(), device="cuda", dtype=cu.dtype), right=True
    )
    for rows, expo, t in ((target, expo_t, temps[owner]), (bonus, expo_b, temps)):
        x = torch.index_select(logits, 0, rows)
        o = torch.empty(rows.numel(), dtype=torch.int32, device="cuda")
        if rows.numel():
            aiter.mixed_sample_outer_exponential(o, x, expo, t, eps=EPS)
            o = torch.where(t <= EPS, _argmax_low(x), o)
        out.append(o)
    return out


def run_triton_chain(logits, expo_t, expo_b, temps, target, bonus, cu):
    """What ATOM's verify runs without the fused op (Sampler, independent noise)."""
    owner = torch.searchsorted(
        cu, torch.arange(target.numel(), device="cuda", dtype=cu.dtype), right=True
    )
    ot = fused_mixed_sample(
        torch.index_select(logits, 0, target),
        expo_t,
        temps[owner],
        eps=EPS,
        greedy_eps=EPS,
    )
    ob = fused_mixed_sample(
        torch.index_select(logits, 0, bonus), expo_b, temps, eps=EPS, greedy_eps=EPS
    )
    return ot, ob


def run_fused(logits, expo_t, expo_b, temps, target, bonus, cu):
    ot = torch.empty(target.numel(), dtype=torch.int32, device="cuda")
    ob = torch.empty(bonus.numel(), dtype=torch.int32, device="cuda")
    aiter.spec_verify_pick(
        ot,
        ob,
        logits,
        expo_t,
        expo_b,
        temps,
        target,
        bonus,
        cu,
        eps=EPS,
        greedy_eps=EPS,
    )
    return ot, ob


def _mismatch(args, got):
    ref_t, ref_b = run_chain(*args)
    temps, target, cu = args[3], args[4], args[6]
    owner = torch.searchsorted(
        cu, torch.arange(target.numel(), device="cuda", dtype=cu.dtype), right=True
    )
    greedy = torch.cat([temps[owner] <= EPS, temps <= EPS])
    diff = torch.cat([got[0] != ref_t, got[1] != ref_b])
    return int((diff & greedy).sum()), int((diff & ~greedy).sum()), int(diff.numel())


def check_accuracy(l_dtype, l_conc, l_v, max_draft):
    for dtype in l_dtype:
        for V in l_v:
            for conc in l_conc:
                for pitch in (0, 64):
                    args = build(conc, max_draft, V, dtype, pitch=pitch, seed=conc + V)
                    g_bad, s_bad, n = _mismatch(args, run_fused(*args))
                    tag = f"{dtype} {V=} {conc=} {pitch=}"
                    assert g_bad == 0, f"greedy mismatch {tag}: {g_bad}/{n}"
                    assert s_bad == 0, f"sampled mismatch {tag}: {s_bad}/{n}"
    aiter.logger.info("spec_verify_pick accuracy passed")


def check_edges():
    V = 4099
    # n_target == 0: bonus picks only.
    args = build(4, 0, V, dtypes.bf16, greedy_frac=0.5, seed=1)
    assert args[4].numel() == 0
    g_bad, s_bad, _ = _mismatch(args, run_fused(*args))
    assert g_bad == 0 and s_bad == 0, "n_target == 0"

    # Greedy key order: NaN above +inf, -0.0 == 0.0, ties to the smallest column,
    # an all -inf row picks 0.
    logits = torch.full((4, V), float("-inf"), device="cuda")
    logits[0, 7] = 0.0
    logits[0, 5] = -0.0
    logits[1, 4000] = float("inf")
    logits[1, 4098] = float("nan")
    logits[2, 3] = 1.0
    logits[2, 2] = 1.0
    temps = torch.full((4,), EPS, device="cuda")
    bonus = torch.arange(4, device="cuda", dtype=torch.int64)
    expo = torch.ones(4, V, device="cuda")
    target = torch.empty(0, dtype=torch.int64, device="cuda")
    cu = torch.zeros(4, dtype=torch.int32, device="cuda")
    for dtype in (dtypes.fp32, dtypes.bf16, dtypes.fp16):
        _, ob = run_fused(logits.to(dtype), expo[:0], expo, temps, target, bonus, cu)
        assert ob.tolist() == [5, 4098, 2, 0], f"{dtype} greedy keys: {ob.tolist()}"

    # Broadcast noise (stride 0) and the route's conversions: int32 index lists,
    # int64 prefix sums, a non-contiguous output -- same picks as the plain call.
    args = build(8, 3, V, dtypes.bf16, seed=2)
    logits, _, _, temps, target, bonus, cu = args
    row = torch.empty(1, V, device="cuda").exponential_(1)
    bcast = (logits, row.expand(target.numel(), V), row.expand(8, V)) + args[3:]
    want = run_fused(*bcast)
    assert _mismatch(bcast, want)[:2] == (0, 0), "broadcast noise"
    ot = torch.empty(target.numel(), 2, dtype=torch.int32, device="cuda")[:, 0]
    ob = torch.empty(8, dtype=torch.int64, device="cuda")
    aiter.spec_verify_pick(
        ot,
        ob,
        logits,
        bcast[1],
        bcast[2],
        temps,
        target.int(),
        bonus.int(),
        cu.long(),
        eps=EPS,
        greedy_eps=EPS,
    )
    assert torch.equal(ot, want[0]) and torch.equal(ob.int(), want[1]), "route casts"

    # Off the GPU: the torch definition; greedy picks are exact.
    cpu = [a.cpu() for a in args]
    ot = torch.empty(target.numel(), dtype=torch.int32)
    ob = torch.empty(8, dtype=torch.int32)
    aiter.spec_verify_pick(ot, ob, *cpu, eps=EPS, greedy_eps=EPS)
    gt, gb = run_fused(*args)
    owner = torch.searchsorted(
        cu, torch.arange(target.numel(), device="cuda", dtype=cu.dtype), right=True
    )
    gmask_t, gmask_b = (temps[owner] <= EPS).cpu(), (temps <= EPS).cpu()
    assert torch.equal(ot[gmask_t], gt.cpu()[gmask_t]), "cpu greedy target"
    assert torch.equal(ob[gmask_b], gb.cpu()[gmask_b]), "cpu greedy bonus"

    # Logits of a dtype the kernel is not built for: the torch definition too.
    ot = torch.empty(target.numel(), dtype=torch.int32, device="cuda")
    ob = torch.empty(8, dtype=torch.int32, device="cuda")
    aiter.spec_verify_pick(ot, ob, logits.double(), *args[1:], eps=EPS, greedy_eps=EPS)
    gmask_t, gmask_b = gmask_t.cuda(), gmask_b.cuda()
    assert torch.equal(ot[gmask_t], gt[gmask_t]), "fp64 greedy target"
    assert torch.equal(ob[gmask_b], gb[gmask_b]), "fp64 greedy bonus"
    aiter.logger.info("spec_verify_pick edge checks passed")


def check_temperatures():
    """T == 0 is greedy; T in (greedy_eps, 1e-5] samples at T = 1e-5 in the
    kernel (split or not) as in the torch definition; a malformed ``temps``
    is a ValueError, not an IndexError or a kernel check."""
    V = 4099
    for conc in (2, 200):  # split rows, whole rows
        args = build(conc, 3, V, dtypes.bf16, greedy_frac=0.0, seed=7 + conc)
        logits, expo_t, expo_b, _, target, bonus, cu = args
        picks = {}
        for t in (0.0, 1e-9, 1e-7, 1e-5):
            temps = torch.full((conc,), t, device="cuda")
            picks[t] = torch.cat(
                run_fused(logits, expo_t, expo_b, temps, target, bonus, cu)
            )
            if t:
                ot = torch.empty(target.numel(), dtype=torch.int32)
                ob = torch.empty(conc, dtype=torch.int32)
                cpu = [a.cpu() for a in (logits, expo_t, expo_b, temps)]
                aiter.spec_verify_pick(
                    ot,
                    ob,
                    *cpu,
                    target.cpu(),
                    bonus.cpu(),
                    cu.cpu(),
                    eps=EPS,
                    greedy_eps=EPS,
                )
                assert torch.equal(
                    torch.cat([ot, ob]), picks[t].cpu()
                ), f"T={t} kernel vs torch definition, {conc=}"
        rows = torch.cat(
            [logits.index_select(0, target), logits.index_select(0, bonus)]
        )
        # No NaN here, so torch's first-index argmax is the tie="low" rule.
        greedy = rows.float().argmax(dim=-1).to(torch.int32)
        assert torch.equal(picks[0.0], greedy), f"T=0 greedy, {conc=}"
        assert torch.equal(picks[1e-9], picks[1e-5]) and torch.equal(
            picks[1e-7], picks[1e-5]
        ), f"T below the 1e-5 clamp, {conc=}"

    logits, expo_t, expo_b, temps, target, bonus, cu = build(4, 2, V, dtypes.bf16)
    for bad in (
        (logits, expo_t, expo_b[:0], temps[:0], target, bonus[:0], cu[:0]),
        (logits, expo_t, expo_b, temps[0], target, bonus, cu),
    ):
        ot = torch.empty(bad[4].numel(), dtype=torch.int32, device="cuda")
        ob = torch.empty(bad[5].numel(), dtype=torch.int32, device="cuda")
        try:
            aiter.spec_verify_pick(ot, ob, *bad, eps=EPS, greedy_eps=EPS)
        except ValueError:
            continue
        raise AssertionError("malformed temps did not raise ValueError")
    aiter.logger.info("spec_verify_pick temperature checks passed")


def check_split_invariance():
    """Split rows (16 workgroups a row + a finishing pass) and whole rows (one
    workgroup a row) must pick bit for bit the same, sampled rows included: the
    split re-runs the one-workgroup race thread for thread. The public op picks
    between them by row count, so this is what makes a pick independent of the
    step's batch size."""
    from aiter.ops.sample import _SPEC_VERIFY_PICK_WS_PER_ROW, _spec_verify_pick_hip

    def pick(args, split):
        target, bonus = args[4], args[5]
        rows = target.numel() + bonus.numel()
        ot = torch.empty(target.numel(), dtype=torch.int32, device="cuda")
        ob = torch.empty(bonus.numel(), dtype=torch.int32, device="cuda")
        ws = torch.full(
            (rows * _SPEC_VERIFY_PICK_WS_PER_ROW,), -1, dtype=torch.int32, device="cuda"
        )
        _spec_verify_pick_hip(ot, ob, *args, ws, split, EPS, EPS)
        return torch.cat([ot, ob])

    n = 0
    for dtype in (dtypes.bf16, dtypes.fp16, dtypes.fp32):
        for V in (129280, 4096 * 9 + 3, 4097, 1000, 160):
            for conc in (1, 8, 64):
                for seed in range(3):
                    args = build(
                        conc, 5, V, dtype, pitch=64 * (seed % 2), seed=seed + V
                    )
                    if seed == 2:
                        # Ties and specials: many equal logits, NaN / +-inf.
                        logits = args[0]
                        logits[:, : V // 2] = logits[:, :1]
                        logits[0, V // 3] = float("nan")
                        logits[-1, V - 1] = float("inf")
                        logits[-1, 0] = float("-inf")
                    a, b = pick(args, False), pick(args, True)
                    assert torch.equal(a, b), f"split rows {dtype} {V=} {conc=} {seed=}"
                    n += a.numel()
    aiter.logger.info("spec_verify_pick split rows == whole rows on %d picks", n)


def check_graph(replays=20):
    V, conc = 129280, 16
    args = build(conc, 5, V, dtypes.bf16, seed=3, ragged=False)
    bufs = [a.clone() for a in args]
    ot = torch.empty(args[4].numel(), dtype=torch.int32, device="cuda")
    ob = torch.empty(conc, dtype=torch.int32, device="cuda")

    def call():
        aiter.spec_verify_pick(ot, ob, *bufs, eps=EPS, greedy_eps=EPS)

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        call()
        s.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            call()
    flips = 0
    for i in range(replays):
        new = build(conc, 5, V, dtypes.bf16, seed=100 + i, ragged=False)
        for buf, src in zip(bufs[:4], new[:4]):
            buf.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        g_bad, s_bad, _ = _mismatch(bufs, (ot, ob))
        assert g_bad == 0, f"graph replay {i}: greedy mismatch"
        flips += s_bad
    assert flips <= replays, f"graph replay: {flips} sampled flips"
    aiter.logger.info("spec_verify_pick graph replay passed (%d sampled flips)", flips)


def graph_us(func, args, calls=100, replays=20):
    """Median us per call of ``func(*args)`` replayed from one CUDA graph of
    ``calls`` calls. The chains are 2 to 15 launches each, which the
    profiler-based ``run_perftest`` cannot split into iterations reliably."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            func(*args)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        for _ in range(calls):
            func(*args)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    times = []
    for _ in range(replays):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / calls)
    return float(torch.tensor(times).median())


@benchmark()
def test_spec_verify_pick_perf(conc, V, dtype, draft=5):
    args = build(conc, draft, V, dtype, greedy_frac=0.0, seed=conc, ragged=False)
    us_triton = graph_us(run_triton_chain, args)
    us_hip = graph_us(run_chain, args)
    us_fused = graph_us(run_fused, args)
    return {
        "M": int(args[4].numel() + conc),
        "triton_chain_us": us_triton,
        "hip_chain_us": us_hip,
        "fused_us": us_fused,
        "speedup_vs_triton": us_triton / us_fused,
        "speedup_vs_hip": us_hip / us_fused,
    }


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter, description=__doc__
)
parser.add_argument(
    "-d", "--dtype", type=str, choices=["bf16", "fp16", "fp32"], nargs="*", default=None
)
parser.add_argument("-c", "--conc", type=int, nargs="*", default=[1, 8, 64, 128])
parser.add_argument("-n", "--vocab", type=int, nargs="*", default=[129280, 4097, 1000])
parser.add_argument("-p", "--perf", action="store_true", help="also time the chains")
args = parser.parse_args()
l_dtype = (
    [dtypes.d_dtypes[d] for d in args.dtype]
    if args.dtype
    else [dtypes.bf16, dtypes.fp16, dtypes.fp32]
)

check_accuracy(l_dtype, args.conc, args.vocab, max_draft=5)
check_edges()
check_temperatures()
check_split_invariance()
check_graph()
if args.perf:
    df = [
        test_spec_verify_pick_perf(conc, 129280, dtypes.bf16)
        for conc in (1, 8, 16, 32, 64, 128, 256)
    ]
    aiter.logger.info(
        "spec_verify_pick perf (median us per call, CUDA graph replay, noise excluded):\n%s",
        pd.DataFrame(df).to_markdown(index=False),
    )
