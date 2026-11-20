# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CUDA-graph time of ``flydsl_rejection_sample_standard`` against the path it
replaces, vLLM's Triton ``rejection_sample`` (gfx950, one GPU; needs vLLM).

    python3 op_tests/op_benchmarks/flydsl/bench_flydsl_rejection_sample.py
    python3 .../bench_flydsl_rejection_sample.py --reqs 8 32 --target bf16

Both the probability-ratio test and the synthetic acceptance mode (``--mode``)
are timed. ``--target bf16``: raw bf16 lm_head rows go straight in; the
baseline also runs the fp32 copy and ``apply_temperature`` vLLM needs first.
``--target fp32``: processed fp32 rows for both.

Each graph replays a call on up to 100 distinct inputs (a working set larger
than the last-level cache, as in a model step); arms run in A B B A order, 7
rounds of 5 replays each, and the table shows the best and the median round in
us per call. Another process sharing the GPU shows up as noise in the median.
"""

import argparse
import statistics

import torch

from aiter.ops.flydsl.rejection_sample import (
    flydsl_rejection_sample_standard,
    warmup_rejection_sample_standard,
)

V = 129280


def graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    return g


def rounds(g, calls, n=7, reps=5):
    ts = []
    for _ in range(n):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / (reps * calls))
    return ts


def abba(graphs, calls):
    names = list(graphs)
    res = {k: [] for k in names}
    for k in names + names[::-1]:
        res[k] += rounds(graphs[k], calls)
    return {k: (min(v), statistics.median(v)) for k, v in res.items()}


def report(title, key, rows):
    cols = list(rows[0][1])
    print(f"\n{title}")
    print(f"| {key} | " + " | ".join(f"{c} best / median us" for c in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for k, t in rows:
        cells = " | ".join(f"{t[c][0]:.2f} / {t[c][1]:.2f}" for c in cols)
        print(f"| {k} | {cells} |", flush=True)


class _DraftCache:
    row_mapping = None

    def __init__(self, values):
        self.values = values


def _rejection_input(b, gen, bf16_target):
    base = torch.randn(b * 6, V, generator=gen, device="cuda") * 2
    base.scatter_(
        1,
        torch.randint(0, V, (b * 6, 16), generator=gen, device="cuda"),
        torch.rand(b * 6, 16, generator=gen, device="cuda") * 10 + 6,
    )
    temp = torch.tensor([0.0, 0.7, 1.0], device="cuda")[
        torch.arange(b, device="cuda") % 3
    ]
    scale = temp.repeat_interleave(6).unsqueeze(1)
    if bf16_target:
        tgt = base.bfloat16()
    else:
        tgt = torch.where(scale > 0, base / scale.clamp_min(1e-6), base)
    rows = base.view(b, 6, V)[:, :5]
    draft = (rows + torch.randn(rows.shape, generator=gen, device="cuda")).bfloat16()
    probs = torch.softmax(draft.float() / temp.clamp_min(0.5).view(b, 1, 1), dim=-1)
    tok = torch.multinomial(probs.view(b * 5, V), 1, generator=gen).view(b, 5)
    tok = torch.where(temp.view(b, 1) == 0, draft.argmax(-1), tok)
    ds = torch.full((b, 6), -1, dtype=torch.int64, device="cuda")
    ds[:, 1:] = tok
    pos = torch.arange(b * 6, device="cuda", dtype=torch.int64) + 1000
    seed = torch.randint(0, 2**40, (b,), generator=gen, device="cuda")
    idx = torch.arange(b, device="cuda", dtype=torch.int32)
    return tgt, draft, ds.view(-1), pos, idx, temp, seed


def bench_rejection(reqs, target, synthetic):
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    warmup_rejection_sample_standard()
    bf16 = target == "bf16"
    out = []
    for b in reqs:
        gen = torch.Generator(device="cuda").manual_seed(b)
        per_input = b * 6 * V * (2 if bf16 else 4) + b * 5 * V * 2
        n = max(4, min(100, -(-(2 << 30) // per_input)))
        inputs = [_rejection_input(b, gen, bf16) for _ in range(n)]
        cu = (torch.arange(b + 1, device="cuda") * 6).int()
        local_pos = torch.arange(6, device="cuda").repeat(b).int()
        eidx = inputs[0][4].repeat_interleave(6)
        rates = (
            torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5], device="cuda")
            if synthetic
            else None
        )

        def flydsl(inputs=inputs, rates=rates):
            for args in inputs:
                flydsl_rejection_sample_standard(*args, rates)

        def vllm_chain(
            inputs=inputs, cu=cu, eidx=eidx, local_pos=local_pos, rates=rates
        ):
            for tgt, dl, ds, pos, idx, temp, seed in inputs:
                if bf16:
                    tgt = tgt.float()
                    apply_temperature(tgt, eidx, temp)
                rejection_sample(
                    tgt,
                    _DraftCache(dl),
                    ds,
                    cu,
                    pos,
                    idx,
                    eidx,
                    local_pos,
                    temp,
                    seed,
                    5,
                    synthetic_conditional_rates=rates,
                )

        graphs = {"vllm": graph(vllm_chain), "flydsl": graph(flydsl)}
        out.append((b, abba(graphs, n)))
        del graphs
        torch.cuda.empty_cache()
    mode = "synthetic acceptance" if synthetic else "ratio test"
    report(
        f"Rejection sampling ({mode}), 5 drafts, vocab 129280, {target} target rows",
        "requests",
        out,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reqs", type=int, nargs="*", default=[8, 16, 32, 64, 128])
    ap.add_argument("--target", nargs="*", default=["bf16", "fp32"])
    ap.add_argument(
        "--mode",
        nargs="*",
        default=["standard", "synthetic"],
        choices=["standard", "synthetic"],
    )
    args = ap.parse_args()
    for mode in args.mode:
        for target in args.target:
            bench_rejection(args.reqs, target, mode == "synthetic")


if __name__ == "__main__":
    main()
