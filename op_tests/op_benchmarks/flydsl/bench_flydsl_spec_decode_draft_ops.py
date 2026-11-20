# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CUDA-graph time of the FlyDSL speculative-decoding draft-step ops against the
paths they replace (gfx950, one GPU).

    python3 op_tests/op_benchmarks/flydsl/bench_flydsl_spec_decode_draft_ops.py
    python3 .../bench_flydsl_spec_decode_draft_ops.py --op router --rows 16 160

Ops and baselines (both baselines need vLLM):

* ``router``: ``flydsl_router_sqrtsoftplus_topk`` (row-major and load-time gate
  layouts) vs vLLM's ``fp32_router_gemm_dispatch`` followed by vLLM's
  ``topk_hash_softplus_sqrt`` or by aiter's ``topk_gating(score_func="sqrtsoftplus")``.
* ``rejection``: ``flydsl_rejection_sample_standard`` vs vLLM's Triton
  ``rejection_sample``, with the probability-ratio test and in the synthetic
  acceptance mode (``--mode``). ``--target bf16``: raw bf16 lm_head rows go
  straight in; the baseline also runs the fp32 copy and ``apply_temperature``
  vLLM needs first. ``--target fp32``: processed fp32 rows for both.

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
from aiter.ops.flydsl.router_sqrtsoftplus_topk import (
    flydsl_router_sqrtsoftplus_topk,
    shuffle_router_gate,
    warmup_router_sqrtsoftplus_topk,
)

H = 5120
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


def bench_router(rows, nin=100):
    import vllm.model_executor.layers.fused_moe.router.gate_linear  # noqa: F401
    from vllm import _custom_ops as ops

    from aiter.ops.topk import topk_gating

    warmup_router_sqrtsoftplus_topk()
    w = (torch.randn(128, H, device="cuda") * 0.02).bfloat16()
    ws = shuffle_router_gate(w)
    bias = torch.randn(128, device="cuda") * 0.1
    out = []
    for m in rows:
        xs = [torch.randn(m, H, device="cuda").bfloat16() for _ in range(nin)]
        bufs = [
            (
                torch.empty(m, 3, device="cuda"),
                torch.empty(m, 3, device="cuda", dtype=torch.int32),
                torch.empty(m, 3, device="cuda", dtype=torch.int32),
            )
            for _ in range(nin)
        ]

        def flydsl(xs=xs, bufs=bufs):
            for x, (tw, ti, _) in zip(xs, bufs):
                flydsl_router_sqrtsoftplus_topk(
                    x, w, bias, 3, 1.5, topk_weights=tw, topk_ids=ti
                )

        def flydsl_shuffled(xs=xs, bufs=bufs):
            for x, (tw, ti, _) in zip(xs, bufs):
                flydsl_router_sqrtsoftplus_topk(
                    x, ws, bias, 3, 1.5, topk_weights=tw, topk_ids=ti,
                    weight_shuffled=True,
                )  # fmt: skip

        def vllm_pair(xs=xs, bufs=bufs):
            for x, (tw, ti, te) in zip(xs, bufs):
                lg = torch.ops.vllm.fp32_router_gemm_dispatch(x, w, False)
                ops.topk_hash_softplus_sqrt(
                    tw, ti, te, lg, True, 1.5, bias, None, None, None, None, 0
                )

        def vllm_gemm_aiter_topk(xs=xs, bufs=bufs):
            for x, (tw, ti, _) in zip(xs, bufs):
                lg = torch.ops.vllm.fp32_router_gemm_dispatch(x, w, False)
                topk_gating(tw, ti, lg, bias, True, 1.5, "sqrtsoftplus")

        graphs = {
            "vllm gemm + topk": graph(vllm_pair),
            "vllm gemm + aiter topk_gating": graph(vllm_gemm_aiter_topk),
            "flydsl": graph(flydsl),
            "flydsl shuffled gate": graph(flydsl_shuffled),
        }
        out.append((m, abba(graphs, nin)))
        del graphs
        torch.cuda.empty_cache()
    report("Router GEMM + sqrt-softplus top-3, 128 experts, hidden 5120", "rows", out)


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
                    tgt, _DraftCache(dl), ds, cu, pos, idx, eidx, local_pos,
                    temp, seed, 5, synthetic_conditional_rates=rates,
                )  # fmt: skip

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
    ap.add_argument(
        "--op", nargs="*", default=["router", "rejection"],
        choices=["router", "rejection"],
    )  # fmt: skip
    ap.add_argument("--rows", type=int, nargs="*", default=None)
    ap.add_argument("--reqs", type=int, nargs="*", default=[8, 16, 32, 64, 128])
    ap.add_argument("--target", nargs="*", default=["bf16", "fp32"])
    ap.add_argument(
        "--mode", nargs="*", default=["standard", "synthetic"],
        choices=["standard", "synthetic"],
    )  # fmt: skip
    args = ap.parse_args()
    if "router" in args.op:
        bench_router(args.rows or [5, 16, 40, 80, 160, 320, 640, 768])
    if "rejection" in args.op:
        for mode in args.mode:
            for target in args.target:
                bench_rejection(args.reqs, target, mode == "synthetic")


if __name__ == "__main__":
    main()
