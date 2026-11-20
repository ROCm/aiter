# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Graph-replay time of the FlyDSL fused DSpark Markov draft sampler against the
paths of vLLM main it replaces (gfx950).

``flydsl_markov_chain_sample`` (5 steps, 129280 x 256) against

- ``vllm_loop``: the per-step loop of vLLM's
  ``DSparkSpeculator._sample_markov_chain`` with the bias fused into
  ``gumbel_sample`` (Markov embedding, bias GEMM through
  ``rocm_unquantized_gemm_impl``, ``gumbel_sample``, token copy; 5 steps),
- ``vllm_fused``: vLLM's one-launch-per-step Triton Markov sampler
  (``vllm.models.deepseek_v41.amd.ops.fused_markov``, used with
  ``--monokernel`` for 1-5 requests on gfx950), for row counts it serves.
  ``sample_sequential`` also gathers the hidden states and runs the draft
  lm_head before its launches; this baseline issues only its launches, from
  the base logits on, with the arguments of vLLM main 9b9bd049a (2026-11-08,
  the last change of ``fused_markov.py``). The kernels' parameter lists are
  asserted, so a changed signature fails instead of timing something else.

Every case captures one CUDA graph of ``--sets`` calls on distinct inputs (the
weights stay; the base logits are larger than the LLC together), and times
``--rounds`` rounds of replays in the order A B B A per candidate pair. The
table holds the median and the minimum us per call.

Run:
    python3 op_tests/op_benchmarks/flydsl/bench_markov_chain_sample.py
    python3 op_tests/op_benchmarks/flydsl/bench_markov_chain_sample.py -n 8 32 64
"""

import argparse
import functools
import os
import statistics

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import markov_chain_sample as MC

SLOTS = 512
BUDGET = 24 << 30  # bytes of distinct inputs per graph


def capture(calls):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in calls[:2]:
            f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in calls:
            f()
    g.replay()
    torch.cuda.synchronize()
    return g


def replay_us(g, n, replays):
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(replays):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / (replays * n)


def time_graphs(graphs, n, rounds, replays):
    """{name: [us per call]}: rounds of A B B A against the first graph."""
    t = {k: [] for k in graphs}
    names = list(graphs)
    for _ in range(rounds):
        for other in names[1:]:
            for k in (names[0], other, other, names[0]):
                t[k].append(replay_us(graphs[k], n, replays))
    return t


def vllm_loop_fn():
    try:
        from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    except Exception as e:  # noqa: BLE001
        aiter.logger.warning("vLLM baseline unavailable: %s", e)
        return None
    NS, V = MC.NSPEC, MC.V
    cols = None

    def loop(base, w1, w2, ids, anchor, idx, pos, temp, seeds, cache, tokens):
        nonlocal cols
        if cols is None:
            cols = [
                torch.tensor(s, dtype=torch.int32, device=base.device)
                for s in range(NS)
            ]
        n = anchor.shape[0]
        base3 = base.view(n, NS, V)
        prev = ids[anchor]
        for s in range(NS):
            bias = rocm_unquantized_gemm_impl(F.embedding(prev, w1), w2)
            out = gumbel_sample(
                base3[:, s],
                idx[:, s],
                temp,
                seeds,
                pos[:, s] - 1,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=cache,
                logits_cache_col=cols[s],
                use_fp64=False,
                bias=bias,
            )
            tokens[:n, s] = out
            prev = out

    return loop


def vllm_fused_fn():
    """vLLM's fused_markov launches from the base logits on (its
    sample_sequential after compute_draft_logits)."""
    try:
        from vllm.models.deepseek_v41.amd.ops import fused_markov as fm
        from vllm.triton_utils import triton
    except Exception as e:  # noqa: BLE001
        aiter.logger.warning("vLLM fused Markov sampler unavailable: %s", e)
        return None, 0

    step_args = [
        "prev_max_ptr", "prev_arg_ptr", "input_ids_ptr", "anchor_idx_ptr",
        "tokens_ptr", "tokens_stride", "w1_ptr", "w2_ptr", "base_ptr",
        "base_stride", "cache_ptr", "cache_stride0", "cache_stride1",
        "idx_map_ptr", "idx_stride", "pos_ptr", "pos_stride", "temp_ptr",
        "seeds_ptr", "out_max_ptr", "out_arg_ptr", "vocab", "num_blocks", "STEP",
        "ROWS", "RANK", "BLOCK", "K_CHUNK", "BLOCKS_POW2", "USE_FP64",
    ]  # fmt: skip
    last_args = [
        "prev_max_ptr", "prev_arg_ptr", "tokens_ptr", "tokens_stride",
        "num_blocks", "STEP", "BLOCKS_POW2",
    ]  # fmt: skip
    if (
        list(fm._markov_step_kernel.arg_names) != step_args
        or list(fm._markov_last_kernel.arg_names) != last_args
    ):
        raise RuntimeError(
            "vLLM's fused_markov kernels changed their parameters since "
            "9b9bd049a; update vllm_fused_fn"
        )

    def fused(base, w1, w2, ids, anchor, idx, pos, temp, seeds, cache, tokens):
        n, ns, vocab = anchor.shape[0], MC.NSPEC, MC.V
        base3 = base.view(n, ns, vocab)
        bufs = fm._scratch(base.device, vocab, False)
        nb = triton.cdiv(vocab, fm.BLOCK)
        nb2 = triton.next_power_of_2(nb)
        for step in range(ns):
            prev_max, prev_arg = bufs[(step + 1) % 2]
            out_max, out_arg = bufs[step % 2]
            fm._markov_step_kernel[(nb,)](
                prev_max,
                prev_arg,
                ids,
                anchor,
                tokens,
                tokens.stride(0),
                w1,
                w2,
                base3[:, step],
                base3.stride(0),
                cache,
                cache.stride(0),
                cache.stride(1),
                idx,
                idx.stride(0),
                pos,
                pos.stride(0),
                temp,
                seeds,
                out_max,
                out_arg,
                vocab,
                nb,
                STEP=step,
                ROWS=n,
                RANK=w1.shape[1],
                BLOCK=fm.BLOCK,
                K_CHUNK=fm.K_CHUNK,
                BLOCKS_POW2=nb2,
                USE_FP64=False,
                num_warps=fm.NUM_WARPS,
            )
        last_max, last_arg = bufs[(ns - 1) % 2]
        fm._markov_last_kernel[(n,)](
            last_max,
            last_arg,
            tokens,
            tokens.stride(0),
            nb,
            STEP=ns - 1,
            BLOCKS_POW2=nb2,
        )

    return fused, fm.MAX_ROWS_GFX950


def bench_markov(ns, sets, rounds, replays):
    dev = torch.device("cuda")
    NS, V, R = MC.NSPEC, MC.V, MC.R
    MC.warmup_markov_chain_sample(dev)
    gen = torch.Generator().manual_seed(1)
    w1 = (torch.randn(V, R, generator=gen) * 0.3).to(torch.bfloat16).to(dev)
    w2 = (torch.randn(V, R, generator=gen) * 0.3).to(torch.bfloat16).to(dev)
    ids = torch.randint(0, V, (1 << 16,), generator=gen).to(dev)
    temp = torch.ones(SLOTS, dtype=torch.float32, device=dev)
    seeds = torch.randint(-(2**62), 2**62, (SLOTS,), generator=gen).to(dev)
    cache = torch.zeros(SLOTS, NS, V, dtype=torch.bfloat16, device=dev)
    loop = vllm_loop_fn()
    fused, fused_max = vllm_fused_fn()
    rows = []
    for n in ns:
        count = max(4, min(sets, BUDGET // (n * NS * V * 2)))
        data = []
        for _ in range(count):
            base = torch.randn(n * NS, V, generator=gen) * 3
            slots = torch.randperm(SLOTS, generator=gen)[:n]
            pos = 131072 + torch.randint(0, 4096, (n, 1), generator=gen)
            data.append(
                {
                    "base": base.to(torch.bfloat16).to(dev),
                    "anchor": torch.randint(0, 1 << 16, (n,), generator=gen).to(dev),
                    "idx": slots.view(n, 1).expand(n, NS).contiguous().to(dev),
                    "pos": (pos + torch.arange(NS)).to(dev),
                }
            )
        tokens = torch.zeros(n, NS, dtype=torch.int64, device=dev)
        cand = {"flydsl": MC.flydsl_markov_chain_sample}
        if loop is not None:
            cand["vllm_loop"] = loop
        if fused is not None and n <= fused_max:
            cand["vllm_fused"] = fused
        row = {"gfx": get_gfx_runtime(), "requests": n, "sets": count}
        # agreement of the draft tokens with the flydsl ones on one input
        d0 = data[0]
        ref_tok = None
        graphs = {}
        for name, fn in cand.items():
            t = torch.zeros(n, NS, dtype=torch.int64, device=dev)
            fn(
                d0["base"],
                w1,
                w2,
                ids,
                d0["anchor"],
                d0["idx"],
                d0["pos"],
                temp,
                seeds,
                cache,
                t,
            )
            if ref_tok is None:
                ref_tok = t
            row[f"{name} tok agree"] = float((t == ref_tok).float().mean())

            graphs[name] = capture(
                [
                    functools.partial(
                        fn,
                        d["base"],
                        w1,
                        w2,
                        ids,
                        d["anchor"],
                        d["idx"],
                        d["pos"],
                        temp,
                        seeds,
                        cache,
                        tokens,
                    )
                    for d in data
                ]
            )
        t = time_graphs(graphs, count, rounds, replays)
        for name, v in t.items():
            row[f"{name} us"] = statistics.median(v)
            row[f"{name} min us"] = min(v)
        rows.append(row)
        aiter.logger.info("%s", row)
    return pd.DataFrame(rows)


def main():
    if get_gfx_runtime() not in ("gfx950",):
        aiter.logger.warning("gfx950 only; skipping on %s", get_gfx_runtime())
        return
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "-n",
        "--requests",
        type=int,
        nargs="*",
        default=[1, 2, 4, 5, 8, 16, 32, 48, 64, 96, 128, 160],
    )
    p.add_argument("--sets", type=int, default=100)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--replays", type=int, default=5)
    a = p.parse_args()
    with torch.inference_mode():
        df = bench_markov(a.requests, a.sets, a.rounds, a.replays)
        aiter.logger.info(
            "Markov draft chain (us per 5-step chain):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
