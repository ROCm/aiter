# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Simulate serving MiMo through ``aiter.ops.unified_attention`` and count JIT work.

A vLLM-style scheduler (continuous batching, chunked prefill, decode first)
drives the public ``unified_attention`` entry with the shapes a serving engine
would pass, for MiMo's global layers and, optionally, its sliding-window layers
with sinks. Every call is timed after a device synchronize, and the JIT work it
triggered is attributed to it:

- FlyDSL: launches that missed aiter's in-process cache of compiled callables
  (each one traces the kernel in Python), split into full compiles and loads
  from the on-disk cache (``FLYDSL_RUNTIME_CACHE_DIR``).
- Triton: kernel specializations compiled or loaded from its cache.

The workload is deterministic for a seed, so runs with a cold cache, a warm
cache, an AOT-built cache (``python -m aiter.aot.flydsl.unified_attention``) or
another backend replay the same calls. With ``FLYDSL_RUNTIME_RUN_ONLY=1`` any
FlyDSL kernel missing from the cache raises instead of compiling.

Usage:
    python op_tests/op_benchmarks/flydsl/bench_unified_attention_jit.py \
        --steps 1500 --json out.json
    python ... --backend triton
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import time
from dataclasses import dataclass, field

import torch

HEAD_DIM, VALUE_DIM, GQA = 192, 128, 16


# --- JIT instrumentation ---------------------------------------------------------


@dataclass
class JitCounters:
    fly_launch_misses: int = 0
    fly_launch_seconds: float = 0.0
    fly_compiles: int = 0
    fly_compile_seconds: float = 0.0
    fly_disk_hits: int = 0
    triton_compiles: int = 0
    triton_compile_seconds: float = 0.0
    kernels: collections.Counter = field(default_factory=collections.Counter)

    def snapshot(self):
        return (
            self.fly_launch_misses,
            self.fly_launch_seconds,
            self.fly_compiles,
            self.fly_compile_seconds,
            self.fly_disk_hits,
            self.triton_compiles,
            self.triton_compile_seconds,
        )


COUNTERS = JitCounters()


def install_hooks():
    """Count FlyDSL launches that miss aiter's compiled-callable cache, the
    backend compiles and disk-cache hits among them, and Triton compiles."""
    try:
        from flydsl._mlir import ir
        from flydsl.compiler import jit_function as jf
    except ImportError:
        jf = None
    if jf is not None:
        dispatch = jf.JitFunction._compile_or_execute

        def counted_dispatch(self, *args, **kwargs):
            # A launch inside another one's trace is inlined, not compiled.
            if ir.Context.current is not None:
                return dispatch(self, *args, **kwargs)
            started = time.perf_counter()
            try:
                return dispatch(self, *args, **kwargs)
            finally:
                COUNTERS.fly_launch_misses += 1
                COUNTERS.fly_launch_seconds += time.perf_counter() - started
                name = self.func.__qualname__.replace(".<locals>", "")
                COUNTERS.kernels[f"flydsl:{name}"] += 1

        jf.JitFunction._compile_or_execute = counted_dispatch

        compile_fn = jf.MlirCompiler.__dict__["compile"].__func__

        def counted_compile(cls, *args, **kwargs):
            started = time.perf_counter()
            try:
                return compile_fn(cls, *args, **kwargs)
            finally:
                COUNTERS.fly_compiles += 1
                COUNTERS.fly_compile_seconds += time.perf_counter() - started

        jf.MlirCompiler.compile = classmethod(counted_compile)

        cache_get = jf.JitCacheManager.get

        def counted_get(self, key):
            value = cache_get(self, key)
            if value is not None:
                COUNTERS.fly_disk_hits += 1
            return value

        jf.JitCacheManager.get = counted_get

    from triton.runtime import jit as triton_jit

    do_compile = triton_jit.JITFunction._do_compile

    def counted_do_compile(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return do_compile(self, *args, **kwargs)
        finally:
            COUNTERS.triton_compiles += 1
            COUNTERS.triton_compile_seconds += time.perf_counter() - started
            COUNTERS.kernels[f"triton:{self.fn.__name__}"] += 1

    triton_jit.JITFunction._do_compile = counted_do_compile


def install_route_spy():
    """Record per call whether FlyDSL served it (the router falls back to
    Triton when ``unified_attention_flydsl`` returns None)."""
    try:
        from aiter.ops.flydsl import unified_attention_bf16 as fly
    except ImportError:
        return lambda: None
    state = {"route": None}
    original = fly.unified_attention_flydsl

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        state["route"] = "flydsl" if result is not None else "triton"
        return result

    fly.unified_attention_flydsl = spy

    def take():
        route, state["route"] = state["route"], None
        return route

    return take


# --- workload ------------------------------------------------------------------


@dataclass
class Request:
    rid: int
    prompt: int
    output: int
    pages: list
    computed: int = 0
    generated: int = 0


class Scheduler:
    """vLLM v1-like: running decodes first (1 + MTP tokens each), then prefill
    chunks of admitted requests within the token budget."""

    def __init__(self, args, rng: random.Random):
        self.args = args
        self.rng = rng
        self.waiting: collections.deque = collections.deque()
        self.running: list[Request] = []
        self.next_id = 0
        self.steps = 0
        self.max_pages = math.ceil(args.max_model_len / args.page)

    def _arrivals(self):
        a = self.args
        rate = a.arrival_rate
        if a.arrival_period:
            # Load swings between idle and twice the mean rate.
            rate *= 1 - math.cos(2 * math.pi * self.steps / a.arrival_period)
        self.steps += 1
        # Poisson arrivals per step.
        n, p, threshold = 0, 1.0, math.exp(-rate)
        while True:
            p *= self.rng.random()
            if p <= threshold:
                break
            n += 1
        for _ in range(n):
            prompt = int(
                min(
                    max(self.rng.lognormvariate(math.log(a.prompt_median), 1.0), 16),
                    a.max_model_len - a.max_output - (a.mtp + 1),
                )
            )
            output = self.rng.randint(a.min_output, a.max_output)
            pages = [self.rng.randrange(1, a.kv_pages) for _ in range(self.max_pages)]
            self.waiting.append(Request(self.next_id, prompt, output, pages))
            self.next_id += 1

    def step(self):
        """One batch: [(request, query tokens)], or [] when idle."""
        a = self.args
        self._arrivals()
        while self.waiting and len(self.running) < a.max_num_seqs:
            self.running.append(self.waiting.popleft())
        budget = a.max_num_batched_tokens
        batch = []
        for r in self.running:
            if r.computed >= r.prompt and budget >= a.mtp + 1:
                batch.append((r, a.mtp + 1))
                budget -= a.mtp + 1
        for r in self.running:
            if r.computed < r.prompt and budget > 0:
                chunk = min(r.prompt - r.computed, budget)
                batch.append((r, chunk))
                budget -= chunk
        return batch

    def commit(self, batch):
        for r, q in batch:
            decoding = r.computed >= r.prompt
            r.computed += q
            if decoding:
                # Every MTP draft accepted: the context grows by q.
                r.generated += q
        self.running = [r for r in self.running if r.generated < r.output]


# --- driver --------------------------------------------------------------------


def build_inputs(batch, args, device, q_pool, out_pool):
    query_lens = [q for _, q in batch]
    tokens = sum(query_lens)
    cu = [0]
    for q in query_lens:
        cu.append(cu[-1] + q)
    seqused = [r.computed + q for r, q in batch]
    table = [r.pages for r, _ in batch]
    return dict(
        q=q_pool[:tokens],
        out=out_pool[:tokens],
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32).to(device, non_blocking=True),
        max_seqlen_q=max(query_lens),
        seqused_k=torch.tensor(seqused, dtype=torch.int32).to(
            device, non_blocking=True
        ),
        max_seqlen_k=max(seqused),
        block_table=torch.tensor(table, dtype=torch.int32).to(
            device, non_blocking=True
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", default=None, choices=[None, "flydsl", "triton"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--page", type=int, default=64, choices=[64, 128])
    p.add_argument("--kv-heads", type=int, default=1, help="KV heads per rank")
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument("--arrival-rate", type=float, default=0.1, help="requests/step")
    p.add_argument(
        "--arrival-period",
        type=int,
        default=1500,
        help="steps per load cycle (idle to peak and back); 0: constant rate",
    )
    p.add_argument("--prompt-median", type=int, default=1500)
    p.add_argument("--min-output", type=int, default=64)
    p.add_argument("--max-output", type=int, default=2048)
    p.add_argument("--mtp", type=int, default=0, help="draft tokens per decode")
    p.add_argument(
        "--layers",
        default="global,swa",
        help="comma list of global (full causal) and swa (window 128 + sinks)",
    )
    p.add_argument("--window", type=int, default=127)
    p.add_argument("--no-sinks", action="store_true", help="SWA layers without sinks")
    p.add_argument("--kv-pages", type=int, default=32768)
    p.add_argument("--stall-ms", type=float, default=50.0, help="report threshold")
    p.add_argument("--json", help="write per-call records here")
    args = p.parse_args()

    install_hooks()
    take_route = install_route_spy()
    from aiter.ops.unified_attention import unified_attention

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    hq = GQA * args.kv_heads
    k_cache = torch.randn(
        args.kv_pages, args.page, args.kv_heads, HEAD_DIM, device=device
    ).bfloat16()
    v_cache = torch.randn(
        args.kv_pages, args.page, args.kv_heads, VALUE_DIM, device=device
    ).bfloat16()
    max_tokens = args.max_num_batched_tokens
    q_pool = torch.randn(max_tokens, hq, HEAD_DIM, device=device).bfloat16()
    out_pool = torch.empty(max_tokens, hq, VALUE_DIM, device=device).bfloat16()
    sinks = None if args.no_sinks else torch.randn(hq, device=device)
    layers = {
        "global": dict(window_size=(-1, -1), sinks=None),
        "swa": dict(window_size=(args.window, 0), sinks=sinks),
    }
    layers = {name: layers[name] for name in args.layers.split(",")}

    sched = Scheduler(args, random.Random(args.seed))
    records = []
    started = time.perf_counter()
    for step in range(args.steps):
        batch = sched.step()
        if not batch:
            continue
        inputs = build_inputs(batch, args, device, q_pool, out_pool)
        decode_only = all(r.computed >= r.prompt for r, _ in batch)
        for layer, opts in layers.items():
            before = COUNTERS.snapshot()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            unified_attention(
                inputs["q"],
                k_cache,
                v_cache,
                inputs["out"],
                inputs["cu_seqlens_q"],
                inputs["max_seqlen_q"],
                inputs["seqused_k"],
                inputs["max_seqlen_k"],
                HEAD_DIM**-0.5,
                True,
                opts["window_size"],
                inputs["block_table"],
                0,
                None,
                None,
                None,
                sinks=opts["sinks"],
                backend=args.backend,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            after = COUNTERS.snapshot()
            delta = [b - a for a, b in zip(before, after)]
            records.append(
                dict(
                    step=step,
                    layer=layer,
                    route=take_route() or args.backend or "triton",
                    decode_only=decode_only,
                    num_seqs=len(batch),
                    num_tokens=int(inputs["q"].shape[0]),
                    max_seqlen_q=inputs["max_seqlen_q"],
                    max_seqlen_k=inputs["max_seqlen_k"],
                    ms=elapsed * 1e3,
                    fly_misses=delta[0],
                    fly_jit_ms=delta[1] * 1e3,
                    fly_compiles=delta[2],
                    fly_disk_hits=delta[4],
                    triton_compiles=delta[5],
                    triton_jit_ms=delta[6] * 1e3,
                )
            )
        sched.commit(batch)
    wall = time.perf_counter() - started

    summarize(records, wall, args)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "wall_s": wall, "calls": records}, f)


def summarize(records, wall, args):
    n = len(records)
    jit_calls = [r for r in records if r["fly_misses"] or r["triton_compiles"]]
    stalls = [r for r in records if r["ms"] >= args.stall_ms]
    routes = collections.Counter(r["route"] for r in records)
    steady = sorted(
        r["ms"] for r in records if not (r["fly_misses"] or r["triton_compiles"])
    )

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")

    print("=" * 78)
    print(
        f"{n} attention calls over {args.steps} steps in {wall:.1f}s "
        f"(routes: {dict(routes)})"
    )
    print(
        f"FlyDSL: {COUNTERS.fly_launch_misses} launch misses "
        f"({COUNTERS.fly_launch_seconds:.1f}s), {COUNTERS.fly_compiles} compiles "
        f"({COUNTERS.fly_compile_seconds:.1f}s), {COUNTERS.fly_disk_hits} disk-cache hits"
    )
    print(
        f"Triton: {COUNTERS.triton_compiles} specializations compiled or loaded "
        f"({COUNTERS.triton_compile_seconds:.1f}s)"
    )
    print(
        f"Calls with JIT work: {len(jit_calls)}  "
        f"(stall time {sum(r['ms'] for r in jit_calls) / 1e3:.1f}s); "
        f"calls >= {args.stall_ms:.0f} ms: {len(stalls)}"
    )
    if jit_calls:
        last = max(r["step"] for r in jit_calls)
        quarters = collections.Counter(
            min(3, 4 * r["step"] // args.steps) for r in jit_calls
        )
        print(
            f"JIT calls per quarter of the run: "
            f"{[quarters.get(i, 0) for i in range(4)]}; last at step {last}"
        )
    print(
        f"Calls without JIT: p50 {pct(steady, 0.5):.3f} ms, "
        f"p99 {pct(steady, 0.99):.3f} ms"
    )
    print("JIT misses per kernel launcher:")
    for name, count in COUNTERS.kernels.most_common():
        print(f"  {count:5d}  {name}")


if __name__ == "__main__":
    main()
