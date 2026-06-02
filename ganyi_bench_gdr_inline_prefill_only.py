# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Self-contained chunked-GDN prefill bench: vLLM vs aiter-Triton vs FlyDSL.

333 unique prefill-only (T, cu_seqlens) shapes from prefill_gdr.log are
embedded as a literal at the bottom of this file — no external CSV needed.
Just run:

    HIP_VISIBLE_DEVICES=7 python benchmarks/bench_chunked_gdr_inline.py

Three vk-layout backends are compared head-to-head per shape:

  * **vllm**   — vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule
                 (FLA upstream Triton, vk layout, no K-fusion across phases)
  * **triton** — aiter.ops.triton.gated_delta_net.gated_delta_rule
                 .chunk_gated_delta_rule_opt_vk
                 (K1+K2 fused triton, K3+K4 fused triton, K5 opt-vk triton,
                  K6 chunk_fwd_o_opt_vk triton)
  * **flydsl** — aiter.ops.flydsl.linear_attention_prefill_kernels
                 .flydsl_gdr_prefill
                 (same K1+K2 / K3+K4 / K6 as 'triton', K5 swapped to flydsl)

Same shape, same random inputs (deterministic seed per shape) → the
deltas are exactly K1+K2 fusion (vllm → triton), K3+K4 fusion (vllm →
triton), and K5 backend (triton → flydsl).

Flags:
    --backends   vllm | triton | flydsl | all   (default: all)
                 also accepts CSV: e.g. --backends vllm,flydsl
    --warmup N   warmup iters per shape         (default: 25)
    --iters N    timed iters per shape          (default: 100)
    --limit N    bench only top-N shapes by log_count
    --top K      console top-K summary
    --seed S     deterministic input seed
    --out PATH   write per-shape results CSV (default: stdout-only)

The aggregate block at the end reports, for every benched-backend pair:
    * Overall speedup (log-weighted by call frequency in the source workload)
    * Geometric mean of per-shape speedups (right average for ratios)
    * Arithmetic mean
    * Median
    * Min / Max
    * Per-shape win/loss/tie count

Shape data was extracted by parsing
/home/gyu_qle/ganyi/ATOM/prefill_gdr.log (28,152 prefill GDN calls),
dedup'd to 407 unique (T, cu_seqlens) tuples, then stripped of
``seqlen==1`` decode segments (they are decode-only steps, not part of
the prefill kernel's workload). Deduping again on the prefill-only
(T, cu_seqlens) collapsed 407 → 333 unique shapes. The ``log_count`` for
each shape is summed across collapsed duplicates so the impact-weighting
in the aggregate still totals 28,152.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch

# Fixed head dims for Qwen3-Next-80B-A3B-Instruct-FP8, TP=1 (constants
# of the model; the source log confirms every prefill call shares them).
QWEN3_NEXT_Hq = 16
QWEN3_NEXT_Hv = 32
QWEN3_NEXT_K = 128
QWEN3_NEXT_V = 128


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


def _load_vllm_backend() -> Callable:
    """FLA upstream vk pipeline as shipped in vLLM (chunk.py — no K-fusion
    across phases, each of K1/K2/K3/K4/K5/K6 launches separately)."""
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
    return chunk_gated_delta_rule


def _load_atom_backend() -> Callable:
    """ATOM's vendored vk port of the vLLM FLA pipeline
    (atom.model_ops.fla_ops.chunk_vk.chunk_gated_delta_rule_vk).

    Should be ~bit-equal to the 'vllm' backend in steady state (verified
    by tests/test_chunk_gated_delta_rule_vk.py). Existing as a separate
    backend lets us catch any latent perf gap from ATOM's local edits
    (the o= inplace plumbing, the optional flydsl dispatch, etc.).

    Run with the flydsl dispatch OFF so we're comparing the ATOM Triton
    pipeline (= vendored vLLM kernels + ATOM's o= wiring) against the
    vLLM Triton pipeline — i.e. isolating only the ATOM-side wiring."""
    import os
    os.environ["ATOM_USE_FLYDSL_GDR_PREFILL"] = "0"
    from atom.model_ops.fla_ops.chunk_vk import chunk_gated_delta_rule_vk
    return chunk_gated_delta_rule_vk


def _load_triton_backend() -> Callable:
    """aiter Triton K1+K2/K3+K4 fused end-to-end vk pipeline."""
    from aiter.ops.triton.gated_delta_net.gated_delta_rule import (
        chunk_gated_delta_rule_opt_vk,
    )
    return chunk_gated_delta_rule_opt_vk


def _load_flydsl_backend() -> Callable:
    """aiter FlyDSL-K5 end-to-end vk pipeline (same K1+K2/K3+K4/K6 as the
    'triton' backend, only K5 differs)."""
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        flydsl_gdr_prefill,
    )
    return flydsl_gdr_prefill


# Single registry so the rest of the bench loop can iterate uniformly.
# Order matters: it determines the column order in the per-shape log line,
# the per-shape CSV, and the canonical baseline (first entry) used for
# the relative-speedup display.
_BACKEND_REGISTRY: dict[str, Callable[[], Callable]] = {
    "vllm": _load_vllm_backend,
    # "atom": _load_atom_backend,
    "triton": _load_triton_backend,
    "flydsl": _load_flydsl_backend,
}


# ---------------------------------------------------------------------------
# Shape table (embedded — parsed at startup from _SHAPES_RAW below)
# ---------------------------------------------------------------------------


def _parse_shapes() -> list[dict]:
    """Parse the embedded shape literal into a list of dicts."""
    shapes = []
    for line in _SHAPES_RAW.strip().splitlines():
        log_count_s, T_s, cu_s = line.split("|", 2)
        cu = [int(x) for x in cu_s.split()]
        T = int(T_s)
        n_seqs = len(cu) - 1
        # Sanity: cu_seqlens[-1] must equal T.
        assert cu[-1] == T, (
            f"embedded data corrupted: cu_seqlens[-1]={cu[-1]} != T={T} "
            f"for shape '{line[:60]}...'"
        )
        shapes.append({
            "T": T,
            "num_seqs": n_seqs,
            "cu_seqlens": cu,
            "log_count": int(log_count_s),
        })
    return shapes


# ---------------------------------------------------------------------------
# Input synthesis and timing
# ---------------------------------------------------------------------------


def _make_inputs(
    T: int,
    cu_seqlens: list[int],
    *,
    seed: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    rng = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(1, T, QWEN3_NEXT_Hq, QWEN3_NEXT_K, dtype=dtype,
                    device=device, generator=rng)
    k = torch.randn(1, T, QWEN3_NEXT_Hq, QWEN3_NEXT_K, dtype=dtype,
                    device=device, generator=rng)
    v = torch.randn(1, T, QWEN3_NEXT_Hv, QWEN3_NEXT_V, dtype=dtype,
                    device=device, generator=rng)
    g = -torch.rand(1, T, QWEN3_NEXT_Hv, dtype=torch.float32,
                    device=device, generator=rng)
    beta = torch.rand(1, T, QWEN3_NEXT_Hv, dtype=dtype,
                      device=device, generator=rng).sigmoid()
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    N = len(cu_seqlens) - 1
    initial_state = torch.randn(
        N, QWEN3_NEXT_Hv, QWEN3_NEXT_V, QWEN3_NEXT_K,
        dtype=torch.float32, device=device, generator=rng,
    )
    return q, k, v, g, beta, cu, initial_state


def _time_call(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    """Median CUDA-event-timed latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        stops[i].record()
    torch.cuda.synchronize()
    times_us = sorted(
        starts[i].elapsed_time(stops[i]) * 1000.0
        for i in range(iters)
    )
    return times_us[len(times_us) // 2]


def _bench_one(
    backend_fn: Callable,
    shape: dict,
    *,
    warmup: int,
    iters: int,
    seed: int,
) -> float:
    q, k, v, g, beta, cu, init = _make_inputs(
        shape["T"], shape["cu_seqlens"], seed=seed,
    )

    def _call():
        backend_fn(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=init.clone(),
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
        )

    return _time_call(_call, warmup=warmup, iters=iters)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_backends_arg(arg: str) -> list[str]:
    """Accept either the shorthand 'all' or a comma-separated list of
    registry keys. Preserves the registry's canonical order so output is
    deterministic regardless of CLI input ordering."""
    if arg == "all":
        return list(_BACKEND_REGISTRY.keys())
    requested = [x.strip() for x in arg.split(",") if x.strip()]
    unknown = [x for x in requested if x not in _BACKEND_REGISTRY]
    if unknown:
        sys.exit(
            f"unknown backend(s): {unknown}. "
            f"Valid: {list(_BACKEND_REGISTRY)} or 'all'."
        )
    # canonical order
    return [k for k in _BACKEND_REGISTRY if k in requested]


def _summarize_speedups(label: str, speedups: list[float], *,
                        a_total: float, b_total: float, a_name: str, b_name: str,
                        n_shapes_total: int):
    """Pretty-print the per-pair aggregate (geomean, arith, median, etc.)."""
    if not speedups or a_total == 0 or b_total == 0:
        print(f"  {label}: no comparable shapes")
        return
    geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    arith_mean = sum(speedups) / len(speedups)
    sorted_s = sorted(speedups)
    median = sorted_s[len(sorted_s) // 2]
    wins = sum(1 for s in speedups if s > 1.0)
    losses = sum(1 for s in speedups if s < 1.0)
    ties = len(speedups) - wins - losses
    p_min, p_max = sorted_s[0], sorted_s[-1]
    overall = a_total / b_total
    faster = b_name if overall > 1.0 else a_name
    print(f"  {label}  ({a_name} / {b_name}):")
    print(f"    {a_name} total wall time: {a_total/1e6:>10.3f} s")
    print(f"    {b_name} total wall time: {b_total/1e6:>10.3f} s")
    print(f"    Overall speedup (log-weighted): {overall:>7.3f}x   "
          f"(faster overall: {faster})")
    print(f"    Geometric mean:                 {geo_mean:>7.3f}x   "
          f"(right average for ratios)")
    print(f"    Arithmetic mean:                {arith_mean:>7.3f}x")
    print(f"    Median:                         {median:>7.3f}x")
    print(f"    Min / Max:                      {p_min:>7.3f}x  /  {p_max:.3f}x")
    print(f"    Per-shape outcome: {b_name} wins {wins} | loses {losses}"
          f"{f' | ties {ties}' if ties else ''}  "
          f"(out of {len(speedups)} comparable shapes, "
          f"{n_shapes_total} benched total)")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional output CSV (default: don't write)")
    ap.add_argument("--backends", default="all",
                    help="comma-separated subset of "
                         f"{list(_BACKEND_REGISTRY)}, or 'all' (default).")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None,
                    help="bench only top-N shapes by log_count")
    ap.add_argument("--sort", choices=("by-count", "by-T", "order"), default="by-count")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shapes = _parse_shapes()
    if args.sort == "by-count":
        shapes.sort(key=lambda s: -s["log_count"])
    elif args.sort == "by-T":
        shapes.sort(key=lambda s: s["T"])
    if args.limit is not None:
        shapes = shapes[: args.limit]
    total_log_calls = sum(s["log_count"] for s in shapes)

    backend_names = _parse_backends_arg(args.backends)
    backend_fns: dict[str, Callable] = {}
    for name in backend_names:
        try:
            backend_fns[name] = _BACKEND_REGISTRY[name]()
        except ImportError as e:
            sys.exit(f"{name} backend import failed: {e}")

    print(f"Device:   {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Shapes:   {len(shapes)} unique (covering {total_log_calls:,} log calls)",
          flush=True)
    print(f"Backends: {backend_names}    warmup={args.warmup}  iters={args.iters}",
          flush=True)
    print("", flush=True)

    # Per-shape results: each row stores us[name] for every benched backend.
    results: list[dict] = []
    t_start = time.time()
    try:
        for i, shape in enumerate(shapes):
            T = shape["T"]
            n_seqs = shape["num_seqs"]
            log_n = shape["log_count"]

            row = {
                "T": T, "num_seqs": n_seqs,
                "cu_seqlens": shape["cu_seqlens"], "log_count": log_n,
                "us": {},
            }
            for name in backend_names:
                row["us"][name] = _bench_one(
                    backend_fns[name], shape,
                    warmup=args.warmup, iters=args.iters, seed=args.seed,
                )
            results.append(row)

            cu_preview = shape["cu_seqlens"]
            cu_str = (
                str(cu_preview) if len(cu_preview) <= 6
                else f"[..., {', '.join(str(x) for x in cu_preview[-3:])}]"
            )
            cells = [f"[{i+1:>3}/{len(shapes)}]",
                     f"T={T:>6}", f"n={n_seqs:>3}",
                     f"cnt={log_n:>5}",
                     f"cu={cu_str}"]
            for name in backend_names:
                cells.append(f"{name}={row['us'][name]:>9.2f}us")
            # If at least two backends ran, show speedups relative to the
            # FIRST benched backend (the canonical-order baseline).
            if len(backend_names) >= 2:
                base = backend_names[0]
                for name in backend_names[1:]:
                    su = row["us"][base] / row["us"][name]
                    cells.append(f"{name}/{base}={su:>5.2f}x")
            print("  ".join(cells), flush=True)
    except KeyboardInterrupt:
        print("\n[interrupted, writing partial results if --out is set]", flush=True)

    elapsed = time.time() - t_start

    # Optional CSV output. Columns: T, num_seqs, cu_seqlens, log_count,
    # <name>_us for each backend, <a>_to_<b>_speedup for each pair,
    # <name>_impact_us for each backend.
    if args.out is not None:
        with args.out.open("w", newline="") as f:
            w = csv.writer(f)
            header = ["T", "num_seqs", "cu_seqlens", "log_count"]
            header += [f"{n}_us" for n in backend_names]
            # Pairs: a/b speedup = us[a] / us[b]. Numerator is the
            # canonical-order earlier backend; positive >1 means the
            # later backend is faster.
            pairs: list[tuple[str, str]] = []
            for i, a in enumerate(backend_names):
                for b in backend_names[i+1:]:
                    pairs.append((a, b))
                    header.append(f"speedup_{b}_over_{a}")
            header += [f"{n}_impact_us" for n in backend_names]
            w.writerow(header)
            for r in results:
                cells = [
                    r["T"], r["num_seqs"],
                    " ".join(str(x) for x in r["cu_seqlens"]),
                    r["log_count"],
                ]
                for n in backend_names:
                    cells.append(f"{r['us'][n]:.2f}" if r["us"].get(n) else "")
                for a, b in pairs:
                    if r["us"].get(a) and r["us"].get(b):
                        cells.append(f"{r['us'][a] / r['us'][b]:.3f}")
                    else:
                        cells.append("")
                for n in backend_names:
                    imp = (r["us"].get(n) or 0) * r["log_count"]
                    cells.append(f"{imp:.0f}" if r["us"].get(n) else "")
                w.writerow(cells)
        print(f"\nWrote: {args.out}  ({len(results)} rows, {elapsed:.1f}s)",
              flush=True)
    else:
        print(f"\nBench done in {elapsed:.1f}s. "
              f"Pass --out PATH to write per-shape results to CSV.", flush=True)

    # Top-K shapes by total impact, using the FIRST benched backend as the
    # canonical impact metric (it's the baseline against which speedups
    # are reported).
    if not results:
        return
    impact_name = backend_names[0]
    top = sorted(
        (r for r in results if r["us"].get(impact_name)),
        key=lambda r: -(r["us"][impact_name] * r["log_count"]),
    )[: args.top]
    print(f"\nTop {min(args.top, len(top))} shapes by {impact_name} total "
          f"impact (log_count × {impact_name}_us):")
    header_cols = [f"{'T':>6}", f"{'n':>3}", f"{'cnt':>5}"]
    for n in backend_names:
        header_cols.append(f"{n+'_us':>12}")
    header_cols.append(f"{impact_name+'_impact_ms':>16}")
    header = "  ".join(header_cols)
    print(header)
    print("-" * len(header))
    for r in top:
        cells = [f"{r['T']:>6}", f"{r['num_seqs']:>3}", f"{r['log_count']:>5}"]
        for n in backend_names:
            u = r["us"].get(n)
            cells.append(f"{u:>12.2f}" if u else f"{'-':>12}")
        imp_ms = r["us"][impact_name] * r["log_count"] / 1000.0
        cells.append(f"{imp_ms:>16,.1f}")
        print("  ".join(cells))

    # Aggregate per-pair summary.
    if len(backend_names) >= 2:
        print(f"\nAggregate over {len(results)} benched shapes "
              f"({total_log_calls:,} log calls):")
        # For each ordered (a, b) pair, report `a / b` so values > 1 mean
        # b is faster (because a took more time per call than b).
        pairs = []
        for i, a in enumerate(backend_names):
            for b in backend_names[i+1:]:
                pairs.append((a, b))
        for a, b in pairs:
            a_total = sum(
                (r["us"].get(a) or 0) * r["log_count"] for r in results
                if r["us"].get(a) and r["us"].get(b)
            )
            b_total = sum(
                (r["us"].get(b) or 0) * r["log_count"] for r in results
                if r["us"].get(a) and r["us"].get(b)
            )
            speedups = [
                r["us"][a] / r["us"][b] for r in results
                if r["us"].get(a) and r["us"].get(b)
            ]
            _summarize_speedups(
                label=f"{a:>6} vs {b:>6}",
                speedups=speedups,
                a_total=a_total, b_total=b_total,
                a_name=a, b_name=b,
                n_shapes_total=len(results),
            )


# ---------------------------------------------------------------------------
# Embedded shape literal — 333 unique prefill-only (T, cu_seqlens) tuples
# from prefill_gdr.log (decode segments with seqlen==1 stripped, then
# deduped with log_count summed). First-occurrence order from the source
# trace is preserved.
# Format: "<log_count>|<T>|<space_separated_cu_seqlens>"
# ---------------------------------------------------------------------------

_SHAPES_RAW = """
2844|5000|0 5000
3096|1000|0 1000
828|15000|0 5000 10000 15000
1764|10000|0 10000
288|7000|0 1000 2000 3000 4000 5000 6000 7000
288|20000|0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
1080|30000|0 10000 20000 30000
576|10000|0 5000 10000
432|3000|0 1000 2000 3000
576|20000|0 10000 20000
252|32767|0 10000 20000 30000 32767
252|32764|0 7233 17233 27233 32764
360|30000|0 5000 10000 15000 20000 25000 30000
216|32764|0 5000 10000 15000 20000 25000 30000 32764
216|32764|0 10000 20000 30000 32764
180|4469|0 4469
144|32767|0 5000 10000 15000 20000 25000 30000 32767
144|32761|0 5000 10000 15000 20000 25000 30000 32761
144|27236|0 2236 7236 12236 17236 22236 27236
252|10000|0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
144|32763|0 10000 20000 30000 32763
144|32760|0 7237 17237 27237 32760
144|32757|0 4477 14477 24477 32757
144|11720|0 1720 11720
144|32758|0 10000 20000 30000 32758
144|32755|0 7242 17242 27242 32755
144|32752|0 4487 14487 24487 32752
108|2233|0 2233
108|12239|0 2239 7239 12239
108|32759|0 5000 10000 15000 20000 25000 30000 32759
180|11000|0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000
108|7236|0 7236
108|32761|0 7236 17236 27236 32761
108|32758|0 4475 14475 24475 32758
108|32749|0 1735 11735 21735 31735 32749
108|32745|0 8986 18986 28986 32745
108|32742|0 6241 16241 26241 32742
108|23499|0 3499 13499 23499
72|32763|0 5000 10000 15000 20000 25000 30000 32763
72|32757|0 2237 7237 12237 17237 22237 27237 32237 32757
72|32750|0 4480 9480 14480 19480 24480 29480 32750
72|32744|0 1730 6730 11730 16730 21730 26730 31730 32744
72|32758|0 2236 7236 12236 17236 22236 27236 32236 32758
72|32751|0 4478 9478 14478 19478 24478 29478 32751
72|32745|0 1727 6727 11727 16727 21727 26727 31727 32745
72|32753|0 2241 7241 12241 17241 22241 27241 32241 32753
72|32746|0 4488 9488 14488 19488 24488 29488 32746
72|32740|0 1742 6742 11742 16742 21742 26742 31742 32740
72|32733|0 4002 9002 14002 19002 24002 29002 32733
72|32727|0 1269 6269 11269 16269 21269 26269 31269 32727
72|32720|0 3542 8542 13542 18542 23542 28542 32720
72|32714|0 822 5822 10822 15822 20822 25822 30822 32714
72|13108|0 3108 8108 13108
144|12000|0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000
108|9000|0 1000 2000 3000 4000 5000 6000 7000 8000 9000
72|32755|0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000 30000 31000 32000 32755
72|18245|0 245 1245 2245 3245 4245 5245 6245 7245 8245 9245 10245 11245 12245 13245 14245 15245 16245 17245 18245
180|32000|0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000 30000 31000 32000
72|18000|0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000
72|6000|0 1000 2000 3000 4000 5000 6000
72|21000|0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000
72|32761|0 4469 14469 24469 32761
72|32758|0 1708 11708 21708 31708 32758
72|32766|0 10000 20000 30000 32766
72|32763|0 7234 17234 27234 32763
72|32760|0 4471 14471 24471 32760
72|32757|0 1711 11711 21711 31711 32757
72|32727|0 10000 20000 30000 32727
72|32736|0 10000 20000 30000 32736
36|32761|0 2233 7233 12233 17233 22233 27233 32233 32761
36|9472|0 4472 9472
36|20000|0 5000 10000 15000 20000
36|3986|0 3986
36|8982|0 3982 8982
36|32760|0 5000 10000 15000 20000 25000 30000 32760
36|32754|0 2240 7240 12240 17240 22240 27240 32240 32754
36|32747|0 4486 9486 14486 19486 24486 29486 32747
36|21739|0 1739 6739 11739 16739 21739
36|32754|0 5000 10000 15000 20000 25000 30000 32754
36|32748|0 2246 7246 12246 17246 22246 27246 32246 32748
36|24498|0 4498 9498 14498 19498 24498
36|22241|0 2241 7241 12241 17241 22241
36|32748|0 5000 10000 15000 20000 25000 30000 32748
36|27252|0 2252 7252 12252 17252 22252 27252
36|32758|0 5000 10000 15000 20000 25000 30000 32758
36|32752|0 2242 7242 12242 17242 22242 27242 32242 32752
36|32745|0 4490 9490 14490 19490 24490 29490 32745
36|11745|0 1745 6745 11745
36|32757|0 5000 10000 15000 20000 25000 30000 32757
36|32751|0 2243 7243 12243 17243 22243 27243 32243 32751
36|32744|0 4492 9492 14492 19492 24492 29492 32744
36|6748|0 1748 6748
36|32751|0 5000 10000 15000 20000 25000 30000 32751
36|32745|0 2249 7249 12249 17249 22249 27249 32249 32745
36|9504|0 4504 9504
36|32755|0 2239 7239 12239 17239 22239 27239 32239 32755
36|32748|0 4484 9484 14484 19484 24484 29484 32748
36|32742|0 1736 6736 11736 16736 21736 26736 31736 32742
36|32735|0 3994 8994 13994 18994 23994 28994 32735
36|32729|0 1259 6259 11259 16259 21259 26259 31259 32729
36|32722|0 3530 8530 13530 18530 23530 28530 32722
36|32716|0 808 5808 10808 15808 20808 25808 30808 32716
36|23092|0 3092 8092 13092 18092 23092
36|32765|0 5000 10000 15000 20000 25000 30000 32765
36|32759|0 2235 7235 12235 17235 22235 27235 32235 32759
36|32752|0 4476 9476 14476 19476 24476 29476 32752
36|32746|0 1724 6724 11724 16724 21724 26724 31724 32746
36|32739|0 3978 8978 13978 18978 23978 28978 32739
36|32733|0 1239 6239 11239 16239 21239 26239 31239 32733
36|32726|0 3506 8506 13506 18506 23506 28506 32726
36|32720|0 780 5780 10780 15780 20780 25780 30780 32720
36|32713|0 3060 8060 13060 18060 23060 28060 32713
36|10347|0 347 5347 10347
36|32738|0 3982 8982 13982 18982 23982 28982 32738
36|32732|0 1244 6244 11244 16244 21244 26244 31244 32732
36|32725|0 3512 8512 13512 18512 23512 28512 32725
36|32719|0 787 5787 10787 15787 20787 25787 30787 32719
36|32712|0 3068 8068 13068 18068 23068 28068 32712
36|5356|0 356 5356
36|32737|0 3986 8986 13986 18986 23986 28986 32737
36|32731|0 1249 6249 11249 16249 21249 26249 31249 32731
36|32724|0 3518 8518 13518 18518 23518 28518 32724
36|32718|0 794 5794 10794 15794 20794 25794 30794 32718
36|32711|0 3076 8076 13076 18076 23076 28076 32711
36|365|0 365
36|32762|0 5000 10000 15000 20000 25000 30000 32762
36|32756|0 2238 7238 12238 17238 22238 27238 32238 32756
36|32749|0 4482 9482 14482 19482 24482 29482 32749
36|32743|0 1733 6733 11733 16733 21733 26733 31733 32743
36|32736|0 3990 8990 13990 18990 23990 28990 32736
36|32730|0 1254 6254 11254 16254 21254 26254 31254 32730
36|32723|0 3524 8524 13524 18524 23524 28524 32723
36|32717|0 801 5801 10801 15801 20801 25801 30801 32717
36|28084|0 3084 8084 13084 18084 23084 28084
36|25000|0 5000 10000 15000 20000 25000
36|32753|0 5000 10000 15000 20000 25000 30000 32753
36|32747|0 2247 7247 12247 17247 22247 27247 32247 32747
36|32740|0 4500 9500 14500 19500 24500 29500 32740
36|32734|0 1760 6760 11760 16760 21760 26760 31760 32734
36|32727|0 4026 9026 14026 19026 24026 29026 32727
36|32721|0 1299 6299 11299 16299 21299 26299 31299 32721
36|32714|0 3578 8578 13578 18578 23578 28578 32714
36|15864|0 864 5864 10864 15864
36|15000|0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000
36|14000|0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000
36|13000|0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000
36|19000|0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000
36|2000|0 1000 2000
36|27000|0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000
36|8000|0 1000 2000 3000 4000 5000 6000 7000 8000
36|4000|0 1000 2000 3000 4000
36|18950|0 8950 18950
36|8954|0 8954
36|21717|0 1717 11717 21717
36|11717|0 1717 11717
36|32754|0 8950 18950 28950 32754
36|32751|0 6196 16196 26196 32751
36|32748|0 3445 13445 23445 32748
36|32745|0 697 10697 20697 30697 32745
36|32741|0 7952 17952 27952 32741
36|15211|0 5211 15211
36|32755|0 1717 11717 21717 31717 32755
36|32751|0 8962 18962 28962 32751
36|32748|0 6211 16211 26211 32748
36|32745|0 3463 13463 23463 32745
36|32742|0 718 10718 20718 30718 32742
36|17976|0 7976 17976
36|32762|0 10000 20000 30000 32762
36|32759|0 7238 17238 27238 32759
36|32756|0 4479 14479 24479 32756
36|32753|0 1723 11723 21723 31723 32753
36|32749|0 8970 18970 28970 32749
36|32746|0 6221 16221 26221 32746
36|32743|0 3475 13475 23475 32743
36|30732|0 732 10732 20732 30732
36|32761|0 10000 20000 30000 32761
36|32758|0 7239 17239 27239 32758
36|32755|0 4481 14481 24481 32755
36|32752|0 1726 11726 21726 31726 32752
36|32748|0 8974 18974 28974 32748
36|32745|0 6226 16226 26226 32745
36|32742|0 3481 13481 23481 32742
36|20739|0 739 10739 20739
36|32750|0 1735 11735 21735 31735 32750
36|32748|0 8985 18985 28985 32748
36|32745|0 6237 16237 26237 32745
36|13492|0 3492 13492
36|32753|0 8954 18954 28954 32753
36|32750|0 6201 16201 26201 32750
36|32747|0 3451 13451 23451 32747
36|32744|0 704 10704 20704 30704 32744
36|32740|0 7960 17960 27960 32740
36|32737|0 5220 15220 25220 32737
36|32734|0 2483 12483 22483 32483 32734
36|32730|0 9749 19749 29749 32730
36|32727|0 7019 17019 27019 32727
36|32724|0 4292 14292 24292 32724
36|32721|0 1568 11568 21568 31568 32721
36|32717|0 8847 18847 28847 32717
36|32714|0 6130 16130 26130 32714
36|32711|0 3416 13416 23416 32711
36|30705|0 705 10705 20705 30705
36|32727|0 7273 17273 27273 32727
36|32727|0 4546 14546 24546 32727
36|32728|0 1819 11819 21819 31819 32728
36|32727|0 9091 19091 29091 32727
36|32727|0 6364 16364 26364 32727
36|32727|0 3637 13637 23637 32727
36|32728|0 910 10910 20910 30910 32728
36|32727|0 8182 18182 28182 32727
36|32727|0 5455 15455 25455 32727
36|32727|0 2728 12728 22728 32727
36|32724|0 7273 17273 27273 32724
36|32721|0 4549 14549 24549 32721
36|32718|0 1828 11828 21828 31828 32718
36|32714|0 9110 19110 29110 32714
36|32711|0 6396 16396 26396 32711
36|32708|0 3685 13685 23685 32708
36|977|0 977
36|27264|0 7264 17264 27264
36|32737|0 7264 17264 27264 32737
36|32737|0 4527 14527 24527 32737
36|32737|0 1790 11790 21790 31790 32737
36|32737|0 9053 19053 29053 32737
36|32737|0 6316 16316 26316 32737
36|32737|0 3579 13579 23579 32737
36|32737|0 842 10842 20842 30842 32737
36|32734|0 8105 18105 28105 32734
36|32731|0 5371 15371 25371 32731
36|32728|0 2640 12640 22640 32640 32728
36|32724|0 9912 19912 29912 32724
36|32721|0 7188 17188 27188 32721
36|32718|0 4467 14467 24467 32718
36|32715|0 1749 11749 21749 31749 32715
36|32711|0 9034 19034 29034 32711
36|32708|0 6323 16323 26323 32708
36|3615|0 3615
36|32729|0 10000 20000 30000 32729
36|7271|0 7271
36|32732|0 10000 20000 30000 32732
36|32732|0 7268 17268 27268 32732
36|32732|0 4536 14536 24536 32732
36|32733|0 1804 11804 21804 31804 32733
36|32732|0 9071 19071 29071 32732
36|32732|0 6339 16339 26339 32732
36|32732|0 3607 13607 23607 32732
36|32733|0 875 10875 20875 30875 32733
36|32732|0 8142 18142 28142 32732
36|32732|0 5410 15410 25410 32732
36|32730|0 2678 12678 22678 32678 32730
36|32726|0 9948 19948 29948 32726
36|32723|0 7222 17222 27222 32723
36|32720|0 4499 14499 24499 32720
36|32717|0 1779 11779 21779 31779 32717
36|32713|0 9062 19062 29062 32713
36|32710|0 6349 16349 26349 32710
36|23639|0 3639 13639 23639
36|32721|0 10000 20000 30000 32721
36|32721|0 7279 17279 27279 32721
36|32721|0 4558 14558 24558 32721
36|32722|0 1837 11837 21837 31837 32722
36|32721|0 9115 19115 29115 32721
36|32721|0 6394 16394 26394 32721
36|32722|0 3673 13673 23673 32722
36|32722|0 951 10951 20951 30951 32722
36|32721|0 8229 18229 28229 32721
36|32721|0 5508 15508 25508 32721
36|32722|0 2787 12787 22787 32722
36|32722|0 65 10065 20065 30065 32722
36|32721|0 7343 17343 27343 32721
36|32721|0 4622 14622 24622 32721
36|32718|0 1901 11901 21901 31901 32718
36|32714|0 9183 19183 29183 32714
36|32711|0 6469 16469 26469 32711
36|32708|0 3758 13758 23758 32708
36|1050|0 1050
36|32723|0 10000 20000 30000 32723
36|32723|0 7277 17277 27277 32723
36|32724|0 4554 14554 24554 32724
36|32724|0 1830 11830 21830 31830 32724
36|32723|0 9106 19106 29106 32723
36|32723|0 6383 16383 26383 32723
36|32724|0 3660 13660 23660 32724
36|32724|0 936 10936 20936 30936 32724
36|32723|0 8212 18212 28212 32723
36|32724|0 5489 15489 25489 32724
36|32724|0 2765 12765 22765 32724
36|32724|0 41 10041 20041 30041 32724
36|32723|0 7317 17317 27317 32723
36|32721|0 4594 14594 24594 32721
36|32718|0 1873 11873 21873 31873 32718
36|32714|0 9155 19155 29155 32714
36|32711|0 6441 16441 26441 32711
36|32708|0 3730 13730 23730 32708
36|1022|0 1022
36|32726|0 10000 20000 30000 32726
36|32727|0 7274 17274 27274 32727
36|32727|0 4547 14547 24547 32727
36|32727|0 1820 11820 21820 31820 32727
36|32726|0 9093 19093 29093 32726
36|32727|0 6367 16367 26367 32727
36|32727|0 3640 13640 23640 32727
36|32727|0 913 10913 20913 30913 32727
36|32727|0 8186 18186 28186 32727
36|32727|0 5459 15459 25459 32727
36|32727|0 2732 12732 22732 32727
36|32727|0 5 10005 20005 30005 32727
36|32724|0 7278 17278 27278 32724
36|32721|0 4554 14554 24554 32721
36|32718|0 1833 11833 21833 31833 32718
36|32714|0 9115 19115 29115 32714
36|32711|0 6401 16401 26401 32711
36|32708|0 3690 13690 23690 32708
36|982|0 982
36|32730|0 10000 20000 30000 32730
36|32730|0 7270 17270 27270 32730
36|32730|0 4540 14540 24540 32730
36|32730|0 1810 11810 21810 31810 32730
36|32730|0 9080 19080 29080 32730
36|32730|0 6350 16350 26350 32730
36|32730|0 3620 13620 23620 32730
36|32731|0 890 10890 20890 30890 32731
36|32730|0 8159 18159 28159 32730
36|32730|0 5429 15429 25429 32730
36|32730|0 2699 12699 22699 32699 32730
36|32727|0 9969 19969 29969 32727
36|32724|0 7242 17242 27242 32724
36|32721|0 4518 14518 24518 32721
36|32718|0 1797 11797 21797 31797 32718
36|32714|0 9079 19079 29079 32714
36|32711|0 6365 16365 26365 32711
36|32708|0 3654 13654 23654 32708
36|946|0 946
"""
if __name__ == "__main__":
    main()
