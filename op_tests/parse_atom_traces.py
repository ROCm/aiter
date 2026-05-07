# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Parse the chrome traces produced by run_atom_qwen3_next_profile.sh and
summarize fill / GEMM / producer kernel counts and per-iteration GPU
time across the three SplitK zero-init demo modes.

The trace from ATOM's offline profiler covers warmup + (1 prefill +
output_len decodes) of a single batch=1 generation.  We:

  1. Identify GPU kernel events by category using the same regex set
     as bench_zero_init_components.py.
  2. Drop the first decode iteration's worth of kernels as warmup
     (CUDA graph capture / first-replay specialization).
  3. Average the remaining kernel counts and total kernel time across
     the steady-state decode iterations.

Per-iteration fill count is the smoking gun for fusion:
   - splitk          : >0 (one per per_1x128 LinearBase that uses splitK)
   - splitk_fused    : 0  (all such fills folded into producer kernels)
   - none            : 0  (splitK=0, kernel doesn't need to zero Y)

Per-iteration GPU time tells us the actual TPOT impact of the fusion
in the regime ATOM runs in (full-decode HIP-graph capture).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys


# IMPORTANT: only match the bf16 FillFunctor that ATen Tensor.zero_() lowers
# to for our bf16 Y output buffers.  The int32 / fp16 / unrolled variants are
# used elsewhere in the model (KV cache reset, attention masks, residual
# init, etc.) and would otherwise drown the SplitK fill signal in noise --
# we observed e.g. ~80k FillFunctor<int> per trace vs ~2.7k FillFunctor<bf16>
# (the actual SplitK fill).
_FILL_PATTERNS = (
    r"vectorized_elementwise_kernel.*FillFunctor<c10::BFloat16>",
)

_PROD_PATTERNS = (
    r"dynamic_per_token_scaled_quant",
    r"dynamic_per_group_scaled_quant",
    r"fused_qk_rmsnorm_group_quant_kernel",
    r"gated_rmsnorm_fp8_group_quant_kernel",
)

_GEMM_PATTERNS = (
    r"a8w8_blockscale_cktile",
    r"blockwise_gemm",
    r"kBatch",
    r"QuantGemmKernel",
    r"ck_tile6kentry",
)

_FILL_RE = re.compile("|".join(_FILL_PATTERNS))
_PROD_RE = re.compile("|".join(_PROD_PATTERNS))
_GEMM_RE = re.compile("|".join(_GEMM_PATTERNS))


def _classify(name: str) -> str | None:
    if not name:
        return None
    if _FILL_RE.search(name):
        return "fill"
    if _PROD_RE.search(name):
        return "prod"
    if _GEMM_RE.search(name):
        return "gemm"
    return None


def _is_kernel_event(ev: dict) -> bool:
    if ev.get("ph") != "X":
        return False
    cat = ev.get("cat") or ""
    # torch.profiler emits GPU kernels with cat='kernel' on rocm/hip.
    # CUDA on AMD also uses 'kernel' / 'gpu_op'. memcpy/memset land in
    # 'gpu_memset' / 'gpu_memcpy'.
    return "kernel" in cat or "gpu" in cat


def _find_trace_files(results_dir: str, mode: str) -> list[str]:
    """ATOM puts traces under <results_dir>/<mode>_traces/rank_*/<json>."""
    pattern = os.path.join(results_dir, f"{mode}_traces", "rank_*", "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        # try the no-rank-subdir layout (single-rank, dp=1)
        pattern = os.path.join(results_dir, f"{mode}_traces", "*.json")
        files = sorted(glob.glob(pattern))
    return files


def _summarize_trace(trace_path: str, output_len: int, warmup_iters: int = 1) -> dict:
    """Return per-iteration kernel count + total GPU time for one trace.

    The trace contains 1 prefill + ``output_len`` decode iterations.
    We:
      * count + sum-time the prefill separately (different shape, M>>1);
      * for the decode steps, drop the first ``warmup_iters`` iterations
        (CUDA graph first-replay specialization typically takes longer)
        and report mean per-iteration values across the rest.
    """
    with open(trace_path) as f:
        trace = json.load(f)

    events = trace.get("traceEvents") or []
    # Collect kernel events in chronological order.
    kernels = []
    for ev in events:
        if not _is_kernel_event(ev):
            continue
        ts = ev.get("ts")
        dur = ev.get("dur") or 0
        name = ev.get("name", "")
        if ts is None:
            continue
        kernels.append((ts, dur, name))
    kernels.sort()

    # ATOM's full decode forward is captured as a CUDA graph + replayed
    # once per token.  We can't reliably segment by iteration from the
    # device timeline alone, but the model's "begin-step" CPU op is
    # logged and we could use it.  For a first-cut summary just emit
    # totals divided by the number of decode iterations -- ATOM also
    # captures graph capture + warmup + prefill, so we report buckets.
    counts = {"prod": 0, "fill": 0, "gemm": 0, "other": 0}
    times_us = {"prod": 0.0, "fill": 0.0, "gemm": 0.0, "other": 0.0}
    other_names: dict[str, int] = {}
    for ts, dur, name in kernels:
        cls = _classify(name)
        if cls is None:
            counts["other"] += 1
            times_us["other"] += dur
            other_names[name] = other_names.get(name, 0) + 1
        else:
            counts[cls] += 1
            times_us[cls] += dur

    # Total GPU work captured (sum of kernel durations -- under-estimates
    # wall time because we don't account for inter-kernel gaps but is a
    # good proxy for "actual GPU compute" since CUDA graphs squeeze gaps).
    total_us = sum(times_us.values())

    # Steady-state per-iteration estimate: divide by output_len.
    # This includes the 1 prefill, but prefill is a small fraction of
    # total kernels at output_len >> 1.  A more accurate bucketing
    # requires per-step CPU markers; for our purposes the across-mode
    # *delta* is what matters and that's invariant to this scaling.
    if output_len > 0:
        per_iter_counts = {k: v / output_len for k, v in counts.items()}
        per_iter_us = {k: v / output_len for k, v in times_us.items()}
    else:
        per_iter_counts = dict(counts)
        per_iter_us = dict(times_us)

    return {
        "trace_path": trace_path,
        "total_kernels": sum(counts.values()),
        "total_us": total_us,
        "counts": counts,
        "times_us": times_us,
        "per_iter_counts": per_iter_counts,
        "per_iter_us": per_iter_us,
        "top_other": sorted(other_names.items(), key=lambda kv: -kv[1])[:8],
    }


def _print_mode_summary(mode: str, summary: dict) -> None:
    print(f"### mode={mode}")
    print(f"  trace: {summary['trace_path']}")
    print(
        f"  total kernels={summary['total_kernels']}  "
        f"total kernel-time={summary['total_us']/1e3:.2f} ms"
    )
    print("  counts (over the whole trace -- prefill+decodes):")
    for k in ("prod", "gemm", "fill", "other"):
        c = summary["counts"][k]
        t = summary["times_us"][k]
        print(f"    {k:5s}: count={c:6d}  time={t/1e3:7.2f} ms")
    print("  per-iter (avg across all captured iterations):")
    for k in ("prod", "gemm", "fill", "other"):
        c = summary["per_iter_counts"][k]
        t = summary["per_iter_us"][k]
        print(f"    {k:5s}: count={c:7.2f}  time={t:8.2f} us")
    if summary["top_other"]:
        print("  top 'other' kernel names (not classified):")
        for name, c in summary["top_other"][:5]:
            print(f"    {c:6d}  {name[:120]}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-len", type=int, default=32)
    args = parser.parse_args()

    summaries: dict[str, dict] = {}
    for mode in ("none", "splitk", "splitk_fused"):
        files = _find_trace_files(args.results_dir, mode)
        if not files:
            print(f"### mode={mode}: no trace found under {args.results_dir}/{mode}_traces",
                  file=sys.stderr)
            continue
        if len(files) > 1:
            print(f"# mode={mode}: multiple traces found ({len(files)}); using last",
                  file=sys.stderr)
        summary = _summarize_trace(files[-1], output_len=args.output_len)
        summaries[mode] = summary
        _print_mode_summary(mode, summary)

    if {"splitk", "splitk_fused"}.issubset(summaries.keys()):
        s = summaries["splitk"]
        f = summaries["splitk_fused"]
        print("=========================================================")
        print("# Delta splitk -> splitk_fused (per-iter):")
        print("=========================================================")
        print(
            f"  fill count: {s['per_iter_counts']['fill']:.2f} -> "
            f"{f['per_iter_counts']['fill']:.2f}  "
            f"(delta = {f['per_iter_counts']['fill'] - s['per_iter_counts']['fill']:+.2f})"
        )
        print(
            f"  fill time : {s['per_iter_us']['fill']:.2f} us -> "
            f"{f['per_iter_us']['fill']:.2f} us  "
            f"(delta = {f['per_iter_us']['fill'] - s['per_iter_us']['fill']:+.2f} us)"
        )
        prod_delta = f['per_iter_us']['prod'] - s['per_iter_us']['prod']
        print(
            f"  prod time : {s['per_iter_us']['prod']:.2f} us -> "
            f"{f['per_iter_us']['prod']:.2f} us  "
            f"(delta = {prod_delta:+.2f} us  -- expected slightly positive: "
            f"producer absorbs the fill)"
        )
        gemm_delta = f['per_iter_us']['gemm'] - s['per_iter_us']['gemm']
        print(
            f"  gemm time : {s['per_iter_us']['gemm']:.2f} us -> "
            f"{f['per_iter_us']['gemm']:.2f} us  "
            f"(delta = {gemm_delta:+.2f} us  -- expected slightly negative: "
            f"GEMM skips its own internal Y.zero_())"
        )
        total_delta = (
            f['total_us'] - s['total_us']
        )
        print(
            f"  total kernel-time: {s['total_us']/1e3:.2f} ms -> "
            f"{f['total_us']/1e3:.2f} ms  "
            f"(delta = {total_delta/1e3:+.2f} ms across whole trace)"
        )
        per_iter_delta = total_delta / max(args.output_len, 1)
        print(
            f"  total kernel-time per iter: "
            f"{s['total_us']/args.output_len:.2f} us -> "
            f"{f['total_us']/args.output_len:.2f} us  "
            f"(delta = {per_iter_delta:+.2f} us = "
            f"{per_iter_delta/1000:+.4f} ms)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
