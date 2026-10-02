#!/usr/bin/env python3
"""Interleaved baseline/fused GEMM1 benchmark for the E64 prefill shape.

Run this only after a host-level GPU/KFD idle check.  Both cases execute from
the same working tree and process; ``AITER_FLYDSL_GEMM1_FUSED_QUANT`` selects
the original GEMM1 + standalone quant pipeline or the fused GEMM1 pipeline.
"""

from __future__ import annotations

import argparse
import importlib
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ENABLE_CK", "0")
os.environ.setdefault("AITER_MOE_EXPERT_BALANCE", "true")
os.environ.setdefault("AITER_LOG_MORE", "1")
os.environ.setdefault("AITER_USE_GROUPED_GEMM", "1")
os.environ.setdefault("AITER_GROUPED_DEBUG", "0")
os.environ.setdefault("AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE", "1")

torch = importlib.import_module("torch")
moe_test = importlib.import_module("my_code.test_flydsl_grouped_gemm_gfx1250")
ActivationType = importlib.import_module("aiter").ActivationType


def _named_kernel_us(trace_df, fragment: str) -> float | None:
    matches = []
    for record in trace_df.to_dict("records"):
        if fragment not in str(record.get("name", "")):
            continue
        try:
            value = float(record["device_time_avg"])
        except (KeyError, TypeError, ValueError):
            continue
        if value > 0:
            matches.append(value)
    if len(matches) != 1:
        return None
    return matches[0]


def _run_case(*, fused: bool, iters: int) -> dict[str, float | bool | None]:
    os.environ["AITER_FLYDSL_GEMM1_FUSED_QUANT"] = "1" if fused else "0"
    (
        out,
        reference,
        _gemm1_reference,
        _gemm1_hash,
        _gemm2_reference_hash,
        _gemm2_hash,
        e2e_us,
        _kernel_us,
        trace_df,
        _route_counts,
    ) = moe_test._run_grouped_via_fused_moe(
        experts=64,
        tokens=1536,
        topk=8,
        model_dim=7168,
        inter_dim=2048,
        data_format="a4w4",
        activation=ActivationType.Silu,
        use_bias=False,
        bench=True,
        kernel_bench=False,
        seed=0,
        warmup=5,
        iters=iters,
        const_init=0.0,
    )
    timings, _kernels, error = moe_test._extract_profiled_gemm_timings(
        trace_df=trace_df,
        data_format="a4w4",
        activation=ActivationType.Silu,
        experts=64,
        model_dim=7168,
        inter_dim=2048,
    )
    if error:
        raise RuntimeError(error)
    quant_us = _named_kernel_us(trace_df, "moe_quant_preshuffled_a_fd2048_")
    logits_diff = moe_test._logits_diff(out, reference)
    rel_l2 = moe_test._rel_l2(out, reference)
    return {
        "gemm1_us": timings["gemm1_us"],
        "quant_us": quant_us,
        "gemm1_quant_us": (
            timings["gemm1_us"] + (quant_us or 0.0)
            if timings["gemm1_us"] is not None
            else None
        ),
        "gemm2_us": timings["gemm2_us"],
        "e2e_us": e2e_us,
        "logits_diff": logits_diff,
        "rel_l2": rel_l2,
        "passed": logits_diff < moe_test.LOGITS_DIFF_TOL,
    }


def _fmt(values: list[float]) -> str:
    return ", ".join(f"{value:.3f}" for value in values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.rounds < 1 or args.iters < 2:
        parser.error("--rounds must be >= 1 and --iters must be >= 2")

    results: dict[str, list[dict[str, float | bool | None]]] = {
        "baseline": [],
        "fused": [],
    }
    for round_index in range(args.rounds):
        order = ("baseline", "fused") if round_index % 2 == 0 else ("fused", "baseline")
        for case in order:
            print(f"\n===== round={round_index + 1} case={case} =====", flush=True)
            result = _run_case(fused=case == "fused", iters=args.iters)
            results[case].append(result)
            print(f"AB_RESULT case={case} round={round_index + 1} {result}", flush=True)

    print("\n| case | GEMM1 samples (us) | quant samples (us) | GEMM1+quant median (us) | GEMM2 median (us) | MoE median (us) | pass |")
    print("|---|---|---|---:|---:|---:|:---:|")
    for case in ("baseline", "fused"):
        rows = results[case]
        gemm1 = [float(row["gemm1_us"]) for row in rows]
        quant = [float(row["quant_us"] or 0.0) for row in rows]
        combined = [float(row["gemm1_quant_us"]) for row in rows]
        gemm2 = [float(row["gemm2_us"]) for row in rows]
        e2e = [float(row["e2e_us"]) for row in rows]
        passed = all(bool(row["passed"]) for row in rows)
        print(
            f"| {case} | {_fmt(gemm1)} | {_fmt(quant)} | "
            f"{statistics.median(combined):.3f} | "
            f"{statistics.median(gemm2):.3f} | "
            f"{statistics.median(e2e):.3f} | {passed} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
