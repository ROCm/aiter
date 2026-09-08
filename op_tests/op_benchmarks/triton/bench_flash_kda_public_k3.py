# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run the formal public-K3 HIP-versus-Triton performance matrix.

This is a deliberately small orchestration layer over
``bench_flash_kda_native.py``.  That benchmark remains the implementation of
input generation, public-wrapper calls, correctness checking, independent
ROCm graph capture, and alternating paired timing.  This runner fixes the
promotion matrix and adds deterministic stratified-bootstrap evidence plus a
fail-closed, three-seed performance decision.

The formal matrix has 26 logical cells at each of seeds 42, 43, and 44:

* single lengths 128 through 16K, including 4K, crossed with literal
  ``initial_state=None`` and nonzero FP32 resume state;
* the remaining core batch/ragged cases;
* all mixed-production cases up to the 64-sequence production limit; and
* both 1024/1025 mixed-boundary cases.

Every cell uses Hq=HV=12, the zero-environment public wrapper for HIP, forced
``backend="triton"`` for the comparator, graph replay, exactly 20 warmup rounds,
120 measured rounds, and 10,000 bootstrap resamples.  The process exits nonzero
after writing evidence
unless every logical cell satisfies all four formal criteria documented in
``_performance_gate``.

``--print-plan`` and ``--static-self-test`` are CPU-only and intentionally do
not import PyTorch or AITER.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA = "flash-kda-public-k3-formal-matrix-v1"
SEEDS = (42, 43, 44)
HEADS = 12
VALUE_HEADS = 12
SINGLE_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
STATE_VARIANTS = ("fresh-none", "resume-fp32")
MIXED_PRODUCTION_DECODES = (7, 8, 32, 63)
FORMAL_MIN_WARMUP = 20
FORMAL_MIN_REPEAT = 120
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260820
MAX_TOLERANCE = 0.04
PUBLIC_BACKEND = "public-zero-env"
TRITON_BACKEND = "forced-triton"
EXPECTED_PLAN_SHA256 = (
    "0aa58828638da3a2d11f7038333e6cce842a39ba0754799c803be6b859ceb464"
)


def _is_kda_control(name: str) -> bool:
    """Return whether an environment variable can alter KDA routing/code."""

    return "KDA" in name.upper() or name == "AITER_TRITON_ONLY"


def _clear_kda_environment() -> tuple[str, ...]:
    """Establish the public zero-environment contract before AITER imports."""

    removed = []
    for name in tuple(os.environ):
        if _is_kda_control(name):
            removed.append(name)
            os.environ.pop(name, None)
    return tuple(sorted(removed))


_REMOVED_KDA_ENVIRONMENT = _clear_kda_environment()
_RUNTIME_LOADED = False


@dataclass(frozen=True)
class CaseSpec:
    logical_name: str
    source_name: str
    family: str
    source_suite: str
    seq_lens: tuple[int, ...]
    resume: bool
    resume_mask: tuple[bool, ...] | None
    state_variant: str

    @property
    def max_seqlen_upper_bound(self) -> int:
        return max(self.seq_lens)

    @property
    def initial_state_is_none(self) -> bool:
        return self.state_variant == "fresh-none"


def _mixed_source(decodes: int) -> tuple[str, tuple[int, ...], tuple[bool, ...]]:
    name = f"mixed-{decodes}d-budget16k"
    return name, (1,) * decodes + (16384 - decodes,), (True,) * decodes + (False,)


def _fixed_specs() -> tuple[CaseSpec, ...]:
    specs: list[CaseSpec] = []
    source_single_names = {
        128: "single-128",
        256: "single-256",
        512: "single-512",
        1024: "single-1k",
        2048: "single-2k",
        4096: "single-4096",
        8192: "single-8k",
        16384: "single-16k",
    }
    for length in SINGLE_LENGTHS:
        for state_variant in STATE_VARIANTS:
            specs.append(
                CaseSpec(
                    logical_name=f"single-{length}-{state_variant}",
                    source_name=source_single_names[length],
                    family="single",
                    source_suite="core+single-4k-extension",
                    seq_lens=(length,),
                    resume=state_variant == "resume-fp32",
                    resume_mask=None,
                    state_variant=state_variant,
                )
            )

    for source_name, seq_lens in (
        ("batch-16x1k", (1024,) * 16),
        ("batch-64x256", (256,) * 64),
        ("ragged-16k", (127, 255, 511, 1023, 2047, 3073, 4095, 5253)),
    ):
        specs.append(
            CaseSpec(
                logical_name=f"{source_name}-fresh-none",
                source_name=source_name,
                family="core-additional",
                source_suite="core",
                seq_lens=seq_lens,
                resume=False,
                resume_mask=None,
                state_variant="fresh-none",
            )
        )
    specs.append(
        CaseSpec(
            logical_name="resume-4x4k-resume-fp32",
            source_name="resume-4x4k",
            family="core-additional",
            source_suite="core",
            seq_lens=(4096,) * 4,
            resume=True,
            resume_mask=None,
            state_variant="resume-fp32",
        )
    )

    for decodes in MIXED_PRODUCTION_DECODES:
        source_name, seq_lens, resume_mask = _mixed_source(decodes)
        specs.append(
            CaseSpec(
                logical_name=f"{source_name}-state-fp32",
                source_name=source_name,
                family="mixed-production",
                source_suite="mixed-production",
                seq_lens=seq_lens,
                resume=False,
                resume_mask=resume_mask,
                state_variant="mixed-fp32",
            )
        )

    for source_name, prefill in (
        ("mixed-15d-prefill-1024", 1024),
        ("mixed-15d-prefill-1025", 1025),
    ):
        specs.append(
            CaseSpec(
                logical_name=f"{source_name}-state-fp32",
                source_name=source_name,
                family="mixed-boundary",
                source_suite="mixed-boundary",
                seq_lens=(1,) * 15 + (prefill,),
                resume=False,
                resume_mask=(True,) * 15 + (False,),
                state_variant="mixed-fp32",
            )
        )
    return tuple(specs)


ALL_SPECS = _fixed_specs()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Durably append one completed-cell or failure record as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _active_kda_environment() -> dict[str, str]:
    return {
        name: os.environ[name]
        for name in sorted(os.environ)
        if _is_kda_control(name)
    }


def _load_benchmark() -> Any:
    """Load the mature GPU benchmark only after CPU-only modes are resolved."""

    global _RUNTIME_LOADED
    if _active_kda_environment():
        raise RuntimeError(
            f"KDA environment was reintroduced: {_active_kda_environment()}"
        )
    module = importlib.import_module(
        "op_tests.op_benchmarks.triton.bench_flash_kda_native"
    )
    expected = (
        _REPO_ROOT / "op_tests/op_benchmarks/triton/bench_flash_kda_native.py"
    ).resolve()
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError(
            f"benchmark import escaped this checkout: {module.__file__} != {expected}"
        )
    _RUNTIME_LOADED = True
    return module


def _source_tuple(case: Any) -> tuple[Any, ...]:
    return (
        case.name,
        tuple(case.seq_lens),
        bool(case.resume),
        None if case.resume_mask is None else tuple(case.resume_mask),
    )


def _spec_source_tuple(spec: CaseSpec) -> tuple[Any, ...]:
    return (spec.source_name, spec.seq_lens, spec.resume, spec.resume_mask)


def _assert_runtime_suite_identity(benchmark: Any) -> dict[str, Any]:
    """Fail if the fixed promotion matrix drifted from its source suites."""

    expected_by_name: dict[str, tuple[Any, ...]] = {}
    for suite_name in ("core", "mixed-production", "mixed-boundary"):
        for case in benchmark.CASE_SUITES[suite_name]:
            expected_by_name[case.name] = _source_tuple(case)

    checked: dict[str, tuple[Any, ...]] = {}
    for spec in ALL_SPECS:
        # The requested 4K single is an explicit extension of historical core.
        if spec.source_name == "single-4096":
            continue
        expected = expected_by_name.get(spec.source_name)
        if expected is None:
            raise RuntimeError(f"source case disappeared: {spec.source_name}")
        actual = _spec_source_tuple(spec)
        # Resume singles deliberately add state to the established fresh shape.
        matches = (
            actual[:2] == expected[:2]
            if spec.family == "single"
            else actual == expected
        )
        if not matches:
            raise RuntimeError(
                f"source case drift for {spec.source_name}: {actual} != {expected}"
            )
        checked[spec.source_name] = actual
    return {
        "runtime_suite_identity_verified": True,
        "verified_source_cases": sorted(checked),
        "single_4k_explicit_extension": True,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("empty percentile input")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _stable_seed(label: str, root_seed: int = BOOTSTRAP_SEED) -> int:
    digest = hashlib.sha256(f"{root_seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _bootstrap(
    strata: dict[str, list[float]], *, label: str, resamples: int
) -> dict[str, Any]:
    """Percentile-bootstrap paired deltas within each invocation-order stratum."""

    if resamples <= 0 or not strata or any(not values for values in strata.values()):
        raise ValueError("bootstrap requires positive resamples and nonempty strata")
    flat = [value for values in strata.values() for value in values]
    if any(not math.isfinite(value) for value in flat):
        raise ValueError("bootstrap input must be finite")

    rng = random.Random(_stable_seed(label))
    medians: list[float] = []
    means: list[float] = []
    wins: list[float] = []
    for _ in range(resamples):
        sample: list[float] = []
        for values in strata.values():
            sample.extend(rng.choice(values) for _ in values)
        medians.append(statistics.median(sample))
        means.append(statistics.fmean(sample))
        wins.append(sum(value < 0.0 for value in sample) / len(sample))
    return {
        "delta_definition": "public-hip-minus-public-triton; negative is HIP win",
        "strata": {key: len(values) for key, values in sorted(strata.items())},
        "samples": len(flat),
        "point_estimate": {
            "p50_delta_us": statistics.median(flat),
            "mean_delta_us": statistics.fmean(flat),
            "hip_win_fraction": sum(value < 0.0 for value in flat) / len(flat),
        },
        "bootstrap_95pct_ci": {
            "method": "stratified percentile bootstrap of paired rounds",
            "resamples": resamples,
            "p50_delta_us": [_percentile(medians, 0.025), _percentile(medians, 0.975)],
            "mean_delta_us": [_percentile(means, 0.025), _percentile(means, 0.975)],
            "hip_win_fraction": [_percentile(wins, 0.025), _percentile(wins, 0.975)],
        },
    }


def _paired_raw_rows(
    raw_rows: list[dict[str, object]], *, spec: CaseSpec, repeat: int
) -> list[dict[str, object]]:
    """Normalize the mature benchmark's invocation rows and attach pair data."""

    aliases = {"native": PUBLIC_BACKEND, "triton": TRITON_BACKEND}
    by_round: dict[int, list[dict[str, object]]] = {}
    for raw in raw_rows:
        backend = str(raw["backend"])
        if backend not in aliases:
            raise RuntimeError(f"{spec.logical_name}: unexpected backend {backend}")
        row = dict(raw)
        row["backend"] = aliases[backend]
        row["logical_name"] = spec.logical_name
        row["source_case"] = spec.source_name
        row["state_variant"] = spec.state_variant
        row["initial_state_literal_none"] = spec.initial_state_is_none
        by_round.setdefault(int(row["round"]), []).append(row)

    if sorted(by_round) != list(range(repeat)):
        raise RuntimeError(f"{spec.logical_name}: incomplete measured rounds")
    normalized: list[dict[str, object]] = []
    for round_index in range(repeat):
        pair = by_round[round_index]
        if len(pair) != 2 or {row["backend"] for row in pair} != {
            PUBLIC_BACKEND,
            TRITON_BACKEND,
        }:
            raise RuntimeError(f"{spec.logical_name}: malformed round {round_index}")
        if sorted(int(row["order"]) for row in pair) != [0, 1]:
            raise RuntimeError(f"{spec.logical_name}: invalid invocation positions")
        pair.sort(key=lambda row: int(row["order"]))
        expected_first = PUBLIC_BACKEND if round_index % 2 == 0 else TRITON_BACKEND
        if pair[0]["backend"] != expected_first:
            raise RuntimeError(
                f"{spec.logical_name}: round {round_index} did not alternate order"
            )
        public = next(row for row in pair if row["backend"] == PUBLIC_BACKEND)
        triton = next(row for row in pair if row["backend"] == TRITON_BACKEND)
        public_ms = float(public["latency_ms"])
        triton_ms = float(triton["latency_ms"])
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (public_ms, triton_ms)
        ):
            raise RuntimeError(
                f"{spec.logical_name}: invalid latency in round {round_index}"
            )
        delta_us = (public_ms - triton_ms) * 1000.0
        speedup = triton_ms / public_ms
        for row in pair:
            row["paired_public_minus_triton_us"] = delta_us
            row["paired_triton_over_public_speedup"] = speedup
            row["paired_public_win"] = public_ms < triton_ms
            normalized.append(row)
    return normalized


def _paired_strata(
    result: dict[str, Any], prefix: str = "order"
) -> dict[str, list[float]]:
    strata: dict[str, list[float]] = {}
    for row in result["raw_timing_samples"]:
        if row["backend"] != PUBLIC_BACKEND:
            continue
        order = int(row["order"])
        strata.setdefault(f"{prefix}-{order}", []).append(
            float(row["paired_public_minus_triton_us"])
        )
    return strata


def _summary_rows(
    rows: list[dict[str, object]], spec: CaseSpec
) -> list[dict[str, object]]:
    aliases = {"native": PUBLIC_BACKEND, "triton": TRITON_BACKEND}
    normalized = []
    for source in rows:
        row = dict(source)
        row["backend"] = aliases[str(source["backend"])]
        row["logical_name"] = spec.logical_name
        row["source_case"] = spec.source_name
        row["state_variant"] = spec.state_variant
        row["initial_state_literal_none"] = spec.initial_state_is_none
        normalized.append(row)
    return normalized


def _position_counts(raw_rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    counts = {
        PUBLIC_BACKEND: {"first": 0, "second": 0},
        TRITON_BACKEND: {"first": 0, "second": 0},
    }
    for row in raw_rows:
        position = "first" if int(row["order"]) == 0 else "second"
        counts[str(row["backend"])][position] += 1
    return counts


def _run_one(
    benchmark: Any,
    spec: CaseSpec,
    *,
    seed: int,
    warmup: int,
    repeat: int,
    tolerance: float,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    active_environment = _active_kda_environment()
    if active_environment:
        raise RuntimeError(
            f"{spec.logical_name}: KDA environment is not clean at cell start: "
            f"{active_environment}"
        )
    case = benchmark.Case(
        spec.logical_name,
        spec.seq_lens,
        resume=spec.resume,
        resume_mask=spec.resume_mask,
    )
    raw_rows: list[dict[str, object]] = []
    rows = benchmark._benchmark_case(
        case,
        heads=HEADS,
        value_heads=VALUE_HEADS,
        selected=("native", "explicit-native", "triton"),
        warmup=warmup,
        repeat=repeat,
        seed=seed,
        check=True,
        tolerance=tolerance,
        execution="graph",
        public_k3=True,
        initial_state_none=spec.initial_state_is_none,
        check_input_state_immutability=True,
        timed_selected=("native", "triton"),
        audit_graph_routes=True,
        raw_rows=raw_rows,
    )
    rows = _summary_rows(rows, spec)
    raw_rows = _paired_raw_rows(raw_rows, spec=spec, repeat=repeat)
    by_backend = {str(row["backend"]): row for row in rows}
    if set(by_backend) != {PUBLIC_BACKEND, TRITON_BACKEND}:
        raise RuntimeError(f"{spec.logical_name}: benchmark result backends changed")
    public = by_backend[PUBLIC_BACKEND]
    triton = by_backend[TRITON_BACKEND]
    if public.get("public_default_bitwise_native") is not True:
        raise RuntimeError(
            f"{spec.logical_name}: public default was not proven bitwise native"
        )
    if public.get("input_initial_state_unchanged") is not True:
        raise RuntimeError(
            f"{spec.logical_name}: initial-state immutability was not proven"
        )
    route_audit = public.pop("graph_route_audit", None)
    if not isinstance(route_audit, dict) or not all(
        route_audit.get(field) is True
        for field in (
            "all_routes_verified",
            "graphs_independent",
            "streams_independent",
            "public_explicit_graph_signatures_equal",
        )
    ):
        raise RuntimeError(f"{spec.logical_name}: graph route audit failed")
    for backend, row in by_backend.items():
        if row.get("output_contract_verified") is not True:
            raise RuntimeError(
                f"{spec.logical_name}/{backend}: output contract was not verified"
            )
        for field in (
            "graph_eager_output_bitwise_equal",
            "graph_eager_final_state_bitwise_equal",
            "final_graph_eager_output_bitwise_equal",
            "final_graph_eager_final_state_bitwise_equal",
        ):
            if row.get(field) is not True:
                raise RuntimeError(f"{spec.logical_name}/{backend}: {field} failed")
    if public.get("input_resume_mask_verified") is not True:
        raise RuntimeError(
            f"{spec.logical_name}: initial-state content mode was not verified"
        )
    if _active_kda_environment():
        raise RuntimeError(
            f"{spec.logical_name}: KDA environment changed during the cell: "
            f"{_active_kda_environment()}"
        )

    positions = _position_counts(raw_rows)
    expected_positions = {"first": repeat // 2, "second": repeat // 2}
    if any(counts != expected_positions for counts in positions.values()):
        raise RuntimeError(
            f"{spec.logical_name}: paired invocation positions unbalanced: {positions}"
        )
    timing = {
        "samples_per_backend": repeat,
        "position_counts": positions,
        "public_p50_ms": float(public["latency_median_ms"]),
        "triton_p50_ms": float(triton["latency_median_ms"]),
        "triton_over_public_speedup_from_p50": float(public["speedup_vs_triton"]),
        "paired_triton_over_public_speedup_p50": float(public["paired_speedup_median"]),
        "public_win_fraction": float(public["paired_win_fraction"]),
    }
    result: dict[str, Any] = {
        "case": spec.logical_name,
        "logical_name": spec.logical_name,
        "source_case": spec.source_name,
        "family": spec.family,
        "source_suite": spec.source_suite,
        "seq_lens": list(spec.seq_lens),
        "seed": seed,
        "heads": HEADS,
        "value_heads": VALUE_HEADS,
        "state_variant": spec.state_variant,
        "initial_state_contract": {
            "literal_none": spec.initial_state_is_none,
            "materialized_dtype": (
                None if spec.initial_state_is_none else "torch.float32"
            ),
            "resume_mask": None if spec.resume_mask is None else list(spec.resume_mask),
        },
        "max_seqlen_upper_bound": spec.max_seqlen_upper_bound,
        "execution": "graph",
        "timed_callable_contract": {
            "hip": "chunk_kimi_delta_attn(...), backend keyword omitted",
            "triton": "chunk_kimi_delta_attn(..., backend='triton')",
            "max_seqlen_upper_bound": spec.max_seqlen_upper_bound,
        },
        "timing": timing,
        "summary_rows": rows,
        "raw_timing_samples": raw_rows,
        "graph_route_audit": route_audit,
        "all_input_state_immutability_checks_passed": True,
        "all_contract_checks_passed": True,
    }
    result["paired_bootstrap"] = _bootstrap(
        _paired_strata(result),
        label=f"{spec.logical_name}:seed-{seed}",
        resamples=bootstrap_resamples,
    )
    return result


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite: {value!r}")
    return result


def _gate_checks(speedup: Any, win_fraction: Any, ci_high_us: Any) -> dict[str, Any]:
    speedup_value = _finite_float(speedup, "p50 speedup")
    win_value = _finite_float(win_fraction, "paired HIP win fraction")
    ci_value = _finite_float(ci_high_us, "p50 delta CI upper bound")
    checks = {
        "p50_speedup": speedup_value,
        "p50_speedup_strictly_above_one": speedup_value > 1.0,
        "paired_hip_win_fraction": win_value,
        "paired_hip_win_fraction_strictly_above_half": win_value > 0.5,
        "p50_delta_95pct_ci_high_us": ci_value,
        "p50_delta_ci_upper_strictly_negative": ci_value < 0.0,
    }
    checks["passed"] = all(
        (
            checks["p50_speedup_strictly_above_one"],
            checks["paired_hip_win_fraction_strictly_above_half"],
            checks["p50_delta_ci_upper_strictly_negative"],
        )
    )
    return checks


def _cross_seed_summary(
    results: list[dict[str, Any]], resamples: int
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["logical_name"], []).append(result)

    cells: list[dict[str, Any]] = []
    for spec in ALL_SPECS:
        seed_results = grouped.get(spec.logical_name, [])
        if sorted(result["seed"] for result in seed_results) != list(SEEDS):
            raise RuntimeError(f"{spec.logical_name}: incomplete promotion seed set")
        strata: dict[str, list[float]] = {}
        per_seed = []
        for result in sorted(seed_results, key=lambda item: item["seed"]):
            strata.update(
                _paired_strata(result, prefix=f"seed-{result['seed']}-order")
            )
            seed_ci = result["paired_bootstrap"]["bootstrap_95pct_ci"][
                "p50_delta_us"
            ]
            seed_gate = _gate_checks(
                result["timing"]["triton_over_public_speedup_from_p50"],
                result["timing"]["public_win_fraction"],
                seed_ci[1],
            )
            result["performance_gate"] = seed_gate
            result["performance_gate_evaluated"] = True
            result["performance_gate_passed"] = seed_gate["passed"]
            per_seed.append(
                {
                    "seed": result["seed"],
                    "speedup_from_p50": result["timing"][
                        "triton_over_public_speedup_from_p50"
                    ],
                    "paired_speedup_p50": result["timing"][
                        "paired_triton_over_public_speedup_p50"
                    ],
                    "hip_win_fraction": result["timing"]["public_win_fraction"],
                    "p50_delta_us_ci": seed_ci,
                    "performance_gate": seed_gate,
                }
            )
        aggregate = _bootstrap(
            strata, label=f"cross-seed:{spec.logical_name}", resamples=resamples
        )
        cross_ci_high = aggregate["bootstrap_95pct_ci"]["p50_delta_us"][1]
        cross_ci_passed = _finite_float(
            cross_ci_high, "cross-seed p50 delta CI upper bound"
        ) < 0.0
        cell = {
            "logical_name": spec.logical_name,
            "family": spec.family,
            "state_variant": spec.state_variant,
            "seq_lens": list(spec.seq_lens),
            "per_seed": per_seed,
            "worst_seed_speedup_from_p50": min(
                item["speedup_from_p50"] for item in per_seed
            ),
            "worst_seed_hip_win_fraction": min(
                item["hip_win_fraction"] for item in per_seed
            ),
            "cross_seed_paired_bootstrap": aggregate,
            "cross_seed_p50_delta_ci_upper_strictly_negative": cross_ci_passed,
        }
        cell["performance_gate_passed"] = all(
            item["performance_gate"]["passed"] for item in per_seed
        ) and cross_ci_passed
        cells.append(cell)

    speedup_cell = min(cells, key=lambda cell: cell["worst_seed_speedup_from_p50"])
    win_cell = min(cells, key=lambda cell: cell["worst_seed_hip_win_fraction"])
    ci_cell = max(
        cells,
        key=lambda cell: cell["cross_seed_paired_bootstrap"]["bootstrap_95pct_ci"][
            "p50_delta_us"
        ][1],
    )
    return {
        "logical_cells": len(cells),
        "seed_cells": len(results),
        "global_worst": {
            "minimum_worst_seed_speedup_from_p50": speedup_cell[
                "worst_seed_speedup_from_p50"
            ],
            "minimum_speedup_cell": speedup_cell["logical_name"],
            "minimum_worst_seed_hip_win_fraction": win_cell[
                "worst_seed_hip_win_fraction"
            ],
            "minimum_win_fraction_cell": win_cell["logical_name"],
            "maximum_cross_seed_p50_delta_ci_high_us": ci_cell[
                "cross_seed_paired_bootstrap"
            ]["bootstrap_95pct_ci"]["p50_delta_us"][1],
            "maximum_ci_high_cell": ci_cell["logical_name"],
        },
        "cells": cells,
    }


def _performance_gate(cross_seed_summary: dict[str, Any]) -> dict[str, Any]:
    """Return the four-part, fail-closed formal performance decision."""

    cells = cross_seed_summary["cells"]
    all_p50 = all(
        cell["worst_seed_speedup_from_p50"] > 1.0 for cell in cells
    )
    all_win = all(
        cell["worst_seed_hip_win_fraction"] > 0.5 for cell in cells
    )
    all_seed_ci = all(
        seed_row["p50_delta_us_ci"][1] < 0.0
        for cell in cells
        for seed_row in cell["per_seed"]
    )
    all_cross_ci = all(
        cell["cross_seed_paired_bootstrap"]["bootstrap_95pct_ci"][
            "p50_delta_us"
        ][1]
        < 0.0
        for cell in cells
    )
    passed = all_p50 and all_win and all_seed_ci and all_cross_ci
    failed_cells = [
        cell["logical_name"] for cell in cells if not cell["performance_gate_passed"]
    ]
    return {
        "evaluated": True,
        "passed": passed,
        "performance_definition": (
            "every logical cell must have HIP faster by p50 in every seed, "
            "paired HIP win fraction > 0.5 in every seed, every per-seed "
            "p50-delta 95% CI upper bound < 0, and cross-seed p50-delta "
            "95% CI upper bound < 0"
        ),
        "all_cells_p50_faster_every_seed": all_p50,
        "all_cells_majority_paired_wins_every_seed": all_win,
        "all_per_seed_p50_delta_ci_upper_below_zero": all_seed_ci,
        "all_cross_seed_p50_delta_ci_upper_below_zero": all_cross_ci,
        "failed_logical_cells": failed_cells,
    }


def _plan(seeds: tuple[int, ...]) -> dict[str, Any]:
    family_counts: dict[str, int] = {}
    for spec in ALL_SPECS:
        family_counts[spec.family] = family_counts.get(spec.family, 0) + 1
    cells = [
        {
            **asdict(spec),
            "seed": seed,
            "heads": HEADS,
            "value_heads": VALUE_HEADS,
            "max_seqlen_upper_bound": spec.max_seqlen_upper_bound,
            "initial_state_literal_none": spec.initial_state_is_none,
        }
        for seed in seeds
        for spec in ALL_SPECS
    ]
    return {
        "schema": SCHEMA,
        "cpu_only": True,
        "seeds": list(seeds),
        "heads": HEADS,
        "value_heads": VALUE_HEADS,
        "logical_cells_per_seed": len(ALL_SPECS),
        "total_seed_cells": len(cells),
        "timed_backends_per_seed_cell": 2,
        "graphs_per_seed_cell": 3,
        "family_counts_per_seed": family_counts,
        "paired_timing_calls_at_w20_r120": len(cells) * 2 * (20 + 120),
        "cells": cells,
    }


def _plan_sha256(plan: dict[str, Any]) -> str:
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _static_self_test() -> dict[str, Any]:
    if _RUNTIME_LOADED:
        raise RuntimeError("CPU self-test unexpectedly loaded the GPU benchmark")
    if len(ALL_SPECS) != 26 or len({spec.logical_name for spec in ALL_SPECS}) != 26:
        raise RuntimeError("fixed matrix cardinality or names changed")
    if {spec.state_variant for spec in ALL_SPECS if spec.family == "single"} != set(
        STATE_VARIANTS
    ):
        raise RuntimeError("single state cross is incomplete")
    if {
        spec.source_name
        for spec in ALL_SPECS
        if spec.family == "mixed-production"
    } != {
        f"mixed-{decodes}d-budget16k" for decodes in MIXED_PRODUCTION_DECODES
    }:
        raise RuntimeError("mixed-production source set changed")
    plan = _plan(SEEDS)
    if plan["total_seed_cells"] != 78:
        raise RuntimeError("formal seed-cell count changed")
    if _plan_sha256(plan) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("formal public-K3 matrix identity changed")

    bootstrap = _bootstrap(
        {"order-0": [-3.0, -2.0], "order-1": [-1.0, -0.5]},
        label="self-test",
        resamples=200,
    )
    if bootstrap["point_estimate"]["hip_win_fraction"] != 1.0:
        raise RuntimeError("bootstrap sign convention changed")
    raw_witness: list[dict[str, object]] = []
    for round_index in range(4):
        order = ("native", "triton") if round_index % 2 == 0 else (
            "triton",
            "native",
        )
        for position, backend in enumerate(order):
            raw_witness.append(
                {
                    "backend": backend,
                    "round": round_index,
                    "order": position,
                    "latency_ms": 1.0 if backend == "native" else 2.0,
                }
            )
    paired_witness = _paired_raw_rows(
        raw_witness, spec=ALL_SPECS[0], repeat=4
    )
    if _position_counts(paired_witness) != {
        PUBLIC_BACKEND: {"first": 2, "second": 2},
        TRITON_BACKEND: {"first": 2, "second": 2},
    }:
        raise RuntimeError("paired timing position balance changed")
    passing = _gate_checks(1.01, 0.51, -0.01)
    if not passing["passed"]:
        raise RuntimeError("strict gate rejected a passing boundary witness")
    for values in ((1.0, 0.51, -0.01), (1.01, 0.5, -0.01), (1.01, 0.51, 0.0)):
        if _gate_checks(*values)["passed"]:
            raise RuntimeError("strict gate accepted an equality boundary")
    return {
        "schema": SCHEMA,
        "self_test": "PASS",
        "gpu_runtime_imported_by_runner": _RUNTIME_LOADED,
        "matrix": plan,
        "matrix_sha256": _plan_sha256(plan),
        "bootstrap_schema": bootstrap,
        "strict_gate_boundary_tested": True,
        "removed_kda_environment_names": list(_REMOVED_KDA_ENVIRONMENT),
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--warmup", type=int, default=FORMAL_MIN_WARMUP)
    parser.add_argument("--repeat", type=int, default=FORMAL_MIN_REPEAT)
    parser.add_argument("--tolerance", type=float, default=MAX_TOLERANCE)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--static-self-test", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args(argv)
    args.seed = tuple(args.seed or SEEDS)
    if args.seed != SEEDS:
        parser.error(
            "formal full matrix requires exactly --seed 42 --seed 43 --seed 44"
        )
    if args.warmup != FORMAL_MIN_WARMUP or args.repeat != FORMAL_MIN_REPEAT:
        parser.error("formal timing requires exactly warmup=20 and repeat=120")
    if args.tolerance != MAX_TOLERANCE:
        parser.error("formal timing requires exactly tolerance=0.04")
    if args.bootstrap_resamples != BOOTSTRAP_RESAMPLES:
        parser.error("formal timing requires exactly 10000 bootstrap resamples")
    if args.static_self_test and args.print_plan:
        parser.error("choose one CPU-only mode")
    if not args.static_self_test and not args.print_plan and args.output is None:
        parser.error(
            "GPU execution requires --output so gate failures retain evidence"
        )
    return args


def _run_gpu(args: argparse.Namespace) -> dict[str, Any]:
    assert args.output is not None
    results: list[dict[str, Any]] = []
    checkpoint = args.output.with_name("partial-results.jsonl")
    if checkpoint.exists():
        raise RuntimeError(f"refusing to overwrite checkpoint: {checkpoint}")
    total_seed_cells = len(ALL_SPECS) * len(args.seed)
    active_seed: int | None = None
    active_spec: CaseSpec | None = None
    phase = "runtime-import"
    try:
        benchmark = _load_benchmark()
        suite_identity = _assert_runtime_suite_identity(benchmark)
        torch = benchmark.torch
        phase = "device-preflight"
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("formal public-K3 benchmark requires a ROCm GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "formal public-K3 benchmark requires one visible GPU"
            )
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch_detail = str(getattr(properties, "gcnArchName", "unknown"))
        if arch_detail.split(":", 1)[0] != "gfx950":
            raise RuntimeError(
                f"formal public-K3 benchmark requires gfx950, got {arch_detail}"
            )
        if properties.multi_processor_count != 256:
            raise RuntimeError(
                "formal public-K3 benchmark requires the 256-CU gfx950 target, "
                f"got {properties.multi_processor_count} CUs"
            )

        phase = "seed-cell-capture"
        with torch.inference_mode():
            for seed in args.seed:
                for index, spec in enumerate(ALL_SPECS, start=1):
                    active_seed = seed
                    active_spec = spec
                    print(
                        f"seed={seed} cell={index}/{len(ALL_SPECS)} "
                        f"{spec.logical_name}",
                        flush=True,
                    )
                    result = _run_one(
                        benchmark,
                        spec,
                        seed=seed,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        tolerance=args.tolerance,
                        bootstrap_resamples=args.bootstrap_resamples,
                    )
                    results.append(result)
                    _append_checkpoint(
                        checkpoint,
                        {
                            "schema": SCHEMA,
                            "event": "seed-cell-complete",
                            "complete": False,
                            "completed_seed_cells": len(results),
                            "total_seed_cells": total_seed_cells,
                            "result": result,
                        },
                    )

        _append_checkpoint(
            checkpoint,
            {
                "schema": SCHEMA,
                "event": "timing-capture-complete",
                "complete": False,
                "capture_complete": True,
                "completed_seed_cells": len(results),
                "total_seed_cells": total_seed_cells,
            },
        )

        # Collect provenance only after compilation/loading, so the reported
        # object is the exact DSO used for every timed HIP graph.
        phase = "jit-provenance"
        environment = benchmark._print_environment(HEADS, VALUE_HEADS)
        if environment.get("module_sha256") is None:
            raise RuntimeError(
                "compiled module_flash_kda_hip.so identity is missing"
            )
        loaded_modules = environment.get("loaded_module_identities", {})
        loaded_flash = loaded_modules.get("module_flash_kda_hip", {})
        if (
            loaded_flash.get("matches_expected_jit_path") is not True
            or loaded_flash.get("sha256") != environment["module_sha256"]
        ):
            raise RuntimeError(
                "loaded module_flash_kda_hip identity does not match "
                "AITER_JIT_DIR"
            )
        active_environment = _active_kda_environment()
        if active_environment:
            raise RuntimeError(
                "KDA environment changed during the formal matrix: "
                f"{active_environment}"
            )
        environment.update(
            {
                "active_kda_environment": active_environment,
                "removed_kda_environment_names": list(
                    _REMOVED_KDA_ENVIRONMENT
                ),
                "formal_runner_path": str(Path(__file__).resolve()),
                "formal_runner_sha256": _sha256(Path(__file__).resolve()),
                "build_mode": "fresh-jit-build-in-isolated-output-directory",
            }
        )

        phase = "cross-seed-analysis"
        cross_seed_summary = _cross_seed_summary(
            results, args.bootstrap_resamples
        )
        performance_gate = _performance_gate(cross_seed_summary)
        payload = {
            "schema": SCHEMA,
            "configuration": {
                "seeds": list(args.seed),
                "warmup": args.warmup,
                "repeat": args.repeat,
                "tolerance": args.tolerance,
                "bootstrap_resamples": args.bootstrap_resamples,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "heads": HEADS,
                "value_heads": VALUE_HEADS,
                "execution": "graph",
                "public_native_backend_keyword_omitted": True,
                "public_triton_backend_keyword": "triton",
                "paired_order": "alternating-position-balanced",
            },
            "plan": _plan(args.seed),
            "plan_sha256": _plan_sha256(_plan(args.seed)),
            "environment": environment,
            "suite_identity": suite_identity,
            "raw_evidence": {
                "embedded_in_each_result": True,
                "checkpoint_jsonl": str(checkpoint),
                "rows_per_seed_cell": args.repeat * 2,
                "total_invocation_rows": len(results) * args.repeat * 2,
                "paired_delta_unit": "microseconds",
            },
            "all_nonperformance_contract_checks_passed": all(
                result["all_contract_checks_passed"] for result in results
            ),
            "results": results,
            "cross_seed_summary": cross_seed_summary,
            "performance_gate": performance_gate,
            "performance_gate_evaluated": True,
            "performance_gate_passed": performance_gate["passed"],
            "capture_complete": True,
        }
    except BaseException as error:
        _append_checkpoint(
            checkpoint,
            {
                "schema": SCHEMA,
                "event": "run-failed",
                "complete": False,
                "phase": phase,
                "completed_seed_cells": len(results),
                "total_seed_cells": total_seed_cells,
                "active_seed": active_seed,
                "active_logical_cell": (
                    None if active_spec is None else active_spec.logical_name
                ),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    return payload


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
    print(rendered, end="", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.static_self_test:
        payload = _static_self_test()
    elif args.print_plan:
        payload = _plan(args.seed)
    else:
        payload = _run_gpu(args)
        assert args.output is not None
        checkpoint = args.output.with_name("partial-results.jsonl")
        try:
            _emit(payload, args.output)
        except BaseException as error:
            _append_checkpoint(
                checkpoint,
                {
                    "schema": SCHEMA,
                    "event": "run-failed",
                    "complete": False,
                    "phase": "result-emission",
                    "completed_seed_cells": len(payload["results"]),
                    "total_seed_cells": len(ALL_SPECS) * len(args.seed),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise
        _append_checkpoint(
            checkpoint,
            {
                "schema": SCHEMA,
                "event": "run-complete",
                "complete": True,
                "capture_complete": True,
                "result_json_written": True,
                "completed_seed_cells": len(payload["results"]),
                "total_seed_cells": len(ALL_SPECS) * len(args.seed),
                "performance_gate_passed": payload[
                    "performance_gate_passed"
                ],
            },
        )
        if payload["performance_gate_passed"] is not True:
            failed = payload["performance_gate"]["failed_logical_cells"]
            raise SystemExit(
                "formal public-K3 performance gate failed for: " + ", ".join(failed)
            )
        return
    _emit(payload, args.output)


if __name__ == "__main__":
    main()
