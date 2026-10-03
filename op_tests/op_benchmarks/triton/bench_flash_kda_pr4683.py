# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Canonical gfx950 comparison for the dense table introduced by PR #4683.

The pull request published six dense, eager FlashKDA rows: ``B=1``,
``T in {8192, 16384}``, ``H in {32, 64, 96}``, and ``K=V=128``.  This runner
keeps that geometry and input contract, but measures the current native HIP
operator against the current-main descendant of that Triton FlashKDA
implementation in the same process. The Triton implementation is selected by
importing its internal ``flash_kda_fwd`` directly with Gluon explicitly
disabled; no shared backend router participates in the result.

The formal run has 18 seed cells (six shapes crossed with seeds 42, 43, and
44).  Each cell checks output relative RMS, performs position-balanced
alternating paired timing, and applies a deterministic stratified bootstrap.
It fails closed unless every seed cell has at least 1.03x speedup by both the
ratio of backend p50s and the median paired ratio, a native paired-win fraction
strictly above 0.5, and upper bounds below zero for both the paired median- and
mean-delta 95% confidence intervals.  Hierarchical cross-seed confidence
intervals are also required for every logical shape.

``--print-plan`` and ``--static-self-test`` are CPU-only and intentionally do
not import PyTorch, Triton, or AITER.  For promotion evidence, invoke the clean
checkout wrapper ``scripts/run_flash_kda_pr4683_perf_acceptance.sh``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA = "flash-kda-pr4683-dense-gfx950-v1"
SEEDS = (42, 43, 44)
SEQUENCE_LENGTHS = (8192, 16384)
HEAD_COUNTS = (32, 64, 96)
HEAD_DIM = 128
BATCH = 1
LOWER_BOUND = -5.0
WARMUP = 20
REPEAT = 120
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260902
CORRECTNESS_TOLERANCE = 0.04
MIN_NATIVE_SPEEDUP = 1.03
MIN_PAIRED_WIN_FRACTION = 0.5
EXPECTED_ARCH = "gfx950"
EXPECTED_COMPUTE_UNITS = 256
NATIVE_BACKEND = "native-hip-direct"
TRITON_BACKEND = "latest-main-triton-direct"
REQUIRED_ROUTE_ENV = {"AITER_FDA_USE_GLUON": "0"}
SOURCE_TABLE_PR_HEAD = "71647be6262f757b99b172a975a3a914674be9ac"
TRITON_SOURCE_SHA256 = (
    "c4f536616802286f4a73557cf08a15d6948eb3af582d67c1909550e9ad9b707e"
)

# Values copied from the gfx950 table in PR #4683.  They are provenance only:
# acceptance compares paired measurements from this checkout and does not gate
# on numbers collected by another host/software stack.
PUBLISHED_PR_TABLE = {
    "B1-T8192-H32-K128-V128": {
        "baseline_ms": 1.0031,
        "triton_flash_ms": 0.6091,
    },
    "B1-T8192-H64-K128-V128": {
        "baseline_ms": 1.6186,
        "triton_flash_ms": 0.9148,
    },
    "B1-T8192-H96-K128-V128": {
        "baseline_ms": 2.3558,
        "triton_flash_ms": 1.2243,
    },
    "B1-T16384-H32-K128-V128": {
        "baseline_ms": 1.9047,
        "triton_flash_ms": 1.1722,
    },
    "B1-T16384-H64-K128-V128": {
        "baseline_ms": 3.1407,
        "triton_flash_ms": 1.7967,
    },
    "B1-T16384-H96-K128-V128": {
        "baseline_ms": 4.6134,
        "triton_flash_ms": 2.4802,
    },
}

EXPECTED_PLAN_SHA256 = (
    "8e651f53b430ef4fe8a947de1ee0ed64687a7d8192369351b65ab71a0b4a1fab"
)
_RUNTIME_LOADED = False


@dataclass(frozen=True)
class ShapeSpec:
    batch: int
    seqlen: int
    heads: int
    key_dim: int = HEAD_DIM
    value_dim: int = HEAD_DIM

    @property
    def name(self) -> str:
        return (
            f"B{self.batch}-T{self.seqlen}-H{self.heads}-"
            f"K{self.key_dim}-V{self.value_dim}"
        )

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.batch,
            self.seqlen,
            self.heads,
            self.key_dim,
            self.value_dim,
        )


SHAPES = tuple(
    ShapeSpec(BATCH, seqlen, heads)
    for seqlen in SEQUENCE_LENGTHS
    for heads in HEAD_COUNTS
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={_REPO_ROOT}", *args],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _bootstrap(
    strata: dict[str, list[float]], *, label: str, resamples: int
) -> dict[str, Any]:
    """Bootstrap paired native-minus-Triton deltas within order strata."""

    if resamples <= 0 or not strata or any(not values for values in strata.values()):
        raise ValueError("bootstrap requires positive resamples and nonempty strata")
    flat = [value for values in strata.values() for value in values]
    if any(not math.isfinite(value) for value in flat):
        raise ValueError("bootstrap samples must be finite")

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
        "delta_definition": "native_hip_ms-minus-latest_main_triton_ms; negative is native win",
        "delta_unit": "microseconds",
        "strata": {name: len(values) for name, values in sorted(strata.items())},
        "samples": len(flat),
        "point_estimate": {
            "p50_delta_us": statistics.median(flat),
            "mean_delta_us": statistics.fmean(flat),
            "native_win_fraction": sum(value < 0.0 for value in flat) / len(flat),
        },
        "bootstrap_95pct_ci": {
            "method": "stratified percentile bootstrap of paired rounds",
            "resamples": resamples,
            "p50_delta_us": [
                _percentile(medians, 0.025),
                _percentile(medians, 0.975),
            ],
            "mean_delta_us": [
                _percentile(means, 0.025),
                _percentile(means, 0.975),
            ],
            "native_win_fraction": [
                _percentile(wins, 0.025),
                _percentile(wins, 0.975),
            ],
        },
    }


def _hierarchical_bootstrap(
    seed_strata: dict[int, dict[str, list[float]]],
    *,
    label: str,
    resamples: int,
) -> dict[str, Any]:
    """Resample seeds, then paired deltas within each invocation-order stratum."""

    if resamples <= 0 or not seed_strata:
        raise ValueError("hierarchical bootstrap requires seeds and resamples")
    if any(
        not strata or any(not values for values in strata.values())
        for strata in seed_strata.values()
    ):
        raise ValueError("every seed and order stratum must be nonempty")
    flat = [
        value
        for strata in seed_strata.values()
        for values in strata.values()
        for value in values
    ]
    if any(not math.isfinite(value) for value in flat):
        raise ValueError("hierarchical bootstrap samples must be finite")

    seeds = sorted(seed_strata)
    rng = random.Random(_stable_seed(label))
    medians: list[float] = []
    means: list[float] = []
    wins: list[float] = []
    for _ in range(resamples):
        sample: list[float] = []
        for _seed_position in seeds:
            selected_seed = rng.choice(seeds)
            for values in seed_strata[selected_seed].values():
                sample.extend(rng.choice(values) for _ in values)
        medians.append(statistics.median(sample))
        means.append(statistics.fmean(sample))
        wins.append(sum(value < 0.0 for value in sample) / len(sample))

    return {
        "delta_definition": "native_hip_ms-minus-latest_main_triton_ms; negative is native win",
        "delta_unit": "microseconds",
        "hierarchy": {
            str(seed): {
                order: len(values)
                for order, values in sorted(seed_strata[seed].items())
            }
            for seed in seeds
        },
        "samples": len(flat),
        "point_estimate": {
            "p50_delta_us": statistics.median(flat),
            "mean_delta_us": statistics.fmean(flat),
            "native_win_fraction": sum(value < 0.0 for value in flat) / len(flat),
        },
        "bootstrap_95pct_ci": {
            "method": (
                "hierarchical percentile bootstrap: resample seeds, then "
                "paired rounds within invocation-order strata"
            ),
            "resamples": resamples,
            "p50_delta_us": [
                _percentile(medians, 0.025),
                _percentile(medians, 0.975),
            ],
            "mean_delta_us": [
                _percentile(means, 0.025),
                _percentile(means, 0.975),
            ],
            "native_win_fraction": [
                _percentile(wins, 0.025),
                _percentile(wins, 0.975),
            ],
        },
    }


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite: {value!r}")
    return result


def _cell_gate(
    *,
    speedup: Any,
    paired_speedup: Any,
    win_fraction: Any,
    bootstrap: dict[str, Any],
    rrms: Any,
) -> dict[str, Any]:
    speedup_value = _finite_float(speedup, "native p50 speedup")
    paired_speedup_value = _finite_float(paired_speedup, "paired native p50 speedup")
    win_value = _finite_float(win_fraction, "native paired-win fraction")
    rrms_value = _finite_float(rrms, "native-vs-Triton output rRMS")
    ci = bootstrap["bootstrap_95pct_ci"]
    p50_high = _finite_float(ci["p50_delta_us"][1], "p50 delta CI high")
    mean_high = _finite_float(ci["mean_delta_us"][1], "mean delta CI high")
    checks = {
        "minimum_native_speedup": MIN_NATIVE_SPEEDUP,
        "native_speedup_from_p50": speedup_value,
        "native_speedup_at_least_minimum": speedup_value >= MIN_NATIVE_SPEEDUP,
        "paired_native_speedup_p50": paired_speedup_value,
        "paired_native_speedup_at_least_minimum": (
            paired_speedup_value >= MIN_NATIVE_SPEEDUP
        ),
        "minimum_paired_win_fraction_exclusive": MIN_PAIRED_WIN_FRACTION,
        "native_paired_win_fraction": win_value,
        "native_paired_win_fraction_strictly_above_half": (
            win_value > MIN_PAIRED_WIN_FRACTION
        ),
        "p50_delta_95pct_ci_high_us": p50_high,
        "p50_delta_ci_upper_strictly_negative": p50_high < 0.0,
        "mean_delta_95pct_ci_high_us": mean_high,
        "mean_delta_ci_upper_strictly_negative": mean_high < 0.0,
        "maximum_output_relative_rms": CORRECTNESS_TOLERANCE,
        "native_vs_triton_output_relative_rms": rrms_value,
        "correctness_within_tolerance": rrms_value <= CORRECTNESS_TOLERANCE,
    }
    checks["passed"] = all(
        (
            checks["native_speedup_at_least_minimum"],
            checks["paired_native_speedup_at_least_minimum"],
            checks["native_paired_win_fraction_strictly_above_half"],
            checks["p50_delta_ci_upper_strictly_negative"],
            checks["mean_delta_ci_upper_strictly_negative"],
            checks["correctness_within_tolerance"],
        )
    )
    return checks


def _plan() -> dict[str, Any]:
    cells = [
        {
            "seed": seed,
            **asdict(spec),
            "logical_name": spec.name,
            "shape": list(spec.shape),
            "execution": "eager",
            "initial_state": None,
            "output_final_state": False,
            "beta_dtype": "torch.float32",
        }
        for seed in SEEDS
        for spec in SHAPES
    ]
    return {
        "schema": SCHEMA,
        "cpu_only": True,
        "source_table": "https://github.com/ROCm/aiter/pull/4683",
        "comparator": {
            "implementation": "latest-main descendant of PR #4683",
            "source_table_pr_head": SOURCE_TABLE_PR_HEAD,
            "expected_triton_source_sha256": TRITON_SOURCE_SHA256,
            "forced_direct_import": True,
            "forced_route_environment": REQUIRED_ROUTE_ENV,
            "gluon_disabled": True,
        },
        "logical_shapes": len(SHAPES),
        "seed_cells": len(cells),
        "seeds": list(SEEDS),
        "warmup_rounds": WARMUP,
        "measured_rounds": REPEAT,
        "timed_backends_per_round": 2,
        "paired_invocations": len(cells) * REPEAT * 2,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "cells": cells,
    }


def _plan_sha256(plan: dict[str, Any]) -> str:
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _static_self_test() -> dict[str, Any]:
    if _RUNTIME_LOADED:
        raise RuntimeError("CPU self-test unexpectedly loaded the GPU runtime")
    if len(SHAPES) != 6 or len({spec.name for spec in SHAPES}) != 6:
        raise RuntimeError("PR #4683 shape matrix identity changed")
    if {spec.seqlen for spec in SHAPES} != set(SEQUENCE_LENGTHS):
        raise RuntimeError("sequence-length coverage changed")
    if {spec.heads for spec in SHAPES} != set(HEAD_COUNTS):
        raise RuntimeError("head-count coverage changed")
    if set(PUBLISHED_PR_TABLE) != {spec.name for spec in SHAPES}:
        raise RuntimeError("published PR table no longer matches the matrix")
    plan = _plan()
    if plan["seed_cells"] != 18:
        raise RuntimeError("formal seed-cell count changed")
    plan_hash = _plan_sha256(plan)
    if plan_hash != EXPECTED_PLAN_SHA256:
        raise RuntimeError(
            f"formal matrix identity changed: {plan_hash} != {EXPECTED_PLAN_SHA256}"
        )

    witness = {
        "native-first": [-4.0, -3.0, -2.0],
        "triton-first": [-3.5, -2.5, -1.5],
    }
    bootstrap = _bootstrap(witness, label="static-self-test", resamples=200)
    hierarchical = _hierarchical_bootstrap(
        {
            42: witness,
            43: {
                name: [value - 0.25 for value in values]
                for name, values in witness.items()
            },
            44: {
                name: [value - 0.50 for value in values]
                for name, values in witness.items()
            },
        },
        label="static-hierarchical-self-test",
        resamples=200,
    )
    if hierarchical["bootstrap_95pct_ci"]["mean_delta_us"][1] >= 0.0:
        raise RuntimeError("hierarchical bootstrap sign convention changed")
    passing = _cell_gate(
        speedup=MIN_NATIVE_SPEEDUP,
        paired_speedup=MIN_NATIVE_SPEEDUP,
        win_fraction=0.51,
        bootstrap=bootstrap,
        rrms=CORRECTNESS_TOLERANCE,
    )
    if not passing["passed"]:
        raise RuntimeError("inclusive speedup/correctness boundary was rejected")

    failing_inputs = (
        (MIN_NATIVE_SPEEDUP - 1.0e-9, 1.04, 0.51, -1.0, -1.0, 0.0),
        (1.04, MIN_NATIVE_SPEEDUP - 1.0e-9, 0.51, -1.0, -1.0, 0.0),
        (1.04, 1.04, 0.5, -1.0, -1.0, 0.0),
        (1.04, 1.04, 0.51, 0.0, -1.0, 0.0),
        (1.04, 1.04, 0.51, -1.0, 0.0, 0.0),
        (
            1.04,
            1.04,
            0.51,
            -1.0,
            -1.0,
            CORRECTNESS_TOLERANCE + 1.0e-9,
        ),
    )
    for speedup, paired_speedup, wins, p50_high, mean_high, rrms in failing_inputs:
        synthetic = {
            "bootstrap_95pct_ci": {
                "p50_delta_us": [-2.0, p50_high],
                "mean_delta_us": [-2.0, mean_high],
            }
        }
        if _cell_gate(
            speedup=speedup,
            paired_speedup=paired_speedup,
            win_fraction=wins,
            bootstrap=synthetic,
            rrms=rrms,
        )["passed"]:
            raise RuntimeError("fail-closed gate accepted a boundary witness")

    return {
        "schema": SCHEMA,
        "self_test": "PASS",
        "gpu_runtime_imported": _RUNTIME_LOADED,
        "matrix": plan,
        "matrix_sha256": plan_hash,
        "bootstrap_witness": bootstrap,
        "hierarchical_bootstrap_witness": hierarchical,
        "gate_boundaries_tested": True,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }


def _load_runtime() -> SimpleNamespace:
    global _RUNTIME_LOADED
    if _RUNTIME_LOADED:
        raise RuntimeError("GPU runtime loader called more than once")

    torch = importlib.import_module("torch")
    triton = importlib.import_module("triton")
    native_module = importlib.import_module("aiter.ops.flash_kda")
    triton_module = importlib.import_module(
        "aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda"
    )
    jit_core = importlib.import_module("aiter.jit.core")

    expected_native = (_REPO_ROOT / "aiter/ops/flash_kda.py").resolve()
    expected_triton = (
        _REPO_ROOT / "aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py"
    ).resolve()
    actual_native = Path(native_module.__file__).resolve()
    actual_triton = Path(triton_module.__file__).resolve()
    if actual_native != expected_native:
        raise RuntimeError(
            f"native import escaped this checkout: {actual_native} != {expected_native}"
        )
    if actual_triton != expected_triton:
        raise RuntimeError(
            f"Triton import escaped this checkout: {actual_triton} != {expected_triton}"
        )
    actual_triton_sha256 = _sha256(actual_triton)
    if actual_triton_sha256 != TRITON_SOURCE_SHA256:
        raise RuntimeError(
            "direct Triton comparator is not the audited latest-main source: "
            f"{actual_triton_sha256} != {TRITON_SOURCE_SHA256}"
        )
    if triton_module.AITER_FDA_USE_GLUON_K1 or triton_module.AITER_FDA_USE_GLUON_K2:
        raise RuntimeError(
            "direct Triton comparator unexpectedly enabled a Gluon KDA route"
        )
    effective_meta_dir = Path(jit_core.AITER_META_DIR).resolve()
    effective_csrc_dir = Path(jit_core.AITER_CSRC_DIR).resolve()
    expected_csrc_dir = (_REPO_ROOT / "csrc").resolve()
    if effective_meta_dir != _REPO_ROOT or effective_csrc_dir != expected_csrc_dir:
        raise RuntimeError(
            "native build inputs escaped this checkout: "
            f"AITER_META_DIR={effective_meta_dir}, "
            f"AITER_CSRC_DIR={effective_csrc_dir}"
        )

    _RUNTIME_LOADED = True
    return SimpleNamespace(
        torch=torch,
        triton=triton,
        native_module=native_module,
        triton_module=triton_module,
        jit_core=jit_core,
        native_fn=native_module.flash_kda_fwd,
        triton_fn=triton_module.flash_kda_fwd,
    )


def _control_environment() -> dict[str, str]:
    exact = {
        "AITER_FDA_USE_GLUON",
        "AITER_TRITON_ONLY",
        "AITER_REBUILD",
    }
    prefixes = ("FLASH_KDA_", "CHUNK_DELTA_ATTN_", "KDA_")
    return {
        name: value
        for name, value in sorted(os.environ.items())
        if name in exact or name.startswith(prefixes)
    }


def _make_inputs(
    runtime: SimpleNamespace, spec: ShapeSpec, seed: int
) -> dict[str, Any]:
    """Reproduce the PR table's seed and dtype contract for one shape."""

    torch = runtime.torch
    torch.manual_seed(seed)
    b, t, h, k_dim, v_dim = spec.shape
    device = torch.device("cuda")
    inputs = {
        "q": torch.randn((b, t, h, k_dim), device=device, dtype=torch.bfloat16),
        "k": torch.randn((b, t, h, k_dim), device=device, dtype=torch.bfloat16),
        "v": torch.randn((b, t, h, v_dim), device=device, dtype=torch.bfloat16),
        "g": (torch.randn((b, t, h, k_dim), device=device, dtype=torch.bfloat16) * 0.1),
        "beta": torch.randn((b, t, h), device=device, dtype=torch.float32),
        "A_log": (torch.randn(h, device=device, dtype=torch.float32).abs() * 0.5),
        "dt_bias": (torch.randn(h * k_dim, device=device, dtype=torch.float32) * 0.1),
        "scale": 1.0 / math.sqrt(k_dim),
    }
    if any(
        tensor.dtype != torch.bfloat16
        for tensor in (inputs["q"], inputs["k"], inputs["v"], inputs["g"])
    ):
        raise RuntimeError(f"{spec.name}: q/k/v/g fixture dtype drifted")
    if any(
        tensor.dtype != torch.float32
        for tensor in (inputs["beta"], inputs["A_log"], inputs["dt_bias"])
    ):
        raise RuntimeError(f"{spec.name}: beta/A_log/dt_bias fixture dtype drifted")
    if any(
        not tensor.is_contiguous()
        for tensor in (
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["A_log"],
            inputs["dt_bias"],
        )
    ):
        raise RuntimeError(f"{spec.name}: benchmark inputs must be contiguous")
    return inputs


def _build_functions(
    runtime: SimpleNamespace, inputs: dict[str, Any]
) -> dict[str, Callable[[], tuple[Any, Any]]]:
    def native() -> tuple[Any, Any]:
        return runtime.native_fn(
            **inputs,
            lower_bound=LOWER_BOUND,
            initial_state=None,
            output_final_state=False,
        )

    def triton_flash() -> tuple[Any, Any]:
        return runtime.triton_fn(
            **inputs,
            lower_bound=LOWER_BOUND,
            initial_state=None,
            output_final_state=False,
            chunks_per_seg=None,
        )

    return {NATIVE_BACKEND: native, TRITON_BACKEND: triton_flash}


def _assert_output_contract(
    runtime: SimpleNamespace,
    spec: ShapeSpec,
    backend: str,
    result: Any,
) -> tuple[Any, None]:
    torch = runtime.torch
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"{spec.name}/{backend}: expected an (output, state) pair")
    output, final_state = result
    expected_shape = (spec.batch, spec.seqlen, spec.heads, spec.value_dim)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"{spec.name}/{backend}: output shape {tuple(output.shape)} "
            f"!= {expected_shape}"
        )
    if output.dtype != torch.bfloat16 or output.device.type != "cuda":
        raise RuntimeError(
            f"{spec.name}/{backend}: output contract is "
            f"{output.dtype}/{output.device}"
        )
    if final_state is not None:
        raise RuntimeError(
            f"{spec.name}/{backend}: output_final_state=False returned a state"
        )
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError(f"{spec.name}/{backend}: output contains non-finite values")
    return output, None


def _relative_rms(runtime: SimpleNamespace, actual: Any, reference: Any) -> float:
    actual_f = actual.detach().float()
    reference_f = reference.detach().float()
    difference = actual_f - reference_f
    value = difference.square().mean().sqrt() / (
        reference_f.square().mean().sqrt().clamp_min(1.0e-8)
    )
    result = float(value.item())
    if not math.isfinite(result):
        raise RuntimeError("output relative RMS is not finite")
    return result


def _time_once(runtime: SimpleNamespace, fn: Callable[[], Any]) -> float:
    torch = runtime.torch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end))
    del result, start, end
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError(f"invalid measured latency: {elapsed!r}")
    return elapsed


def _run_cell(
    runtime: SimpleNamespace,
    spec: ShapeSpec,
    seed: int,
) -> dict[str, Any]:
    if _control_environment():
        raise RuntimeError(
            f"{spec.name}/seed-{seed}: routing environment is not clean: "
            f"{_control_environment()}"
        )

    torch = runtime.torch
    inputs = _make_inputs(runtime, spec, seed)
    functions = _build_functions(runtime, inputs)

    # A fresh native JIT's first call enters the tensor-descriptor ABI and only
    # then makes the additive raw-v3 symbol importable. Prime that build once,
    # explicitly bind raw-v3, and compare the second native call so correctness
    # covers the exact ABI used by every timed invocation, including cell 1.
    native_prime = _assert_output_contract(
        runtime, spec, NATIVE_BACKEND, functions[NATIVE_BACKEND]()
    )
    torch.cuda.synchronize()
    del native_prime
    raw_binding = runtime.native_module._get_raw_pointer_binding()
    if raw_binding is None or raw_binding[1] != 3:
        version = None if raw_binding is None else raw_binding[1]
        raise RuntimeError(
            f"{spec.name}: timed native path requires raw-v3, got {version}"
        )

    # Compile Triton and compare the exact raw-v3/direct-Triton outputs before
    # either path enters warmup or timing.
    eager_outputs: dict[str, tuple[Any, None]] = {}
    for backend in (NATIVE_BACKEND, TRITON_BACKEND):
        eager_outputs[backend] = _assert_output_contract(
            runtime, spec, backend, functions[backend]()
        )
        torch.cuda.synchronize()
    rrms = _relative_rms(
        runtime,
        eager_outputs[NATIVE_BACKEND][0],
        eager_outputs[TRITON_BACKEND][0],
    )
    del eager_outputs
    torch.cuda.synchronize()

    # Warm in the same alternating, balanced order as the measured rounds.
    for round_index in range(WARMUP):
        order = (
            (NATIVE_BACKEND, TRITON_BACKEND)
            if round_index % 2 == 0
            else (TRITON_BACKEND, NATIVE_BACKEND)
        )
        for backend in order:
            result = functions[backend]()
            del result
        torch.cuda.synchronize()

    raw_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    samples = {NATIVE_BACKEND: [], TRITON_BACKEND: []}
    strata = {"native-first": [], "triton-first": []}
    for round_index in range(REPEAT):
        order = (
            (NATIVE_BACKEND, TRITON_BACKEND)
            if round_index % 2 == 0
            else (TRITON_BACKEND, NATIVE_BACKEND)
        )
        latencies: dict[str, float] = {}
        for position, backend in enumerate(order):
            latency = _time_once(runtime, functions[backend])
            latencies[backend] = latency
            samples[backend].append(latency)
            raw_rows.append(
                {
                    "round": round_index,
                    "position": position,
                    "backend": backend,
                    "latency_ms": latency,
                }
            )
        delta_us = (latencies[NATIVE_BACKEND] - latencies[TRITON_BACKEND]) * 1000.0
        speedup = latencies[TRITON_BACKEND] / latencies[NATIVE_BACKEND]
        stratum = "native-first" if order[0] == NATIVE_BACKEND else "triton-first"
        strata[stratum].append(delta_us)
        paired_rows.append(
            {
                "round": round_index,
                "first_backend": order[0],
                "native_ms": latencies[NATIVE_BACKEND],
                "triton_ms": latencies[TRITON_BACKEND],
                "native_minus_triton_us": delta_us,
                "triton_over_native_speedup": speedup,
                "native_win": delta_us < 0.0,
            }
        )

    expected_position_count = REPEAT // 2
    positions = {
        NATIVE_BACKEND: {
            "first": sum(
                row["backend"] == NATIVE_BACKEND and row["position"] == 0
                for row in raw_rows
            ),
            "second": sum(
                row["backend"] == NATIVE_BACKEND and row["position"] == 1
                for row in raw_rows
            ),
        },
        TRITON_BACKEND: {
            "first": sum(
                row["backend"] == TRITON_BACKEND and row["position"] == 0
                for row in raw_rows
            ),
            "second": sum(
                row["backend"] == TRITON_BACKEND and row["position"] == 1
                for row in raw_rows
            ),
        },
    }
    expected_positions = {
        "first": expected_position_count,
        "second": expected_position_count,
    }
    if any(counts != expected_positions for counts in positions.values()):
        raise RuntimeError(
            f"{spec.name}: backend positions are unbalanced: {positions}"
        )

    summaries: dict[str, dict[str, float]] = {}
    for backend, values in samples.items():
        summaries[backend] = {
            "p10_ms": _percentile(values, 0.10),
            "p50_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
            "p90_ms": _percentile(values, 0.90),
        }
    native_p50 = summaries[NATIVE_BACKEND]["p50_ms"]
    triton_p50 = summaries[TRITON_BACKEND]["p50_ms"]
    native_speedup = triton_p50 / native_p50
    native_win_fraction = sum(row["native_win"] for row in paired_rows) / REPEAT
    paired_speedup_p50 = statistics.median(
        row["triton_over_native_speedup"] for row in paired_rows
    )
    bootstrap = _bootstrap(
        strata,
        label=f"{spec.name}:seed-{seed}",
        resamples=BOOTSTRAP_RESAMPLES,
    )
    gate = _cell_gate(
        speedup=native_speedup,
        paired_speedup=paired_speedup_p50,
        win_fraction=native_win_fraction,
        bootstrap=bootstrap,
        rrms=rrms,
    )
    published = PUBLISHED_PR_TABLE[spec.name]
    result = {
        "logical_name": spec.name,
        "seed": seed,
        "shape": list(spec.shape),
        "input_contract": {
            "q_k_v_g_dtype": "torch.bfloat16",
            "beta_dtype": "torch.float32",
            "A_log_dtype": "torch.float32",
            "dt_bias_dtype": "torch.float32",
            "initial_state": None,
            "output_final_state": False,
            "cu_seqlens": None,
            "dense": True,
            "max_seqlen_upper_bound": (
                "native derives exact dense T; direct Triton has no hint argument"
            ),
            "scale": inputs["scale"],
            "lower_bound": LOWER_BOUND,
        },
        "execution": "eager",
        "timed_callable_contract": {
            NATIVE_BACKEND: "aiter.ops.flash_kda.flash_kda_fwd",
            TRITON_BACKEND: (
                "aiter.ops.triton._triton_kernels.chunk_delta_attn."
                "flash_kda.flash_kda_fwd"
            ),
            "caller_selects_backend": True,
            "triton_chunks_per_seg": None,
        },
        "correctness": {
            "definition": "rms(native_fp32-triton_fp32)/rms(triton_fp32)",
            "native_vs_triton_output_relative_rms": rrms,
            "maximum_relative_rms": CORRECTNESS_TOLERANCE,
            "passed": rrms <= CORRECTNESS_TOLERANCE,
        },
        "timing": {
            "warmup_rounds": WARMUP,
            "measured_rounds": REPEAT,
            "position_counts": positions,
            "summaries": summaries,
            "triton_over_native_speedup_from_p50": native_speedup,
            "paired_triton_over_native_speedup_p50": paired_speedup_p50,
            "native_paired_win_fraction": native_win_fraction,
        },
        "published_pr4683_context": {
            **published,
            "measured_triton_p50_delta_pct": 100.0
            * (triton_p50 / published["triton_flash_ms"] - 1.0),
            "used_as_acceptance_threshold": False,
        },
        "paired_bootstrap": bootstrap,
        "raw_timing_samples": raw_rows,
        "paired_rounds": paired_rows,
        "performance_gate": gate,
        "performance_gate_passed": gate["passed"],
    }
    del functions, inputs, samples
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _source_provenance() -> dict[str, Any]:
    relative_paths = [
        Path("aiter/ops/flash_kda.py"),
        Path("aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py"),
        Path("aiter/jit/core.py"),
        Path("aiter/jit/optCompilerConfig.json"),
        Path("csrc/include/flash_kda.h"),
        Path("csrc/pybind/flash_kda_pybind.cu"),
    ]
    for pattern in (
        "csrc/kernels/flash_kda/*.cu",
        "csrc/kernels/flash_kda/*.hpp",
        "csrc/kernels/flash_kda/gfx950/*.cu",
        "csrc/kernels/flash_kda/gfx950/*.hpp",
        "aiter/ops/triton/_triton_kernels/chunk_delta_attn/**/*.py",
    ):
        relative_paths.extend(
            path.relative_to(_REPO_ROOT) for path in sorted(_REPO_ROOT.glob(pattern))
        )
    relative_paths = list(dict.fromkeys(relative_paths))
    identities = {}
    combined = hashlib.sha256()
    for relative in relative_paths:
        absolute = (_REPO_ROOT / relative).resolve()
        if not absolute.is_file():
            raise RuntimeError(f"provenance source is missing: {absolute}")
        digest = _sha256(absolute)
        identities[str(relative)] = {"path": str(absolute), "sha256": digest}
        combined.update(str(relative).encode())
        combined.update(b"\0")
        combined.update(digest.encode())
        combined.update(b"\n")
    return {
        "combined_sha256": combined.hexdigest(),
        "files": identities,
    }


def _runtime_provenance(runtime: SimpleNamespace) -> dict[str, Any]:
    torch = runtime.torch
    triton = runtime.triton
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch_detail = str(getattr(properties, "gcnArchName", "unknown"))
    arch = arch_detail.split(":", 1)[0]

    native_jit_module = runtime.jit_core.get_module("module_flash_kda_hip_v2")
    native_jit_path = Path(native_jit_module.__file__).resolve()
    expected_jit_dir_text = os.environ.get("AITER_JIT_DIR")
    expected_jit_path = (
        Path(expected_jit_dir_text).resolve() / "module_flash_kda_hip_v2.so"
        if expected_jit_dir_text
        else None
    )
    if expected_jit_path is not None and native_jit_path != expected_jit_path:
        raise RuntimeError(
            "loaded native JIT module does not match AITER_JIT_DIR: "
            f"{native_jit_path} != {expected_jit_path}"
        )
    raw_binding = runtime.native_module._get_raw_pointer_binding()
    raw_abi_version = None if raw_binding is None else raw_binding[1]
    if raw_abi_version != 3:
        raise RuntimeError(
            f"native timed module did not retain raw-v3 binding: {raw_abi_version}"
        )

    native_source = Path(runtime.native_module.__file__).resolve()
    triton_source = Path(runtime.triton_module.__file__).resolve()
    triton_package = Path(triton.__file__).resolve()
    torch_package = Path(torch.__file__).resolve()
    benchmark_path = Path(__file__).resolve()
    controlled_names = (
        "AITER_AOT_IMPORT",
        "AITER_FDA_USE_GLUON",
        "AITER_JIT_DIR",
        "AITER_META_DIR",
        "AITER_REBUILD",
        "AITER_TRITON_ONLY",
        "CK_DIR",
        "GPU_ARCHS",
        "HIP_VISIBLE_DEVICES",
        "HIP_KITTENS_DIR",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "MAX_JOBS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPUS_GEN_CO_DIR",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONOPTIMIZE",
        "PYTHONPATH",
        "TRITON_CACHE_DIR",
    )
    return {
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "tree": _git("rev-parse", "HEAD^{tree}"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "status_porcelain": _git(
                "status", "--porcelain=v1", "--untracked-files=all"
            ),
            "remotes": _git("remote", "-v"),
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "python_executable_sha256": _sha256(Path(sys.executable).resolve()),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "gpu": {
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "arch": arch,
            "arch_detail": arch_detail,
            "compute_units": properties.multi_processor_count,
            "visible_device_count": torch.cuda.device_count(),
        },
        "software": {
            "torch_version": torch.__version__,
            "torch_hip_version": torch.version.hip,
            "torch_package": str(torch_package),
            "triton_version": triton.__version__,
            "triton_package": str(triton_package),
        },
        "modules": {
            "native_build_roots": {
                "aiter_meta_dir": str(Path(runtime.jit_core.AITER_META_DIR).resolve()),
                "aiter_csrc_dir": str(Path(runtime.jit_core.AITER_CSRC_DIR).resolve()),
                "both_match_checkout": (
                    Path(runtime.jit_core.AITER_META_DIR).resolve() == _REPO_ROOT
                    and Path(runtime.jit_core.AITER_CSRC_DIR).resolve()
                    == (_REPO_ROOT / "csrc").resolve()
                ),
            },
            "native_python": {
                "path": str(native_source),
                "sha256": _sha256(native_source),
                "callable_module": runtime.native_fn.__module__,
            },
            "latest_main_triton_python": {
                "path": str(triton_source),
                "sha256": _sha256(triton_source),
                "callable_module": runtime.triton_fn.__module__,
                "forced_direct_import": True,
                "source_table_pr_head": SOURCE_TABLE_PR_HEAD,
                "expected_source_sha256": TRITON_SOURCE_SHA256,
                "matches_expected_source": (
                    _sha256(triton_source) == TRITON_SOURCE_SHA256
                ),
                "gluon_k1_enabled": runtime.triton_module.AITER_FDA_USE_GLUON_K1,
                "gluon_k2_enabled": runtime.triton_module.AITER_FDA_USE_GLUON_K2,
                "triton_only_route_verified": not (
                    runtime.triton_module.AITER_FDA_USE_GLUON_K1
                    or runtime.triton_module.AITER_FDA_USE_GLUON_K2
                ),
            },
            "native_jit": {
                "path": str(native_jit_path),
                "sha256": _sha256(native_jit_path),
                "expected_path": (
                    None if expected_jit_path is None else str(expected_jit_path)
                ),
                "matches_expected_jit_path": (
                    expected_jit_path is None or native_jit_path == expected_jit_path
                ),
                "raw_abi_version": raw_abi_version,
            },
            "benchmark": {
                "path": str(benchmark_path),
                "sha256": _sha256(benchmark_path),
            },
        },
        "source_provenance": _source_provenance(),
        "environment": {name: os.environ.get(name) for name in controlled_names},
        "active_route_control_environment": _control_environment(),
    }


def _cross_seed_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["logical_name"], []).append(result)

    logical_cells = []
    for spec in SHAPES:
        seed_results = sorted(
            grouped.get(spec.name, []), key=lambda result: result["seed"]
        )
        if [result["seed"] for result in seed_results] != list(SEEDS):
            raise RuntimeError(f"{spec.name}: incomplete formal seed set")
        seed_strata: dict[int, dict[str, list[float]]] = {}
        for result in seed_results:
            seed = result["seed"]
            order_strata = seed_strata.setdefault(seed, {})
            for row in result["paired_rounds"]:
                order = (
                    "native-first"
                    if row["first_backend"] == NATIVE_BACKEND
                    else "triton-first"
                )
                order_strata.setdefault(order, []).append(
                    float(row["native_minus_triton_us"])
                )
        bootstrap = _hierarchical_bootstrap(
            seed_strata,
            label=f"cross-seed:{spec.name}",
            resamples=BOOTSTRAP_RESAMPLES,
        )
        p50_high = bootstrap["bootstrap_95pct_ci"]["p50_delta_us"][1]
        mean_high = bootstrap["bootstrap_95pct_ci"]["mean_delta_us"][1]
        seed_cells_passed = all(
            result["performance_gate_passed"] for result in seed_results
        )
        cell = {
            "logical_name": spec.name,
            "seeds": list(SEEDS),
            "worst_seed_speedup": min(
                result["timing"]["triton_over_native_speedup_from_p50"]
                for result in seed_results
            ),
            "worst_seed_paired_speedup": min(
                result["timing"]["paired_triton_over_native_speedup_p50"]
                for result in seed_results
            ),
            "worst_seed_native_win_fraction": min(
                result["timing"]["native_paired_win_fraction"]
                for result in seed_results
            ),
            "maximum_seed_output_relative_rms": max(
                result["correctness"]["native_vs_triton_output_relative_rms"]
                for result in seed_results
            ),
            "all_seed_cells_passed": seed_cells_passed,
            "cross_seed_paired_bootstrap": bootstrap,
            "cross_seed_p50_delta_ci_upper_strictly_negative": p50_high < 0.0,
            "cross_seed_mean_delta_ci_upper_strictly_negative": mean_high < 0.0,
        }
        cell["performance_gate_passed"] = all(
            (
                seed_cells_passed,
                cell["cross_seed_p50_delta_ci_upper_strictly_negative"],
                cell["cross_seed_mean_delta_ci_upper_strictly_negative"],
            )
        )
        logical_cells.append(cell)

    failed = [
        cell["logical_name"]
        for cell in logical_cells
        if not cell["performance_gate_passed"]
    ]
    return {
        "logical_cells": len(logical_cells),
        "seed_cells": len(results),
        "passed_logical_cells": len(logical_cells) - len(failed),
        "failed_logical_cells": failed,
        "minimum_seed_speedup": min(
            cell["worst_seed_speedup"] for cell in logical_cells
        ),
        "minimum_seed_native_win_fraction": min(
            cell["worst_seed_native_win_fraction"] for cell in logical_cells
        ),
        "maximum_output_relative_rms": max(
            cell["maximum_seed_output_relative_rms"] for cell in logical_cells
        ),
        "cells": logical_cells,
    }


def _append_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run_gpu(output: Path) -> dict[str, Any]:
    checkpoint = output.with_name("partial-results.jsonl")
    if output.exists() or checkpoint.exists():
        raise RuntimeError("refusing to overwrite result or checkpoint evidence")
    if _control_environment() != REQUIRED_ROUTE_ENV:
        raise RuntimeError(
            "routing environment must select the pure Triton comparator before "
            f"runtime import: expected {REQUIRED_ROUTE_ENV}, got "
            f"{_control_environment()}"
        )

    runtime = _load_runtime()
    torch = runtime.torch
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("the formal PR #4683 benchmark requires a ROCm GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("select exactly one visible ROCm GPU")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = str(getattr(properties, "gcnArchName", "unknown")).split(":", 1)[0]
    if (
        arch != EXPECTED_ARCH
        or properties.multi_processor_count != EXPECTED_COMPUTE_UNITS
    ):
        raise RuntimeError(
            "acceptance requires the 256-CU gfx950 reference target; "
            f"got {arch}/{properties.multi_processor_count} CU"
        )
    results: list[dict[str, Any]] = []
    active_spec: ShapeSpec | None = None
    active_seed: int | None = None
    phase = "seed-cell-timing"
    started = time.time()
    try:
        with torch.inference_mode():
            for seed in SEEDS:
                for index, spec in enumerate(SHAPES, start=1):
                    active_spec = spec
                    active_seed = seed
                    print(
                        f"seed={seed} shape={index}/{len(SHAPES)} {spec.name}",
                        flush=True,
                    )
                    result = _run_cell(runtime, spec, seed)
                    results.append(result)
                    _append_checkpoint(
                        checkpoint,
                        {
                            "schema": SCHEMA,
                            "event": "seed-cell-complete",
                            "complete": False,
                            "completed_seed_cells": len(results),
                            "total_seed_cells": len(SHAPES) * len(SEEDS),
                            "result": result,
                        },
                    )
                    timing = result["timing"]
                    print(
                        f"  HIP {timing['summaries'][NATIVE_BACKEND]['p50_ms']:.4f} ms; "
                        f"Triton {timing['summaries'][TRITON_BACKEND]['p50_ms']:.4f} ms; "
                        f"speedup {timing['triton_over_native_speedup_from_p50']:.3f}x; "
                        f"wins {timing['native_paired_win_fraction']:.1%}; "
                        "rRMS "
                        f"{result['correctness']['native_vs_triton_output_relative_rms']:.3e}; "
                        f"gate {'PASS' if result['performance_gate_passed'] else 'FAIL'}",
                        flush=True,
                    )

        phase = "provenance"
        provenance = _runtime_provenance(runtime)
        if provenance["active_route_control_environment"] != REQUIRED_ROUTE_ENV:
            raise RuntimeError("route-control environment changed during the run")
        phase = "cross-seed-analysis"
        cross_seed = _cross_seed_summary(results)
        performance_passed = not cross_seed["failed_logical_cells"]
        payload = {
            "schema": SCHEMA,
            "source_pr": "https://github.com/ROCm/aiter/pull/4683",
            "source_table_pr_head": SOURCE_TABLE_PR_HEAD,
            "started_unix": started,
            "finished_unix": time.time(),
            "configuration": {
                "seeds": list(SEEDS),
                "warmup_rounds": WARMUP,
                "measured_rounds": REPEAT,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "correctness_tolerance": CORRECTNESS_TOLERANCE,
                "minimum_native_speedup": MIN_NATIVE_SPEEDUP,
                "minimum_native_paired_win_fraction_exclusive": (
                    MIN_PAIRED_WIN_FRACTION
                ),
                "paired_delta_ci_requirements": {
                    "p50_95pct_ci_upper_strictly_negative": True,
                    "mean_95pct_ci_upper_strictly_negative": True,
                },
                "execution": "eager",
                "paired_order": "alternating-position-balanced",
                "native_backend": NATIVE_BACKEND,
                "triton_backend": TRITON_BACKEND,
                "triton_forced_direct_internal_import": True,
                "triton_gluon_disabled": True,
            },
            "plan": _plan(),
            "plan_sha256": _plan_sha256(_plan()),
            "published_pr_table": PUBLISHED_PR_TABLE,
            "published_values_used_as_acceptance_threshold": False,
            "environment": provenance,
            "results": results,
            "cross_seed_summary": cross_seed,
            "performance_gate": {
                "evaluated": True,
                "passed": performance_passed,
                "definition": (
                    "Every shape/seed cell requires both ratio-of-p50 and paired "
                    "p50 native speedup >= 1.03, paired native win fraction > "
                    "0.5, paired p50 and mean delta 95% CI upper bounds < 0, "
                    "and output rRMS <= 0.04; each logical shape also requires "
                    "hierarchical cross-seed p50 and mean delta 95% CI upper "
                    "bounds < 0."
                ),
                "failed_logical_cells": cross_seed["failed_logical_cells"],
            },
            "performance_gate_evaluated": True,
            "performance_gate_passed": performance_passed,
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
                "total_seed_cells": len(SHAPES) * len(SEEDS),
                "active_seed": active_seed,
                "active_logical_cell": (
                    None if active_spec is None else active_spec.name
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
        if output.exists():
            raise RuntimeError(f"refusing to overwrite output: {output}")
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
    print(rendered, end="", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--static-self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.print_plan and args.static_self_test:
        parser.error("choose one CPU-only mode")
    if not args.print_plan and not args.static_self_test and args.output is None:
        parser.error("GPU execution requires --output to preserve gate evidence")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.static_self_test:
        _emit(_static_self_test(), args.output)
        return
    if args.print_plan:
        _emit(_plan(), args.output)
        return

    assert args.output is not None
    payload = _run_gpu(args.output)
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
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    if payload["performance_gate_passed"] is not True:
        failed = payload["performance_gate"]["failed_logical_cells"]
        _append_checkpoint(
            checkpoint,
            {
                "schema": SCHEMA,
                "event": "run-failed",
                "complete": False,
                "capture_complete": True,
                "result_json_written": True,
                "phase": "performance-gate",
                "completed_seed_cells": len(payload["results"]),
                "total_seed_cells": len(SHAPES) * len(SEEDS),
                "performance_gate_passed": False,
                "failed_logical_cells": failed,
            },
        )
        raise SystemExit(
            "formal PR #4683 performance gate failed for: " + ", ".join(failed)
        )
    _append_checkpoint(
        checkpoint,
        {
            "schema": SCHEMA,
            "event": "run-complete",
            "complete": True,
            "capture_complete": True,
            "completed_seed_cells": len(payload["results"]),
            "total_seed_cells": len(SHAPES) * len(SEEDS),
            "result_json_written": True,
            "performance_gate_passed": True,
        },
    )


if __name__ == "__main__":
    main()
